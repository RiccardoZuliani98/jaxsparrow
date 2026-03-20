"""
setup_dense_solver.py
=====================
Dense differentiable QP solver.

Thin wrapper around :func:`solver_common.build_solver` that supplies
dense-specific converters and creates the dense numpy solver /
differentiator callables.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import (
    DEFAULT_CONSTRUCTOR_OPTIONS,
    ConstructorOptions,
    ConstructorOptionsFull,
)
from jaxsparrow._solver_dense._solvers import create_dense_qp_solver, DenseQPSolverFn
from jaxsparrow._solver_dense._differentiators import (
    create_dense_kkt_differentiator_fwd,
    create_dense_kkt_differentiator_rev,
    DenseKKTDifferentiatorFwd,
    DenseKKTDifferentiatorRev,
)
from jaxsparrow._solver_dense._types import DenseQPIngredientsNP
from jaxsparrow._solver_common import (
    build_solver, 
    make_expected_shapes, 
    compute_required_keys, 
    compute_dynamic_keys,
)
from jaxsparrow._solver_dense._converters import (
    dense_primal_converter,
    dense_tangent_converter,
    dense_grad_to_jax,
)
from jaxsparrow._utils._fd_recorder import FiniteDifferenceRecorder


logger: logging.Logger = logging.getLogger(__name__)


def setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: Optional[DenseQPIngredientsNP] = None,
    options: Optional[ConstructorOptions] = None,
):
    """Set up a differentiable dense QP solver.

    This is the main entry point for creating a JAX-compatible dense
    QP solver with forward- and reverse-mode differentiation. It
    validates inputs, creates the NumPy-level solver and KKT
    differentiator closures, and delegates to
    :func:`build_solver` to produce the final JAX custom-VJP
    primitive.

    The returned object can be called directly with JAX arrays and
    supports ``jax.grad``, ``jax.jacobian``, and ``jax.vmap``.

    Args:
        n_var: Number of decision variables.
        n_ineq: Number of inequality constraints (``G x <= h``).
            Zero if there are no inequality constraints.
        n_eq: Number of equality constraints (``A x = b``).
            Zero if there are no equality constraints.
        fixed_elements: QP parameters that are constant across calls
            and should not be differentiated through. Keys are any
            subset of ``{"P", "q", "A", "b", "G", "h"}``. Fixed
            parameters are baked into the solver/differentiator
            closures at construction time. Remaining parameters must
            be supplied as dynamic arguments at each call.
        options: Constructor-level options controlling the solver
            backend, differentiator backend, dtype, and optional
            finite-difference checking. Missing keys are filled from
            ``DEFAULT_CONSTRUCTOR_OPTIONS``.

    Returns:
        A JAX-compatible callable that solves the QP and supports
        automatic differentiation. See :func:`build_solver` for
        the full return type and calling convention.

    Raises:
        ValueError: If ``solver_type`` or ``differentiator_type``
            in *options* is not a supported value.
        AssertionError: If any fixed element has a shape that does
            not match the expected shape for the given problem
            dimensions.
    """

    # ── Parse options ────────────────────────────────────────────────
    options_parsed: ConstructorOptionsFull = parse_options(options, DEFAULT_CONSTRUCTOR_OPTIONS)

    # ── Validate fixed elements ──────────────────────────────────────
    expected_shapes: dict[str, tuple[int, ...]] = make_expected_shapes(n_var, n_eq, n_ineq)
    fixed_keys_set: set[str] = set()

    if fixed_elements is not None:
        fixed_keys_set = set(fixed_elements.keys())
        for key, val in fixed_elements.items():
            assert val.shape == expected_shapes[key]  # type: ignore

    # gather all required keys
    required_keys: Sequence[str] = compute_required_keys(n_eq, n_ineq)

    # gather keys required at runtime (i.e., not fixed).
    # Only these flow through JAX's traced path.
    dynamic_keys: Sequence[str] = compute_dynamic_keys(required_keys, fixed_keys_set)

    # ── Create numpy solver ──────────────────────────────────────────
    solve_qp_numpy: DenseQPSolverFn
    if options_parsed["solver_type"] == "qp_solvers":
        solve_qp_numpy = create_dense_qp_solver(
            n_eq=n_eq,
            n_ineq=n_ineq,
            options=options_parsed["solver"],
            fixed_elements=fixed_elements,
        )
    else:
        raise ValueError("Only qp_solvers is available as 'solver_type'")
 
    # ── Create differentiators ───────────────────────────────────────
    diff_forward_numpy: DenseKKTDifferentiatorFwd
    diff_reverse_numpy: DenseKKTDifferentiatorRev
    if options_parsed["differentiator_type"] in ("kkt_fwd", "kkt_rev"):
        diff_forward_numpy = create_dense_kkt_differentiator_fwd(
            n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
            options=options_parsed["differentiator"],
            fixed_elements=fixed_elements,
        )
        diff_reverse_numpy = create_dense_kkt_differentiator_rev(
            n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
            options=options_parsed["differentiator"],
            fixed_elements=fixed_elements,
            dynamic_keys=dynamic_keys,
        )
    else:
        raise ValueError("Only differentiator available is 'kkt'")
 
    # ── FD recorder ──────────────────────────────────────────────────
    fd_check: FiniteDifferenceRecorder = FiniteDifferenceRecorder(
        enabled=options_parsed.get("fd_check", False),
        eps=options_parsed.get("fd_eps", 1e-6),
    )
 
    # ── Log ──────────────────────────────────────────────────────────
    logger.info(
        f"Setting up dense QP with {n_var} variables, "
        f"{n_eq} equalities, {n_ineq} inequalities."
    )
    logger.info(f"Fixed variables: {fixed_keys_set or 'none'}")
    logger.info(f"Dynamic variables: {dynamic_keys or 'none'}")
 
    # ── Delegate to common builder ───────────────────────────────────
    return build_solver(
        n_var=n_var,
        n_ineq=n_ineq,
        n_eq=n_eq,
        options_parsed=options_parsed,
        fixed_keys_set=fixed_keys_set,
        solve_qp_numpy=solve_qp_numpy,
        diff_forward_numpy=diff_forward_numpy,
        diff_reverse_numpy=diff_reverse_numpy,
        primal_converter=dense_primal_converter,
        tangent_converter=dense_tangent_converter,
        grad_to_jax=dense_grad_to_jax,
        fd_check=fd_check,
    )