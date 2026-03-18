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
from typing import Optional

from src.utils.parsing_utils import parse_options
from src.options_common import (
    DEFAULT_CONSTRUCTOR_OPTIONS,
    ConstructorOptions,
)
from src.solver_dense.solvers import create_dense_qp_solver
from src.solver_dense.differentiators import (
    create_dense_kkt_differentiator_fwd,
    create_dense_kkt_differentiator_rev,
)
from src.solver_dense.types import DenseQPIngredientsNP
from src.solver_common import build_solver
from src.solver_dense.converters import (
    dense_primal_converter,
    dense_tangent_converter,
    dense_grad_to_jax
)
from src.utils.fd_recorder import FiniteDifferenceRecorder


def setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: Optional[DenseQPIngredientsNP] = None,
    options: Optional[ConstructorOptions] = None,
):
    logger = logging.getLogger(__name__)

    # ── Parse options ────────────────────────────────────────────────
    options_parsed = parse_options(options, DEFAULT_CONSTRUCTOR_OPTIONS)

    # ── Validate fixed elements ──────────────────────────────────────
    expected_shapes: dict[str, tuple[int, ...]] = {
        "P": (n_var, n_var),
        "q": (n_var,),
        "A": (n_eq, n_var),
        "b": (n_eq,),
        "G": (n_ineq, n_var),
        "h": (n_ineq,),
    }
    fixed_keys_set: set[str] = set()

    if fixed_elements is not None:
        fixed_keys_set = set(fixed_elements.keys())
        for key, val in fixed_elements.items():
            assert val.shape == expected_shapes[key]  # type: ignore

    # gather all required keys
    required_keys: tuple[str,...] = ("P", "q")
    if n_eq > 0:
        required_keys += ("A", "b")
    if n_ineq > 0:
        required_keys += ("G", "h")

    # gather keys required at runtime (i.e., not fixed).
    # Only these flow through JAX's traced path.
    dynamic_keys = tuple(k for k in required_keys if k not in fixed_keys_set)

    # ── Create numpy solver ──────────────────────────────────────────
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
    fd_check = FiniteDifferenceRecorder(
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