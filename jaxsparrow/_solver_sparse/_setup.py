"""
solver_sparse/setup.py
======================
Sparse differentiable solver.

Thin wrapper around :func:`solver_common.build_solver` that supplies
sparse-specific converters and creates the sparse numpy solver /
differentiator callables.

Usage
-----
>>> from jax.experimental.sparse import BCOO
>>> import jax.numpy as jnp
>>>
>>> # Define sparsity patterns (values don't matter, only structure)
>>> P_pattern = BCOO.fromdense(jnp.eye(3))
>>> G_pattern = BCOO.fromdense(jnp.ones((2, 3)))
>>>
>>> solver = setup_sparse_solver(
...     n_var=3, n_ineq=2,
...     sparsity_patterns={"P": P_pattern, "G": G_pattern},
... )
>>>
>>> # At solve time, pass BCOO matrices for P/G and dense arrays for q/h
>>> result = solver(P=P_bcoo, q=q_arr, G=G_bcoo, h=h_arr)
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
from jax.experimental.sparse import BCOO

from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import (
    DEFAULT_CONSTRUCTOR_OPTIONS,
    ConstructorOptions,
)
from jaxsparrow._solver_sparse._solvers import create_sparse_solver
from jaxsparrow._solver_sparse._differentiators import (
    create_sparse_kkt_differentiator_fwd,
    create_sparse_kkt_differentiator_rev,
)
from jaxsparrow._solver_sparse._types import SparseIngredientsNP
from jaxsparrow._solver_sparse._converters import (
    build_sparsity_info,
    is_sparse_key,
    make_sparse_primal_converter,
    make_sparse_tangent_converter,
    make_sparse_grad_to_jax_forward,
    make_sparse_grad_to_jax_reverse
)
from jaxsparrow._solver_common import (
    build_solver,
    make_expected_shapes,
    compute_required_keys,
    compute_dynamic_keys,
)
from jaxsparrow._utils._fd_recorder import FiniteDifferenceRecorder


def setup_sparse_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    sparsity_patterns: Optional[dict[str, BCOO]] = None,
    fixed_elements: Optional[SparseIngredientsNP] = None,
    options: Optional[ConstructorOptions] = None,
):
    """Build a differentiable sparse solver.

    Constructs a JAX-traceable callable that solves quadratic programs
    of the form::

        min  0.5 * x^T P x + q^T x
        s.t. A x = b
             G x <= h

    where P, A, G are sparse (JAX ``BCOO`` at call time, SciPy CSC
    internally) and q, b, h are dense vectors.

    The returned solver supports ``jax.jvp``, ``jax.vjp``, and
    ``jax.vmap`` via ``pure_callback`` with ``custom_jvp`` or
    ``custom_vjp``, depending on the configured differentiator mode.

    Sparsity patterns must be provided for every matrix key that will
    be passed dynamically (i.e., not fixed at setup). Only the
    ``.indices`` of each pattern are used; the ``.data`` values are
    ignored.

    Args:
        n_var: Number of decision variables.
        n_ineq: Number of inequality constraints. Zero if there are
            none.
        n_eq: Number of equality constraints. Zero if there are none.
        sparsity_patterns: Mapping from sparse key name (``"P"``, and
            optionally ``"A"``, ``"G"``) to a ``BCOO`` matrix encoding
            the sparsity structure. Required for every matrix key that
            is dynamic (not in *fixed_elements*).
        fixed_elements: ingredients that remain constant across
            calls. Matrices should be ``scipy.sparse.csc_matrix``;
            vectors should be ``ndarray``. Keys present here are
            excluded from JAX's traced path and should not be passed
            again at solve time.
        options: Solver and differentiator options. Unspecified keys
            are filled from ``DEFAULT_CONSTRUCTOR_OPTIONS``. Notable
            keys include ``"differentiator_type"`` (``"kkt_fwd"`` or
            ``"kkt_rev"``), ``"solver_type"``, ``"dtype"``,
            ``"fd_check"``, and ``"fd_eps"``.

    Returns:
        A solver callable with signature
        ``solver(*, P=..., q=..., ..., warmstart=None) -> SolverOutput``.
        Matrices (P, A, G) should be passed as JAX ``BCOO``; vectors
        (q, b, h) as regular ``jax.Array``. The returned ``SolverOutput``
        is a dict with keys ``"x"``, ``"lam"``, ``"mu"``.

        The callable also exposes ``.timings`` (a ``TimingRecorder``)
        and ``.fd_check`` (a ``FiniteDifferenceRecorder``) for
        diagnostics.

    Raises:
        ValueError: If a dynamic sparse key has no corresponding entry
            in *sparsity_patterns*.
        ValueError: If ``"differentiator_type"`` is unknown.
        ValueError: If ``"solver_type"`` is unknown.
        AssertionError: If any fixed element or sparsity pattern has a
            shape that doesn't match the declared problem dimensions.
    """
    logger = logging.getLogger(__name__)

    # ── Parse options ────────────────────────────────────────────────
    options_parsed = parse_options(options, DEFAULT_CONSTRUCTOR_OPTIONS)
    _dtype = options_parsed["dtype"]

    # ── Validate inputs ──────────────────────────────────────────────
    if sparsity_patterns is None:
        sparsity_patterns = {}

    expected_shapes = make_expected_shapes(n_var, n_eq, n_ineq)
    fixed_keys_set: set[str] = set()

    if fixed_elements is not None:
        fixed_keys_set = set(fixed_elements.keys())
        for key, val in fixed_elements.items():
            shape = val.shape  # works for both csc_matrix and ndarray #type:ignore
            assert shape == expected_shapes[key], (
                f"Fixed element '{key}' has shape {shape}, "
                f"expected {expected_shapes[key]}"
            )

    # Validate sparsity patterns
    for key, pattern in sparsity_patterns.items():
        assert tuple(pattern.shape) == expected_shapes[key], (
            f"Sparsity pattern '{key}' has shape {pattern.shape}, "
            f"expected {expected_shapes[key]}"
        )

    # Check that every dynamic sparse key has a sparsity pattern
    required_keys = compute_required_keys(n_eq, n_ineq)
    dynamic_keys = compute_dynamic_keys(required_keys, fixed_keys_set)

    for k in dynamic_keys:
        if is_sparse_key(k) and k not in sparsity_patterns:
            raise ValueError(
                f"Dynamic sparse key '{k}' requires a sparsity pattern. "
                f"Provide it in sparsity_patterns or fix it via fixed_elements."
            )

    # ── Build sparsity info ──────────────────────────────────────────
    sparsity_info = build_sparsity_info(sparsity_patterns)

    # ── Build converters ─────────────────────────────────────────────
    primal_converter  = make_sparse_primal_converter(sparsity_info)
    tangent_converter = make_sparse_tangent_converter(sparsity_info)
    if options_parsed["differentiator_type"] == "kkt_fwd":
        grad_to_jax   = make_sparse_grad_to_jax_forward(sparsity_info)
    elif options_parsed["differentiator_type"] == "kkt_rev":
        grad_to_jax   = make_sparse_grad_to_jax_reverse(sparsity_info)
    else:
        raise ValueError("Allowed differentiator keys are 'kkt_rev' and 'kkt_fwd', "
            f"got {options_parsed['differentiator_type']}")

    # ── VJP backward shapes ──────────────────────────────────────────
    # For sparse keys: gradient is w.r.t. the nnz nonzero values (1-D).
    # For dense keys:  gradient has the full expected shape.
    vjp_bwd_shapes: dict[str, jax.ShapeDtypeStruct] = {}
    for k in dynamic_keys:
        if is_sparse_key(k):
            if k not in sparsity_info:
                raise ValueError(
                    f"Key '{k}' is marked as sparse but has no entry "
                    f"in sparsity_info (available: {list(sparsity_info)})"
                )
            nnz = sparsity_info[k]["nnz"]
            vjp_bwd_shapes[k] = jax.ShapeDtypeStruct((nnz,), _dtype)
        else:
            vjp_bwd_shapes[k] = jax.ShapeDtypeStruct(
                expected_shapes[k], _dtype
            )

    # ── Create numpy solver ──────────────────────────────────────────
    #TODO: this should be improved. qp_solvers should be an option inside
    # create_sparse_solver
    if options_parsed["solver_type"] == "qp_solvers":
        solver_numpy = create_sparse_solver(
            n_eq=n_eq,
            n_ineq=n_ineq,
            options=options_parsed["solver"],
            fixed_elements=fixed_elements,
        )
    else:
        raise ValueError("Only qp_solvers is available as 'solver_type'")

    # ── Create differentiators ───────────────────────────────────────
    if options_parsed["differentiator_type"] in ("kkt_fwd", "kkt_rev"):
        diff_forward_numpy = create_sparse_kkt_differentiator_fwd(
            n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
            options=options_parsed["differentiator"],
            fixed_elements=fixed_elements,
        )
        diff_reverse_numpy = create_sparse_kkt_differentiator_rev(
            n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
            options=options_parsed["differentiator"],
            fixed_elements=fixed_elements,
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
        f"Setting up sparse solver with {n_var} variables, "
        f"{n_eq} equalities, {n_ineq} inequalities."
    )
    logger.info(f"Fixed variables: {fixed_keys_set or 'none'}")
    for k, si in sparsity_info.items():
        logger.info(
            f"Sparsity pattern '{k}': shape={si['shape']}, nnz={si['nnz']}"
        )

    # ── Delegate to common builder ───────────────────────────────────
    return build_solver(
        n_var=n_var,
        n_ineq=n_ineq,
        n_eq=n_eq,
        options_parsed=options_parsed,
        fixed_keys_set=fixed_keys_set,
        solver_numpy=solver_numpy,
        diff_forward_numpy=diff_forward_numpy,
        diff_reverse_numpy=diff_reverse_numpy,
        primal_converter=primal_converter,
        tangent_converter=tangent_converter,
        grad_to_jax=grad_to_jax,
        vjp_bwd_shapes=vjp_bwd_shapes,
        fd_check=fd_check,
    )