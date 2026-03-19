"""
solver_sparse/setup.py
======================
Sparse differentiable QP solver.

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

from src.utils.parsing_utils import parse_options
from src.options_common import (
    DEFAULT_CONSTRUCTOR_OPTIONS,
    ConstructorOptions,
)
from src.solver_sparse.solvers import create_sparse_qp_solver
from src.solver_sparse.differentiators import (
    create_sparse_kkt_differentiator_fwd,
    create_sparse_kkt_differentiator_rev,
)
from src.solver_sparse.types import SparseQPIngredientsNP
from src.solver_sparse.converters import (
    build_sparsity_info,
    is_sparse_key,
    make_sparse_primal_converter,
    make_sparse_tangent_converter,
    make_sparse_grad_to_jax_forward,
    make_sparse_grad_to_jax_reverse
)
from src.solver_common import (
    build_solver,
    make_expected_shapes,
    compute_required_keys,
    compute_dynamic_keys,
)
from src.utils.fd_recorder import FiniteDifferenceRecorder


def setup_sparse_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    sparsity_patterns: Optional[dict[str, BCOO]] = None,
    fixed_elements: Optional[SparseQPIngredientsNP] = None,
    options: Optional[ConstructorOptions] = None,
):
    """
    Build a differentiable sparse QP solver.

    Parameters
    ----------
    n_var, n_ineq, n_eq : int
        Problem dimensions.
    sparsity_patterns : dict[str, BCOO]
        Sparsity patterns for sparse keys ("P", and optionally "A", "G").
        Only the ``.indices`` are used; values are ignored.
        Must be provided for every matrix key that is *dynamic* (not fixed).
    fixed_elements : SparseQPIngredientsNP, optional
        QP ingredients that are constant across calls.  Matrices should
        be ``scipy.sparse.csc_matrix``; vectors should be ``ndarray``.
    options : ConstructorOptions, optional
        Solver and differentiator options.

    Returns
    -------
    solver
        Callable with signature ``solver(*, P=..., q=..., ...) -> QPOutput``.
        Matrices (P, A, G) should be passed as JAX ``BCOO``; vectors
        (q, b, h) as regular ``jax.Array``.
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
        raise KeyError("Allowed differentiator keys are 'kkt_rev' and 'kkt_fwd', " \
            f"got {options_parsed["differentiator_type"]}")

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
    if options_parsed["solver_type"] == "qp_solvers":
        solve_qp_numpy = create_sparse_qp_solver(
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
        f"Setting up sparse QP with {n_var} variables, "
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
        solve_qp_numpy=solve_qp_numpy,
        diff_forward_numpy=diff_forward_numpy,
        diff_reverse_numpy=diff_reverse_numpy,
        primal_converter=primal_converter,
        tangent_converter=tangent_converter,
        grad_to_jax=grad_to_jax,
        vjp_bwd_shapes=vjp_bwd_shapes,
        fd_check=fd_check,
    )
