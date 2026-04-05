"""
solver_sparse/setup.py
======================
Sparse differentiable solver.

Thin wrapper around :func:`solver_common.build_solver` that supplies
sparse-specific converters and creates the sparse numpy solver /
differentiator callables.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, Dict, Any

import numpy as np
import jax
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix

from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._utils._sparse_utils import bcoo_to_csc
from jaxsparrow._options_common import (
    DEFAULT_CONSTRUCTOR_OPTIONS,
    ConstructorOptions,
)
from jaxsparrow._solver_sparse._solvers import (
    create_sparse_solver,
    _resolve_backend_defaults as _resolve_solver_defaults,
)
from jaxsparrow._solver_sparse._differentiators import (
    create_sparse_kkt_differentiator_fwd,
    create_sparse_kkt_differentiator_rev,
)
from jaxsparrow._solver_sparse._types import SparseIngredientsNP, SparseIngredients
from jaxsparrow._solver_sparse._converters import (
    build_sparsity_info,
    is_sparse_key,
    make_sparse_primal_converter,
    make_sparse_tangent_converter,
    make_sparse_residual_extractor,
    make_sparse_residual_reconstructor,
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
    fixed_elements: Optional[Union[SparseIngredientsNP, SparseIngredients, Dict[str, jax.Array | BCOO]]] = None,
    options: Optional[ConstructorOptions | Dict[str, Any]] = None,
):
    """Build a differentiable sparse solver.

    Args:
        n_var: Number of decision variables.
        n_ineq: Number of inequality constraints.
        n_eq: Number of equality constraints.
        sparsity_patterns: BCOO matrices encoding sparsity structure
            for each dynamic sparse key.
        fixed_elements: Parameters constant across calls.
        options: Constructor-level options.

    Returns:
        A JAX-compatible callable supporting automatic differentiation.
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

        # ── Convert JAX types to NumPy/SciPy if needed ───────────────
        _, solver_defaults = _resolve_solver_defaults(
            options_parsed["solver"],
        )
        solver_options_parsed = parse_options(
            options_parsed["solver"], solver_defaults,
        )
        assert "dtype" in solver_options_parsed
        _solver_dtype = solver_options_parsed["dtype"]

        fixed_elements_converted: SparseIngredientsNP = {}
        for key, val in fixed_elements.items():
            if isinstance(val, BCOO):
                fixed_elements_converted[key] = bcoo_to_csc(
                    val, dtype=_solver_dtype,
                )
            elif isinstance(val, jax.Array):
                fixed_elements_converted[key] = np.asarray(
                    val, dtype=_solver_dtype,
                )
            else:
                fixed_elements_converted[key] = val

        fixed_elements = fixed_elements_converted

        fixed_keys_set = set(fixed_elements.keys())
        for key, val in fixed_elements.items():
            shape = val.shape  # type: ignore
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
    primal_converter = make_sparse_primal_converter(sparsity_info)
    tangent_converter = make_sparse_tangent_converter(sparsity_info)
    residual_extractor = make_sparse_residual_extractor(sparsity_info)
    residual_reconstructor = make_sparse_residual_reconstructor(sparsity_info)

    # ── VJP backward shapes ─────────────────────────────────────────
    # Gradient shapes: (nnz,) for sparse keys, full shape for dense.
    vjp_bwd_shapes: dict[str, jax.ShapeDtypeStruct] = {}
    for k in dynamic_keys:
        if is_sparse_key(k) and k in sparsity_info:
            nnz = sparsity_info[k]["nnz"]
            vjp_bwd_shapes[k] = jax.ShapeDtypeStruct((nnz,), _dtype)
        else:
            vjp_bwd_shapes[k] = jax.ShapeDtypeStruct(
                expected_shapes[k], _dtype,
            )

    # ── VJP residual shapes ──────────────────────────────────────────
    # Residual shapes: (nnz,) for sparse keys (only .data is saved),
    # full shape for dense keys.
    vjp_residual_shapes: dict[str, jax.ShapeDtypeStruct] = {}
    for k in dynamic_keys:
        if is_sparse_key(k) and k in sparsity_info:
            nnz = sparsity_info[k]["nnz"]
            vjp_residual_shapes[k] = jax.ShapeDtypeStruct((nnz,), _dtype)
        else:
            vjp_residual_shapes[k] = jax.ShapeDtypeStruct(
                expected_shapes[k], _dtype,
            )

    # ── Create numpy solver ──────────────────────────────────────────
    solver_numpy = create_sparse_solver(
        n_eq=n_eq,
        n_ineq=n_ineq,
        options=options_parsed["solver"],
        fixed_elements=fixed_elements,
    )

    # ── Create differentiators ───────────────────────────────────────
    diff_forward_numpy = create_sparse_kkt_differentiator_fwd(
        n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
        options=options_parsed["differentiator"],
        fixed_elements=fixed_elements,
    )
    diff_reverse_numpy = create_sparse_kkt_differentiator_rev(
        n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
        options=options_parsed["differentiator"],
        fixed_elements=fixed_elements,
        dynamic_keys=dynamic_keys,
        sparsity_info=sparsity_info,
    )

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
        vjp_bwd_shapes=vjp_bwd_shapes,
        residual_extractor=residual_extractor,
        residual_reconstructor=residual_reconstructor,
        vjp_residual_shapes=vjp_residual_shapes,
        fd_check=fd_check,
    )