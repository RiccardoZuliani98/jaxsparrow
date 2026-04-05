"""
solver_sparse/setup.py
======================
Sparse differentiable solver.

Thin wrapper around :func:`build_solver` that supplies sparse-specific
stacker/unstacker functions and creates the sparse numpy solver /
differentiator callables.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, Dict, Any

import numpy as np
import jax
from jax.experimental.sparse import BCOO

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
    make_sparse_stacker,
    make_sparse_unstacker_primals,
    make_sparse_unstacker_tangents,
    make_fwd_diff_stacker,
    make_rev_diff_stacker,
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
    """Build a differentiable sparse solver."""
    logger = logging.getLogger(__name__)

    # ── Parse options ────────────────────────────────────────────────
    options_parsed = parse_options(options, DEFAULT_CONSTRUCTOR_OPTIONS)
    _dtype = options_parsed["dtype"]

    if sparsity_patterns is None:
        sparsity_patterns = {}

    expected_shapes = make_expected_shapes(n_var, n_eq, n_ineq)
    fixed_keys_set: set[str] = set()

    if fixed_elements is not None:
        _, solver_defaults = _resolve_solver_defaults(
            options_parsed["solver"],
        )
        solver_options_parsed = parse_options(
            options_parsed["solver"], solver_defaults,
        )
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

    for key, pattern in sparsity_patterns.items():
        assert tuple(pattern.shape) == expected_shapes[key], (
            f"Sparsity pattern '{key}' has shape {pattern.shape}, "
            f"expected {expected_shapes[key]}"
        )

    required_keys = compute_required_keys(n_eq, n_ineq)
    dynamic_keys = compute_dynamic_keys(required_keys, fixed_keys_set)

    for k in dynamic_keys:
        if is_sparse_key(k) and k not in sparsity_patterns:
            raise ValueError(
                f"Dynamic sparse key '{k}' requires a sparsity pattern."
            )

    # ── Build sparsity info ──────────────────────────────────────────
    sparsity_info = build_sparsity_info(sparsity_patterns)

    # ── Build stacker / unstackers ───────────────────────────────────
    stack_fn, packed_length, layout = make_sparse_stacker(
        dynamic_keys, sparsity_info, expected_shapes,
    )
    unstack_primals_fn = make_sparse_unstacker_primals(
        dynamic_keys, layout, sparsity_info,
    )
    unstack_tangents_fn = make_sparse_unstacker_tangents(
        dynamic_keys, layout, sparsity_info,
    )

    # Forward-diff packer (primals + tangents → one vector)
    fwd_stack_fn, fwd_unstack_fn = make_fwd_diff_stacker(packed_length)

    # Reverse-diff packer (residuals + sol + cotangents → one vector)
    rev_stack_fn, rev_unstack_fn, rev_total_length = make_rev_diff_stacker(
        packed_length, n_var, n_ineq, n_eq,
    )

    # ── VJP backward shapes ─────────────────────────────────────────
    vjp_bwd_shapes: dict[str, jax.ShapeDtypeStruct] = {}
    for k in dynamic_keys:
        if is_sparse_key(k) and k in sparsity_info:
            nnz = sparsity_info[k]["nnz"]
            vjp_bwd_shapes[k] = jax.ShapeDtypeStruct((nnz,), _dtype)
        else:
            vjp_bwd_shapes[k] = jax.ShapeDtypeStruct(
                expected_shapes[k], _dtype,
            )

    # ── Create numpy solver ──────────────────────────────────────────
    solver_numpy = create_sparse_solver(
        n_eq=n_eq, n_ineq=n_ineq,
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

    fd_check = FiniteDifferenceRecorder(
        enabled=options_parsed.get("fd_check", False),
        eps=options_parsed.get("fd_eps", 1e-6),
    )

    logger.info(
        f"Setting up sparse solver: {n_var} vars, "
        f"{n_eq} eq, {n_ineq} ineq. "
        f"Packed length: {packed_length}. "
        f"Rev total length: {rev_total_length}."
    )

    return build_solver(
        n_var=n_var,
        n_ineq=n_ineq,
        n_eq=n_eq,
        options_parsed=options_parsed,
        fixed_keys_set=fixed_keys_set,
        solver_numpy=solver_numpy,
        diff_forward_numpy=diff_forward_numpy,
        diff_reverse_numpy=diff_reverse_numpy,
        stack_fn=stack_fn,
        unstack_primals_fn=unstack_primals_fn,
        unstack_tangents_fn=unstack_tangents_fn,
        packed_length=packed_length,
        fwd_stack_fn=fwd_stack_fn,
        fwd_unstack_fn=fwd_unstack_fn,
        rev_stack_fn=rev_stack_fn,
        rev_unstack_fn=rev_unstack_fn,
        rev_total_length=rev_total_length,
        vjp_bwd_shapes=vjp_bwd_shapes,
        fd_check=fd_check,
    )