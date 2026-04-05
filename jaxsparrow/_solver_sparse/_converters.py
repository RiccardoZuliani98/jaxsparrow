"""
solver_sparse/converters.py
===========================
JAX ↔ numpy conversions for the sparse path.

All dynamic QP ingredients are packed into a single flat 1-D vector
on the JAX side before passing through ``pure_callback``.  This
reduces ``device_put`` overhead from N calls to 1.

Three packing layouts exist:

- **Solve**: ingredients only (P.data, q, A.data, b, G.data, h).
- **Forward diff**: primals ++ tangents (two copies of the solve
  layout, concatenated).  Tangents may be batched ``(batch, len)``.
- **Reverse diff**: residuals ++ solution ++ cotangents.  Solution
  is ``x ++ lam ++ mu``, cotangents are ``g_x ++ g_lam ++ g_mu``.
  Cotangents may be batched.

The stacker runs on the JAX side (``jnp.concatenate``), the
unstacker runs inside the callback (numpy slicing + CSC rebuild).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy import ndarray
from jax import Array
from jax.experimental.sparse import BCOO
import jax
import jax.numpy as jnp
from scipy.sparse import csc_matrix

from jaxsparrow._solver_common import EXPECTED_NDIM

# ── Which keys are sparse (matrices) vs dense (vectors) ─────────────

SPARSE_KEYS: frozenset[str] = frozenset({"P", "A", "G"})

def is_sparse_key(key: str) -> bool:
    return key in SPARSE_KEYS


# ── Sparsity info ────────────────────────────────────────────────────

SparsityEntry = dict
SparsityInfo  = dict[str, SparsityEntry]


def build_sparsity_info(
    sparsity_patterns: dict[str, BCOO],
) -> SparsityInfo:
    """Extract index arrays from BCOO patterns. Used at construction."""
    info: SparsityInfo = {}
    for key, bcoo in sparsity_patterns.items():
        if bcoo.ndim != 2:
            raise ValueError(
                f"Sparsity pattern for '{key}' must be a 2-D matrix, "
                f"got {bcoo.ndim}-D array with shape {bcoo.shape}"
            )
        indices = np.asarray(bcoo.indices)
        info[key] = {
            "rows":  indices[:, 0].astype(np.int32),
            "cols":  indices[:, 1].astype(np.int32),
            "shape": tuple(bcoo.shape),
            "nnz":   indices.shape[0],
        }
    return info


# ── Layout computation ───────────────────────────────────────────────

def _compute_ingredient_layout(
    dynamic_keys: Sequence[str],
    sparsity_info: SparsityInfo,
    expected_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, dict[str, tuple[int, int, tuple[int, ...]]]]:
    """Compute the flat packing layout for QP ingredients.

    Returns:
        ``(total_length, layout)`` where layout maps each key to
        ``(start, end, original_shape)``.  For sparse keys the
        shape is ``(nnz,)``; for dense keys it's the expected shape.
    """
    layout: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    offset = 0
    for k in dynamic_keys:
        if is_sparse_key(k) and k in sparsity_info:
            n = sparsity_info[k]["nnz"]
            shape = (n,)
        else:
            shape = expected_shapes[k]
            n = 1
            for s in shape:
                n *= s
        layout[k] = (offset, offset + n, shape)
        offset += n
    return offset, layout


# ── JAX-side stacker ─────────────────────────────────────────────────

def make_sparse_stacker(
    dynamic_keys: Sequence[str],
    sparsity_info: SparsityInfo,
    expected_shapes: dict[str, tuple[int, ...]],
):
    """Create a JAX-side stacker and compute the packing layout.

    Returns:
        ``(stack_fn, ingredient_length, layout)``
    """
    ingredient_length, layout = _compute_ingredient_layout(
        dynamic_keys, sparsity_info, expected_shapes,
    )

    def stack_fn(
        keys: Sequence[str],
        vals: Sequence[Any],
    ) -> jax.Array:
        """Pack dynamic JAX vals into a single flat vector."""
        parts = []
        for k, v in zip(keys, vals):
            if isinstance(v, BCOO):
                parts.append(v.data.ravel())
            else:
                parts.append(v.ravel())
        return jnp.concatenate(parts)

    return stack_fn, ingredient_length, layout


# ── Numpy-side unstackers ────────────────────────────────────────────

def make_sparse_unstacker_primals(
    dynamic_keys: Sequence[str],
    layout: dict[str, tuple[int, int, tuple[int, ...]]],
    sparsity_info: SparsityInfo,
):
    """Create unstacker: flat 1-D numpy vector → dict of primals."""

    def unstack_fn(flat: ndarray, dtype) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k in dynamic_keys:
            start, end, shape = layout[k]
            chunk = flat[start:end]
            if is_sparse_key(k) and k in sparsity_info:
                si = sparsity_info[k]
                result[k] = csc_matrix(
                    (chunk, (si["rows"], si["cols"])),
                    shape=si["shape"],
                )
            else:
                result[k] = chunk.reshape(shape)
        return result

    return unstack_fn


def make_sparse_unstacker_tangents(
    dynamic_keys: Sequence[str],
    layout: dict[str, tuple[int, int, tuple[int, ...]]],
    sparsity_info: SparsityInfo,
):
    """Create unstacker: flat numpy vector → dict of tangents.

    Handles both unbatched ``(total,)`` and batched ``(batch, total)``.
    """

    def unstack_fn(flat: ndarray, dtype) -> dict[str, Any]:
        result: dict[str, Any] = {}
        batched = flat.ndim == 2

        for k in dynamic_keys:
            start, end, shape = layout[k]

            if batched:
                chunk = flat[:, start:end]
            else:
                chunk = flat[start:end]

            if is_sparse_key(k) and k in sparsity_info:
                si = sparsity_info[k]
                rows, cols, sp_shape = si["rows"], si["cols"], si["shape"]
                if batched:
                    result[k] = [
                        csc_matrix((chunk[i], (rows, cols)), shape=sp_shape)
                        for i in range(chunk.shape[0])
                    ]
                else:
                    result[k] = csc_matrix(
                        (chunk, (rows, cols)), shape=sp_shape,
                    )
            else:
                if batched:
                    result[k] = chunk.reshape(chunk.shape[0], *shape)
                else:
                    result[k] = chunk.reshape(shape)

        return result

    return unstack_fn


# ── Full-callback packers ────────────────────────────────────────────
#
# These build the JAX-side and numpy-side functions for packing
# ALL arguments of each callback into a single flat vector.

def make_fwd_diff_stacker(ingredient_length: int):
    """Create JAX-side stacker for forward diff callback.

    Packs ``[packed_primals, packed_tangents]`` into one vector.
    The first ``ingredient_length`` elements are primals, the rest
    are tangents.
    """
    _il = ingredient_length

    def stack_fwd(packed_primals: jax.Array, packed_tangents: jax.Array) -> jax.Array:
        return jnp.concatenate([packed_primals, packed_tangents])

    def unstack_fwd(flat: ndarray):
        """Split into primals (always 1-D) and tangents (1-D or 2-D).

        With ``legacy_vectorized``, the flat vector is:
        - unbatched: ``(2 * ingredient_length,)``
        - batched: ``(batch, 2 * ingredient_length)``

        Primals are identical across the batch, so we squeeze them.
        """
        if flat.ndim == 2:
            primals = flat[0, :_il]        # squeeze batch
            tangents = flat[:, _il:]       # keep batch
        else:
            primals = flat[:_il]
            tangents = flat[_il:]
        return primals, tangents

    return stack_fwd, unstack_fwd


def make_rev_diff_stacker(
    ingredient_length: int,
    n_var: int,
    n_ineq: int,
    n_eq: int,
):
    """Create JAX-side stacker for reverse diff callback.

    Packs ``[packed_residuals, x, lam, mu, g_x, g_lam, g_mu]`` into
    one vector.

    Layout::

        [0 .. il)              packed_residuals  (ingredient_length)
        [il .. il+nv)          x_sol             (n_var)
        [il+nv .. il+nv+ni)    lam_sol           (n_ineq)
        [il+nv+ni .. il+nv+ni+ne)  mu_sol        (n_eq)
        [... .. ...+nv)        g_x               (n_var)
        [... .. ...+ni)        g_lam             (n_ineq)
        [... .. ...+ne)        g_mu              (n_eq)
    """
    _il = ingredient_length
    _nv, _ni, _ne = n_var, n_ineq, n_eq
    _sol_len = _nv + _ni + _ne
    _residual_and_sol_len = _il + _sol_len
    _total_len = _residual_and_sol_len + _sol_len  # cotangents same size as sol

    def stack_rev(
        packed_residuals: jax.Array,
        x_sol: jax.Array,
        lam_sol: jax.Array,
        mu_sol: jax.Array,
        g_x: jax.Array,
        g_lam: jax.Array,
        g_mu: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate([
            packed_residuals, x_sol, lam_sol, mu_sol,
            g_x, g_lam, g_mu,
        ])

    def unstack_rev(flat: ndarray, dtype):
        """Split the flat vector back into components.

        With ``legacy_vectorized``, the flat vector is:
        - unbatched: ``(total_len,)``
        - batched: ``(batch, total_len)``

        Residuals and solution are identical across the batch
        (squeeze).  Cotangents keep the batch dim.
        """
        if flat.ndim == 2:
            # Batched: residuals/sol shared, cotangents batched
            flat_1d = flat[0]  # for residuals and solution
            batch = flat.shape[0]

            packed_res = flat_1d[:_il].astype(dtype)
            o = _il
            x_np   = flat_1d[o:o+_nv].astype(dtype)
            lam_np = flat_1d[o+_nv:o+_nv+_ni].astype(dtype)
            mu_np  = flat_1d[o+_nv+_ni:o+_nv+_ni+_ne].astype(dtype)

            o2 = _residual_and_sol_len
            g_x   = flat[:, o2:o2+_nv].astype(dtype)
            g_lam = flat[:, o2+_nv:o2+_nv+_ni].astype(dtype)
            g_mu  = flat[:, o2+_nv+_ni:o2+_nv+_ni+_ne].astype(dtype)
        else:
            packed_res = flat[:_il].astype(dtype)
            o = _il
            x_np   = flat[o:o+_nv].astype(dtype)
            lam_np = flat[o+_nv:o+_nv+_ni].astype(dtype)
            mu_np  = flat[o+_nv+_ni:o+_nv+_ni+_ne].astype(dtype)

            o2 = _residual_and_sol_len
            g_x   = flat[o2:o2+_nv].astype(dtype)
            g_lam = flat[o2+_nv:o2+_nv+_ni].astype(dtype)
            g_mu  = flat[o2+_nv+_ni:o2+_nv+_ni+_ne].astype(dtype)

        return packed_res, x_np, lam_np, mu_np, g_x, g_lam, g_mu

    return stack_rev, unstack_rev, _total_len