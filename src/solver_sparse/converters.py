"""
solver_sparse/converters.py
===========================
JAX ↔ numpy conversions for the sparse QP path.

*Primals*:  BCOO matrices → scipy CSC;  dense vectors → ndarray.
*Tangents*: BCOO tangent → dense ndarray (the KKT RHS needs dense ops).
*Grads*:    dense matrix gradients → 1-D array of nonzero-value gradients
            (matching the BCOO ``.data`` layout so JAX can propagate them).

The caller must supply a ``sparsity_info`` dict at setup time that maps
each sparse key to the index arrays needed for extraction. This is built
once by :func:`build_sparsity_info` from the user-supplied sparsity
patterns.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from jax import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from scipy.sparse import csc_matrix

from src.solver_common import EXPECTED_NDIM

# ── Which keys are sparse (matrices) vs dense (vectors) ─────────────

SPARSE_KEYS: frozenset[str] = frozenset({"P", "A", "G"})


def is_sparse_key(key: str) -> bool:
    return key in SPARSE_KEYS


# ── Sparsity info ────────────────────────────────────────────────────
#
# Built once at setup from the user-supplied BCOO patterns.
# For each sparse key we store:
#   - "rows", "cols": int arrays from the BCOO indices (for CSC construction)
#   - "shape": tuple (m, n) of the matrix
#   - "nnz": number of nonzeros

SparsityEntry = dict  # {"rows": ndarray, "cols": ndarray, "shape": tuple, "nnz": int}
SparsityInfo  = dict[str, SparsityEntry]


def build_sparsity_info(
    sparsity_patterns: dict[str, BCOO],
) -> SparsityInfo:
    """
    Extract index arrays from BCOO patterns.

    Parameters
    ----------
    sparsity_patterns : dict[str, BCOO]
        Mapping from sparse key name ("P", "A", "G") to a BCOO matrix
        whose ``.indices`` encode the sparsity pattern.  The ``.data``
        values are ignored — only the structure matters.

    Returns
    -------
    SparsityInfo
        Per-key dict with "rows", "cols", "shape", "nnz".
    """
    info: SparsityInfo = {}
    for key, bcoo in sparsity_patterns.items():
        indices = np.asarray(bcoo.indices)           # (nnz, 2)
        info[key] = {
            "rows":  indices[:, 0].astype(np.int32),
            "cols":  indices[:, 1].astype(np.int32),
            "shape": tuple(bcoo.shape),
            "nnz":   indices.shape[0],
        }
    return info


# ── Converter factories ──────────────────────────────────────────────
#
# Each factory takes the sparsity_info built at setup and returns a
# converter function with the signature expected by solver_common.


def make_sparse_primal_converter(
    sparsity_info: SparsityInfo,
):
    """
    Return a primal converter: (key, jax_val, dtype) -> ndarray | csc_matrix.

    Sparse keys (P, A, G):
        Extracts ``.data`` from the BCOO, builds a CSC using the
        pre-stored index arrays.
    Dense keys (q, b, h):
        Plain ``np.asarray``, squeezing batch-1 leading dims.
    """

    def converter(key: str, val, dtype) -> ndarray | csc_matrix:
        if is_sparse_key(key) and key in sparsity_info:
            si = sparsity_info[key]
            # BCOO .data is a (nnz,) or (1, nnz) array
            data = np.asarray(val.data, dtype=dtype).ravel()
            return csc_matrix(
                (data, (si["rows"], si["cols"])),
                shape=si["shape"],
            )
        else:
            # Dense vector path
            arr = np.asarray(val, dtype=dtype)
            if arr.ndim > EXPECTED_NDIM[key]:
                arr = arr[0]
            return arr

    return converter


def make_sparse_tangent_converter(
    sparsity_info: SparsityInfo,
):
    """
    Return a tangent converter: (key, jax_val, dtype) -> ndarray.

    For sparse keys the tangent is a BCOO with the same sparsity
    pattern. We convert to a dense ndarray because the KKT RHS
    computation uses dense linear algebra.

    The batch dimension (if present) is preserved.
    """

    def converter(key: str, val, dtype) -> ndarray:
        if is_sparse_key(key) and key in sparsity_info:
            # BCOO tangent → dense ndarray
            # .todense() handles both batched and unbatched BCOO
            if hasattr(val, "todense"):
                return np.asarray(val.todense(), dtype=dtype)
            else:
                return np.asarray(val, dtype=dtype)
        else:
            return np.asarray(val, dtype=dtype)

    return converter


def make_sparse_grad_to_jax(
    sparsity_info: SparsityInfo,
):
    """
    Return a grad-to-JAX converter: (key, numpy_grad, dtype) -> jax.Array.

    For sparse keys the differentiator returns a full dense gradient
    matrix.  We extract only the entries at the sparsity pattern
    positions and return a 1-D array of length ``nnz`` — this is what
    JAX expects as the gradient w.r.t. the BCOO ``.data`` attribute.

    For dense keys this is a plain ``jnp.asarray``.
    """

    def converter(key: str, val: ndarray, dtype) -> Array:
        if is_sparse_key(key) and key in sparsity_info:
            si = sparsity_info[key]
            rows, cols = si["rows"], si["cols"]
            if val.ndim == 2:
                # Unbatched: (m, n) → (nnz,)
                return jnp.asarray(val[rows, cols], dtype=dtype)
            else:
                # Batched: (batch, m, n) → (batch, nnz)
                return jnp.asarray(val[:, rows, cols], dtype=dtype)
        else:
            return jnp.asarray(val, dtype=dtype)

    return converter
