"""
solver_sparse/converters.py
===========================
JAX ↔ numpy conversions for the sparse path.

*Primals*:     BCOO matrices → scipy CSC;  dense vectors → ndarray.
*Tangents*:    BCOO tangent → CSC (unbatched) or list[CSC] (batched).
*Residuals*:   BCOO → .data only (extract); .data + cached pattern → CSC
               (reconstruct).

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

from jaxsparrow._solver_common import EXPECTED_NDIM

# ── Which keys are sparse (matrices) vs dense (vectors) ─────────────

SPARSE_KEYS: frozenset[str] = frozenset({"P", "A", "G"})

def is_sparse_key(key: str) -> bool:
    return key in SPARSE_KEYS


# ── Sparsity info ────────────────────────────────────────────────────

SparsityEntry = dict  # {"rows": ndarray, "cols": ndarray, "shape": tuple, "nnz": int}
SparsityInfo  = dict[str, SparsityEntry]


def build_sparsity_info(
    sparsity_patterns: dict[str, BCOO],
) -> SparsityInfo:
    """Extract index arrays from BCOO patterns. Used at construction.

    Args:
        sparsity_patterns: Mapping from sparse key name to a BCOO
            matrix whose ``.indices`` encode the sparsity pattern.

    Returns:
        Per-key dict with ``"rows"``, ``"cols"``, ``"shape"``,
        ``"nnz"``.
    """
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


# ── Converter factories ──────────────────────────────────────────────


def make_sparse_primal_converter(sparsity_info: SparsityInfo):
    """Create a converter: JAX primal → numpy/scipy.

    Sparse keys: extract ``.data``, squeeze batch dim, build CSC.
    Dense keys: ``np.asarray`` with squeeze.
    """
    def converter(key: str, val, dtype) -> ndarray | csc_matrix:
        if is_sparse_key(key):
            if key not in sparsity_info:
                raise ValueError(
                    f"Key '{key}' is marked as sparse but has no entry "
                    f"in sparsity_info (available: {list(sparsity_info)})"
                )
            si = sparsity_info[key]
            data = np.asarray(val, dtype=dtype)
            while data.ndim > 1:
                data = data[0]
            return csc_matrix(
                (data, (si["rows"], si["cols"])),
                shape=si["shape"],
            )
        else:
            arr = np.asarray(val, dtype=dtype)
            if arr.ndim > EXPECTED_NDIM[key]:
                arr = arr[0]
            return arr

    return converter


def make_sparse_tangent_converter(sparsity_info: SparsityInfo):
    """Create a converter: JAX tangent → numpy/scipy.

    Sparse unbatched: single CSC matrix.
    Sparse batched: ``list[csc_matrix]``.
    Dense: ``np.asarray`` preserving batch dim.
    """
    def converter(key: str, val, dtype) -> ndarray | csc_matrix | list[csc_matrix]:
        if is_sparse_key(key):
            if key not in sparsity_info:
                raise ValueError(
                    f"Key '{key}' is marked as sparse but has no entry "
                    f"in sparsity_info (available: {list(sparsity_info)})"
                )
            si = sparsity_info[key]
            rows, cols, shape = si["rows"], si["cols"], si["shape"]
            data = np.asarray(val, dtype=dtype)

            if data.ndim == 1:
                return csc_matrix((data, (rows, cols)), shape=shape)
            elif data.ndim == 2:
                return [
                    csc_matrix((data[i], (rows, cols)), shape=shape)
                    for i in range(data.shape[0])
                ]
            else:
                raise ValueError(
                    f"Unexpected tangent data ndim={data.ndim} for key '{key}'"
                )
        else:
            return np.asarray(val, dtype=dtype)

    return converter


def make_sparse_residual_extractor(sparsity_info: SparsityInfo):
    """Create a function: JAX primal → minimal JAX residual.

    Sparse keys: return ``val.data`` (just the nnz values).
    Dense keys: return ``val`` unchanged.

    This reduces the data saved as VJP residuals — constant sparsity
    indices are cached in the converter and don't need to travel
    through JAX's residual machinery.
    """
    def extractor(key: str, val) -> Array:
        if is_sparse_key(key) and key in sparsity_info:
            return val.data
        return val

    return extractor


def make_sparse_residual_reconstructor(sparsity_info: SparsityInfo):
    """Create a function: minimal residual → numpy primal.

    Sparse keys: rebuild CSC from the 1-D data vector plus cached
    indices.
    Dense keys: ``np.asarray`` with squeeze.

    This is the inverse of ``make_sparse_residual_extractor`` — it
    reconstructs the full numpy primal from the compressed residual.
    """
    def reconstructor(key: str, val, dtype) -> ndarray | csc_matrix:
        if is_sparse_key(key) and key in sparsity_info:
            si = sparsity_info[key]
            data = np.asarray(val, dtype=dtype)
            while data.ndim > 1:
                data = data[0]
            return csc_matrix(
                (data, (si["rows"], si["cols"])),
                shape=si["shape"],
            )
        else:
            arr = np.asarray(val, dtype=dtype)
            if arr.ndim > EXPECTED_NDIM[key]:
                arr = arr[0]
            return arr

    return reconstructor