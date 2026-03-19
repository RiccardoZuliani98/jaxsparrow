"""
solver_sparse/converters.py
===========================
JAX ↔ numpy conversions for the sparse QP path.

*Primals*:  BCOO matrices → scipy CSC;  dense vectors → ndarray.
*Tangents*: BCOO tangent → dense ndarray (the KKT RHS needs dense ops).
*Grads*:    gradient arrays → 1-D array of nonzero-value gradients
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
    Extract index arrays from BCOO patterns. Used at construction.

    Args:
        sparsity_patterns : dict[str, BCOO]
            Mapping from sparse key name ("P", "A", "G") to a BCOO matrix
            whose ``.indices`` encode the sparsity pattern.  The ``.data``
            values are ignored — only the structure matters.

    Returns:
        SparsityInfo
            Per-key dict with "rows", "cols", "shape", "nnz".
    """
    info: SparsityInfo = {}
    for key, bcoo in sparsity_patterns.items():
        if bcoo.ndim != 2:
            raise ValueError(
                f"Sparsity pattern for '{key}' must be a 2-D matrix, "
                f"got {bcoo.ndim}-D array with shape {bcoo.shape}"
            )
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


def make_sparse_primal_converter(sparsity_info: SparsityInfo):
    """Create a converter that turns primal values into NumPy/SciPy arrays.

    Builds a closure that, given a key and a JAX value, returns either a
    SciPy CSC sparse matrix (for sparse keys) or a dense NumPy array
    (for everything else). Batch dimensions introduced by
    ``jax.expand_dims`` are squeezed automatically, this happens because
    the primals are not supposed to be vectorized through expand_dims, but
    this is unavoidable because of how the code is written.
    The output is either a numpy array (for dense ingredients), or a scipy
    csc matrix (for sparse ingredients). The input is a jax Array or a 
    jax BCOO matrix, respectively.

    Args:
        sparsity_info: Per-key mapping produced by ``build_sparsity_info``,
            containing ``"rows"``, ``"cols"``, ``"shape"``, and ``"nnz"``
            for each sparse key.

    Returns:
        A converter function with signature
        ``(key: str, val, dtype) -> ndarray | csc_matrix``.

    Raises:
        ValueError: If a key is identified as sparse by ``is_sparse_key``
            but has no corresponding entry in *sparsity_info*.
    """
    def converter(key: str, val, dtype) -> ndarray | csc_matrix:
        if is_sparse_key(key):
            if key not in sparsity_info:
                raise ValueError(
                    f"Key '{key}' is marked as sparse but has no entry "
                    f"in sparsity_info (available: {list(sparsity_info)})"
                )
            si = sparsity_info[key]
            # BCOO may have a batch dim from expand_dims — squeeze it
            # since primals are identical across batch
            data = np.asarray(val.data, dtype=dtype)
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
    """Create a converter that turns tangent values into NumPy/SciPy arrays.

    Similar to the primal converter, but must handle batched tangents
    produced by ``vmap_method="expand_dims"`` in the JVP path. When JAX
    vmaps the JVP rule, each tangent acquires a leading batch dimension
    while the primals remain shared. For sparse keys this means
    ``val.data`` may be either ``(nnz,)`` (unbatched) or
    ``(batch, nnz)`` (batched).

    Since SciPy sparse matrices are strictly 2-D, the unbatched case
    returns a CSC matrix while the batched case materializes into a
    dense ``(batch, m, n)`` array so the downstream numpy differentiator
    can iterate over batch elements.

    Args:
        sparsity_info: Per-key mapping produced by ``build_sparsity_info``,
            containing ``"rows"``, ``"cols"``, ``"shape"``, and ``"nnz"``
            for each sparse key.

    Returns:
        A converter function with signature
        ``(key: str, val, dtype) -> ndarray | csc_matrix``.

    Raises:
        ValueError: If a key is identified as sparse by ``is_sparse_key``
            but has no corresponding entry in *sparsity_info*.
        ValueError: If a sparse tangent has an unexpected number of
            dimensions (neither 1 nor 2).
    """
    def converter(key: str, val, dtype) -> ndarray | csc_matrix:
        if is_sparse_key(key):
            if key not in sparsity_info:
                raise ValueError(
                    f"Key '{key}' is marked as sparse but has no entry "
                    f"in sparsity_info (available: {list(sparsity_info)})"
                )
            si = sparsity_info[key]
            data = np.asarray(val.data, dtype=dtype)

            if data.ndim == 1:
                # Unbatched: (nnz,) → CSC
                return csc_matrix(
                    (data, (si["rows"], si["cols"])),
                    shape=si["shape"],
                )
            elif data.ndim == 2:
                # Batched: (batch, nnz) → dense (batch, m, n)
                # scipy.sparse is 2-D only
                batch_size = data.shape[0]
                #TODO: is this the fastest thing we can do here? Could we instead build
                # a sparse matrix for this?
                out = np.zeros((batch_size, *si["shape"]), dtype=dtype)
                out[:, si["rows"], si["cols"]] = data
                return out
            else:
                raise ValueError(
                    f"Unexpected tangent data ndim={data.ndim} for key '{key}'"
                )
        else:
            arr = np.asarray(val, dtype=dtype)
            # if arr.ndim > EXPECTED_NDIM[key]:
            #     arr = arr[0]
            return arr

    return converter


def make_sparse_grad_to_jax_reverse(sparsity_info: SparsityInfo):
    """Convert reverse-mode gradients to JAX arrays.

    The reverse differentiator returns gradients for sparse keys as
    arrays of length ``nnz`` aligned with the BCOO ``.data`` layout,
    either ``(nnz,)`` unbatched or ``(batch, nnz)`` batched. These
    are passed through directly via ``jnp.asarray``.

    Args:
        sparsity_info: Per-key mapping produced by ``build_sparsity_info``.

    Returns:
        A converter with signature ``(key, numpy_grad, dtype) -> jax.Array``.
    """
    def converter(key: str, val: ndarray, dtype) -> Array:
        return jnp.asarray(val, dtype=dtype)

    return converter


def make_sparse_grad_to_jax_forward(sparsity_info: SparsityInfo):
    """Convert forward-mode gradients to JAX arrays.

    The forward differentiator returns gradients for sparse keys as
    full dense matrices, either ``(m, n)`` unbatched or
    ``(batch, m, n)`` batched. This converter extracts only the entries
    at the sparsity-pattern positions so that the result aligns with
    the BCOO ``.data`` layout.

    Args:
        sparsity_info: Per-key mapping produced by ``build_sparsity_info``.

    Returns:
        A converter with signature ``(key, numpy_grad, dtype) -> jax.Array``.
    """
    def converter(key: str, val: ndarray, dtype) -> Array:
        if is_sparse_key(key):
            if key not in sparsity_info:
                raise ValueError(
                    f"Key '{key}' is marked as sparse but has no entry "
                    f"in sparsity_info (available: {list(sparsity_info)})"
                )
            si = sparsity_info[key]
            rows, cols = si["rows"], si["cols"]
            if val.ndim == 2:
                return jnp.asarray(val[rows, cols], dtype=dtype)
            else:
                # Batched: (batch, m, n) → (batch, nnz)
                return jnp.asarray(val[:, rows, cols], dtype=dtype)
        else:
            return jnp.asarray(val, dtype=dtype)

    return converter