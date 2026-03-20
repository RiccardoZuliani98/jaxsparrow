"""
utils/_sparse_utils.py
======================
Sparse matrix conversion utilities for the JAX ↔ SciPy boundary.
"""

import numpy as np
from numpy import ndarray
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix


def bcoo_to_csc(mat: BCOO, dtype: type[np.floating]) -> csc_matrix:
    """Convert a JAX BCOO sparse matrix to a SciPy CSC sparse matrix.

    Extracts the stored indices and data from the BCOO representation,
    casts the data to the requested NumPy dtype, and constructs a
    ``csc_matrix`` with the same shape.

    Args:
        mat: A JAX BCOO sparse matrix (2-D, with ``indices`` of shape
            ``(nnz, 2)`` and ``data`` of shape ``(nnz,)``).
        dtype: Target NumPy floating-point dtype for the nonzero
            values.

    Returns:
        A SciPy ``csc_matrix`` with the same shape and sparsity
        pattern as *mat*.
    """
    idx: ndarray = np.asarray(mat.indices)
    data: ndarray = np.asarray(mat.data, dtype=dtype)
    rows: ndarray = idx[:, 0]
    cols: ndarray = idx[:, 1]
    return csc_matrix((data, (rows, cols)), shape=mat.shape)