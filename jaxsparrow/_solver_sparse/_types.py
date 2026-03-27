"""
solver_sparse/types.py
======================
Type aliases for the sparse solver path.

Matrices (P, A, G) are ``scipy.sparse.csc_matrix``.
Vectors (q, b, h) remain dense ``ndarray``.
"""

from typing import TypedDict, Union
from numpy import ndarray
from scipy.sparse import csc_matrix
from jaxtyping import Float
from jax.experimental.sparse import BCOO
from jax import Array

# --------------------------------------------------------------------------
# Numpy-side ingredients (what the solver / differentiator receive)
# --------------------------------------------------------------------------

class SparseIngredientsNP(TypedDict, total=False):
    P: csc_matrix
    q: Float[ndarray, "n_var"]
    A: csc_matrix
    b: Float[ndarray, "n_eq"]
    G: csc_matrix
    h: Float[ndarray, "n_ineq"]

# --------------------------------------------------------------------------
# Jax-side ingredients (used as inputs-outputs)
# --------------------------------------------------------------------------

class SparseIngredients(TypedDict, total=False):
    P: BCOO
    q: Float[Array, "n_var"]
    A: BCOO
    b: Float[Array, "n_eq"]
    G: BCOO
    h: Float[Array, "n_ineq"]

class SparseIngredientsFull(TypedDict):
    P: BCOO
    q: Float[Array, "n_var"]
    A: BCOO
    b: Float[Array, "n_eq"]
    G: BCOO
    h: Float[Array, "n_ineq"]

# --------------------------------------------------------------------------
# Tangents for the forward (JVP) path — numpy side.
#
# Unbatched sparse tangents are CSC matrices (same structure as primals).
# Batched sparse tangents are list[csc_matrix] — one CSC per batch
# element, all sharing the same sparsity pattern (rows, cols, shape)
# but with different data arrays. This avoids materializing a dense
# (batch, m, n) tensor.
# Dense vector tangents are ndarrays in both cases.
# --------------------------------------------------------------------------

SparseOrDenseOrList = Union[csc_matrix, ndarray, list[csc_matrix]]

class SparseIngredientsTangentsNP(TypedDict, total=False):
    P: SparseOrDenseOrList  # csc_matrix (unbatched) or list[csc_matrix] (batched)
    q: ndarray               # (n_var,) or (batch, n_var)
    A: SparseOrDenseOrList  # csc_matrix (unbatched) or list[csc_matrix] (batched)
    b: ndarray               # (n_eq,) or (batch, n_eq)
    G: SparseOrDenseOrList  # csc_matrix (unbatched) or list[csc_matrix] (batched)
    h: ndarray               # (n_ineq,) or (batch, n_ineq)