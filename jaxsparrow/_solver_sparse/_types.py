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

class SparseIngredientsNPFull(TypedDict):
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
# Batched sparse tangents are dense ndarrays because SciPy sparse is
# strictly 2-D — the tangent converter materializes (batch, nnz) into
# (batch, m, n) so the downstream differentiator can iterate over
# batch elements.
# Dense vector tangents are ndarrays in both cases.
# --------------------------------------------------------------------------

SparseOrDense = Union[csc_matrix, ndarray]

class SparseIngredientsTangentsNP(TypedDict, total=False):
    P: SparseOrDense  # csc_matrix (unbatched) or ndarray (batch, n_var, n_var)
    q: ndarray         # (n_var,) or (batch, n_var)
    A: SparseOrDense  # csc_matrix (unbatched) or ndarray (batch, n_eq, n_var)
    b: ndarray         # (n_eq,) or (batch, n_eq)
    G: SparseOrDense  # csc_matrix (unbatched) or ndarray (batch, n_ineq, n_var)
    h: ndarray         # (n_ineq,) or (batch, n_ineq)