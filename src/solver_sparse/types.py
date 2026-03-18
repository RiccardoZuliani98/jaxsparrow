"""
solver_sparse/types.py
======================
Type aliases for the sparse QP solver path.

Matrices (P, A, G) are ``scipy.sparse.csc_matrix``.
Vectors (q, b, h) remain dense ``ndarray``.
"""

from typing import TypedDict, Union
from numpy import ndarray
from scipy.sparse import csc_matrix
from jaxtyping import Float

# --------------------------------------------------------------------------
# Numpy-side ingredients (what the solver / differentiator receive)
# --------------------------------------------------------------------------

class SparseQPIngredientsNP(TypedDict, total=False):
    P: csc_matrix
    q: Float[ndarray, "n_var"]
    A: csc_matrix
    b: Float[ndarray, "n_eq"]
    G: csc_matrix
    h: Float[ndarray, "n_ineq"]

class SparseQPIngredientsNPFull(TypedDict):
    P: csc_matrix
    q: Float[ndarray, "n_var"]
    A: csc_matrix
    b: Float[ndarray, "n_eq"]
    G: csc_matrix
    h: Float[ndarray, "n_ineq"]

# --------------------------------------------------------------------------
# Tangents for the forward (JVP) path — numpy side.
#
# Tangents for sparse matrices are dense ndarray (or batched),
# because dP, dA, dG are perturbations of the *nonzero values*
# expanded back to dense form for the KKT RHS computation.
# --------------------------------------------------------------------------

class SparseQPIngredientsTangentsNP(TypedDict, total=False):
    P: Float[ndarray, "n_var n_var"]    | Float[ndarray, "n_batch n_var n_var"]
    q: Float[ndarray, "n_var"]          | Float[ndarray, "n_batch n_var"]
    A: Float[ndarray, "n_eq n_var"]     | Float[ndarray, "n_batch n_eq n_var"]
    b: Float[ndarray, "n_eq"]           | Float[ndarray, "n_batch n_eq"]
    G: Float[ndarray, "n_ineq n_var"]   | Float[ndarray, "n_batch n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]         | Float[ndarray, "n_batch n_ineq"]
