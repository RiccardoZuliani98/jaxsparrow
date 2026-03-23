"""
solver_sparse/_types.py
=======================
TypedDict definitions for sparse ingredients.

These types describe the parameter dictionaries flowing through the
sparse solver and differentiator pipeline:

- **JAX-side** (``SparseIngredients``, ``SparseIngredientsFull``):
  used in the public API where inputs are JAX BCOO matrices (for
  sparse parameters) and JAX arrays (for dense vectors).
- **NumPy-side** (``SparseIngredientsNP``, ``SparseIngredientsNPFull``):
  used inside the NumPy solver/differentiator closures after
  conversion from JAX. Sparse matrices are SciPy CSC, dense vectors
  are NumPy ndarrays.
- **Tangents** (``SparseIngredientsTangentsNP``): tangent vectors
  for forward-mode differentiation. For sparse keys, unbatched
  tangents are CSC matrices; batched tangents are materialised as
  dense ndarrays because SciPy sparse is strictly 2-D. Dense vector
  tangents are ndarrays in both cases.

The ``total=False`` variants allow partial dictionaries (e.g. only
the dynamic keys), while ``total=True`` variants require all six
parameters to be present.
"""

from jaxtyping import Float
from numpy import ndarray
from jax import Array
from typing import TypedDict, NamedTuple, Protocol

class SolverOutput(TypedDict):
    """Solver output (Jax)."""
    x:      Float[Array, "n_var"]       |  Float[Array, "n_batch n_var"]
    lam:    Float[Array, "n_ineq"]      |  Float[Array, "n_batch n_ineq"]
    mu:     Float[Array, "n_eq"]        |  Float[Array, "n_batch n_eq"]

class SolverOutputNP(NamedTuple):
    """Solver output (NumPy)."""
    x:      Float[ndarray, "n_var"]     |  Float[ndarray, "n_batch n_var"]
    lam:    Float[ndarray, "n_ineq"]    |  Float[ndarray, "n_batch n_ineq"]
    mu:     Float[ndarray, "n_eq"]      |  Float[ndarray, "n_batch n_eq"]

class SolverDiffOutFwd(TypedDict):
    """Reverse-mode differentiation output (NumPy)."""
    x   : Float[Array, "n_var"]         |  Float[Array, "n_batch n_var"] 
    lam : Float[Array, "n_ineq"]        |  Float[Array, "n_batch n_ineq"]
    mu  : Float[Array, "n_eq"]          |  Float[Array, "n_batch n_eq"]

class SolverDiffOutFwdNP(NamedTuple):
    """Forward-mode differentiation output (NumPy)."""
    x_np   : Float[ndarray, "n_var"]    |  Float[ndarray, "n_batch n_var"] 
    lam_np : Float[ndarray, "n_ineq"]   |  Float[ndarray, "n_batch n_ineq"]
    mu_np  : Float[ndarray, "n_eq"]     |  Float[ndarray, "n_batch n_eq"]

class SolverDiffOutRev(TypedDict):
    """Reverse-mode differentiation output (Jax)."""
    x   : Float[Array, "n_var"]         |  Float[Array, "n_batch n_var"] 
    lam : Float[Array, "n_ineq"]        |  Float[Array, "n_batch n_ineq"]
    mu  : Float[Array, "n_eq"]          |  Float[Array, "n_batch n_eq"]

class SolverDiffOutRevNP(NamedTuple):
    """Reverse-mode differentiation output (NumPy)."""
    x_np   : Float[ndarray, "n_var"]    |  Float[ndarray, "n_batch n_var"] 
    lam_np : Float[ndarray, "n_ineq"]   |  Float[ndarray, "n_batch n_ineq"]
    mu_np  : Float[ndarray, "n_eq"]     |  Float[ndarray, "n_batch n_eq"]

class Solver(Protocol):

    def __call__(
        self, **kwargs: ndarray,
    ) -> tuple[SolverOutputNP, dict[str, float]]: 
        """
        Solve a convex optimization program.
        
        Args:
            **kwargs: ingredients (P, q, A, b, G, h) as ndarrays.
            
        Returns:
            Tuple of (solution, timing_dict) where solution contains
            x (primal), lam (inequality multipliers), mu (equality multipliers).
            
        Raises:
            ValueError: If required ingredients are missing.
            RuntimeError: If solver fails to find a solution.
        """
        ...