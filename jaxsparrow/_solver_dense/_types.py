"""
solver_dense/_types.py
======================
TypedDict definitions for dense QP ingredients.

These types describe the parameter dictionaries flowing through the
dense QP solver and differentiator pipeline:

- **JAX-side** (``DenseQPIngredients``, ``DenseQPIngredientsFull``):
  used in the public API where inputs are JAX arrays.
- **NumPy-side** (``DenseQPIngredientsNP``, ``DenseQPIngredientsNPFull``):
  used inside the NumPy solver/differentiator closures after
  conversion from JAX.
- **Tangents** (``DenseQPIngredientsTangentsNP``): tangent vectors
  for forward-mode differentiation, which may carry an extra leading
  batch dimension when produced by ``vmap``.

The ``total=False`` variants allow partial dictionaries (e.g. only
the dynamic keys), while ``total=True`` variants require all six
QP parameters to be present.
"""

from typing import TypedDict
from jax import Array
from numpy import ndarray
from jaxtyping import Float


class DenseQPIngredients(TypedDict, total=False):
    """Partial dictionary of dense QP parameters as JAX arrays.

    Used for passing subsets of QP ingredients (e.g. only dynamic
    parameters) through the JAX-level API.
    """
    P: Float[Array, "n_var n_var"]
    q: Float[Array, "n_var"]
    A: Float[Array, "n_eq n_var"]
    b: Float[Array, "n_eq"]
    G: Float[Array, "n_ineq n_var"]
    h: Float[Array, "n_ineq"]


class DenseQPIngredientsFull(TypedDict):
    """Complete dictionary of dense QP parameters as JAX arrays.

    All six keys are required. Represents a fully-specified QP::

        min  0.5 x^T P x + q^T x
        s.t. A x = b
             G x <= h
    """
    P: Float[Array, "n_var n_var"]
    q: Float[Array, "n_var"]
    A: Float[Array, "n_eq n_var"]
    b: Float[Array, "n_eq"]
    G: Float[Array, "n_ineq n_var"]
    h: Float[Array, "n_ineq"]


class DenseQPIngredientsNP(TypedDict, total=False):
    """Partial dictionary of dense QP parameters as NumPy arrays.

    Used inside solver/differentiator closures for dynamic or fixed
    parameter subsets after conversion from JAX.
    """
    P: Float[ndarray, "n_var n_var"]
    q: Float[ndarray, "n_var"]
    A: Float[ndarray, "n_eq n_var"]
    b: Float[ndarray, "n_eq"]
    G: Float[ndarray, "n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]


class DenseQPIngredientsNPFull(TypedDict):
    """Complete dictionary of dense QP parameters as NumPy arrays.

    All six keys are required. This is the merged form used
    internally after combining fixed and dynamic elements.
    """
    P: Float[ndarray, "n_var n_var"]
    q: Float[ndarray, "n_var"]
    A: Float[ndarray, "n_eq n_var"]
    b: Float[ndarray, "n_eq"]
    G: Float[ndarray, "n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]


class DenseQPIngredientsTangentsNP(TypedDict, total=False):
    """Tangent vectors for dense QP parameters as NumPy arrays.

    Each value is either unbatched (same shape as the corresponding
    primal) or batched with a leading ``n_batch`` dimension when
    produced by ``vmap``.
    """
    P: Float[ndarray, "n_var n_var"]    |  Float[ndarray, "n_batch n_var n_var"]
    q: Float[ndarray, "n_var"]          |  Float[ndarray, "n_batch n_var"]
    A: Float[ndarray, "n_eq n_var"]     |  Float[ndarray, "n_batch n_eq n_var"]
    b: Float[ndarray, "n_eq"]           |  Float[ndarray, "n_batch n_eq"]
    G: Float[ndarray, "n_ineq n_var"]   |  Float[ndarray, "n_batch n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]         |  Float[ndarray, "n_batch n_ineq"]
