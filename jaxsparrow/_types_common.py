"""
_types_common.py
================
Type definitions for solver inputs, outputs, and differentiation
results.

- **SolverOutput**: Dict returned by the public ``solver()`` callable.
- **SolverOutputNP**: Named tuple used internally by numpy backends.
- **SolverDiffOutFwdNP / SolverDiffOutRevNP**: Named tuples used
  internally by the numpy differentiator backends.
- **Solver**: Protocol for numpy QP solver backends.
"""

from jaxtyping import Float
from numpy import ndarray
from jax import Array
from typing import TypedDict, NamedTuple, Protocol, Union

ArrayLike = Union[Array, ndarray]


class SolverOutput(TypedDict):
    """Solver output dictionary.

    Inside ``pure_callback``, values are numpy ndarrays; after the
    callback boundary, JAX converts them automatically.
    """
    x:   Float[ArrayLike, "n_var"]   | Float[ArrayLike, "n_batch n_var"]
    lam: Float[ArrayLike, "n_ineq"]  | Float[ArrayLike, "n_batch n_ineq"]
    mu:  Float[ArrayLike, "n_eq"]    | Float[ArrayLike, "n_batch n_eq"]


class SolverOutputNP(NamedTuple):
    """Solver output (NumPy), used internally by solver backends."""
    x:   Float[ndarray, "n_var"]   | Float[ndarray, "n_batch n_var"]
    lam: Float[ndarray, "n_ineq"]  | Float[ndarray, "n_batch n_ineq"]
    mu:  Float[ndarray, "n_eq"]    | Float[ndarray, "n_batch n_eq"]


class SolverDiffOutFwdNP(NamedTuple):
    """Forward-mode differentiation output (NumPy), used internally
    by differentiator backends."""
    x_np:   Float[ndarray, "n_var"]   | Float[ndarray, "n_batch n_var"]
    lam_np: Float[ndarray, "n_ineq"]  | Float[ndarray, "n_batch n_ineq"]
    mu_np:  Float[ndarray, "n_eq"]    | Float[ndarray, "n_batch n_eq"]


class SolverDiffOutRevNP(NamedTuple):
    """Reverse-mode differentiation output (NumPy), used internally
    by differentiator backends."""
    x_np:   Float[ndarray, "n_var"]   | Float[ndarray, "n_batch n_var"]
    lam_np: Float[ndarray, "n_ineq"]  | Float[ndarray, "n_batch n_ineq"]
    mu_np:  Float[ndarray, "n_eq"]    | Float[ndarray, "n_batch n_eq"]


class Solver(Protocol):
    """Protocol for numpy QP solver backends."""

    def __call__(
        self, **kwargs: ndarray,
    ) -> tuple[SolverOutputNP, dict[str, float]]:
        """Solve a convex optimization program.

        Args:
            **kwargs: Ingredients (P, q, A, b, G, h) as ndarrays.

        Returns:
            Tuple of (solution, timing_dict).
        """
        ...