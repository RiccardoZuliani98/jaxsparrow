"""
_options_common.py
==================
Common options types for the QP solver and differentiator pipelines.

These types define the configuration dictionaries used across both
dense and sparse solver paths:

- **SolverOptions**: Configuration for the numerical QP solver backend.
  Declares the ``backend`` and ``dtype`` fields that are common to
  all solver backends; all other keys are backend-specific.
- **DifferentiatorOptions**: Configuration for the differentiation backend.
  Declares the ``backend`` field that selects the concrete backend
  implementation; all other keys are backend-specific.
- **ConstructorOptions**: Top-level options controlling the overall
  differentiable solver construction (differentiation mode, solver
  selection, debugging flags, etc.).

The ``total=False`` variants allow partial dictionaries where missing
keys are filled with defaults. The ``total=True`` variants represent
the fully resolved options after merging user-supplied values with
defaults.
"""

from typing import TypedDict, Final, Literal, Union, Dict, Any
import numpy as np
import jax.numpy as jnp


# ----------------------------------------------------------------------
# Base option types
# ----------------------------------------------------------------------

class SolverOptions(TypedDict, total=False):
    """Configuration options for the numerical QP solver.

    The ``backend`` field selects the solver *protocol* — the
    library or interface used to solve the QP (e.g.,
    ``"qpsolvers"``).  The concrete solver *within* that protocol
    is chosen by backend-specific keys (e.g., ``solver_name`` for
    the ``qpsolvers`` backend selects ``"piqp"``, ``"osqp"``, etc.).

    The ``dtype`` field sets the NumPy floating-point dtype for
    all arrays and is used by both the solver backend and by
    ``setup_dense_solver`` for fixed-element conversion.

    All remaining keys are backend-specific and documented in the
    corresponding options classes (e.g.,
    :class:`DenseQpSolverOptions`).

    Missing keys are filled from backend-specific defaults.
    """
    backend: str
    dtype:   type[np.floating]


class DifferentiatorOptions(TypedDict, total=False):
    """Configuration options for the differentiation backend.

    The ``backend`` field selects the differentiation *algorithm*
    (e.g., ``"dense_kkt"`` for standard KKT differentiation,
    ``"dense_dbd"`` for the regularised Differentiable-by-Design
    method, ``"sparse_kkt"`` for the sparse variant).

    All remaining keys are backend-specific and documented in the
    corresponding options classes (e.g., :class:`DenseKKTDiffOptions`,
    :class:`DenseDBDDiffOptions`).

    Missing keys are filled from backend-specific defaults.
    """
    backend: str


# ----------------------------------------------------------------------
# Top-level constructor options
# ----------------------------------------------------------------------

class ConstructorOptions(TypedDict, total=False):
    """Partial constructor options for the differentiable solver.

    All keys are optional; missing keys are filled from
    ``DEFAULT_CONSTRUCTOR_OPTIONS``. After merging, the resolved
    options are represented by :class:`ConstructorOptionsFull`.

    Attributes:
        diff_mode: Differentiation mode for the solver.
            - ``"fwd"``: Forward-mode differentiation (JVP)
            - ``"rev"``: Reverse-mode differentiation (VJP)
        solver: Options passed to the QP solver backend.
            Dictionary keys depend on the chosen solver.
        differentiator: Options passed to the differentiation backend.
            Dictionary keys depend on the chosen backend.
        dtype: Floating-point dtype for all numerical computations.
            Must be a JAX dtype (e.g., ``jnp.float64``, ``jnp.float32``).
        bool_dtype: Boolean dtype for masks (e.g., active constraints).
        verbose: Whether to print solver and differentiator progress.
        debug: Enable debug mode with additional checks and logging.
        fd_check: Enable finite-difference checking for gradient
            verification (only for testing/debugging).
        fd_eps: Step size for finite-difference approximation.
    """
    diff_mode: Literal["fwd", "rev"]
    solver: Union[SolverOptions, Dict[str, Any]]
    differentiator: Union[DifferentiatorOptions, Dict[str, Any]]
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool
    fd_check: bool
    fd_eps: float


class ConstructorOptionsFull(TypedDict):
    """Fully resolved constructor options after merging with defaults.

    All keys are required. This type represents the exact configuration
    used to build the differentiable solver after all defaults have
    been applied.

    Attributes:
        diff_mode: Differentiation mode (``"fwd"`` or ``"rev"``).
        solver: Fully resolved solver options (merged with defaults).
        differentiator: Fully resolved differentiator options (merged
            with defaults).
        dtype: Floating-point dtype for computations.
        bool_dtype: Boolean dtype for masks.
        verbose: Enable verbose output.
        debug: Enable debug mode.
        fd_check: Enable finite-difference verification.
        fd_eps: Finite-difference step size.
    """
    diff_mode: Literal["fwd", "rev"]
    solver: SolverOptions
    differentiator: DifferentiatorOptions
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool
    fd_check: bool
    fd_eps: float


# ----------------------------------------------------------------------
# Default configuration
# ----------------------------------------------------------------------

DEFAULT_CONSTRUCTOR_OPTIONS: Final[ConstructorOptionsFull] = {
    "diff_mode": "fwd",
    "solver": {},
    "differentiator": {},
    "dtype": jnp.float64,
    "bool_dtype": jnp.bool_,
    "verbose": True,
    "debug": True,
    "fd_check": False,
    "fd_eps": 1e-6,
}