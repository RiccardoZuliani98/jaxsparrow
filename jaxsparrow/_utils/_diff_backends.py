"""
utils/_differentiator_backends.py
=================================
Abstract differentiator backend protocol and concrete implementations
for the sparse path.

The two-phase protocol separates differentiator lifecycle into:

1. **setup** — one-time initialization: receive the fixed QP
   ingredients, cast and store them, pre-compute zero tangents,
   extract sparsity indices, select the linear solver.

2. **differentiate_fwd / differentiate_rev** — per-call: receive
   the solution and dynamic ingredients, compute the active set,
   assemble the KKT system, solve it, and extract output
   tangents (fwd) or parameter cotangents (rev).

Backends
--------
- ``SparseKKTDifferentiatorBackend``: assembles the KKT system in
  sparse CSC form and solves with a configurable sparse linear
  solver (default: ``splu``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union, TypeVar

from numpy import ndarray

from jaxsparrow._solver_sparse._types import (
    SparseIngredientsNP, 
    SparseIngredientsTangentsNP
)
from jaxsparrow._solver_dense._types import (
    DenseIngredientsNP, 
    DenseIngredientsTangentsNP
)

from jaxsparrow._solver_sparse._converters import SparsityInfo
from jaxsparrow._types_common import (
    SolverOutputNP, 
    SolverDiffOutFwdNP, 
    SolverDiffOutRevNP
)

# Type variables for more precise typing
SparseT = TypeVar('SparseT', bound=SparseIngredientsNP)
DenseT = TypeVar('DenseT', bound=DenseIngredientsNP)

# Union types for compatibility with existing code
IngredientsNP = Union[SparseIngredientsNP, DenseIngredientsNP]
IngredientsTangentsNP = Union[SparseIngredientsTangentsNP, DenseIngredientsTangentsNP]


# =====================================================================
# Abstract protocol
# =====================================================================

class DifferentiatorBackend(ABC):
    """Abstract base class for differentiator backends.

    Subclasses implement a two-phase lifecycle:

    1. :meth:`setup` — one-time initialization with fixed elements.
    2. :meth:`differentiate_fwd` / :meth:`differentiate_rev` —
       per-call forward or reverse differentiation.

    Backend implementations should handle both sparse and dense
    ingredients appropriately, raising TypeError if the ingredient
    type doesn't match the backend's expected format.

    Shape conventions:
    - Single problem: all arrays have shape (n,)
    - Batched problems: arrays have shape (batch_size, n) where
      batch_size is provided at call time
    - Gradient outputs follow the same batching convention
    """

    @abstractmethod
    def setup(
        self,
        fixed_elements: Optional[IngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """One-time initialization of the differentiator.

        This method should store any ingredients that remain constant
        across all differentiation calls, pre-compute sparsity patterns,
        and initialize the linear solver.

        Args:
            fixed_elements: QP ingredients constant across calls.
                Must match the backend's expected type (sparse or dense).
                If None, all ingredients are assumed to be dynamic.
            dynamic_keys: Keys for which gradients are needed.
                None means gradients for all keys will be computed.
                This allows the backend to optimize by pre-computing
                only the necessary derivatives.
            sparsity_info: Per-key sparsity info from BCOO patterns.
                Required for dynamic sparse keys to know the sparsity
                structure. Ignored for dense backends.

        Returns:
            Timing dictionary with keys:
                - "setup": time spent in setup phase

        Raises:
            TypeError: If fixed_elements type doesn't match the backend
                (e.g., dense ingredients passed to sparse backend)
            ValueError: If required ingredients for differentiation
                are missing or invalid
        """
        ...

    @abstractmethod
    def differentiate_fwd(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: IngredientsNP,
        dyn_tangents_np: IngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]:
        """Forward-mode (JVP) differentiation.

        Computes the Jacobian-vector product for the QP solution
        with respect to the dynamic ingredients.

        Args:
            sol_np: Solution from the QP solver containing:
                - x: primal variables, shape (n_vars,) or (batch_size, n_vars)
                - lam: Lagrange multipliers for equality constraints,
                  shape (n_eq,) or (batch_size, n_eq)
                - mu: Lagrange multipliers for inequality constraints,
                  shape (n_ineq,) or (batch_size, n_ineq)
                - status: solver status code
                - obj_val: objective value at solution
            dyn_primals_np: Dynamic ingredients at which to evaluate
                the derivative. Must contain all keys needed for
                KKT matrix assembly (e.g., Q, c, A, b, G, h).
                Shape follows the batch_size convention.
            dyn_tangents_np: Tangents for the dynamic ingredients.
                Must have the same structure as dyn_primals_np,
                with each array having the same shape as the
                corresponding primal.
            batch_size: Number of problems in batch. Determines
                whether inputs are treated as single problems
                (batch_size=1) or batched (batch_size>1).

        Returns:
            Tuple of:
                - SolverDiffOutFwdNP: Contains:
                    - x_t: tangent of primal variables
                    - lam_t: tangent of equality multipliers
                    - mu_t: tangent of inequality multipliers
                    - obj_val_t: tangent of objective value
                  All tangents have same shape as the corresponding
                  primal variables.
                - Timing dictionary with keys:
                    - "assemble": time to assemble KKT system
                    - "solve": time to solve linear system
                    - "extract": time to extract tangents

        Raises:
            ValueError: If batch_size doesn't match the shapes of
                dyn_primals_np or dyn_tangents_np
            RuntimeError: If linear solver fails
        """
        ...

    @abstractmethod
    def differentiate_rev(
        self,
        dyn_primals_np: IngredientsNP,
        x_np: ndarray,
        lam_np: ndarray,
        mu_np: ndarray,
        g_x: ndarray,
        g_lam: ndarray,
        g_mu: ndarray,
        batch_size: int,
    ) -> tuple[SolverDiffOutRevNP, dict[str, float]]:
        """Reverse-mode (VJP) differentiation.

        Computates the vector-Jacobian product for the QP solution
        with respect to the dynamic ingredients.

        Args:
            dyn_primals_np: Dynamic ingredients at which to evaluate
                the derivative. Must contain all keys needed for
                KKT matrix assembly (e.g., Q, c, A, b, G, h).
                Shape follows the batch_size convention.
            x_np: Primal variables at solution, shape (n_vars,)
                or (batch_size, n_vars)
            lam_np: Equality multipliers at solution, shape (n_eq,)
                or (batch_size, n_eq)
            mu_np: Inequality multipliers at solution, shape (n_ineq,)
                or (batch_size, n_ineq)
            g_x: Cotangent (adjoint) of primal variables.
                Shape must match x_np.
            g_lam: Cotangent of equality multipliers.
                Shape must match lam_np.
            g_mu: Cotangent of inequality multipliers.
                Shape must match mu_np.
            batch_size: Number of problems in batch. Determines
                whether inputs are treated as single problems
                (batch_size=1) or batched (batch_size>1).

        Returns:
            Tuple of:
                - SolverDiffOutRevNP: Dictionary mapping ingredient
                  keys to their cotangents. Each cotangent has the
                  same shape as the corresponding ingredient in
                  dyn_primals_np.
                - Timing dictionary with keys:
                    - "assemble": time to assemble KKT system
                    - "solve": time to solve linear system
                    - "extract": time to extract cotangents

        Raises:
            ValueError: If batch_size doesn't match the shapes of
                input arrays or if cotangent shapes don't match
                primal shapes
            RuntimeError: If linear solver fails
        """
        ...


# =====================================================================
# Registry and factory
# =====================================================================

_DIFF_BACKEND_REGISTRY: dict[str, type[DifferentiatorBackend]] = {}


def register_differentiator_backend(
    name: str, cls: type[DifferentiatorBackend],
) -> None:
    """Register a new differentiator backend.

    Args:
        name: Unique identifier for the backend (e.g., "kkt", "custom")
        cls: Backend class that implements DifferentiatorBackend

    Raises:
        TypeError: If cls is not a subclass of DifferentiatorBackend
        ValueError: If name is already registered (overwrites allowed
            but warning may be emitted in future versions)

    Example:
        >>> class MyBackend(DifferentiatorBackend):
        ...     ...
        >>> register_differentiator_backend("my_backend", MyBackend)
    """
    if not (isinstance(cls, type) and issubclass(cls, DifferentiatorBackend)):
        raise TypeError(
            f"Expected a DifferentiatorBackend subclass, got {cls!r}"
        )
    
    if name in _DIFF_BACKEND_REGISTRY:
        # Consider adding a warning for duplicate registration
        import warnings
        warnings.warn(
            f"Overwriting existing differentiator backend: {name!r}",
            UserWarning,
            stacklevel=2
        )
    
    _DIFF_BACKEND_REGISTRY[name] = cls


def get_differentiator_backend(name: str, **kwargs: Any) -> DifferentiatorBackend:
    """Instantiate a differentiator backend by name.

    Args:
        name: Registered backend name (e.g. ``"kkt"``).
        **kwargs: Passed to the backend constructor.
            Common kwargs include:
            - solver: Linear solver to use (e.g., "splu", "umfpack")
            - use_csc: Whether to use CSC format (sparse backends)
            - tol: Tolerance for linear solver

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If *name* is not registered.

    Example:
        >>> backend = get_differentiator_backend("kkt", solver="umfpack")
        >>> backend.setup(fixed_elements=...)
    """
    if name not in _DIFF_BACKEND_REGISTRY:
        available = sorted(_DIFF_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown differentiator backend: {name!r}. "
            f"Available backends: {available}"
        )
    
    backend_class = _DIFF_BACKEND_REGISTRY[name]
    return backend_class(**kwargs)


def list_available_backends() -> list[str]:
    """List all registered differentiator backends.

    Returns:
        Sorted list of registered backend names.

    Example:
        >>> backends = list_available_backends()
        >>> print(f"Available: {backends}")
    """
    return sorted(_DIFF_BACKEND_REGISTRY.keys())