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
from typing import Any, Optional, Sequence, Union

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

IngredientsNP = Union[SparseIngredientsNP,DenseIngredientsNP]
IngredientsTangentsNP = Union[SparseIngredientsTangentsNP,DenseIngredientsTangentsNP]


# =====================================================================
# Abstract protocol
# =====================================================================

class DifferentiatorBackend(ABC):
    """Abstract base class for differentiator backends.

    Subclasses implement a two-phase lifecycle:

    1. :meth:`setup` — one-time initialization with fixed elements.
    2. :meth:`differentiate_fwd` / :meth:`differentiate_rev` —
       per-call forward or reverse differentiation.
    """

    @abstractmethod
    def setup(
        self,
        fixed_elements: Optional[IngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """One-time initialization.

        Args:
            fixed_elements: QP ingredients constant across calls.
            dynamic_keys: Keys for which gradients are needed.
                ``None`` means all keys.
            sparsity_info: Per-key sparsity info from BCOO patterns
                (for dynamic sparse keys).

        Returns:
            Timing dict.
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
        """Forward-mode (JVP) differentiation."""
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
        """Reverse-mode (VJP) differentiation."""
        ...


# =====================================================================
# Registry and factory
# =====================================================================

_DIFF_BACKEND_REGISTRY: dict[str, type[DifferentiatorBackend]] = {}


def register_differentiator_backend(
    name: str, cls: type[DifferentiatorBackend],
) -> None:
    """Register a new differentiator backend."""
    if not (isinstance(cls, type) and issubclass(cls, DifferentiatorBackend)):
        raise TypeError(
            f"Expected a DifferentiatorBackend subclass, got {cls!r}"
        )
    _DIFF_BACKEND_REGISTRY[name] = cls


def get_differentiator_backend(name: str, **kwargs: Any) -> DifferentiatorBackend:
    """Instantiate a differentiator backend by name.

    Args:
        name: Registered backend name (e.g. ``"kkt"``).
        **kwargs: Passed to the backend constructor.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _DIFF_BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown differentiator backend: {name!r}. "
            f"Available: {sorted(_DIFF_BACKEND_REGISTRY)}."
        )
    return _DIFF_BACKEND_REGISTRY[name](**kwargs)