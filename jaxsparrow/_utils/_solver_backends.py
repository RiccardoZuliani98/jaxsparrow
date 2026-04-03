"""
utils/_solver_backends.py
=========================
Abstract QP solver backend protocol and concrete implementations.

The two-phase protocol separates solver lifecycle into:

1. **setup** — one-time initialization: receive the fixed QP
   ingredients as a dict, cast and store them, perform symbolic
   analysis, pre-factorize, allocate workspace.

2. **solve** — per-call: receive runtime QP ingredients as
   keyword arguments, merge them with the stored fixed elements,
   build the problem, run the numerical solver and return the
   primal/dual solution.

Backends
--------
- ``QpSolversBackend``: wraps the ``qpsolvers`` library (stateless,
  rebuilds the ``Problem`` object on every solve). This is the
  default and reproduces the existing behaviour.

Future backends (OSQP, PIQP, Clarabel, …) can exploit the
setup/solve split for significant speedups by reusing symbolic
factorizations and only updating numerical values between solves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Callable, Optional, Union, cast

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix, issparse

from qpsolvers import Problem, solve_problem

from jaxsparrow._solver_sparse._types import SparseIngredientsNP
from jaxsparrow._solver_dense._types import DenseIngredientsNP
from jaxsparrow._options_common import SolverOptions


# =====================================================================
# Abstract protocol
# =====================================================================

class SolverBackend(ABC):
    """Abstract base class for solver backends.

    Subclasses implement a two-phase lifecycle:

    1. :meth:`__init__` — receive fully resolved options.
    2. :meth:`setup` — one-time structural initialization.
    3. :meth:`solve` — per-call numerical solve.

    The ``__init__`` call receives the fully resolved options dict
    (after merging user-supplied values with backend-specific
    defaults).  Concrete backends should narrow the type annotation
    to their own ``*OptionsFull`` TypedDict so the type checker
    knows which keys are available.

    The ``setup`` call receives the fixed QP ingredients as a
    dict.  Backends should cast the values to the configured dtype,
    store them, and perform any symbolic analysis or workspace
    allocation.

    The ``solve`` call receives runtime QP ingredients as keyword
    arguments, merges them with the stored fixed elements, builds
    the problem, runs the solver, and returns the raw solution
    arrays plus a timing dict.
    """

    @abstractmethod
    def __init__(self, options: SolverOptions) -> None:
        """Receive fully resolved backend options.

        Args:
            options: Fully resolved options dict (all keys present).
                Concrete backends should narrow this type to their
                own ``*OptionsFull`` TypedDict.
        """
        ...

    @abstractmethod
    def setup(
        self,
        fixed_elements: Optional[SparseIngredientsNP | DenseIngredientsNP] = None,
    ) -> dict[str, float]:
        """One-time structural initialization.

        Called once at construction with the QP ingredients that
        remain constant across calls.  Backends should cast sparse
        matrices to CSC and dense vectors to 1-D arrays of the
        configured dtype, store the results, and perform any
        symbolic analysis or workspace allocation.

        Args:
            fixed_elements: Dict of fixed QP ingredients.  Keys are
                a subset of ``{"P", "q", "A", "b", "G", "h"}``.
                Sparse matrices may arrive in any format; dense
                vectors may have extra dimensions.  ``None`` is
                equivalent to an empty dict.

        Returns:
            A timing dict with backend-specific keys (e.g.
            ``"symbolic_factorization"``, ``"workspace_alloc"``).
            May be empty for stateless backends.
        """
        ...

    @abstractmethod
    def solve(
        self,
        **kwargs: ndarray,
    ) -> tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], dict[str, float]]:
        """Run the numerical solve.

        Receives runtime QP ingredients as keyword arguments (those
        not fixed at construction).  An optional ``"warmstart"`` key
        may supply an initial guess for the primal variable.

        Implementations should merge *kwargs* with the stored fixed
        elements, build the problem, and run the solver.

        Args:
            **kwargs: Runtime QP ingredients plus an optional
                ``warmstart`` array of shape ``(n_var,)``.

        Returns:
            A tuple ``(x, y, z, timing)`` where:

            - ``x``: primal solution, shape ``(n_var,)``, or
              ``None`` if the solver failed.
            - ``y``: equality multipliers, shape ``(n_eq,)``, or
              ``None``.
            - ``z``: inequality multipliers, shape ``(n_ineq,)``,
              or ``None``.
            - ``timing``: dict with per-phase timing keys.
        """
        ...


# =====================================================================
# qpsolvers backend (stateless, default)
# =====================================================================

from jaxsparrow._solver_dense._options import DenseQpSolverOptionsFull
from jaxsparrow._solver_sparse._options import SparseQpSolverOptionsFull


class QpSolversBackend(SolverBackend):
    """Stateless backend wrapping the ``qpsolvers`` library.

    Rebuilds the ``Problem`` object on every :meth:`solve` by
    merging the stored fixed elements with the runtime keyword
    arguments.  This reproduces the existing behaviour and serves
    as the baseline implementation.

    Args:
        options: Fully resolved solver options dict.  Expected
            keys: ``"solver_name"`` (str), ``"dtype"``
            (NumPy floating dtype).  The ``"backend"`` key is
            ignored (already used for dispatch).
    """

    def __init__(self, options: DenseQpSolverOptionsFull| SparseQpSolverOptionsFull) -> None:
        self._solver_name: str = options["solver_name"]
        self._dtype: type[np.floating] = options["dtype"] #type: ignore

        # Fixed elements stored at setup time
        self._fixed: SparseIngredientsNP = {}

    def _store_matrix(
        self, val: Union[ndarray, csc_matrix],
    ) -> Union[ndarray, csc_matrix]:
        """Cast a matrix to the configured dtype, keeping format."""
        if issparse(val):
            return csc_matrix(val, dtype=self._dtype)
        return np.asarray(val, dtype=self._dtype)

    def _store_vector(self, val: ndarray) -> ndarray:
        """Cast a vector to the configured dtype and squeeze."""
        return np.atleast_1d(np.asarray(val, dtype=self._dtype).squeeze())

    # ── Lifecycle ────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[SparseIngredientsNP] = None,
    ) -> dict[str, float]:
        """Cast and store the fixed QP ingredients.

        No pre-computation is performed for this stateless backend.
        """
        start: float = perf_counter()

        self._fixed = {}
        for k, v in (fixed_elements or {}).items():
            if issparse(v):
                self._fixed[k] = self._store_matrix(v) #type: ignore
            else:
                self._fixed[k] = self._store_vector(v) #type: ignore

        return {"setup": perf_counter() - start}

    def solve(
        self,
        **kwargs: ndarray,
    ) -> tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], dict[str, float]]:
        """Merge fixed + runtime elements, build a ``qpsolvers.Problem``, and solve."""
        t: dict[str, float] = {}

        # ── Build problem ────────────────────────────────────────────
        start: float = perf_counter()
        warmstart: Optional[ndarray] = kwargs.pop("warmstart", None)

        # Merge fixed + runtime
        merged = cast(SparseIngredientsNP, {**self._fixed, **kwargs})

        assert "P" in merged and "q" in merged, (
            "P and q are required" \
            "Provide them via fixed_elements or as dynamic arguments."
        )

        b_val = merged.get("b")
        h_val = merged.get("h")

        prob: Problem = Problem(
            P=merged["P"],
            q=np.atleast_1d(merged["q"]),
            A=merged.get("A"),
            b=np.atleast_1d(b_val) if b_val is not None else None,
            G=merged.get("G"),
            h=np.atleast_1d(h_val) if h_val is not None else None,
        )
        t["problem_setup"] = perf_counter() - start

        # ── Solve ────────────────────────────────────────────────────
        start = perf_counter()
        sol = solve_problem(
            prob,
            solver=self._solver_name,
            initvals=warmstart,
        )
        t["solver"] = perf_counter() - start

        if not sol.found:
            return None, None, None, t

        return sol.x, sol.y, sol.z, t


# =====================================================================
# Registry and factory
# =====================================================================

# Type alias for backend constructors: each takes a SolverOptions
# dict and returns a SolverBackend instance.
SolverBackendFactory = Callable[[SolverOptions], SolverBackend]

_BACKEND_REGISTRY: dict[str, SolverBackendFactory] = {
    "qpsolvers": QpSolversBackend,
}


def register_backend(name: str, cls: SolverBackendFactory) -> None:
    """Register a new QP solver backend.

    Args:
        name: Short identifier for the backend (e.g. ``"osqp"``,
            ``"piqp"``).
        cls: The backend class or callable.  Must accept a single
            ``options: SolverOptions`` argument and return a
            :class:`SolverBackend` instance.

    Raises:
        TypeError: If *cls* is not a subclass of :class:`SolverBackend`.
    """
    if isinstance(cls, type) and not issubclass(cls, SolverBackend):
        raise TypeError(
            f"Expected a SolverBackend subclass, got {cls!r}"
        )
    _BACKEND_REGISTRY[name] = cls


def get_backend(name: str, options: SolverOptions) -> SolverBackend:
    """Instantiate a QP solver backend by name.

    Args:
        name: Registered backend name.
        options: Fully resolved solver options dict.  Passed
            directly to the backend constructor.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown QP backend: {name!r}. "
            f"Available: {sorted(_BACKEND_REGISTRY)}."
        )
    return _BACKEND_REGISTRY[name](options)