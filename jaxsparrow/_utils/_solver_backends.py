"""
utils/_qp_backends.py
=====================
Abstract QP solver backend protocol and concrete implementations.

The three-phase protocol separates solver lifecycle into:

1. **setup** — one-time initialization: receive the full problem
   structure (sparsity patterns, dimensions, fixed elements),
   perform symbolic analysis, pre-factorize, allocate workspace.

2. **solve** — per-call: run the numerical solver using the
   current parameter state and return the primal/dual solution.

Backends
--------
- ``QpSolversBackend``: wraps the ``qpsolvers`` library (stateless,
  rebuilds the ``Problem`` object on every solve). This is the
  default and reproduces the existing behaviour.

Future backends (OSQP, PIQP, Clarabel, …) can exploit the
setup/update split for significant speedups by reusing symbolic
factorizations and only updating numerical values between solves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Optional, Union

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix, issparse

from qpsolvers import Problem, solve_problem


# =====================================================================
# Abstract protocol
# =====================================================================

class SolverBackend(ABC):
    """Abstract base class for solver backends.

    Subclasses implement a three-phase lifecycle:

    1. :meth:`setup` — one-time structural initialization.
    2. :meth:`solve` — per-call numerical solve.

    The ``setup`` call receives the full problem structure (matrices
    with their sparsity patterns, vectors, dimensions). Backends may
    store references, perform symbolic factorization, allocate
    workspace, etc.

    The ``solve`` call runs the solver and returns the raw solution
    arrays plus a timing dict.
    """

    @abstractmethod
    def setup(
        self,
        P: Optional[Union[ndarray, csc_matrix]] = None,
        q: Optional[ndarray] = None,
        A: Optional[Union[ndarray, csc_matrix]] = None,
        b: Optional[ndarray] = None,
        G: Optional[Union[ndarray, csc_matrix]] = None,
        h: Optional[ndarray] = None,
    ) -> dict[str, float]:
        """One-time structural initialization.

        Called once at construction with the full problem structure.
        Backends should store the sparsity patterns (for matrices)
        and initial values, and perform any symbolic analysis or
        workspace allocation.

        Args:
            P: Objective Hessian. Sparse CSC or dense.
            q: Objective linear term, shape ``(n_var,)``.
            A: Equality constraint matrix, or ``None``.
            b: Equality constraint RHS, or ``None``.
            G: Inequality constraint matrix, or ``None``.
            h: Inequality constraint RHS, or ``None``.

        Returns:
            A timing dict with backend-specific keys (e.g.
            ``"symbolic_factorization"``, ``"workspace_alloc"``).
            May be empty for stateless backends.
        """
        ...

    @abstractmethod
    def solve(
        self,
        warmstart: Optional[ndarray] = None,
    ) -> tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], dict[str, float]]:
        """Run the numerical solve.

        Uses the current parameter state (from the most recent
        :meth:`setup` + :meth:`update` calls).

        Args:
            warmstart: Optional initial guess for the primal
                variable ``x``, shape ``(n_var,)``.

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

class QpSolversBackend(SolverBackend):
    """Stateless backend wrapping the ``qpsolvers`` library.

    Rebuilds the ``Problem`` object on every :meth:`solve`. This
    reproduces the existing behaviour and serves as the baseline
    implementation.

    The ``setup`` and ``update`` calls simply store parameter values
    internally; no pre-computation is performed.

    Args:
        solver_name: Backend solver name passed to
            ``qpsolvers.solve_problem`` (e.g. ``"piqp"``, ``"osqp"``,
            ``"clarabel"``).
        dtype: NumPy floating-point dtype for all arrays.
    """

    def __init__(self, solver_name: str, dtype: type[np.floating] = np.float64) -> None:
        self._solver_name: str = solver_name
        self._dtype: type[np.floating] = dtype

        # Current parameter state
        self._P: Optional[Union[ndarray, csc_matrix]] = None
        self._q: Optional[ndarray] = None
        self._A: Optional[Union[ndarray, csc_matrix]] = None
        self._b: Optional[ndarray] = None
        self._G: Optional[Union[ndarray, csc_matrix]] = None
        self._h: Optional[ndarray] = None

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
        P: Optional[Union[ndarray, csc_matrix]] = None,
        q: Optional[ndarray] = None,
        A: Optional[Union[ndarray, csc_matrix]] = None,
        b: Optional[ndarray] = None,
        G: Optional[Union[ndarray, csc_matrix]] = None,
        h: Optional[ndarray] = None,
    ) -> dict[str, float]:
        """Store the full problem. No pre-computation for this backend."""
        start: float = perf_counter()

        self._P = self._store_matrix(P) if P is not None else None
        self._q = self._store_vector(q) if q is not None else None
        self._A = self._store_matrix(A) if A is not None else None
        self._b = self._store_vector(b) if b is not None else None
        self._G = self._store_matrix(G) if G is not None else None
        self._h = self._store_vector(h) if h is not None else None

        return {"setup": perf_counter() - start}

    def solve(
        self,
        warmstart: Optional[ndarray] = None,
    ) -> tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], dict[str, float]]:
        """Build a ``qpsolvers.Problem`` and solve it."""
        t: dict[str, float] = {}

        # ── Build Problem ────────────────────────────────────────────
        start: float = perf_counter()
        prob: Problem = Problem(
            P=self._P,
            q=np.atleast_1d(self._q),
            A=self._A,
            b=np.atleast_1d(self._b) if self._b is not None else self._b,
            G=self._G,
            h=np.atleast_1d(self._h) if self._h is not None else self._h,
        )
        t["problem_build"] = perf_counter() - start

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

_BACKEND_REGISTRY: dict[str, type[SolverBackend]] = {
    "qpsolvers": QpSolversBackend,
}


def register_backend(name: str, cls: type[SolverBackend]) -> None:
    """Register a new QP solver backend.

    Args:
        name: Short identifier for the backend (e.g. ``"osqp"``,
            ``"piqp"``).
        cls: The backend class (must subclass ``QPSolverBackend``).

    Raises:
        TypeError: If *cls* is not a subclass of ``QPSolverBackend``.
    """
    if not (isinstance(cls, type) and issubclass(cls, SolverBackend)):
        raise TypeError(
            f"Expected a QPSolverBackend subclass, got {cls!r}"
        )
    _BACKEND_REGISTRY[name] = cls


def get_backend(name: str, **kwargs: Any) -> SolverBackend:
    """Instantiate a QP solver backend by name.

    Args:
        name: Registered backend name.
        **kwargs: Passed to the backend constructor.

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
    return _BACKEND_REGISTRY[name](**kwargs)