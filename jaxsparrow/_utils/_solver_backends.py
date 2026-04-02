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

- ``QOCOBackend``: wraps the QOCO solver
  (https://github.com/qoco-org/qoco-python).  Exploits the
  setup/solve split: the first call performs a full ``setup`` and
  subsequent calls use ``update_vector_data`` /
  ``update_matrix_data`` for efficient re-solves.

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

    def __init__(self, options: DenseQpSolverOptionsFull) -> None:
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
# QOCO backend (stateful, setup/solve split)
# =====================================================================

class QOCOBackend(SolverBackend):
    """Backend wrapping the QOCO conic solver for QP problems.

    QOCO solves second-order cone programs with quadratic
    objectives.  A standard QP (with linear inequality and
    equality constraints) is a special case where the cone is
    the non-negative orthant (no second-order cone constraints).

    This backend exploits QOCO's setup/solve split:

    - The **first** call to :meth:`solve` performs a full
      ``qoco.QOCO.setup()`` which includes symbolic analysis
      and workspace allocation.
    - **Subsequent** calls use ``update_vector_data`` and/or
      ``update_matrix_data`` to update only the numerical
      values before re-solving, avoiding redundant symbolic
      work.

    Problem mapping (QP → QOCO)::

        QP standard form           QOCO standard form
        ─────────────────          ──────────────────
        min ½ x'Px + q'x          min ½ x'Px + c'x
        s.t. Ax = b               s.t. Ax = b
             Gx ≤ h                    Gx ≤_C h

    The cone C is set to R^l_+ (the non-negative orthant with
    ``l = n_ineq``, ``nsoc = 0``, ``q = []``), so the conic
    inequality reduces to element-wise ``Gx ≤ h``.

    QOCO names the linear cost vector ``c``; this backend maps
    the QP ``q`` vector to QOCO's ``c``.

    Args:
        options: Fully resolved solver options dict.  Expected
            keys: ``"dtype"`` (NumPy floating dtype).  Optional
            QOCO settings (e.g. ``"verbose"``, ``"abstol"``,
            ``"reltol"``) can be passed and are forwarded to
            ``qoco.QOCO.update_settings``.
    """

    def __init__(self, options: SolverOptions) -> None:
        self._dtype: type[np.floating] = options.get("dtype", np.float64) #type: ignore

        # QOCO solver settings forwarded to update_settings.
        # Filter out keys consumed by the backend protocol.
        _RESERVED = frozenset({"backend", "solver_name", "dtype"})
        self._qoco_settings: dict = {
            k: v for k, v in options.items()
            if k not in _RESERVED
        }
        # Default to quiet output unless the caller asked for verbose.
        self._qoco_settings.setdefault("verbose", 0)

        # Fixed elements stored at setup time
        self._fixed: SparseIngredientsNP = {}

        # Problem dimensions (set during first solve)
        self._n: int = 0       # number of variables
        self._n_eq: int = 0    # number of equality constraints
        self._n_ineq: int = 0  # number of inequality constraints

        # QOCO solver instance — lazily created on first solve
        self._solver: Optional["qoco.QOCO"] = None  # type: ignore[name-defined]
        self._setup_done: bool = False

    # ── Helpers ──────────────────────────────────────────────────────

    def _to_csc(
        self, val: Union[ndarray, csc_matrix],
    ) -> csc_matrix:
        """Ensure a matrix is CSC with the configured dtype."""
        if issparse(val):
            return csc_matrix(val, dtype=self._dtype)
        return csc_matrix(np.asarray(val, dtype=self._dtype))

    def _to_vec(self, val: ndarray) -> ndarray:
        """Cast a vector to the configured dtype and squeeze."""
        return np.atleast_1d(np.asarray(val, dtype=self._dtype).squeeze())

    # ── Lifecycle ────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[SparseIngredientsNP] = None,
    ) -> dict[str, float]:
        """Cast and store the fixed QP ingredients.

        The actual QOCO solver setup is deferred to the first
        :meth:`solve` call, because dimensions may not be known
        until all ingredients (fixed + dynamic) are available.
        """
        start: float = perf_counter()

        self._fixed = {}
        for k, v in (fixed_elements or {}).items():
            if issparse(v) or (isinstance(v, ndarray) and v.ndim == 2):
                self._fixed[k] = self._to_csc(v) #type: ignore
            else:
                self._fixed[k] = self._to_vec(v) #type: ignore

        # Reset solver so next solve triggers a fresh QOCO setup
        self._solver = None
        self._setup_done = False

        return {"setup": perf_counter() - start}

    def _do_qoco_setup(self, merged: dict) -> dict[str, float]:
        """Perform the one-time QOCO solver setup."""
        import qoco

        t: dict[str, float] = {}
        start = perf_counter()

        P_csc: Optional[csc_matrix] = self._to_csc(merged["P"]) if "P" in merged else None
        c_vec: ndarray = self._to_vec(merged["q"])  # QP 'q' → QOCO 'c'

        A_csc: Optional[csc_matrix] = None
        b_vec: Optional[ndarray] = None
        G_csc: Optional[csc_matrix] = None
        h_vec: Optional[ndarray] = None

        if "A" in merged:
            A_csc = self._to_csc(merged["A"])
            b_vec = self._to_vec(merged["b"]) if "b" in merged else np.zeros(A_csc.shape[0], dtype=self._dtype)
            self._n_eq = A_csc.shape[0]
        else:
            self._n_eq = 0

        if "G" in merged:
            G_csc = self._to_csc(merged["G"])
            h_vec = self._to_vec(merged["h"]) if "h" in merged else np.zeros(G_csc.shape[0], dtype=self._dtype)
            self._n_ineq = G_csc.shape[0]
        else:
            self._n_ineq = 0

        self._n = c_vec.shape[0]

        self._solver = qoco.QOCO()
        self._solver.setup(
            n=self._n,
            m=self._n_ineq,       # total cone dimension = n_ineq (orthant only)
            p=self._n_eq,
            P=P_csc,
            c=c_vec,
            A=A_csc,
            b=b_vec,
            G=G_csc,
            h=h_vec,
            l=self._n_ineq,       # non-negative orthant dimension
            nsoc=0,               # no second-order cones
            q=None,               # no SOC sizes
            **self._qoco_settings,
        )
        self._setup_done = True
        t["qoco_setup"] = perf_counter() - start
        return t

    def _update_data(self, merged: dict) -> dict[str, float]:
        """Update QOCO solver data for a re-solve (same sparsity)."""
        t: dict[str, float] = {}
        start = perf_counter()

        assert self._solver is not None

        # Determine which matrices need updating (compare against fixed)
        mat_updates: dict[str, Optional[ndarray]] = {}
        for key in ("P", "A", "G"):
            if key in merged and key not in self._fixed:
                # Dynamic matrix — extract upper-triangular data for P
                mat_csc = self._to_csc(merged[key])
                if key == "P":
                    from scipy.sparse import triu
                    mat_csc = triu(mat_csc, format="csc")
                mat_updates[key] = mat_csc.data.astype(self._dtype)

        if mat_updates:
            self._solver.update_matrix_data(
                P=mat_updates.get("P"),
                A=mat_updates.get("A"),
                G=mat_updates.get("G"),
            )

        # Update vectors
        vec_updates: dict[str, Optional[ndarray]] = {}
        if "q" in merged and "q" not in self._fixed:
            vec_updates["c"] = self._to_vec(merged["q"])
        if "b" in merged and "b" not in self._fixed:
            vec_updates["b"] = self._to_vec(merged["b"])
        if "h" in merged and "h" not in self._fixed:
            vec_updates["h"] = self._to_vec(merged["h"])

        if vec_updates:
            self._solver.update_vector_data(
                c=vec_updates.get("c"),
                b=vec_updates.get("b"),
                h=vec_updates.get("h"),
            )

        t["qoco_update"] = perf_counter() - start
        return t

    def solve(
        self,
        **kwargs: ndarray,
    ) -> tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], dict[str, float]]:
        """Merge fixed + runtime elements, solve with QOCO.

        On the first call the full QOCO setup is performed.
        Subsequent calls update only the changed data and re-solve.
        """
        t: dict[str, float] = {}

        # Pop warmstart (QOCO does not support external warmstart,
        # but we accept the key for API compatibility).
        _: Optional[ndarray] = kwargs.pop("warmstart", None)

        # Merge fixed + runtime
        merged = {**self._fixed, **kwargs}

        assert "P" in merged and "q" in merged, (
            "P and q are required. "
            "Provide them via fixed_elements or as dynamic arguments."
        )

        # ── Setup or update ──────────────────────────────────────────
        if not self._setup_done:
            t.update(self._do_qoco_setup(merged))
        else:
            t.update(self._update_data(merged))

        # ── Solve ────────────────────────────────────────────────────
        start = perf_counter()
        result = self._solver.solve()  # type: ignore[union-attr]
        t["solver"] = perf_counter() - start

        if result.status not in ("QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"):
            return None, None, None, t

        x = np.array(result.x, dtype=self._dtype)

        # QOCO returns y for equality duals, z for inequality (cone) duals
        y: Optional[ndarray] = None
        z: Optional[ndarray] = None

        if self._n_eq > 0 and result.y is not None and len(result.y) > 0:
            y = np.array(result.y, dtype=self._dtype)
        if self._n_ineq > 0 and result.z is not None and len(result.z) > 0:
            z = np.array(result.z, dtype=self._dtype)

        return x, y, z, t


# =====================================================================
# Registry and factory
# =====================================================================

# Type alias for backend constructors: each takes a SolverOptions
# dict and returns a SolverBackend instance.
SolverBackendFactory = Callable[[SolverOptions], SolverBackend]

_BACKEND_REGISTRY: dict[str, SolverBackendFactory] = {
    "qpsolvers": QpSolversBackend, #type: ignore
    "qoco": QOCOBackend, #type: ignore
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