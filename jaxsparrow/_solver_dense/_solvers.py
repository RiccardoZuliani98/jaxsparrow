"""
solver_dense/_solvers.py
========================
Dense solver factory.

Creates a closure that solves a convex problem in standard-form::

    min  0.5 x^T P x + q^T x
    s.t. A x = b
         G x <= h

using a pluggable :class:`SolverBackend`.  The backend owns
dtype casting and storage of fixed elements, merging of fixed and
dynamic parameters, and ``qpsolvers.Problem`` construction.  The
closure extracts the primal/dual solution, and returns timing 
information.
"""

from time import perf_counter
import numpy as np
from numpy import ndarray
from jaxtyping import Float, Bool
from typing import cast, Optional, Protocol
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import SolverOptions
from jaxsparrow._solver_dense._options import (
    SOLVER_OPTIONS_DEFAULTS,
    DEFAULT_SOLVER_BACKEND,
)
from jaxsparrow._solver_dense._types import DenseIngredientsNP
from jaxsparrow._types_common import SolverOutputNP
from jaxsparrow._utils._solver_backends import SolverBackend, get_backend


# ── Callable protocol for the returned closure ──────────────────────

class DenseSolverFn(Protocol):
    """Signature of the closure returned by :func:`create_dense_solver`."""

    def __call__(self, **kwargs: ndarray) -> tuple[SolverOutputNP, dict[str, float]]: ...


# ── Helpers ──────────────────────────────────────────────────────────

def _resolve_backend_defaults(
    options: Optional[SolverOptions],
) -> tuple[str, SolverOptions]:
    """Determine the solver backend name and its matching defaults.

    The backend is read from ``options["backend"]`` if present,
    otherwise ``DEFAULT_SOLVER_BACKEND`` is used.  The returned
    defaults dict is the one registered in
    :data:`SOLVER_OPTIONS_DEFAULTS` for that backend.

    Returns:
        ``(backend_name, default_options)``

    Raises:
        KeyError: If the resolved backend name has no entry in
            :data:`SOLVER_OPTIONS_DEFAULTS`.
    """
    if options is not None and "backend" in options:
        backend_name: str = options["backend"]
    else:
        backend_name = DEFAULT_SOLVER_BACKEND

    if backend_name not in SOLVER_OPTIONS_DEFAULTS:
        raise KeyError(
            f"Unknown solver backend {backend_name!r}.  "
            f"Available backends: {sorted(SOLVER_OPTIONS_DEFAULTS)}"
        )

    return backend_name, SOLVER_OPTIONS_DEFAULTS[backend_name]


# ── Factory ──────────────────────────────────────────────────────────

def create_dense_solver(
    n_eq: int,
    n_ineq: int,
    options: Optional[SolverOptions] = None,
    fixed_elements: Optional[DenseIngredientsNP] = None,
) -> DenseSolverFn:
    """Create a dense solver closure.

    Builds a callable that passes dynamic parameters to a
    :class:`DenseSolverBackend`, which merges them with the stored
    fixed elements, solves the problem, and returns the raw solution.
    The closure then extracts the primal/dual variables, and returns 
    timing information.

    The solver lifecycle is delegated to a :class:`DenseSolverBackend`:

    - **setup** is called once at construction time.  The backend
      receives *fixed_elements* and is responsible for casting
      dense arrays to the configured dtype, storing the results,
      and performing any pre-computation.
    - **solve** is called at each invocation with the runtime
      ingredients supplied via ``**kwargs``.  The backend merges
      them with the stored fixed elements, builds the
      ``qpsolvers.Problem``, and runs the numerical solver.

    Each backend has its own default options; user-supplied keys
    override the backend-specific defaults.

    Args:
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Solver options.  The ``"backend"`` key selects the
            solver backend protocol (default: ``"qpsolvers"``).
            Remaining keys are merged with that backend's defaults.
        fixed_elements: ingredients that remain constant across
            calls (e.g. constraint matrices that do not change).
            Passed directly to the backend's :meth:`setup` call;
            the backend is responsible for dtype casting and storage.
            Any key present here should *not* be passed again at
            call time.

    Returns:
        A callable with signature::

            solver_numpy(**kwargs: ndarray)
                -> tuple[SolverOutputNP, dict[str, float]]

        where ``SolverOutputNP`` is ``(x, lam, mu)`` and
        the dict contains per-phase timing keys: ``"setup.*"``
        (from construction), ``"solve.*"``, ``"retrieve"``.
    """

    # ── Resolve backend and parse options ────────────────────────────
    backend_name, defaults = _resolve_backend_defaults(options)
    options_parsed = parse_options(options, defaults)
    _dtype: type[np.floating] = options_parsed["dtype"]

    # ── Create backend ───────────────────────────────────────────────
    backend: SolverBackend = get_backend(
        backend_name,
        options=options_parsed,
    )

    # ── Setup: pass fixed elements to the backend (once, now) ────────
    #
    # The backend owns dtype casting and storage of fixed elements.

    _setup_timing: dict[str, float] = backend.setup(
        fixed_elements=fixed_elements,
    )

    # ─────────────────────────────────────────────────────────────────

    def solver_numpy(**kwargs: ndarray) -> tuple[SolverOutputNP, dict[str, float]]:
        """Solve a dense solver and return the primal/dual solution.

        Passes runtime ingredients to the backend, which merges
        them with the stored fixed elements, builds the problem,
        and runs the solver.

        Args:
            **kwargs: Dynamic parameters (any subset of
                ``P, q, A, b, G, h``) as NumPy arrays, plus an
                optional ``warmstart`` array for the primal initial
                guess. These are merged with *fixed_elements*
                provided at construction time.

        Returns:
            A tuple ``(sol, timing)`` where ``sol`` is
            ``(x, lam, mu)`` and ``timing`` maps phase
            names to elapsed seconds.

        Raises:
            AssertionError: If the solver fails to find a
                solution.
        """

        # preallocate dictionary with computation times
        t: dict[str, float] = {}

        # Propagate setup timings
        t.update({f"setup.{k}": v for k, v in _setup_timing.items()})

        # ── Solve ────────────────────────────────────────────────────
        x_raw, y_raw, z_raw, solve_timing = backend.solve(**kwargs)
        t.update({f"solve.{k}": v for k, v in solve_timing.items()})

        assert x_raw is not None, "solver failed to find a solution."

        # ── Extract solution ─────────────────────────────────────────
        start: float = perf_counter()

        x: Float[ndarray, "n_var"] = np.asarray(x_raw, dtype=_dtype).reshape(-1)

        mu: Float[ndarray, "n_eq"]
        if n_eq > 0:
            mu = np.asarray(y_raw, dtype=_dtype).reshape(-1)
        else:
            mu = np.empty(0, dtype=_dtype)

        lam: Float[ndarray, "n_ineq"]
        if n_ineq > 0:
            lam = np.asarray(z_raw, dtype=_dtype).reshape(-1)
        else:
            lam = np.empty(0, dtype=_dtype)
        t["retrieve"] = perf_counter() - start

        return cast(SolverOutputNP, (x, lam, mu)), t

    return solver_numpy