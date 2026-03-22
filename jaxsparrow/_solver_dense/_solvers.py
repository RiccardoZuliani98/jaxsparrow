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
closure extracts the primal/dual solution, computes the active-set
mask, and returns timing information.
"""

from time import perf_counter
import numpy as np
from numpy import ndarray
from jaxtyping import Float, Bool
from typing import cast, Optional, Protocol
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import SolverOptions
from jaxsparrow._solver_dense._types import DenseIngredientsNP
from jaxsparrow._types_common import SolverOutputNP
from jaxsparrow._utils._solver_backends import SolverBackend, get_backend


# ── Solver options ───────────────────────────────────────────────────

class DenseSolverOptions(SolverOptions):
    """Partial solver options for the dense path.

    All keys are optional; missing keys are filled from
    ``DEFAULT_SOLVER_OPTIONS`` via :func:`parse_options`.
    """
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float


class DenseSolverOptionsFull(SolverOptions, total=True):
    """Complete solver options for the dense path.

    All keys are required. This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        solver_name: Backend solver name passed to ``qpsolvers``
            (e.g. ``"piqp"``, ``"osqp"``, ``"clarabel"``).
        dtype: NumPy floating-point dtype for all arrays.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
    """
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float


DEFAULT_SOLVER_OPTIONS: DenseSolverOptionsFull = {
    "solver_name": "piqp",
    "dtype": np.float64,
    "bool_dtype": np.bool_,
    "cst_tol": 1e-8,
}


# ── Callable protocol for the returned closure ──────────────────────

class DenseSolverFn(Protocol):
    """Signature of the closure returned by :func:`create_dense_solver`."""

    def __call__(self, **kwargs: ndarray) -> tuple[SolverOutputNP, dict[str, float]]: ...


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
    The closure then extracts the primal/dual variables, computes
    the active-set mask, and returns timing information.

    The solver lifecycle is delegated to a :class:`DenseSolverBackend`:

    - **setup** is called once at construction time.  The backend
      receives *fixed_elements* and is responsible for casting
      dense arrays to the configured dtype, storing the results,
      and performing any pre-computation.
    - **solve** is called at each invocation with the runtime
      ingredients supplied via ``**kwargs``.  The backend merges
      them with the stored fixed elements, builds the
      ``qpsolvers.Problem``, and runs the numerical solver.

    Args:
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Solver options (solver backend name, dtype,
            constraint tolerance). Defaults are filled for missing
            keys from ``DEFAULT_SOLVER_OPTIONS``.
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

        where ``SolverOutputNP`` is ``(x, lam, mu, active)`` and
        the dict contains per-phase timing keys: ``"setup.*"``
        (from construction), ``"solve.*"``, ``"retrieve"``,
        ``"active_set"``.
    """

    # parse options
    options_parsed: DenseSolverOptionsFull = parse_options(options, DEFAULT_SOLVER_OPTIONS)
    _dtype: type[np.floating] = options_parsed["dtype"]
    _bool_dtype: type[np.bool_] = options_parsed["bool_dtype"]

    # ── Create backend ───────────────────────────────────────────────

    backend_name: str = options_parsed.get("backend", "qpsolvers")
    backend: SolverBackend = get_backend(
        backend_name,
        solver_name=options_parsed["solver_name"],
        dtype=_dtype,
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
            ``(x, lam, mu, active)`` and ``timing`` maps phase
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

        # ── Active set ───────────────────────────────────────────────
        start = perf_counter()
        active: Bool[ndarray, "n_ineq"]
        if n_ineq > 0:
            # G and h must have been provided (fixed or runtime);
            # look up from kwargs first, falling back to fixed_elements.
            merged_G = kwargs.get("G", (fixed_elements or {}).get("G"))
            merged_h = kwargs.get("h", (fixed_elements or {}).get("h"))
            assert merged_G is not None and merged_h is not None, (
                "G and h are required when n_ineq > 0"
            )
            active = np.asarray(
                np.abs(merged_G @ x_raw - merged_h)
                <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start

        return cast(SolverOutputNP, (x, lam, mu, active)), t

    return solver_numpy