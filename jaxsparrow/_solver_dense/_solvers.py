"""
solver_dense/_solvers.py
========================
Dense QP solver factory.

Creates a closure that solves a standard-form QP::

    min  0.5 x^T P x + q^T x
    s.t. A x = b
         G x <= h

using any backend supported by ``qpsolvers``. The closure merges
fixed and dynamic parameters, calls the solver, and returns the
primal/dual solution together with an active-set mask and timing
information.
"""

from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
from numpy import ndarray
from jaxtyping import Float, Bool
from typing import cast, Optional, Protocol
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import SolverOptions
from jaxsparrow._solver_dense._types import DenseQPIngredientsNP, DenseQPIngredientsNPFull
from jaxsparrow._types_common import QPOutputNP


# ── Solver options ───────────────────────────────────────────────────

class DenseQPSolverOptions(SolverOptions):
    """Partial solver options for the dense QP path.

    All keys are optional; missing keys are filled from
    ``DEFAULT_SOLVER_OPTIONS`` via :func:`parse_options`.
    """
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float


class DenseQPSolverOptionsFull(SolverOptions, total=True):
    """Complete solver options for the dense QP path.

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


DEFAULT_SOLVER_OPTIONS: DenseQPSolverOptionsFull = {
    "solver_name": "piqp",
    "dtype": np.float64,
    "bool_dtype": np.bool_,
    "cst_tol": 1e-8,
}


# ── Callable protocol for the returned closure ──────────────────────

class DenseQPSolverFn(Protocol):
    """Signature of the closure returned by :func:`create_dense_qp_solver`."""

    def __call__(self, **kwargs: ndarray) -> tuple[QPOutputNP, dict[str, float]]: ...


# ── Factory ──────────────────────────────────────────────────────────

def create_dense_qp_solver(
    n_eq: int,
    n_ineq: int,
    options: Optional[SolverOptions] = None,
    fixed_elements: Optional[DenseQPIngredientsNP] = None,
) -> DenseQPSolverFn:
    """Create a dense QP solver closure.

    Builds a callable that merges fixed and dynamic QP parameters,
    solves the resulting QP via ``qpsolvers``, and returns the
    primal/dual solution along with an active-set boolean mask.

    The closure accepts dynamic parameters as keyword arguments
    (e.g. ``P=..., q=...``) and optionally a ``warmstart`` array
    for the primal initial guess.

    Args:
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Solver options (solver backend name, dtype,
            constraint tolerance). Defaults are filled for missing
            keys from ``DEFAULT_SOLVER_OPTIONS``.
        fixed_elements: QP ingredients that are constant across
            calls. Stored as dense NumPy arrays and merged with
            dynamic arguments at solve time.

    Returns:
        A callable with signature::

            solve_qp_numpy(**kwargs: ndarray)
                -> tuple[QPOutputNP, dict[str, float]]

        where ``QPOutputNP`` is ``(x, lam, mu, active)`` and the
        dict contains per-phase timing keys: ``"problem_setup"``,
        ``"solve"``, ``"retrieve"``, ``"active_set"``.
    """

    # parse options
    options_parsed: DenseQPSolverOptionsFull = parse_options(options, DEFAULT_SOLVER_OPTIONS)
    _dtype: type[np.floating] = options_parsed["dtype"]
    _bool_dtype: type[np.bool_] = options_parsed["bool_dtype"]

    _fixed: DenseQPIngredientsNP
    if fixed_elements is not None:
        _fixed = cast(
            DenseQPIngredientsNP,
            {k: np.array(v, dtype=_dtype).squeeze() for k, v in fixed_elements.items()},
        )
    else:
        _fixed = {}

    def solve_qp_numpy(**kwargs: ndarray) -> tuple[QPOutputNP, dict[str, float]]:
        """Solve a dense QP and return the primal/dual solution.

        Args:
            **kwargs: Dynamic QP parameters (any subset of
                ``P, q, A, b, G, h``) as NumPy arrays, plus an
                optional ``warmstart`` array for the primal initial
                guess. These are merged with *fixed_elements*
                provided at construction time.

        Returns:
            A tuple ``(sol, timing)`` where ``sol`` is
            ``(x, lam, mu, active)`` and ``timing`` maps phase
            names to elapsed seconds.

        Raises:
            AssertionError: If the QP solver fails to find a
                solution.
        """

        # preallocate dictionary with computation times
        t: dict[str, float] = {}

        # Build qpsolvers Problem
        start: float = perf_counter()

        # get warmstart if present
        warmstart: Optional[ndarray] = kwargs.pop("warmstart", None)

        # merge with fixed elements
        merged: DenseQPIngredientsNPFull = cast(DenseQPIngredientsNPFull, _fixed | kwargs)

        # form problem with the warmstart removed
        prob: Problem = Problem(**merged)
        t["problem_setup"] = perf_counter() - start

        # Solve QP
        start = perf_counter()
        sol = solve_problem(
            prob,
            solver=options_parsed["solver_name"],
            initvals=warmstart,
        )
        assert sol.found, "QP solver failed to find a solution."
        t["solve"] = perf_counter() - start

        # Recover primal / dual variables
        start = perf_counter()
        x: Float[ndarray, "n_var"] = (
            np.asarray(sol.x, dtype=_dtype).reshape(-1)
        )

        mu: Float[ndarray, "n_eq"]
        if n_eq > 0:
            mu = np.asarray(sol.y, dtype=_dtype).reshape(-1)
        else:
            mu = np.empty(0, dtype=_dtype)

        lam: Float[ndarray, "n_ineq"]
        if n_ineq > 0:
            lam = np.asarray(sol.z, dtype=_dtype).reshape(-1)
        else:
            lam = np.empty(0, dtype=_dtype)
        t["retrieve"] = perf_counter() - start

        # Determine active set: |Gx − h| <= tolerance
        start = perf_counter()
        active: Bool[ndarray, "n_ineq"]
        if n_ineq > 0:
            active = np.asarray(
                np.abs(merged["G"] @ sol.x - merged["h"])
                <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start

        return cast(QPOutputNP, (x, lam, mu, active)), t

    return solve_qp_numpy
