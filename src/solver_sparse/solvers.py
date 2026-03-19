"""
solver_sparse/solvers.py
========================
Numpy-level QP solver for the sparse path.

Wraps ``qpsolvers`` which natively accepts ``scipy.sparse.csc_matrix``
for P, G, and A.  Vectors remain dense ndarray.
"""

from time import perf_counter
import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix, issparse
from qpsolvers import Problem, solve_problem
from jaxtyping import Float, Bool
from typing import cast, Optional

from src.utils.parsing_utils import parse_options
from src.options_common import SolverOptions
from src.solver_sparse.types import SparseQPIngredientsNP, SparseQPIngredientsNPFull
from src.types_common import QPOutputNP
from src.solver_sparse.options import DEFAULT_SOLVER_OPTIONS


def create_sparse_qp_solver(
    n_eq: int,
    n_ineq: int,
    options: Optional[SolverOptions] = None,
    fixed_elements: Optional[SparseQPIngredientsNP] = None,
):
    """Build a numpy-level sparse QP solver closure.

    Creates a callable that solves quadratic programs of the form::

        min  0.5 * x^T P x + q^T x
        s.t. A x = b
             G x <= h

    where P, A, G are ``scipy.sparse.csc_matrix`` and q, b, h are
    dense ``ndarray``. Elements provided via *fixed_elements* are
    baked into the closure and reused across calls; remaining
    ingredients are supplied at call time as keyword arguments.

    Args:
        n_eq: Number of equality constraints. Zero if there are none.
        n_ineq: Number of inequality constraints. Zero if there are none.
        options: Solver-specific options (solver backend name, dtype,
            constraint tolerance, etc.). Defaults are filled in for any
            keys not provided.
        fixed_elements: QP ingredients that remain constant across
            calls. Sparse matrices are stored as CSC; dense vectors
            are squeezed and cast to the configured dtype. Any key
            present here should *not* be passed again at call time.

    Returns:
        A callable with signature
        ``(**kwargs) -> tuple[QPOutputNP, dict[str, float]]``.
        The first element is ``(x, lam, mu, active)`` and the second
        is a timing dict with keys ``"problem_setup"``,
        ``"solve"``, ``"retrieve"``, and ``"active_set"``.
    """

    options_parsed = parse_options(options, DEFAULT_SOLVER_OPTIONS)
    _dtype = options_parsed["dtype"]
    _bool_dtype = options_parsed["bool_dtype"]

    # Pre-store fixed elements
    if fixed_elements is not None:
        _fixed: SparseQPIngredientsNP = {}
        for k, v in fixed_elements.items():
            if issparse(v):
                _fixed[k] = csc_matrix(v, dtype=_dtype)
            else:
                _fixed[k] = np.asarray(v, dtype=_dtype).squeeze()
    else:
        _fixed = {}

    def solve_qp_numpy(**kwargs) -> tuple[QPOutputNP, dict[str, float]]:
        """Solve a single QP instance.

        Merges *kwargs* with any fixed elements provided at
        construction, builds the problem, and delegates to the
        configured solver backend.

        Args:
            **kwargs: Runtime QP ingredients (those not fixed at
                setup). An optional ``"warmstart"`` key may supply
                an initial guess for the primal variable *x*.

        Returns:
            A tuple of ``(x, lam, mu, active)`` and a timing dict.

        Raises:
            AssertionError: If the underlying solver fails to find
                a solution.
        """

        t: dict[str, float] = {}

        # ── Build problem ────────────────────────────────────────────
        start = perf_counter()
        warmstart = kwargs.pop("warmstart", None)

        # Merge fixed + runtime
        merged = cast(SparseQPIngredientsNPFull, {**_fixed, **kwargs})

        prob = Problem(
            P=merged["P"],
            q=merged["q"],
            A=merged.get("A"),
            b=merged.get("b"),
            G=merged.get("G"),
            h=merged.get("h"),
        )
        t["problem_setup"] = perf_counter() - start

        # ── Solve ────────────────────────────────────────────────────
        start = perf_counter()
        sol = solve_problem(
            prob,
            solver=options_parsed["solver_name"],
            initvals=warmstart,
        )
        assert sol.found, "QP solver failed to find a solution."
        t["solve"] = perf_counter() - start

        # ── Extract solution ─────────────────────────────────────────
        start = perf_counter()
        x: Float[ndarray, "n_var"] = np.asarray(sol.x, dtype=_dtype).reshape(-1)

        mu: Float[ndarray, "n_eq"] = (
            np.asarray(sol.y, dtype=_dtype).reshape(-1)
            if n_eq > 0
            else np.empty(0, dtype=_dtype)
        )

        lam: Float[ndarray, "n_ineq"] = (
            np.asarray(sol.z, dtype=_dtype).reshape(-1)
            if n_ineq > 0
            else np.empty(0, dtype=_dtype)
        )
        t["retrieve"] = perf_counter() - start

        # ── Active set ───────────────────────────────────────────────
        start = perf_counter()
        if n_ineq > 0:
            G = merged["G"]
            Gx = G @ sol.x
            h = merged["h"]
            active: Bool[ndarray, "n_ineq"] = np.asarray(
                np.abs(Gx - h) <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start

        return cast(QPOutputNP, (x, lam, mu, active)), t

    return solve_qp_numpy