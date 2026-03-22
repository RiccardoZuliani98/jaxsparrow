"""
solver_sparse/_solvers.py
=========================
Numpy-level QP solver for the sparse path.

Uses the :class:`SolverBackend` protocol to delegate problem
setup and solving to a pluggable backend.
The default backend (``QpSolversBackend``) wraps the ``qpsolvers``
library and reproduces the original stateless behaviour. Future
backends can exploit the setup/solve split for pre-factorization.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Protocol, Union, cast

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix, issparse
from jaxtyping import Float, Bool

from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import SolverOptions
from jaxsparrow._solver_sparse._types import SparseIngredientsNP
from jaxsparrow._types_common import SolverOutputNP, Solver
from jaxsparrow._solver_sparse._options import DEFAULT_SOLVER_OPTIONS
from jaxsparrow._utils._solver_backends import SolverBackend, get_backend


# ── Factory ──────────────────────────────────────────────────────────

def create_sparse_qp_solver(
    n_eq: int,
    n_ineq: int,
    options: Optional[SolverOptions] = None,
    fixed_elements: Optional[SparseIngredientsNP] = None,
) -> Solver:
    """Build a numpy-level sparse QP solver closure.

    Creates a callable that solves quadratic programs of the form::

        min  0.5 x^T P x + q^T x
        s.t. A x = b
             G x <= h

    where ``P``, ``A``, ``G`` are ``scipy.sparse.csc_matrix`` and
    ``q``, ``b``, ``h`` are dense ``ndarray``.

    The solver lifecycle is delegated to a :class:`SolverBackend`:

    - **setup** is called once at construction with the full problem
      structure from *fixed_elements*, allowing the backend to do
      symbolic analysis or workspace allocation.
    - **solve** is called at each invocation to run the numerical
      solver.

    Args:
        n_eq: Number of equality constraints. Zero if there are none.
        n_ineq: Number of inequality constraints. Zero if there are
            none.
        options: Solver-specific options. Recognised keys:

            - ``"solver_name"``: backend solver name (e.g. ``"piqp"``).
            - ``"backend"``: backend protocol name (default:
              ``"qpsolvers"``). Controls which :class:`SolverBackend`
              implementation is used.
            - ``"dtype"``, ``"bool_dtype"``, ``"cst_tol"``: as before.

            Defaults are filled for missing keys.
        fixed_elements: QP ingredients that define the problem
            structure. Sparse matrices are stored as CSC; dense
            vectors are squeezed and cast. These are passed to the
            backend's ``setup()`` call. All problem matrices and
            vectors (P, q, A, b, G, h) must be provided here.

    Returns:
        A callable with signature
        ``(**kwargs) -> tuple[SolverOutputNP, dict[str, float]]``.
        The first element is ``(x, lam, mu, active)`` and the second
        is a timing dict with keys ``"setup.*"`` (from construction),
        ``"solve.*"``, ``"retrieve"``, ``"active_set"``.
    """

    options_parsed = parse_options(options, DEFAULT_SOLVER_OPTIONS)
    _dtype: type[np.floating] = options_parsed["dtype"]
    _bool_dtype: type[np.bool_] = options_parsed["bool_dtype"]

    # ── Prepare fixed elements ───────────────────────────────────────

    _fixed: SparseIngredientsNP = {}
    if fixed_elements is not None:
        for k, v in fixed_elements.items():
            if issparse(v):
                _fixed[k] = csc_matrix(v, dtype=_dtype)
            else:
                _fixed[k] = np.asarray(v, dtype=_dtype).squeeze()

    # ── Create backend ───────────────────────────────────────────────

    backend_name: str = options_parsed.get("backend", "qpsolvers")
    backend: SolverBackend = get_backend(
        backend_name,
        solver_name=options_parsed["solver_name"],
        dtype=_dtype,
    )

    # ── Setup: pass fixed elements to the backend (once, now) ────────

    _setup_timing: dict[str, float] = backend.setup(
        P=_fixed.get("P"),
        q=_fixed.get("q"),
        A=_fixed.get("A"),
        b=_fixed.get("b"),
        G=_fixed.get("G"),
        h=_fixed.get("h"),
    )

    # ─────────────────────────────────────────────────────────────────

    def solve_qp_numpy(**kwargs: ndarray) -> tuple[SolverOutputNP, dict[str, float]]:
        """Solve a single QP instance.

        Delegates to the backend's solve cycle and extracts the
        solution.

        Args:
            **kwargs: Optional runtime overrides. A ``"warmstart"``
                key may supply an initial guess for the primal
                variable.

        Returns:
            A tuple ``(x, lam, mu, active)`` and a timing dict.

        Raises:
            AssertionError: If the solver fails to find a solution.
        """
        t: dict[str, float] = {}

        # Propagate setup timings
        t.update({f"setup.{k}": v for k, v in _setup_timing.items()})

        # ── Solve ────────────────────────────────────────────────────
        x_raw, y_raw, z_raw, solve_timing = backend.solve(**kwargs)
        t.update({f"solve.{k}": v for k, v in solve_timing.items()})

        assert x_raw is not None, "QP solver failed to find a solution."

        # ── Extract solution ─────────────────────────────────────────
        start: float = perf_counter()

        x: Float[ndarray, "n_var"] = np.asarray(x_raw, dtype=_dtype).reshape(-1)

        mu: Float[ndarray, "n_eq"] = (
            np.asarray(y_raw, dtype=_dtype).reshape(-1)
            if n_eq > 0 and y_raw is not None
            else np.empty(0, dtype=_dtype)
        )

        lam: Float[ndarray, "n_ineq"] = (
            np.asarray(z_raw, dtype=_dtype).reshape(-1)
            if n_ineq > 0 and z_raw is not None
            else np.empty(0, dtype=_dtype)
        )
        t["retrieve"] = perf_counter() - start

        # ── Active set ───────────────────────────────────────────────
        start = perf_counter()
        active: Bool[ndarray, "n_ineq"]
        if n_ineq > 0:
            assert "G" in _fixed and "h" in _fixed
            G_mat = _fixed["G"]
            Gx: ndarray = G_mat @ x_raw
            h_vec: ndarray = np.asarray(_fixed["h"], dtype=_dtype).ravel()
            active = np.asarray(
                np.abs(Gx - h_vec) <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start

        return cast(SolverOutputNP, (x, lam, mu, active)), t

    return solve_qp_numpy