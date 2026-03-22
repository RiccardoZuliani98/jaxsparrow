"""
solver_sparse/_solvers.py
=========================
Numpy-level solver for the sparse path.

Uses the :class:`SolverBackend` protocol to delegate problem
setup and solving to a pluggable backend.
The default backend (``SolversBackend``) wraps the ``qpsolvers``
library and reproduces the original stateless behaviour. Future
backends can exploit the setup/solve split for pre-factorization.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, cast

import numpy as np
from numpy import ndarray
from jaxtyping import Float, Bool

from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import SolverOptions
from jaxsparrow._solver_sparse._types import SparseIngredientsNP
from jaxsparrow._types_common import SolverOutputNP, Solver
from jaxsparrow._solver_sparse._options import DEFAULT_SOLVER_OPTIONS
from jaxsparrow._utils._solver_backends import SolverBackend, get_backend


# ── Factory ──────────────────────────────────────────────────────────

def create_sparse_solver(
    n_eq: int,
    n_ineq: int,
    options: Optional[SolverOptions] = None,
    fixed_elements: Optional[SparseIngredientsNP] = None,
) -> Solver:
    """Build a numpy-level sparse solver closure.

    Creates a callable that solves quadratic programs of the form::

        min  0.5 x^T P x + q^T x
        s.t. A x = b
             G x <= h

    where ``P``, ``A``, ``G`` are ``scipy.sparse.csc_matrix`` and
    ``q``, ``b``, ``h`` are dense ``ndarray``.

    The solver lifecycle is delegated to a :class:`SolverBackend`:

    - **setup** is called once at construction time. The backend
      receives *fixed_elements* and is responsible for casting
      sparse matrices to CSC and dense vectors to the configured
      dtype, storing the results, and performing any symbolic
      analysis or workspace allocation.
    - **solve** is called at each invocation with the runtime
      ingredients supplied via ``**kwargs``. The backend merges
      them with the stored fixed elements, builds the problem,
      and runs the numerical solver.

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
        fixed_elements: ingredients that remain constant across
            calls (e.g. constraint matrices that do not change).
            Passed directly to the backend's :meth:`setup` call;
            the backend is responsible for dtype casting and storage.
            Any key present here should *not* be passed again at
            call time.

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
        """Solve a single problem instance.

        Passes runtime ingredients to the backend, which merges
        them with the stored fixed elements, builds the problem,
        and runs the solver.

        Args:
            **kwargs: Runtime ingredients (those not fixed at
                construction). An optional ``"warmstart"`` key may
                supply an initial guess for the primal variable.

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

        assert x_raw is not None, "Solver failed to find a solution."

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
            # G and h must have been provided (fixed or runtime);
            # re-merge to get the effective values.
            merged_G = kwargs.get("G", (fixed_elements or {}).get("G"))
            merged_h = kwargs.get("h", (fixed_elements or {}).get("h"))
            assert merged_G is not None and merged_h is not None, (
                "G and h are required when n_ineq > 0"
            )
            Gx: ndarray = merged_G @ x_raw
            h_vec: ndarray = np.asarray(merged_h, dtype=_dtype).ravel()
            active = np.asarray(
                np.abs(Gx - h_vec) <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start

        return cast(SolverOutputNP, (x, lam, mu, active)), t

    return solver_numpy