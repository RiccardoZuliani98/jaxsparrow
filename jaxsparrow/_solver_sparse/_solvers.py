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
from jaxtyping import Float

from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._options_common import SolverOptions
from jaxsparrow._solver_sparse._types import SparseIngredientsNP
from jaxsparrow._types_common import SolverOutputNP, Solver
from jaxsparrow._solver_sparse._options import (
    SOLVER_OPTIONS_DEFAULTS,
    DEFAULT_SOLVER_BACKEND,
)
from jaxsparrow._utils._solver_backends import SolverBackend, get_backend


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
              implementation is used. Currently only ``"qpsolvers"``
              is supported.
            - ``"dtype"``: NumPy floating-point dtype for arrays.

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
        The first element is ``(x, lam, mu)`` and the second
        is a timing dict with keys ``"setup.*"`` (from construction),
        ``"solve.*"``, ``"retrieve"``.
    """

    # ── Resolve backend and parse options ────────────────────────────
    backend_name, defaults = _resolve_backend_defaults(options)
    options_parsed = parse_options(options, defaults)

    # dtype is guaranteed present after merging with backend defaults.
    _dtype: type[np.floating] = options_parsed.get("dtype", np.float64)

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
        """Solve a single problem instance.

        Passes runtime ingredients to the backend, which merges
        them with the stored fixed elements, builds the problem,
        and runs the solver.

        Args:
            **kwargs: Runtime ingredients (those not fixed at
                construction). An optional ``"warmstart"`` key may
                supply an initial guess for the primal variable.

        Returns:
            A tuple ``(x, lam, mu)`` and a timing dict.

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

        return cast(SolverOutputNP, (x, lam, mu)), t

    return solver_numpy