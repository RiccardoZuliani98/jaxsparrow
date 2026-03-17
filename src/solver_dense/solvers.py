from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
from numpy import ndarray
from jaxtyping import Float, Bool
from typing import cast

DEFAULT_SOLVER_OPTIONS = {
    "solver_name":"piqp",
    "dtype":np.float64,
    "bool_dtype":np.bool_,
    "cst_tol": 1e-8
}

def create_dense_qp_solver(n_eq,n_ineq,options=None,fixed_elements=None):

    # parse options
    if options is not None:
        options_parsed = DEFAULT_SOLVER_OPTIONS | options
    else:
        options_parsed = DEFAULT_SOLVER_OPTIONS

    if fixed_elements is not None:
        _fixed = cast(
            dict[str,ndarray],
            {k: np.array(v, dtype=options_parsed["dtype"]).squeeze() for k, v in fixed_elements.items()}
        )
    else:
        _fixed = {}

    #TODO: change output
    def solve_qp_numpy(**kwargs: ndarray) -> tuple[
            Float[ndarray, " nv"],      # x
            Float[ndarray, " ni"],      # lam
            Float[ndarray, " ne"],      # mu
            Bool[ndarray, " ni"],       # active
            dict[str, float],           # timing
        ]:
            """Solve the QP in pure numpy via ``qpsolvers``.

            Only dynamic elements needed in ``kwargs``, since kwargs is
            merged with _fixed in this function.

            When the problem has no equality constraints (``n_eq == 0``),
            ``A``, ``b``, and ``mu`` are absent / empty. Likewise, when
            there are no inequality constraints (``n_ineq == 0``), ``G``,
            ``h``, ``lam``, and ``active`` are absent / empty.

            Args:
                **kwargs: Numpy arrays for the QP ingredients:

                    - ``P`` (nv, nv): Positive semi-definite cost matrix.
                    - ``q`` (nv,): Linear cost vector.
                    - ``A`` (ne, nv): Equality constraint matrix
                    (required when ``n_eq > 0``).
                    - ``b`` (ne,): Equality constraint vector
                    (required when ``n_eq > 0``).
                    - ``G`` (ni, nv): Inequality constraint matrix
                    (required when ``n_ineq > 0``).
                    - ``h`` (ni,): Inequality constraint vector
                    (required when ``n_ineq > 0``).
                    - ``warmstart``: Warmstart value for primal variable,
                    optional, otherwise no warmstart is used.

            Returns:
                A tuple ``(x, lam, mu, active, t)`` where:

                    - ``x`` (nv,): Primal solution.
                    - ``lam`` (ni,): Dual variables for inequality
                    constraints. Empty when ``n_ineq == 0``.
                    - ``mu`` (ne,): Dual variables for equality
                    constraints. Empty when ``n_eq == 0``.
                    - ``active`` (ni,): Boolean mask of active inequality
                    constraints. Empty when ``n_ineq == 0``.
                    - ``t``: Timing dict with keys ``problem_setup``,
                    ``solve``, ``retrieve``, ``active_set``.

            Raises:
                AssertionError: If the solver fails or array shapes are
                    inconsistent with declared problem dimensions.
            """

            # preallocate dictionary with computation times
            t: dict[str, float] = {}

            # Build qpsolvers Problem
            start = perf_counter()

            # get warmstart if present
            warmstart = kwargs.pop("warmstart",None)

            # merge with fixed elements
            merged = _fixed | kwargs

            # form problem with the warmstart removed
            prob = Problem(**merged)
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
            x: Float[ndarray, " nv"] = (
                np.asarray(sol.x, dtype=options_parsed["dtype"]).reshape(-1)
            )

            if n_eq > 0:
                mu: Float[ndarray, " ne"] = (
                    np.asarray(sol.y, dtype=options_parsed["dtype"]).reshape(-1)
                )
            else:
                mu = np.empty(0, dtype=options_parsed["dtype"])

            if n_ineq > 0:
                lam: Float[ndarray, " ni"] = (
                    np.asarray(sol.z, dtype=options_parsed["dtype"]).reshape(-1)
                )
            else:
                lam = np.empty(0, dtype=options_parsed["dtype"])
            t["retrieve"] = perf_counter() - start

            # Determine active set: |Gx − h| <= tolerance
            start = perf_counter()
            if n_ineq > 0:
                active: Bool[ndarray, " ni"] = np.asarray(
                    np.abs(merged["G"] @ sol.x - merged["h"])
                    <= options_parsed["cst_tol"],
                    dtype=options_parsed["bool_dtype"],
                ).reshape(-1)
            else:
                active = np.empty(0, dtype=options_parsed["bool_dtype"])
            t["active_set"] = perf_counter() - start

            return x, lam, mu, active, t
    
    return solve_qp_numpy