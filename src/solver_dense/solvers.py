from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
from numpy import ndarray
from jaxtyping import Float, Bool
from typing import cast, Optional
from src.utils.parsing_utils import parse_options
from src.solver_dense.solver_dense_options import SolverOptions
from src.solver_dense.solver_dense_types import DenseQPIngredientsNP, DenseQPIngredientsNPFull, QPOutputNP

#TODO: docstrintgs
class DenseQPSolverOptions(SolverOptions):
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float

class DenseQPSolverOptionsFull(SolverOptions,total=True):
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float

DEFAULT_SOLVER_OPTIONS : DenseQPSolverOptionsFull = {
    "solver_name":"piqp",
    "dtype":np.float64,
    "bool_dtype":np.bool_,
    "cst_tol": 1e-8
}

#TODO: active set should be determined when differentiating
def create_dense_qp_solver(
    n_eq:int,
    n_ineq:int,
    options:Optional[SolverOptions]=None,
    fixed_elements:Optional[DenseQPIngredientsNP]=None):

    # parse options
    options_parsed = parse_options(options ,DEFAULT_SOLVER_OPTIONS)

    if fixed_elements is not None:
        _fixed = cast(
            DenseQPIngredientsNPFull,
            {k: np.array(v, dtype=options_parsed["dtype"]).squeeze() for k, v in fixed_elements.items()}
        )
    else:
        _fixed = {}

    #TODO: change output
    def solve_qp_numpy(**kwargs: ndarray) -> tuple[QPOutputNP,dict[str, float]]:

        # preallocate dictionary with computation times
        t: dict[str, float] = {}

        # Build qpsolvers Problem
        start = perf_counter()

        # get warmstart if present
        warmstart = kwargs.pop("warmstart",None)

        # merge with fixed elements
        merged = cast(DenseQPIngredientsNPFull, _fixed | kwargs)

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
        x: Float[ndarray, "n_var"] = (
            np.asarray(sol.x, dtype=options_parsed["dtype"]).reshape(-1)
        )

        if n_eq > 0:
            mu: Float[ndarray, "n_eq"] = (
                np.asarray(sol.y, dtype=options_parsed["dtype"]).reshape(-1)
            )
        else:
            mu = np.empty(0, dtype=options_parsed["dtype"])

        if n_ineq > 0:
            lam: Float[ndarray, "n_ineq"] = (
                np.asarray(sol.z, dtype=options_parsed["dtype"]).reshape(-1)
            )
        else:
            lam = np.empty(0, dtype=options_parsed["dtype"])
        t["retrieve"] = perf_counter() - start

        # Determine active set: |Gx − h| <= tolerance
        start = perf_counter()
        if n_ineq > 0:
            active: Bool[ndarray, "n_ineq"] = np.asarray(
                np.abs(merged["G"] @ sol.x - merged["h"])
                <= options_parsed["cst_tol"],
                dtype=options_parsed["bool_dtype"],
            ).reshape(-1)
        else:
            active : Bool[ndarray, "n_ineq"] = np.empty(0, dtype=options_parsed["bool_dtype"])
        t["active_set"] = perf_counter() - start

        return cast(QPOutputNP, (x, lam, mu, active)), t
    
    return solve_qp_numpy