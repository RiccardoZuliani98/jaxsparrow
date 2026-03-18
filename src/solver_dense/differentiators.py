from numpy import ndarray
from time import perf_counter
from jaxtyping import Float
import numpy as np
from typing import cast, Optional
from src.solver_dense.solver_dense_types import DenseQPIngredientsNP, DenseQPIngredientsNPFull, QPOutputNP, QPDiffOutNP, DenseQPIngredientsTangentsNP
from src.solver_dense.solver_dense_options import DifferentiatorOptions
from src.utils.parsing_utils import parse_options

class DenseKKTfwdOptions(DifferentiatorOptions):
    dtype:          type[np.floating]

class DenseKKTfwdOptionsFull(DifferentiatorOptions,total=True):
    dtype:          type[np.floating]

DEFAULT_DIFF_OPTIONS : DenseKKTfwdOptionsFull = {
    "dtype": np.float64,
}

#TODO: annotate output
#TODO: docstrings
def create_dense_kkt_differentiator_fwd(
    n_var:int,
    n_eq:int,
    n_ineq:int,
    options:Optional[DifferentiatorOptions]=None,
    fixed_elements:Optional[DenseQPIngredientsNP]=None
):
    
    # parse options
    options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
    
    # store fixed elements, this can happen in two situations:
    # 1. if the user passes some fixed elements in "fixed_elements" argument,
    # 2. if there are no equality / inequality constraints
    _fixed : DenseQPIngredientsNP = {}
    _d_fixed : DenseQPIngredientsTangentsNP = {}

    if fixed_elements is not None:

        # store fixed elements
        _fixed = cast(DenseQPIngredientsNP, {k: np.array(v, dtype=options_parsed["dtype"]).squeeze() for k, v in fixed_elements.items()})

        # store zero differentials for such elements
        _d_fixed = cast(DenseQPIngredientsTangentsNP, {k: np.zeros_like(v, dtype=options_parsed["dtype"]).squeeze() for k, v in fixed_elements.items()})

    # add zero matrices / vectors if equality / inequality constraints are missing
    if n_eq == 0:
        _fixed["A"] = np.zeros((0, n_var), dtype=options_parsed["dtype"])
        _fixed["b"] = np.zeros((0,), dtype=options_parsed["dtype"])
        _d_fixed["A"] = np.zeros((0, n_var), dtype=options_parsed["dtype"])
        _d_fixed["b"] = np.zeros((0,), dtype=options_parsed["dtype"])
    if n_ineq == 0:
        _fixed["G"] = np.zeros((0, n_var), dtype=options_parsed["dtype"])
        _fixed["h"] = np.zeros((0,), dtype=options_parsed["dtype"])
        _d_fixed["G"] = np.zeros((0, n_var), dtype=options_parsed["dtype"])
        _d_fixed["h"] = np.zeros((0,), dtype=options_parsed["dtype"])

    # store a batched version where the first dimension is expanded
    _d_fixed_batched = cast(DenseQPIngredientsTangentsNP, {k: np.expand_dims(v, 0) for k,v in _d_fixed.items()}) #type: ignore
    
    # choose lienar system solver
    def _solve_linear_system(a,b):
        return np.linalg.lstsq(a,b)[0]


    def kkt_differentiator_fwd(
        sol_np: QPOutputNP,
        dyn_primals_np:DenseQPIngredientsNP,
        dyn_tangents_np:DenseQPIngredientsTangentsNP,
        batch_size: int
    ) -> tuple[QPDiffOutNP,dict[str,float]]:
        
        # start timing
        t : dict[str, float] = {}
        start = perf_counter()

        # bool representing batching for simplicity
        batched = batch_size > 0

        # extract solution
        x_np, lam_np, mu_np, active_np = sol_np


        # ── Parse ingredients ────────────────────────────────────────
        
        # merge dynamic elements with fixed ones computed at function 
        # generation, make sure to use batched version if needed.
        if batched:
            d_np = cast(dict[str,ndarray],_d_fixed_batched | dyn_tangents_np)
        else:
            d_np = cast(dict[str,ndarray],_d_fixed | dyn_tangents_np)

        # merge qp ingredients too
        prob_np = cast(DenseQPIngredientsNPFull, _fixed | dyn_primals_np)

        # count active constraints
        n_active = int(np.sum(active_np))


        # ── Build LHS ────────────────────────────────────────────────

        # construct H matrix
        H_parts: list[Float[ndarray, "_ nv"]] = []
        if n_eq > 0:
            H_parts.append(prob_np["A"])
        if n_ineq > 0 and n_active > 0:
            H_parts.append(prob_np["G"][active_np, :])

        if H_parts:
            H_np: Float[ndarray, "nh nv"] = np.vstack(H_parts)
        else:
            H_np = np.empty((0, n_var), dtype=options_parsed["dtype"])

        n_h = H_np.shape[0]

        # build lhs
        lhs: Float[ndarray, "nv_nh nv_nh"] = np.block([
            [prob_np["P"],  H_np.T],
            [H_np,  np.zeros((n_h, n_h), dtype=options_parsed["dtype"])],
        ])


        # ── Build RHS (differs between batched and unbatched) ────────
        
        # start with gradient of cost function only
        dL_np = d_np["P"] @ x_np + d_np["q"]

        # list of blocks in KKT
        rhs_pieces = []

        # branch based on batching mode
        if batched:

            if n_eq > 0:

                # add equality constraints to gradient of Lagrangian
                dL_np = dL_np + d_np["A"].transpose(0, 2, 1) @ mu_np

                # add equality block of KKT conditions
                rhs_pieces.append( d_np["A"] @ x_np - d_np["b"] )
            
            if n_ineq > 0 and n_active > 0:

                dG_active = d_np["G"][:, active_np, :]

                # add inequality constraints to gradient of Lagrangian
                dL_np = dL_np + dG_active.transpose(0, 2, 1) @ lam_np[active_np]

                # add inequality block of KKT conditions
                rhs_pieces.append( dG_active @ x_np - d_np["h"][:, active_np] )

            # form rhs
            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in [dL_np] + rhs_pieces
            ], axis=1).T

        else:

            if n_eq > 0:

                # add equality constraints to gradient of Lagrangian
                dL_np += d_np["A"].T @ mu_np

                # add equality block of KKT conditions
                rhs_pieces.append( d_np["A"] @ x_np - d_np["b"] )
            
            if n_ineq > 0 and n_active > 0:

                dG_active = d_np["G"][active_np, :]

                # add inequality constraints to gradient of Lagrangian
                dL_np += dG_active.T @ lam_np[active_np]

                # add inequality block of KKT conditions
                rhs_pieces.append( dG_active @ x_np - d_np["h"][active_np] )

            # form rhs
            rhs = np.hstack([dL_np] + rhs_pieces)

        t["build_system"] = perf_counter() - start


        # ── Solve the linear system ──────────────────────────────────

        start = perf_counter()
        sol = _solve_linear_system(lhs, -rhs)
        t["lin_solve"] = perf_counter() - start


        # ── Extract dx, dlam, dmu from the solution ──────────────────

        if batch_size > 0:
            dx_np = sol[:n_var, :].T
            dmu_np = (
                sol[n_var:n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=options_parsed["dtype"])
            )
            dlam_np = np.zeros(
                (batch_size, n_ineq), dtype=options_parsed["dtype"]
            )
            if n_ineq > 0 and n_active > 0:
                dlam_np[:, active_np] = sol[n_var + n_eq:, :].T
        else:
            dx_np = sol[:n_var]
            dmu_np = (
                sol[n_var:n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=options_parsed["dtype"])
            )
            dlam_np = np.zeros(n_ineq, dtype=options_parsed["dtype"])
            if n_ineq > 0 and n_active > 0:
                dlam_np[active_np] = sol[n_var + n_eq:]

        return cast(QPDiffOutNP, (dx_np, dlam_np, dmu_np)), t
    
    return kkt_differentiator_fwd