from typing import Optional, cast
from mpc_types import DEFAULT_SOLVER_OPTIONS, SolverOptions, SolverOptionsFull
from jax import custom_jvp, ShapeDtypeStruct, pure_callback
import jax.numpy as jnp
from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
import jax
import logging

#TODO: we should allow the user not to pass e.g. A,b, or G,h
#TODO: add warmstart


def _parse_options(options:Optional[SolverOptions]) -> SolverOptionsFull:

    if options is None:
        return DEFAULT_SOLVER_OPTIONS

    allowed = set(DEFAULT_SOLVER_OPTIONS.keys())
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(
            f"Unknown option key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )
    return cast(
        SolverOptionsFull, {**DEFAULT_SOLVER_OPTIONS, **options}
    )

def setup_dense_solver(
    n_var: int,
    n_ineq: int,
    n_eq: int,
    options: Optional[SolverOptions] = None,
):
    
    logger = logging.getLogger(__name__)
    
    options_parsed = _parse_options(options)
    dtype = options_parsed['dtype']

    _fwd_shapes = {
        "x": jax.ShapeDtypeStruct((n_var,), jnp.float64),
        "lam": jax.ShapeDtypeStruct((n_ineq,), jnp.float64),
        "mu": jax.ShapeDtypeStruct((n_eq,), jnp.float64),
        "active": jax.ShapeDtypeStruct((n_ineq,), jnp.bool_),
    }
    _bwd_shapes = (
        jax.ShapeDtypeStruct((n_var,),         jnp.float64),  # dx
        jax.ShapeDtypeStruct((n_eq,),          jnp.float64),  # dmu
        jax.ShapeDtypeStruct((n_ineq,),        jnp.float64),  # dlam
        jax.ShapeDtypeStruct((n_ineq,),        jnp.bool_),    # active
        _fwd_shapes,
    )


    # =================================================================
    # SETUP SOLVER
    # =================================================================

    def _convert_qp_ingredients_to_numpy(P, q, A, b, G, h):

        # Convert vectors
        start = perf_counter()
        q_np = np.asarray(q, dtype=dtype)
        b_np = np.asarray(b, dtype=dtype)
        h_np = np.asarray(h, dtype=dtype)
        t_convert = {"convert_vectors": perf_counter() - start}

        # Convert matrices
        start = perf_counter()
        P_np = np.asarray(P, dtype=dtype)
        A_np = np.asarray(A, dtype=dtype)
        G_np = np.asarray(G, dtype=dtype)
        t_convert["convert_matrices"] = perf_counter() - start

        return (
            P_np, 
            q_np, 
            A_np, 
            b_np, 
            G_np, 
            h_np, 
            t_convert
        )

    def _convert_solution_to_jax(x, lam, mu, active):
        start = perf_counter()
        sol =  {
            "x": jnp.array(x,dtype=dtype), 
            "lam":jnp.array(lam,dtype=dtype),
            "mu":jnp.array(mu,dtype=dtype),
            "active":jnp.array(active,dtype=jnp.bool_)
        }
        t_convert = perf_counter() - start
        return sol, t_convert

    def _solve_qp_numpy(P, q, A, b, G, h):

        t_dict = {}

        # Build qpsolvers Problem
        start = perf_counter()
        prob_dict = {"P": P,"q": q,"A": A,"b": b,"G": G,"h": h}
        prob = Problem(**prob_dict)
        t_dict["problem_setup"] = perf_counter() - start

        # Solve QP
        start = perf_counter()
        sol = solve_problem(prob, solver=options_parsed["solver"])
        assert sol.found
        t_dict["solve"] = perf_counter() - start

        # Recover primal/dual variables
        start = perf_counter()
        x = np.asarray(sol.x, dtype=dtype).reshape(-1)
        mu = np.asarray(sol.y, dtype=dtype).reshape(-1)
        lam = np.asarray(sol.z, dtype=dtype).reshape(-1)
        t_dict["retrieve"] = perf_counter() - start

        # Determine active set: Gx − h >= tolerance
        start = perf_counter()
        active = np.asarray(np.abs(G @ sol.x - h) <= options_parsed["jac_tol"], dtype=np.bool_).reshape(-1)
        t_dict["active"] = perf_counter() - start

        return x, lam, mu, active, t_dict

    def _solve_qp(P, q, A, b, G, h):

        start = perf_counter()

        P_np, q_np, A_np, b_np, G_np, h_np, t_dict_convert = _convert_qp_ingredients_to_numpy(P, q, A, b, G, h)

        x_np, lam_np, mu_np, active_np, t_dict_solve = _solve_qp_numpy(P_np, q_np, A_np, b_np, G_np, h_np)

        sol, t_sol_convert = _convert_solution_to_jax(x_np, lam_np, mu_np, active_np)

        t_full = perf_counter() - start

        if options_parsed["verbose"]:
            logger.info(
                f"DenseQP time {t_full:.3e}s -- "
                f"solve={t_dict_solve["solve"]:.3e}s  "
                f"active set={t_dict_solve["active"]:.3e}s  "
                f"retrieve={t_dict_solve["retrieve"]:.3e}s  "
                f"solve={t_dict_solve["solve"]:.3e}s  "
                f"problem setup={t_dict_solve["problem_setup"]:.3e}s  "
                f"conversion vectors={t_dict_convert["convert_vectors"]:.3e}s  "
                f"conversion matrices={t_dict_convert["convert_matrices"]:.3e}s  "
                f"conversion solution={t_sol_convert:.3e}"
            )

        return sol

    def _solve_qp_callback(P, q, A, b, G, h):
        return pure_callback(
            _solve_qp,
            _fwd_shapes,
            P, q, A, b, G, h,
        )

    # =================================================================
    # SETUP DIFFERENTIATOR
    # =================================================================

    def _kkt_diff(P, q, A, b, G, h, dP, dq, dA, db, dG, dh):

        # this function takes in jax.numpy arrays and matrices,
        # the tangent arguments dQ, dq, dF, df, dG, dg can be vectorized,
        # in which case they are passed

        print("Hi, I am kkt differentiator and I am running.")

        P_np, q_np, A_np, b_np, G_np, h_np, t_dict = _convert_qp_ingredients_to_numpy(P, q, A, b, G, h)

        # Get primal/dual solution (forward eval)
        x_np, lam_np, mu_np, active_np, t_dict_solve = _solve_qp_numpy(P_np, q_np, A_np, b_np, G_np, h_np)
        t_dict.update(t_dict_solve)

        # Convert vectors
        start = perf_counter()
        dq_np = np.asarray(dq, dtype=dtype).squeeze()
        db_np = np.asarray(db, dtype=dtype).squeeze()
        dh_np = np.asarray(dh, dtype=dtype).squeeze()
        t_dict["convert_vector_derivatives"] = perf_counter() - start

        # Convert matrices
        start = perf_counter()
        dP_np = np.asarray(dP, dtype=dtype).squeeze()
        dA_np = np.asarray(dA, dtype=dtype).squeeze()
        dG_np = np.asarray(dG, dtype=dtype).squeeze()
        t_dict["convert_matrix_derivatives"] = perf_counter() - start

        # Stack constraints
        H_np = np.vstack((A_np, G_np[active_np,:]))  # (n_eq + n_ineq, n_var)

        # Derivative of Lagrangian:
        dL_np = dP_np @ x_np + dq_np + dA_np.T @ mu_np + dG_np[active_np,:].T @ lam_np[active_np]

        # RHS = concatenation of dL and constraint derivatives
        rhs = np.concatenate([dL_np, dA_np @ x_np - db_np, dG_np[active_np,:] @ x_np - dh_np[active_np]])
        n_h = H_np.shape[0]
        lhs = np.block([[P_np, H_np.T],[H_np, np.zeros((n_h,n_h))]])

        sol                 = np.linalg.solve(lhs, -rhs)
        dx_np               = sol[:n_var]
        dmu_np              = sol[n_var : n_var + n_eq]
        dlam_a_np           = sol[n_var + n_eq :]
        dlam_np             = np.zeros(n_ineq)
        dlam_np[active_np]  = dlam_a_np

        

        return dx_np, dmu_np, dlam_np, active_np, res
    
    def _kkt_diff_batched(P, q, A, b, G, h, dP, dq, dA, db, dG, dh):

        # this function takes in jax.numpy arrays and matrices,
        # the tangent arguments dQ, dq, dF, df, dG, dg can be vectorized,
        # in which case they are passed

        print("Hi, I am kkt differentiator and I am running.")

        P_np, q_np, A_np, b_np, G_np, h_np, t_dict = _convert_qp_ingredients_to_numpy(P, q, A, b, G, h)

        # Get primal/dual solution (forward eval)
        x_np, lam_np, mu_np, active_np, t_dict_solve = _solve_qp_numpy(P_np, q_np, A_np, b_np, G_np, h_np)
        t_dict.update(t_dict_solve)

        # Convert vectors
        start = perf_counter()
        dq_np = np.asarray(dq, dtype=dtype).squeeze()
        db_np = np.asarray(db, dtype=dtype).squeeze()
        dh_np = np.asarray(dh, dtype=dtype).squeeze()
        t_dict["convert_vector_derivatives"] = perf_counter() - start

        # Convert matrices
        start = perf_counter()
        dP_np = np.asarray(dP, dtype=dtype).squeeze()
        dA_np = np.asarray(dA, dtype=dtype).squeeze()
        dG_np = np.asarray(dG, dtype=dtype).squeeze()
        t_dict["convert_matrix_derivatives"] = perf_counter() - start

        # Stack constraints
        H_np = np.vstack((A_np, G_np[active_np,:]))  # (n_eq + n_ineq, n_var)

        # Derivative of Lagrangian:
        dL_np = dP_np @ x_np + dq_np + dA_np.T @ mu_np + dG_np[active_np,:].T @ lam_np[active_np]

        # RHS = concatenation of dL and constraint derivatives
        rhs = np.concatenate([dL_np, dA_np @ x_np - db_np, dG_np[active_np,:] @ x_np - dh_np[active_np]])
        n_h = H_np.shape[0]
        lhs = np.block([[P_np, H_np.T],[H_np, np.zeros((n_h,n_h))]])

        sol                 = np.linalg.solve(lhs, -rhs)
        dx_np               = sol[:n_var]
        dmu_np              = sol[n_var : n_var + n_eq]
        dlam_a_np           = sol[n_var + n_eq :]
        dlam_np             = np.zeros(n_ineq)
        dlam_np[active_np]  = dlam_a_np

        res = {"x": np.expand_dims(x_np,0),"mu": np.expand_dims(mu_np,0),"lam": np.expand_dims(lam_np,0),"active": np.expand_dims(active_np,0)}

        return np.expand_dims(dx_np,0), np.expand_dims(dmu_np,0), np.expand_dims(dlam_np,0), np.expand_dims(active_np,0), res
    
    def _kkt_diff_callback(primals, tangents):
        return pure_callback(
            _kkt_diff,
            _bwd_shapes,
            *primals, 
            *tangents,
            vmap_method="expand_dims"
        )

    @custom_jvp
    def solver(P, q, A, b, G, h):
        return _solve_qp_callback(P, q, A, b, G, h)
    
    @solver.defjvp
    def solver_jvp(primals, tangents):

        dx, dmu, dlam, active, res = _kkt_diff_callback(primals, tangents)

        tangents_out = {
            "x":      dx,
            "lam":    dlam,
            "mu":     dmu,
            "active": jnp.zeros_like(active,dtype=jax.dtypes.float0),
        }

        return res, tangents_out

    return solver