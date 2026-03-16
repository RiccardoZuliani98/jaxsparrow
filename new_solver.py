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
        jax.ShapeDtypeStruct((n_ineq,),        jax.dtypes.float0),    # active
        _fwd_shapes,
    )


    # =================================================================
    # SETUP SOLVER
    # =================================================================

    def _convert_qp_ingredients_to_numpy(P, q, A, b, G, h):

        # Convert vectors
        start = perf_counter()
        q_np = np.asarray(q, dtype=dtype).squeeze()
        b_np = np.asarray(b, dtype=dtype).squeeze()
        h_np = np.asarray(h, dtype=dtype).squeeze()
        t_convert = {"convert_vectors": perf_counter() - start}

        # Convert matrices
        start = perf_counter()
        P_np = np.asarray(P, dtype=dtype).squeeze()
        A_np = np.asarray(A, dtype=dtype).squeeze()
        G_np = np.asarray(G, dtype=dtype).squeeze()
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

    def _convert_solution_to_jax(x, lam, mu, active, batch_size=0, n_var=0, n_eq=0, n_ineq=0):
        start = perf_counter()
        if batch_size > 0:
            sol =  {
                "x": jnp.broadcast_to(jnp.array(x,dtype=dtype),(batch_size, n_var)),
                "lam":jnp.broadcast_to(jnp.array(lam,dtype=dtype),(batch_size,n_ineq)),
                "mu":jnp.broadcast_to(jnp.array(mu,dtype=dtype),(batch_size,n_eq)),
                "active":jnp.broadcast_to(jnp.array(active,dtype=jnp.bool_),(batch_size,n_ineq))
            }
        else:
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

        # detect batching: q (primals[1]) is always 1D; expand_dims prepends batch axis
        # _batch_size = primals[1].shape[0] if _is_batched else None
        # if options_parsed["verbose"]:
        # print(f"[_kkt_diff_callback] batched={_is_batched}")

        # detect batching
        batched = P[0].ndim == 2

        # this function takes in jax.numpy arrays and matrices,
        # the tangent arguments dQ, dq, dF, df, dG, dg can be vectorized,
        # in which case they are passed

        logger.info("Hi, I am kkt differentiator and I am running.")

        if batched:

            assert set(elem.shape[0] for elem in (P, q, A, b, G, h)) == {1}
            
            assert P.shape[1:] == (n_var,n_var)
            assert q.shape[1:] == (n_var,)
            assert A.shape[1:] == (n_eq,n_var)
            assert b.shape[1:] == (n_eq,)
            assert G.shape[1:] == (n_ineq,n_var)
            assert h.shape[1:] == (n_ineq,)

            assert set(elem.ndim for elem in (dP, dA, dG)) == {3}
            assert set(elem.ndim for elem in (dq, db, dh)) == {2}
            
            assert dP.shape[1:] == (n_var,n_var)
            assert dq.shape[1:] == (n_var,)
            assert dA.shape[1:] == (n_eq,n_var)
            assert db.shape[1:] == (n_eq,)
            assert dG.shape[1:] == (n_ineq,n_var)
            assert dh.shape[1:] == (n_ineq,)
        
        else:

            assert P.shape == (n_var,n_var)
            assert q.shape == (n_var,)
            assert A.shape == (n_eq,n_var)
            assert b.shape == (n_eq,)
            assert G.shape == (n_ineq,n_var)
            assert h.shape == (n_ineq,)

            assert dP.shape == (n_var,n_var)
            assert dq.shape == (n_var,)
            assert dA.shape == (n_eq,n_var)
            assert db.shape == (n_eq,)
            assert dG.shape == (n_ineq,n_var)
            assert dh.shape == (n_ineq,)

        start = perf_counter()

        # convert to numpy and squeeze any additional dimension
        P_np, q_np, A_np, b_np, G_np, h_np, t_dict = _convert_qp_ingredients_to_numpy(P, q, A, b, G, h)

        # extra shape must be removed here
        assert P_np.shape == (n_var,n_var)
        assert q_np.shape == (n_var,)
        assert A_np.shape == (n_eq,n_var)
        assert b_np.shape == (n_eq,)
        assert G_np.shape == (n_ineq,n_var)
        assert h_np.shape == (n_ineq,)
        
        # Get primal/dual solution (forward eval)
        x_np, lam_np, mu_np, active_np, t_dict_solve = _solve_qp_numpy(P_np, q_np, A_np, b_np, G_np, h_np)

        assert x_np.shape == (n_var,)
        assert mu_np.shape == (n_eq,)
        assert lam_np.shape == (n_ineq,)
        assert active_np.shape == (n_ineq,)

        # Convert vectors
        start = perf_counter()
        dq_np = np.asarray(dq, dtype=dtype)
        db_np = np.asarray(db, dtype=dtype)
        dh_np = np.asarray(dh, dtype=dtype)
        t_dict["convert_vector_derivatives"] = perf_counter() - start

        # Convert matrices
        start = perf_counter()
        dP_np = np.asarray(dP, dtype=dtype)
        dA_np = np.asarray(dA, dtype=dtype)
        dG_np = np.asarray(dG, dtype=dtype)
        t_dict["convert_matrix_derivatives"] = perf_counter() - start
        
        if batched:

            assert set(elem.ndim for elem in (dP, dA, dG)) == {3}
            assert set(elem.ndim for elem in (dq, db, dh)) == {2}
            
            assert dP_np.shape[1:] == (n_var,n_var)
            assert dq_np.shape[1:] == (n_var,)
            assert dA_np.shape[1:] == (n_eq,n_var)
            assert db_np.shape[1:] == (n_eq,)
            assert dG_np.shape[1:] == (n_ineq,n_var)
            assert dh_np.shape[1:] == (n_ineq,)
        
        else:

            assert dP_np.shape == (n_var,n_var)
            assert dq_np.shape == (n_var,)
            assert dA_np.shape == (n_eq,n_var)
            assert db_np.shape == (n_eq,)
            assert dG_np.shape == (n_ineq,n_var)
            assert dh_np.shape == (n_ineq,)

        # differentiate
        start = perf_counter()

        # H_np = np.vstack((A_np, G_np[active_np,:]))  # (n_eq + n_ineq, n_var)
        # dL_np = dP_np @ x_np + dq_np + dA_np.T @ mu_np + dG_np[active_np,:].T @ lam_np[active_np]
        # rhs = np.concatenate([dL_np, dA_np @ x_np - db_np, dG_np[active_np,:] @ x_np - dh_np[active_np]])
        # n_h = H_np.shape[0]
        # lhs = np.block([[P_np, H_np.T],[H_np, np.zeros((n_h,n_h))]])
        
        H_np = np.vstack((A_np, G_np[active_np, :]))   # (n_eq + n_active, n_var)
        n_h  = H_np.shape[0]
        lhs  = np.block([[P_np, H_np.T], [H_np, np.zeros((n_h, n_h))]])

        if batched:

            # expected dimensions
            # dP_np.shape = (batch,n_var,n_var)
            # dq_np.shape == (batch,n_var,) 
            # dA_np.shape == (batch,n_eq,n_var)
            # db_np.shape == (batch,n_eq,)
            # dG_np.shape == (batch,n_ineq,n_var)
            # dh_np.shape == (batch,n_ineq,)
            # x_np.shape == (n_var,)
            # mu_np.shape == (n_eq,)
            # lam_np.shape == (n_ineq,)
            # active_np.shape == (n_ineq,)
            
            # note that batch dimension of each element may differ,
            # but that's not a problem, numpy will correctly broadcast
            # to the correct output dimension

            # all elements here are (batch, n_var)
            dL_np = (dP_np @ x_np
                     + dq_np
                     + dA_np.transpose(0, 2, 1) @ mu_np
                     + dG_np[:, active_np, :].transpose(0, 2, 1) @ lam_np[active_np])

            rhs_pieces = [
                dL_np,
                dA_np @ x_np - db_np,
                dG_np[:, active_np, :] @ x_np - dh_np[:, active_np],
            ]

            # determine batch size
            batch_size = max(p.shape[0] for p in rhs_pieces)

            # form rhs
            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1])) if p.shape[0] == 1 else p
                for p in rhs_pieces
            ], axis=1).T

            assert rhs.shape == (n_var + n_eq + int(np.sum(active_np)), batch_size)
        
        else:
            
            # expected dimensions
            # dP_np.shape = (n_var,n_var)
            # dq_np.shape == (n_var,) 
            # dA_np.shape == (n_eq,n_var)
            # db_np.shape == (n_eq,)
            # dG_np.shape == (n_ineq,n_var)
            # dh_np.shape == (n_ineq,)
            # x_np.shape == (n_var,)
            # mu_np.shape == (n_eq,)
            # lam_np.shape == (n_ineq,)
            # active_np.shape == (n_ineq,)

            dL_np = dP_np @ x_np + dq_np + dA_np.T @ mu_np + dG_np[active_np, :].T @ lam_np[active_np]

            assert dL_np.shape == (n_var,)

            rhs = np.hstack((dL_np, dA_np @ x_np - db_np, dG_np[active_np, :] @ x_np - dh_np[active_np]))

            assert rhs.shape == (n_var + n_eq + int(np.sum(active_np)), )

            batch_size = 0

        t_dict["diff_setup"] = perf_counter() - start
        start = perf_counter()
        sol = np.linalg.solve(lhs, -rhs)
        
        if batched:
            dx_np              = sol[:n_var, :].T             # (B, n_var)
            dmu_np             = sol[n_var : n_var + n_eq, :].T  # (B, n_eq)
            dlam_np            = np.zeros((batch_size, n_ineq))
            dlam_np[:, active_np] = sol[n_var + n_eq :, :].T  # (B, n_active)

            assert dx_np.shape == (batch_size,n_var)
            assert dmu_np.shape == (batch_size,n_eq)
            assert dlam_np.shape == (batch_size,n_ineq)

        else:
            dx_np               = sol[:n_var]
            dmu_np              = sol[n_var : n_var + n_eq]
            dlam_np             = np.zeros(n_ineq)
            dlam_np[active_np]  = sol[n_var + n_eq :]

            assert dx_np.shape == (n_var,)
            assert dmu_np.shape == (n_eq,)
            assert dlam_np.shape == (n_ineq,)

        t_dict["diff_lin_solve"] = perf_counter() - start

        # convert to jax and add extra first dimension if in batched mode
        res, t_sol_convert = _convert_solution_to_jax(x_np, lam_np, mu_np, active_np, batch_size, n_var, n_eq, n_ineq)

        t_full = perf_counter() - start

        if options_parsed["verbose"]:
            logger.info(
                f"DenseQP jvp time {t_full:.3e}s -- "
                f"solve={t_dict_solve["solve"]:.3e}s  "
                f"active set={t_dict_solve["active"]:.3e}s  "
                f"retrieve={t_dict_solve["retrieve"]:.3e}s  "
                f"solve={t_dict_solve["solve"]:.3e}s  "
                f"problem setup={t_dict_solve["problem_setup"]:.3e}s  "
                f"conversion vectors={t_dict["convert_vectors"]:.3e}s  "
                f"conversion vector derivatives={t_dict["convert_vector_derivatives"]:.3e}s  "
                f"conversion matrices={t_dict["convert_matrices"]:.3e}s  "
                f"conversion matrix derivatives={t_dict["convert_matrix_derivatives"]:.3e}s  "
                f"conversion solution={t_sol_convert:.3e}"
                f"derivative setup={t_dict["diff_setup"]:.3e}"
                f"derivative linear solve={t_dict["diff_lin_solve"]:.3e}"
            )

        return dx_np, dmu_np, dlam_np, np.zeros(res["active"].shape, dtype=jax.dtypes.float0), res
    
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

        dx, dmu, dlam, dactive, res = _kkt_diff_callback(primals, tangents)

        tangents_out = {
            "x":      dx,
            "lam":    dlam,
            "mu":     dmu,
            "active": dactive
        }

        return res, tangents_out

    return solver