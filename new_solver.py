from typing import Optional, cast
from mpc_types import DEFAULT_SOLVER_OPTIONS, SolverOptions, SolverOptionsFull
from jax import custom_jvp, ShapeDtypeStruct, pure_callback
import jax.numpy as jnp
from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
import jax

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

    def _solve_qp(Q, q, F, f, G, g):

        # Convert vectors
        start = perf_counter()
        q_vec = np.asarray(q, dtype=dtype)
        f_vec = np.asarray(f, dtype=dtype)
        g_vec = np.asarray(g, dtype=dtype)
        t_to_numpy = perf_counter() - start

        # Convert matrices
        start = perf_counter()

        Q_mat = np.asarray(Q, dtype=dtype)
        F_mat = np.asarray(F, dtype=dtype)
        G_mat = np.asarray(G, dtype=dtype)

        t_convert = perf_counter() - start

        # Build qpsolvers Problem
        prob_dict = {
            "P": Q_mat,
            "q": q_vec,
            "A": F_mat,
            "b": f_vec,
            "G": G_mat,
            "h": g_vec,
        }
        prob = Problem(**prob_dict)

        # Solve QP
        start = perf_counter()
        sol = solve_problem(prob, solver=options_parsed["solver"])
        assert sol.found
        t_solve = perf_counter() - start

        # Recover primal/dual variables
        start = perf_counter()
        x = np.asarray(sol.x, dtype=dtype).reshape(-1)
        mu = np.asarray(sol.y, dtype=dtype).reshape(-1)
        lam = np.asarray(sol.z, dtype=dtype).reshape(-1)
        t_retrieve = perf_counter() - start

        # Determine active set: Gx − h >= tolerance
        start = perf_counter()
        active = np.asarray(np.abs(G_mat @ sol.x - g_vec) <= options_parsed["jac_tol"]).reshape(-1)
        t_active = perf_counter() - start

        print(
            f"DenseQP solve:  solve={t_solve:.3e}s  "
            f"convert={t_convert:.3e}s  retrieve={t_retrieve:.3e}s  "
            f"active={t_active:.3e}s  to_numpy={t_to_numpy:.3e}s"
        )

        return {"x": x, "lam": lam, "mu": mu, "active": active}

    def _solve_qp_callback(Q, q, F, f, G, g):
        return pure_callback(
            _solve_qp,
            _fwd_shapes,
            Q, q, F, f, G, g,
        )

    # =================================================================
    # SETUP DIFFERENTIATOR
    # =================================================================

    def _kkt_diff(Q, q, F, f, G, g, dQ, dq, dF, df, dG, dg):

        print("Hi, I am kkt differentiator and I am running.")

        # Convert vectors
        start = perf_counter()
        q = np.asarray(q, dtype=dtype)
        f = np.asarray(f, dtype=dtype)
        g = np.asarray(g, dtype=dtype)
        dq = np.asarray(dq, dtype=dtype)
        df = np.asarray(df, dtype=dtype)
        dg = np.asarray(dg, dtype=dtype)
        t_to_numpy = perf_counter() - start

        # Convert matrices
        start = perf_counter()

        Q = np.asarray(Q, dtype=dtype)
        F = np.asarray(F, dtype=dtype)
        G = np.asarray(G, dtype=dtype)
        dQ = np.asarray(dQ, dtype=dtype)
        dF = np.asarray(dF, dtype=dtype)
        dG = np.asarray(dG, dtype=dtype)
        t_convert = perf_counter() - start

        # Get primal/dual solution (forward eval)
        res = _solve_qp(Q, q, F, f, G, g)
        x = res["x"]
        lam = res["lam"]
        mu = res["mu"]
        active = np.array(res["active"],dtype=np.bool_)

        # Stack constraints
        #TODO: this converts to numpy again, best to have a conversion utility
        # before running _solve_qp and then pass numpy stuff to _solve_qp,
        # then repeat this for the forward callback
        H = np.vstack((F, G[active,:]))  # (n_eq + n_ineq, n_var)

        # Derivative of Lagrangian:
        dL = dQ @ x + dq + dF.T @ mu + dG[active,:].T @ lam[active]

        # RHS = concatenation of dL and constraint derivatives
        rhs = np.concatenate([dL, dF @ x - df, dG[active,:] @ x - dg[active]])
        lhs = np.block([[Q, H.T],[H, np.zeros((H.shape[0],H.shape[0]))]])

        sol      = np.linalg.solve(lhs, -rhs)
        dx       = sol[:n_var]
        dmu      = sol[n_var : n_var + n_eq]
        dlam_a   = sol[n_var + n_eq :]
        dlam          = np.zeros(n_ineq)
        dlam[active]  = dlam_a

        return dx, dmu, dlam, active, res
    
    def _kkt_diff_callback(Q, q, F, f, G, g, dQ, dq, dF, df, dG, dg):
        return pure_callback(
            _kkt_diff,
            _bwd_shapes,
            Q, q, F, f, G, g, 
            dQ, dq, dF, df, dG, dg
        )

    @custom_jvp
    def solver(Q, q, F, f, G, g):
        return _solve_qp_callback(Q, q, F, f, G, g)
    
    @solver.defjvp
    def solver_jvp(primals, tangents):

        #TODO: vectorize when multiple tangents are passed
        dx, dmu, dlam, active, res = _kkt_diff_callback(*primals, *tangents)

        tangents_out = {
            "x":      dx,
            "lam":    dlam,
            "mu":     dmu,
            "active": jnp.zeros_like(active,dtype=jax.dtypes.float0),
        }

        return res, tangents_out

    return solver