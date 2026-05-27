from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
from jax import jit, jvp, vjp, vmap, block_until_ready
from jax.experimental.sparse import BCOO
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from jaxsparrow import setup_sparse_solver
import qpax

jax.config.update("jax_enable_x64", True)

# ranges of parameters
n_runs      = 10
horizon     = [10, 15, 20, 25, 30]
cl_horizon  = 10

def main(n_runs:int, horizon:int, cl_horizon:int) -> dict:
    """Runs a performance benchmark for JaxSparrow and QPAX over a closed-loop MPC rollout.

    This function builds the dense and sparse matrices for an MPC problem instance,
    warms up the JIT compilation compilation paths, compiles the VJP/JVP differentiation 
    pipelines, and records execution times across multiple randomized tracking runs.

    Args:
        n_runs: Number of timed simulation iterations.
        horizon: The prediction horizon length (N) for the controllers.
        cl_horizon: The total time steps for the closed-loop simulation rollout.

    Returns:
        A dictionary containing cost evaluations, gradients, and lists of the 
        raw execution times for each run to allow statistical processing.
    """

    # =========================================================
    # 1. SETUP MPC
    # =========================================================

    A_dyn = jnp.array([[1, 1], [0, 1]])
    B_dyn = jnp.array([[0], [1]])

    xmax, xmin = 5, -5
    umax, umin = 0.5, -0.5
    cost_state, cost_input = 1, 0.1

    nx, nu = B_dyn.shape
    N = horizon
    nz = (N + 1) * nx + N * nu

    # --- Cost (Dense & Sparse) ---
    P_dense = jnp.diag(
        jnp.hstack((
            jnp.ones((N + 1) * nx) * cost_state,
            jnp.ones(N * nu) * cost_input,
        ))
    )
    P = BCOO.fromdense(P_dense)
    q = jnp.zeros(nz)

    # --- Inequality constraints (Dense & Sparse) ---
    G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
    G = BCOO.fromdense(G_dense)

    h = jnp.hstack((
        jnp.ones((N + 1) * nx) * xmax,
        jnp.ones(N * nu) * umax,
        -jnp.ones((N + 1) * nx) * xmin,
        -jnp.ones(N * nu) * umin,
    ))

    # --- Equality constraints (Dense & Sparse) ---
    S = jnp.diag(jnp.ones(N), -1)
    Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
    Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
    Au = jnp.kron(Su, -B_dyn)
    Aeq_dense = jnp.hstack((Ax, Au))
    Aeq = BCOO.fromdense(Aeq_dense)

    _b_template = jnp.zeros((N + 1) * nx)

    @jit
    def beq(x_init):
        return _b_template.at[:nx].set(x_init)

    neq = Aeq_dense.shape[0]
    nineq = G_dense.shape[0]

    # =========================================================
    # 2. SOLVER DEFINITIONS
    # =========================================================

    # --- JaxSparrow Solvers (Sparse) ---
    sparsity_patterns = {"P": P, "A": Aeq, "G": G}

    solver_sparrow_vjp = setup_sparse_solver(
        n_var=nz, n_ineq=nineq, n_eq=neq,
        # sparsity_patterns=sparsity_patterns,
        fixed_elements={"P": P, "A": Aeq, "G": G, "h":h, "q":q},
        options={"diff_mode": "rev"},
    )

    solver_sparrow_jvp = setup_sparse_solver(
        n_var=nz, n_ineq=nineq, n_eq=neq,
        # sparsity_patterns=sparsity_patterns,
        fixed_elements={"P": P, "A": Aeq, "G": G, "h":h, "q":q},
        options={"diff_mode": "fwd"},
    )

    def solve_mpc_sparrow_vjp(x_init):
        return solver_sparrow_vjp(b=beq(x_init))

    def solve_mpc_sparrow_jvp(x_init):
        return solver_sparrow_jvp(b=beq(x_init))

    # --- QPAX Solver (Dense) ---
    def solve_mpc_qpax(x_init):
        return qpax.solve_qp_primal(P_dense, q, Aeq_dense, beq(x_init), G_dense, h, solver_tol=1e-8)


    # =========================================================
    # 3. CLOSED-LOOP ROLLOUTS & OBJECTIVES
    # =========================================================

    def cost_fun(x_cl, u_cl):
        return jnp.dot(x_cl, x_cl) * cost_state + jnp.dot(u_cl, u_cl) * cost_input

    # --- JaxSparrow Rollouts ---
    def closed_loop_sparrow_vjp(x0):
        x_cl = [x0]
        u_cl = []
        for t in range(cl_horizon):
            u_cl.append(
                solve_mpc_sparrow_vjp(x_cl[-1])["x"][(horizon + 1) * nx : (horizon + 1) * nx + nu]
            )
            x_cl.append(A_dyn @ x_cl[-1] + B_dyn @ u_cl[-1])
        return jnp.hstack(x_cl), jnp.hstack(u_cl)

    def closed_loop_sparrow_jvp(x0):
        x_cl = [x0]
        u_cl = []
        for t in range(cl_horizon):
            u_cl.append(
                solve_mpc_sparrow_jvp(x_cl[-1])["x"][(horizon + 1) * nx : (horizon + 1) * nx + nu]
            )
            x_cl.append(A_dyn @ x_cl[-1] + B_dyn @ u_cl[-1])
        return jnp.hstack(x_cl), jnp.hstack(u_cl)

    def cl_cost_sparrow_vjp(x0):
        x_cl, u_cl = closed_loop_sparrow_vjp(x0)
        return cost_fun(x_cl, u_cl)

    def cl_cost_sparrow_jvp(x0):
        x_cl, u_cl = closed_loop_sparrow_jvp(x0)
        return cost_fun(x_cl, u_cl)

    # --- QPAX Rollout ---
    def closed_loop_qpax(x0):
        x_cl = [x0]
        u_cl = []
        for t in range(cl_horizon):
            sol_x = solve_mpc_qpax(x_cl[-1])
            u_cl.append(sol_x[(horizon + 1) * nx : (horizon + 1) * nx + nu])
            x_cl.append(A_dyn @ x_cl[-1] + B_dyn @ u_cl[-1])
        return jnp.hstack(x_cl), jnp.hstack(u_cl)

    def cl_cost_qpax(x0):
        x_cl, u_cl = closed_loop_qpax(x0)
        return cost_fun(x_cl, u_cl)


    # =========================================================
    # 4. DIFFERENTIATION PATHS
    # =========================================================

    # --- JaxSparrow VJP ---
    @jit
    def solve_and_differentiate_sparrow(x0):
        cost, vjp_func = vjp(cl_cost_sparrow_vjp, x0)
        jac = vjp_func(1.0)[0]
        return cost, jac

    # --- JaxSparrow JVP ---
    def jvp_func_base(x0, dx0):
        return jvp(cl_cost_sparrow_jvp, (x0,), (dx0,))

    jvp_func_sparrow = jit(vmap(jvp_func_base, in_axes=(None, 0)))

    # --- QPAX VJP ---
    @jit
    def solve_and_differentiate_qpax(x0):
        cost, vjp_func = vjp(cl_cost_qpax, x0)
        jac = vjp_func(1.0)[0]
        return cost, jac


    # =========================================================
    # 5. WARMUP & VALIDATION
    # =========================================================
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, n_runs)

    x0_warmup = jax.random.uniform(keys[0], shape=(2,), minval=-2.0, maxval=2.0)
    e_mat = jnp.eye(x0_warmup.shape[0], dtype=x0_warmup.dtype)

    # Warmup pipelines
    _ = jvp_func_sparrow(x0_warmup, e_mat)
    _ = solve_and_differentiate_sparrow(0.9 * x0_warmup)
    _ = solve_and_differentiate_qpax(0.9 * x0_warmup)

    # --- Cross-Solver Verification ---
    cost_s, jac_s = solve_and_differentiate_sparrow(x0_warmup)
    cost_q, jac_q = solve_and_differentiate_qpax(x0_warmup)


    # =========================================================
    # 6. PERFORMANCE BENCHMARK RUNS
    # =========================================================
    elapsed_sparrow_vjp, elapsed_sparrow_jvp, elapsed_qpax_vjp = [], [] , []

    solver_sparrow_vjp.timings.reset()
    solver_sparrow_jvp.timings.reset()

    # 1. JaxSparrow VJP
    for i in range(n_runs):
        xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
        start = perf_counter()
        result = solve_and_differentiate_sparrow(xi)
        block_until_ready(result)
        elapsed_sparrow_vjp.append(perf_counter() - start)

    # 2. JaxSparrow JVP
    for i in range(n_runs):
        xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
        start = perf_counter()
        result = jvp_func_sparrow(xi, e_mat)
        block_until_ready(result)
        elapsed_sparrow_jvp.append(perf_counter() - start)

    # 3. QPAX VJP
    for i in range(n_runs):
        xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
        start = perf_counter()
        result = solve_and_differentiate_qpax(xi)
        block_until_ready(result)
        elapsed_qpax_vjp.append(perf_counter() - start)

    return {
        "jaxsparrow_cost":  cost_s,
        "qpax_cost":        cost_q,
        "jaxsparrow_grad":  jac_s,
        "qpax_grad":        jac_q,
        "grad_diff":        jnp.linalg.norm(jac_s - jac_q),
        "jaxsparrow_vjp":   elapsed_sparrow_vjp,
        "jaxsparrow_jvp":   elapsed_sparrow_jvp,
        "qpax_vjp":         elapsed_qpax_vjp
    }

# =========================================================
# EXPERIMENT EXECUTION & PLOTTING LOOP
# =========================================================
if __name__ == "__main__":
    # Initialize metric containers
    metrics = {
        "sparrow_vjp": {"means": [], "vars": []},
        "sparrow_jvp": {"means": [], "vars": []},
        "qpax_vjp":    {"means": [], "vars": []}
    }

    print("--- STARTING HORIZON SIMULATION LOOP ---")
    for h in horizon:
        print(f"Executing simulation for horizon length: {h}...")
        results = main(n_runs=n_runs, horizon=h, cl_horizon=cl_horizon)
        
        # Calculate statistical mean & variance for each solver configuration
        for raw_key, target_key in [("jaxsparrow_vjp", "sparrow_vjp"), 
                                    ("jaxsparrow_jvp", "sparrow_jvp"), 
                                    ("qpax_vjp", "qpax_vjp")]:
            raw_data = np.array(results[raw_key])
            metrics[target_key]["means"].append(np.mean(raw_data))
            metrics[target_key]["vars"].append(np.var(raw_data))

    # --- PLOT RESULTS ---
    plt.figure(figsize=(10, 6))
    
    # Plot definitions
    config = {
        "sparrow_vjp": {"label": "JaxSparrow VJP", "color": "royalblue", "marker": "o"},
        "sparrow_jvp": {"label": "JaxSparrow JVP", "color": "forestgreen", "marker": "s"},
        "qpax_vjp":    {"label": "QPAX VJP",       "color": "crimson",     "marker": "^"}
    }
    
    for key, style in config.items():
        plt.errorbar(
            horizon, 
            metrics[key]["means"], 
            yerr=metrics[key]["vars"], 
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            capsize=5, 
            linestyle="-", 
            linewidth=1.5
        )

    plt.xlabel("Horizon Length (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Computational Time (Mean and Variance) Over Horizons")
    plt.xticks(horizon)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save visualization to disk
    plt.savefig("mpc_horizon_benchmark.png", dpi=300)
    plt.show()