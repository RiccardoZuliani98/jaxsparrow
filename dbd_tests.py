import pytest
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
import numpy.testing as npt
from jaxsparrow import setup_sparse_solver
import jax.random as jrandom
from jax import vjp

from _dbd_wrapper import RegularizedQPSolver

def generate_mpc_problem(
    horizon=50, 
    cost_state=1.0, 
    cost_input=0.1, 
    duplicate_eq=False
):
    """Factory function to generate an MPC problem."""
    A_dyn = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B_dyn = jnp.array([[0.0], [1.0]])

    xmax, xmin = 5.0, -5.0
    umax, umin = 0.5, -0.5

    nx, nu = B_dyn.shape
    N = horizon
    nz = (N + 1) * nx + N * nu

    # Cost (Diagonal) - Controls strong convexity based on arguments
    P_dense = jnp.diag(
        jnp.hstack((
            jnp.ones((N + 1) * nx) * cost_state,
            jnp.ones(N * nu) * cost_input,
        ))
    )
    P = BCOO.fromdense(P_dense)
    q = jnp.zeros(nz)

    # Inequality constraints (Box)
    G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
    G = BCOO.fromdense(G_dense)

    h = jnp.hstack((
        jnp.ones((N + 1) * nx) * xmax,
        jnp.ones(N * nu) * umax,
        -jnp.ones((N + 1) * nx) * xmin,
        -jnp.ones(N * nu) * umin,
    ))

    # Equality constraints (Dynamics)
    S = jnp.diag(jnp.ones(N), -1)
    Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
    Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
    Au = jnp.kron(Su, -B_dyn)
    
    Aeq_dense = jnp.hstack((Ax, Au))

    if duplicate_eq:
        Aeq_dense = jnp.vstack((Aeq_dense, Aeq_dense))

    Aeq = BCOO.fromdense(Aeq_dense)

    def beq(x_init):
        b_base = jnp.hstack((x_init, jnp.zeros(N * nx)))
        if duplicate_eq:
            return jnp.hstack((b_base, b_base))
        return b_base

    return {
        "nz": nz, "neq": Aeq_dense.shape[0], "nineq": G_dense.shape[0],
        "P": P, "q": q, "A": Aeq, "G": G, "h": h, "beq": beq
    }

# ── Parameterized Pytest ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "cost_state, cost_input, duplicate_eq, scenario",
    [
        # Strict LICQ Maintained
        (1.0, 0.1, False, "Strongly Convex, Strict LICQ"),
        (0.0, 0.1, False, "Not Strongly Convex (Zero State Cost), Strict LICQ"),
        (0.0, 0.0, False, "Not Strongly Convex (Zero State/Input Cost), Strict LICQ"),
        
        # LICQ Lost (Duplicated Equality Constraints)
        (1.0, 0.1, True, "Strongly Convex, LICQ Lost"),
        (0.0, 0.1, True, "Not Strongly Convex (Zero State Cost), LICQ Lost"),
        (0.0, 0.0, True, "Not Strongly Convex (Zero State/Input Cost), LICQ Lost"),
    ]
)
def ALT_test_regularized_solver_convergence_from_cold_start():
    """
    Validates that the regularized solver converges to the true base optimal solution
    when cold-started from zero, given enough proximal iterations.
    """
    # 1. Setup strongly convex problem with strict LICQ
    mpc = generate_mpc_problem(
        cost_state=1.0, 
        cost_input=0.1, 
        duplicate_eq=False
    )
    
    x_init = jnp.array([-3.0, -1.0])
    b = mpc["beq"](x_init)

    sparsity_patterns = {"P": mpc["P"], "A": mpc["A"], "G": mpc["G"]}

    # 2. Setup and solve Base Problem (Ground Truth)
    base_solver = setup_sparse_solver(
        n_var=mpc["nz"], n_ineq=mpc["nineq"], n_eq=mpc["neq"],
        sparsity_patterns=sparsity_patterns,
        options={"solver": {"backend": "piqp"}}
    )
    
    sol_base = base_solver(
        P=mpc["P"], q=mpc["q"], A=mpc["A"], b=b, G=mpc["G"], h=mpc["h"]
    )

    # 3. Setup Regularized Solver for Convergence
    rho_val = 1e-4  # Using a slightly stronger rho for stable proximal steps
    NUM_ITERATIONS = 20  # Increased iterations to allow convergence from cold start
    
    reg_solver = RegularizedQPSolver(
        n_x=mpc["nz"], n_in=mpc["nineq"], n_eq=mpc["neq"],
        num_steps=NUM_ITERATIONS,
        sparsity_patterns=sparsity_patterns,
        rho=rho_val
    )

    # 4. Initialize Cold-Start Vectors (all zeros)
    bar_x_zero = jnp.zeros(mpc["nz"])
    bar_lam_zero = jnp.zeros(mpc["nineq"])
    bar_mu_zero = jnp.zeros(mpc["neq"])

    # 5. Solve Regularized Problem
    sol_reg = reg_solver.solve(
        P=mpc["P"], q=mpc["q"], A=mpc["A"], b=b, G=mpc["G"], h=mpc["h"],
        bar_x=bar_x_zero, 
        bar_lam=bar_lam_zero, 
        bar_mu=bar_mu_zero
    )

    # 6. Assertions
    # We use slightly looser tolerances (1e-3) because asymptotic convergence 
    # of proximal methods can leave a minor tail error after a finite number of steps.
    npt.assert_allclose(
        sol_reg["x"], sol_base["x"], 
        atol=1e-3, rtol=1e-3, 
        err_msg="Primal solution (x) failed to converge to base optimal from a cold start."
    )
    
    npt.assert_allclose(
        sol_reg["lam"], sol_base["lam"], 
        atol=1e-3, rtol=1e-3, 
        err_msg="Inequality duals (lam) failed to converge to base optimal from a cold start."
    )
    
    npt.assert_allclose(
        sol_reg["mu"], sol_base["mu"], 
        atol=1e-3, rtol=1e-3, 
        err_msg="Equality duals (mu) failed to converge to base optimal from a cold start."
    )

def ALT_test_regularized_solver_with_fixed_elements_api():
    """
    Validates that providing structural matrices (P, A, G) and static vectors (q, h) 
    via the `fixed_elements` argument yields the exact same solution as providing 
    them dynamically to the solve function.
    """
    # 1. Setup strongly convex problem
    mpc = generate_mpc_problem(
        cost_state=1.0, 
        cost_input=0.1, 
        duplicate_eq=False
    )
    
    x_init = jnp.array([-3.0, -1.0])
    b = mpc["beq"](x_init)

    sparsity_patterns = {"P": mpc["P"], "A": mpc["A"], "G": mpc["G"]}
    
    # Cold-start reference
    bar_x = jnp.zeros(mpc["nz"])
    bar_lam = jnp.zeros(mpc["nineq"])
    bar_mu = jnp.zeros(mpc["neq"])
    
    rho_val = 1e-6
    NUM_ITERATIONS = 3

    # ── Run 1: Dynamic Solver ──
    reg_solver_dynamic = RegularizedQPSolver(
        n_x=mpc["nz"], n_in=mpc["nineq"], n_eq=mpc["neq"],
        num_steps=NUM_ITERATIONS,
        sparsity_patterns=sparsity_patterns,
        rho=rho_val
    )
    
    sol_dynamic = reg_solver_dynamic.solve(
        P=mpc["P"], q=mpc["q"], A=mpc["A"], b=b, G=mpc["G"], h=mpc["h"],
        bar_x=bar_x, bar_lam=bar_lam, bar_mu=bar_mu
    )

    # ── Run 2: Fixed Elements Solver ──
    # Convert JAX BCOO/arrays to standard NumPy arrays so scipy.sparse.csc_matrix 
    # inside the wrapper can ingest them smoothly.
    fixed_elements_dict = {
        "P": np.array(mpc["P"].todense()),
        "A": np.array(mpc["A"].todense()),
        "G": np.array(mpc["G"].todense()),
        "q": np.array(mpc["q"]),
        "h": np.array(mpc["h"])
        # 'b' is excluded because it depends on the current state in MPC
    }
    
    reg_solver_fixed = RegularizedQPSolver(
        n_x=mpc["nz"], n_in=mpc["nineq"], n_eq=mpc["neq"],
        num_steps=NUM_ITERATIONS,
        fixed_elements=fixed_elements_dict,
        rho=rho_val
    )
    
    # Solve passing ONLY the dynamic parameter 'b'
    sol_fixed = reg_solver_fixed.solve(
        b=b, 
        bar_x=bar_x, bar_lam=bar_lam, bar_mu=bar_mu
    )

    # ── Assertions ──
    # Using strict tolerances (1e-8) because mathematically, both code paths 
    # should formulate the exact same backend matrices.
    npt.assert_allclose(
        sol_fixed["x"], sol_dynamic["x"], 
        atol=1e-8, rtol=1e-8, 
        err_msg="Primal solution (x) diverged when using the fixed_elements API."
    )
    
    npt.assert_allclose(
        sol_fixed["lam"], sol_dynamic["lam"], 
        atol=1e-8, rtol=1e-8, 
        err_msg="Inequality duals (lam) diverged when using the fixed_elements API."
    )
    
    npt.assert_allclose(
        sol_fixed["mu"], sol_dynamic["mu"], 
        atol=1e-8, rtol=1e-8, 
        err_msg="Equality duals (mu) diverged when using the fixed_elements API."
    )

def ALT_test_vjp_regularized_vs_base():
    """
    Validates that the VJP of the regularized solver matches the VJP of the 
    base solver under strongly convex and strict LICQ conditions.
    """
    # 1. Setup problem
    mpc = generate_mpc_problem(cost_state=1.0, cost_input=0.1, duplicate_eq=False)
    
    x_init = jnp.array([-3.0, -1.0])
    b_val = mpc["beq"](x_init)
    sparsity_patterns = {"P": mpc["P"], "A": mpc["A"], "G": mpc["G"]}

    # 2. Setup Solvers
    base_solver = setup_sparse_solver(
        n_var=mpc["nz"], n_ineq=mpc["nineq"], n_eq=mpc["neq"],
        sparsity_patterns=sparsity_patterns,
        options={
            "solver": {"backend": "piqp"},
            "diff_mode":"rev"
        }
    )
    
    reg_solver = RegularizedQPSolver(
        n_x=mpc["nz"], n_in=mpc["nineq"], n_eq=mpc["neq"],
        num_steps=5, # Enough steps to reach the true optimum
        sparsity_patterns=sparsity_patterns,
        rho=1e-8
    )

    # Constant warmstarts
    bar_x = jnp.zeros(mpc["nz"])
    bar_lam = jnp.zeros(mpc["nineq"])
    bar_mu = jnp.zeros(mpc["neq"])

    # 3. Define differentiable closures
    def solve_base_fn(q, b, h):
        sol = base_solver(P=mpc["P"], q=q, A=mpc["A"], b=b, G=mpc["G"], h=h)
        return sol["x"], sol["lam"], sol["mu"]

    def solve_reg_fn(q, b, h):
        sol = reg_solver.solve(
            P=mpc["P"], q=q, A=mpc["A"], b=b, G=mpc["G"], h=h, 
            bar_x=bar_x, bar_lam=bar_lam, bar_mu=bar_mu
        )
        return sol["x"], sol["lam"], sol["mu"]

    # 4. Generate random cotangents (v_x, v_lam, v_mu)
    key = jrandom.PRNGKey(42)
    k1, k2, k3 = jrandom.split(key, 3)
    
    v_x = jrandom.normal(k1, (mpc["nz"],))
    v_lam = jrandom.normal(k2, (mpc["nineq"],))
    v_mu = jrandom.normal(k3, (mpc["neq"],))

    # 5. Compute VJPs
    _, vjp_base_fn = vjp(solve_base_fn, mpc["q"], b_val, mpc["h"])
    _, vjp_reg_fn = vjp(solve_reg_fn, mpc["q"], b_val, mpc["h"])

    cotangents = (v_x, v_lam, v_mu)
    dq_base, db_base, dh_base = vjp_base_fn(cotangents)
    dq_reg, db_reg, dh_reg = vjp_reg_fn(cotangents)

    # 6. Assertions
    npt.assert_allclose(dq_reg, dq_base, atol=1e-3, rtol=1e-3, err_msg="VJP w.r.t q diverged.")
    npt.assert_allclose(db_reg, db_base, atol=1e-3, rtol=1e-3, err_msg="VJP w.r.t b diverged.")
    npt.assert_allclose(dh_reg, dh_base, atol=1e-3, rtol=1e-3, err_msg="VJP w.r.t h diverged.")

@pytest.mark.parametrize(
    "cost_state, cost_input, duplicate_eq, scenario",
    [
        (1.0, 0.1, False, "Strongly Convex, Strict LICQ"),
        (0.0, 0.0, False, "Not Strongly Convex, Strict LICQ"),
        (1.0, 0.1, True, "Strongly Convex, LICQ Lost"),
        (0.0, 0.0, True, "Not Strongly Convex, LICQ Lost"),
    ]
)
def test_vjp_regularized_vs_finite_difference(cost_state, cost_input, duplicate_eq, scenario):
    """
    Validates the VJP of the regularized solver against central finite difference 
    approximations across varying convexity and LICQ conditions.
    """
    mpc = generate_mpc_problem(cost_state=cost_state, cost_input=cost_input, duplicate_eq=duplicate_eq)
    
    x_init = jnp.array([-3.0, -1.0])
    b_val = mpc["beq"](x_init)
    sparsity_patterns = {"P": mpc["P"], "A": mpc["A"], "G": mpc["G"]}

    reg_solver = RegularizedQPSolver(
        n_x=mpc["nz"], n_in=mpc["nineq"], n_eq=mpc["neq"],
        num_steps=3,
        sparsity_patterns=sparsity_patterns,
        rho=1e-8
    )

    # 1. Run a nominal solve with zero initialization to get the primal-dual guess
    init_x = jnp.zeros(mpc["nz"])
    init_lam = jnp.zeros(mpc["nineq"])
    init_mu = jnp.zeros(mpc["neq"])

    nominal_sol = reg_solver.solve(
        P=mpc["P"], q=mpc["q"], A=mpc["A"], b=b_val, G=mpc["G"], h=mpc["h"], 
        bar_x=init_x, bar_lam=init_lam, bar_mu=init_mu
    )

    # Use the nominal solution as the initialization for subsequent evaluations
    bar_x = nominal_sol["x"]
    bar_lam = nominal_sol["lam"]
    bar_mu = nominal_sol["mu"]

    def solve_reg_fn(q, b, h):
        sol = reg_solver.solve(
            P=mpc["P"], q=q, A=mpc["A"], b=b, G=mpc["G"], h=h, 
            bar_x=bar_x, bar_lam=bar_lam, bar_mu=bar_mu
        )
        return sol["x"], sol["lam"], sol["mu"]

    # 2. Random Keys for Cotangents and Perturbation Directions
    key = jrandom.PRNGKey(101)
    k_vx, k_vlam, k_vmu, k_dq, k_db, k_dh = jrandom.split(key, 6)
    
    # Cotangents (for defining the scalar projection)
    v_x = jrandom.normal(k_vx, (mpc["nz"],))
    v_lam = jrandom.normal(k_vlam, (mpc["nineq"],))
    v_mu = jrandom.normal(k_vmu, (mpc["neq"],))

    # Perturbation directions (for the directional derivative)
    dir_q = jrandom.normal(k_dq, (mpc["nz"],))
    dir_b = jrandom.normal(k_db, (mpc["neq"],))
    dir_h = jrandom.normal(k_dh, (mpc["nineq"],))

    # 3. Compute Directional Derivative via VJP
    _, vjp_reg_fn = vjp(solve_reg_fn, mpc["q"], b_val, mpc["h"])
    grad_q, grad_b, grad_h = vjp_reg_fn((v_x, v_lam, v_mu))
    
    # The dot product of the VJP gradients and the perturbation directions
    dir_deriv_vjp = jnp.vdot(grad_q, dir_q) + jnp.vdot(grad_b, dir_b) + jnp.vdot(grad_h, dir_h)

    # 4. Compute Directional Derivative via Central Finite Difference
    eps = 1e-9
    
    def scalar_loss(q, b, h):
        x, lam, mu = solve_reg_fn(q, b, h)
        return jnp.vdot(v_x, x) + jnp.vdot(v_lam, lam) + jnp.vdot(v_mu, mu)

    loss_plus = scalar_loss(mpc["q"] + eps * dir_q, b_val + eps * dir_b, mpc["h"] + eps * dir_h)
    loss_minus = scalar_loss(mpc["q"] - eps * dir_q, b_val - eps * dir_b, mpc["h"] - eps * dir_h)
    
    dir_deriv_fd = (loss_plus - loss_minus) / (2 * eps)

    # 5. Assertions
    npt.assert_allclose(
        dir_deriv_vjp, dir_deriv_fd, 
        atol=1e-3, rtol=1e-3, 
        err_msg=f"VJP diverged from Finite Difference for scenario: {scenario}"
    )

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))