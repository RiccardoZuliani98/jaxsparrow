import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from time import perf_counter
from jax import jit, jvp

from _dbd_wrapper import RegularizedQPSolver

# ── 2. MPC Problem Setup (from attached file) ───────────────────────

horizon = 50

A_dyn = jnp.array([[1, 1], [0, 1]])
B_dyn = jnp.array([[0], [1]])

xmax, xmin = 5, -5
umax, umin = 0.5, -0.5
cost_state, cost_input = 1, 0.1

nx, nu = B_dyn.shape
N = horizon
nz = (N + 1) * nx + N * nu

# Cost (diagonal → sparse)
P_dense = jnp.diag(
    jnp.hstack((
        jnp.ones((N + 1) * nx) * cost_state,
        jnp.ones(N * nu) * cost_input,
    ))
)
P = BCOO.fromdense(P_dense)
q = jnp.zeros(nz)

# Inequality constraints (box → sparse)
G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
G = BCOO.fromdense(G_dense)

h = jnp.hstack((
    jnp.ones((N + 1) * nx) * xmax,
    jnp.ones(N * nu) * umax,
    -jnp.ones((N + 1) * nx) * xmin,
    -jnp.ones(N * nu) * umin,
))

# Equality constraints (dynamics → sparse)
S = jnp.diag(jnp.ones(N), -1)
Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B_dyn)
Aeq_dense = jnp.hstack((Ax, Au))
Aeq = BCOO.fromdense(Aeq_dense)

def beq(x_init):
    return jnp.hstack((x_init, jnp.zeros(N * nx)))

neq = Aeq_dense.shape[0]
nineq = G_dense.shape[0]

# ── 3. Initialization of Base and Regularized Solvers ────────────────

from jaxsparrow import setup_sparse_solver

sparsity_patterns = {"P": P, "A": Aeq, "G": G}

# --- Base (Unregularized) Solver ---
base_solver = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
    options={"solver": {"backend": "piqp"}}
)

def solve_mpc_base(x_init):
    return base_solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# --- Regularized Solver ---
rho_val = 1e-6 # Using a slightly larger rho to see convergence behavior
NUM_ITERATIONS = 3

reg_solver = RegularizedQPSolver(
    n_x=nz, n_in=nineq, n_eq=neq,
    num_steps=NUM_ITERATIONS,  # Updated: Passed to constructor
    sparsity_patterns=sparsity_patterns,
    rho=rho_val,
    # options={"solver": {"backend": "piqp"}}
)

# Initial state and perturbation
x0 = jnp.array([-3.0, -1.0])
epsilon = 0.1
dx0 = jnp.array([epsilon, 0.0])

# Cold-start references
bar_x = jnp.zeros(nz)
bar_lam = jnp.zeros(nineq)
bar_mu = jnp.zeros(neq)

# Updated: Cleaned up signature since num_steps is bound to the solver
@jit
def solve_mpc_reg(x_init):
    return reg_solver.solve(
        P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h,
        bar_x=bar_x, bar_lam=bar_lam, bar_mu=bar_mu
    )

# ── 4. JVP Definitions ──────────────────────────────────────────────

@jit
def jvp_base(x_init, dx_init):
    return jvp(solve_mpc_base, (x_init,), (dx_init,))

# Updated: Removed the static argname lambda workaround
@jit
def jvp_reg(x_init, dx_init):
    return jvp(solve_mpc_reg, (x_init,), (dx_init,))

# ── 5. Execution and Comparison ─────────────────────────────────────

print("Compiling and running base solver...")
# Warmup
_, _ = jvp_base(x0, 100 * dx0)
start = perf_counter()
sol_base, dsol_base = jvp_base(x0, dx0)
print(f"Base elapsed: {perf_counter() - start:.4f}s")

print(f"\nCompiling and running regularized solver ({NUM_ITERATIONS} steps)...")
# Warmup
_, _ = jvp_reg(x0, 100 * dx0)
start = perf_counter()
sol_reg, dsol_reg = jvp_reg(x0, dx0)
print(f"Regularized elapsed: {perf_counter() - start:.4f}s")

# --- Validation: Finite Difference for Regularized ---
sol_perturbed = solve_mpc_reg(x0 + dx0)
dx_reg_analytic = dsol_reg["x"] / epsilon
dx_reg_fd = (sol_perturbed["x"] - sol_reg["x"]) / epsilon

fd_error = jnp.linalg.norm(dx_reg_analytic - dx_reg_fd) / jnp.linalg.norm(dx_reg_fd)

# --- Validation: Match against Base Problem ---
x_error = jnp.linalg.norm(sol_reg["x"] - sol_base["x"]) / jnp.linalg.norm(sol_base["x"])

# Updated: Changed "z" to "lam" and "y" to "mu" to match wrapper output
lam_error = jnp.linalg.norm(jnp.round(sol_reg["lam"],decimals=8) - jnp.round(sol_base["lam"],decimals=8)) / (jnp.linalg.norm(sol_base["lam"]) + 1e-8)
mu_error = jnp.linalg.norm(jnp.round(sol_reg["mu"],decimals=8) - jnp.round(sol_base["mu"],decimals=8)) / (jnp.linalg.norm(sol_base["mu"]) + 1e-8)

dx_error = jnp.linalg.norm(dsol_reg["x"] - dsol_base["x"]) / jnp.linalg.norm(dsol_base["x"])

print(f"\n--- Validation Results ---")
print(f"Reg Solver JVP vs Finite Difference Error: {fd_error:.2e}")
print(f"\n--- Base vs Regularized (Num Steps = {NUM_ITERATIONS}, rho = {rho_val}) ---")
print(f"Primal Solution (x) Relative Error: {x_error:.2e}")
print(f"Inequality Dual (lam) Relative Error: {lam_error:.2e}")
print(f"Equality Dual (mu) Relative Error: {mu_error:.2e}")
print(f"Derivative (dx) Relative Error: {dx_error:.2e}")