import jax.numpy as jnp
import jax
from jax import jvp, vmap
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

jax.config.update("jax_enable_x64", True)

horizon = 30

A = jnp.array([[1,1],[0,1]])
B = jnp.array([[0],[1]])

xmax = 5
xmin = -5
umax = 0.5
umin = -0.5

cost_state = 1
cost_input = 0.1

nx, nu = B.shape
N = horizon

nz = (N+1)*nx + N*nu

# cost
P = jnp.diag(
    jnp.hstack((
        jnp.ones((N+1)*nx) * cost_state,
        jnp.ones(N*nu) * cost_input
    ))
)

q = jnp.zeros(nz)

# inequality constraints
G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))

h = jnp.hstack((
    jnp.ones((N+1)*nx)*xmax,
    jnp.ones(N*nu)*umax,
    -jnp.ones((N+1)*nx)*xmin,
    -jnp.ones(N*nu)*umin
))

# subdiagonal shift matrix
S = jnp.diag(jnp.ones(N), -1)

# state part
Ax = jnp.kron(jnp.eye(N+1), jnp.eye(nx)) + jnp.kron(S, -A)

# input part
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B)

Aeq = jnp.hstack((Ax, Au))

# parameterized RHS
def beq(x_init):
    return jnp.hstack((
        x_init,
        jnp.zeros(N*nx)
    ))

neq = Aeq.shape[0]
nineq = G.shape[0]

from solver_dense.solver_dense import setup_dense_solver

epsilon = 0.1

x0 = jnp.array([-3.0, -1.0])
dx0 = jnp.array([epsilon, 0.0])

solver = setup_dense_solver(n_var=nz, n_ineq=nineq, n_eq=neq)
solve_mpc = lambda x_init: solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# ── Vmapped JVP: evaluate both +dx0 and -dx0 in one call ────────
jvp_func_base = lambda x0, dx0: jvp(solve_mpc, (x0,), (dx0,))
jvp_func = vmap(jvp_func_base, in_axes=(None, 0))

perturbations = jnp.vstack((dx0, -dx0))          # (2, nx)
sol_batch, dsol_batch = jvp_func(x0, perturbations)

# sol_batch["x"]  has shape (2, nz) — but both rows are the same
# dsol_batch["x"] has shape (2, nz) — one for +dx0, one for -dx0

# ── Extract base solution (same for both perturbation dirs) ──────
x_opt_flat = sol_batch["x"][0]                    # (nz,)
x_opt = x_opt_flat[:(N+1)*nx].reshape(-1, nx).T   # (nx, N+1)
x1_opt, x2_opt = x_opt[0], x_opt[1]

# ── Extract linearized tangents for +dx0 and -dx0 ───────────────
labels = ["+dx0", "-dx0"]
dsol_x = dsol_batch["x"]                          # (2, nz)

# ── Compute true perturbed solutions for comparison ──────────────
sol_plus  = solve_mpc(x0 + dx0)
sol_minus = solve_mpc(x0 - dx0)
perturbed_sols = [sol_plus, sol_minus]
signs = [1.0, -1.0]

# ── Helper: reshape state trajectory ─────────────────────────────
def reshape_traj(z_flat):
    return z_flat[:(N+1)*nx].reshape(-1, nx).T     # (nx, N+1)

# ── Compute errors ───────────────────────────────────────────────
print("=" * 60)
for i, label in enumerate(labels):
    dx_jvp = dsol_x[i] / epsilon                   # normalized tangent
    dx_fd  = (perturbed_sols[i]["x"] - x_opt_flat) / epsilon
    rel_err = jnp.linalg.norm(dx_jvp - dx_fd) / jnp.linalg.norm(dx_fd)
    cos_sim = jnp.dot(dx_jvp, dx_fd) / (
        jnp.linalg.norm(dx_jvp) * jnp.linalg.norm(dx_fd)
    )
    print(f"[{label}]  Relative error: {rel_err:.6e},  "
          f"Cosine similarity: {cos_sim:.10f}")
print("=" * 60)

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (label, sign) in enumerate(zip(labels, signs)):
    ax = axes[i]

    # base trajectory
    ax.plot(x1_opt, x2_opt, 'k-o', markersize=3, label='Nominal')

    # true perturbed trajectory
    traj_true = reshape_traj(perturbed_sols[i]["x"])
    ax.plot(traj_true[0], traj_true[1], 's--', markersize=3,
            label=f'Perturbed ({label})')

    # linearized approximation
    traj_approx = reshape_traj(x_opt_flat + dsol_x[i])
    ax.plot(traj_approx[0], traj_approx[1], '^:', markersize=3,
            label=f'Linear approx ({label})')

    # cosmetics
    dx_jvp = dsol_x[i] / epsilon
    dx_fd  = (perturbed_sols[i]["x"] - x_opt_flat) / epsilon
    rel_err = jnp.linalg.norm(dx_jvp - dx_fd) / jnp.linalg.norm(dx_fd)
    cos_sim = jnp.dot(dx_jvp, dx_fd) / (
        jnp.linalg.norm(dx_jvp) * jnp.linalg.norm(dx_fd)
    )
    ax.set_title(f'Perturbation {label}\n'
                 f'rel err = {rel_err:.2e}, cos sim = {cos_sim:.8f}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Vmapped JVP: MPC trajectories (horizon={N}, '
    f'$\\epsilon$={epsilon})',
    fontsize=13,
)
fig.tight_layout()
plt.show()