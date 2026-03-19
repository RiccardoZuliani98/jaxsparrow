"""
test_mpc_sparse.py
==================
Single-shot MPC solve + JVP correctness check — sparse mode.

Mirrors test_mpc.py but uses BCOO matrices for P, G, A and the
sparse solver path.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit, jvp
from jax.experimental.sparse import BCOO
from time import perf_counter

jax.config.update("jax_enable_x64", True)

# ── MPC problem data ────────────────────────────────────────────────
horizon = 50

A_dyn = jnp.array([[1, 1], [0, 1]])
B_dyn = jnp.array([[0], [1]])

xmax, xmin = 5, -5
umax, umin = 0.5, -0.5
cost_state, cost_input = 1, 0.1

nx, nu = B_dyn.shape
N = horizon
nz = (N + 1) * nx + N * nu

# ── Cost (diagonal → sparse) ────────────────────────────────────────
P_dense = jnp.diag(
    jnp.hstack((
        jnp.ones((N + 1) * nx) * cost_state,
        jnp.ones(N * nu) * cost_input,
    ))
)
P = BCOO.fromdense(P_dense)
q = jnp.zeros(nz)

# ── Inequality constraints (box → sparse) ───────────────────────────
G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
G = BCOO.fromdense(G_dense)

h = jnp.hstack((
    jnp.ones((N + 1) * nx) * xmax,
    jnp.ones(N * nu) * umax,
    -jnp.ones((N + 1) * nx) * xmin,
    -jnp.ones(N * nu) * umin,
))

# ── Equality constraints (dynamics → sparse) ────────────────────────
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

# ── Build sparse solver ─────────────────────────────────────────────
from src.solver_sparse.setup import setup_sparse_solver

sparsity_patterns = {"P": P, "A": Aeq, "G": G}

epsilon = 0.1
x0 = jnp.array([-3.0, -1.0])
dx0 = jnp.array([epsilon, 0.0])

solver = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
)

# ── Solve ────────────────────────────────────────────────────────────
def solve_mpc_base(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

solve_mpc = jit(solve_mpc_base)
sol1 = solve_mpc(x0)
sol2 = solve_mpc(x0 + dx0)

# ── JVP ──────────────────────────────────────────────────────────────
def jvp_function_base(x0, dx0):
    return jvp(solve_mpc, (x0,), (dx0,))

jvp_func = jit(jvp_function_base)
sol, dsol = jvp_func(x0, 100 * dx0)  # warmup

start = perf_counter()
sol, dsol = jvp_func(x0, dx0)
print(f"Elapsed: {perf_counter() - start}")

# ── Solver with fixed elements ───────────────────────────────────────
from scipy.sparse import csc_matrix as sp_csc

solver_fixed = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
    fixed_elements={
        "P": sp_csc(np.array(P_dense)),
        "q": np.array(q),
    },
)

sol1_fixed = solver_fixed(P=P, q=q, A=Aeq, b=beq(x0), G=G, h=h)
sol2_fixed = solver_fixed(A=Aeq, b=beq(x0), G=G, h=h)

def solver2(x_init):
    return solver_fixed(A=Aeq, b=beq(x_init), G=G, h=h)

start = perf_counter()
sol, dsol = jvp_func(x0, dx0)
print(f"Elapsed: {perf_counter() - start}")

# ── Extract trajectories ─────────────────────────────────────────────
x_opt = sol["x"][:(horizon + 1) * nx].reshape(-1, nx).T
x1_opt, x2_opt = x_opt[0, :].squeeze(), x_opt[1, :].squeeze()
u_opt = sol["x"][(horizon + 1) * nx:]

x_opt_approx = x_opt + dsol["x"][:(horizon + 1) * nx].reshape(-1, nx).T
x1_opt_approx = x_opt_approx[0, :].squeeze()
x2_opt_approx = x_opt_approx[1, :].squeeze()

# ── Perturbed solve for FD comparison ────────────────────────────────
start = perf_counter()
sol_perturbed = solve_mpc(x0 + dx0)
print(f"Elapsed: {perf_counter() - start}")

x_opt_perturbed = sol_perturbed["x"][:(horizon + 1) * nx].reshape(-1, nx).T
x1_opt_perturbed = x_opt_perturbed[0, :].squeeze()
x2_opt_perturbed = x_opt_perturbed[1, :].squeeze()

dx = dsol["x"] / epsilon
dx_fd = (sol_perturbed["x"] - sol["x"]) / epsilon

error = jnp.linalg.norm(dx - dx_fd) / jnp.linalg.norm(dx_fd)
cos_sim = jnp.dot(dx, dx_fd) / (jnp.linalg.norm(dx_fd) * jnp.linalg.norm(dx))

print(f"Relative error {error}, cosine similarity: {cos_sim}")

import matplotlib.pyplot as plt

plt.plot(x1_opt, x2_opt, label="Original")
plt.plot(x1_opt_perturbed, x2_opt_perturbed, label="Perturbed")
plt.plot(x1_opt_approx, x2_opt_approx, label="Approx")
plt.legend()
# plt.show()
