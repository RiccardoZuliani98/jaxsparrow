"""
test_mpc_batch_sparse.py
========================
Batched JVP via vmap — sparse mode.

Mirrors test_mpc_batch.py but uses BCOO matrices for P, G, A and the
sparse solver path.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
from jax import jvp, vmap, jit
from jax.experimental.sparse import BCOO
from time import perf_counter
import logging

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
x0 = jnp.array([-2.0, -1.0])
dx0 = jnp.array([epsilon, 0.0])

solver = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
)

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# ── JVP + vmap setup ────────────────────────────────────────────────
def jvp_func_base(x0, dx0):
    return jvp(solve_mpc, (x0,), (dx0,))

jvp_func = jit(vmap(jvp_func_base, in_axes=(None, 0)))

perturbations = jnp.vstack((dx0, -dx0))  # (2, nx)

# ── Warmup ───────────────────────────────────────────────────────────
solve_mpc(x0)
jvp_func(x0, perturbations)

# ── Timed forward solves ─────────────────────────────────────────────
print("Forward solves:")
n_runs = 5
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, n_runs)

for i in range(n_runs):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    start = perf_counter()
    sol = solve_mpc(xi)
    elapsed = perf_counter() - start
    print(f"  Run {i}: {elapsed:.6f}s")

# ── Timed vmapped JVP ───────────────────────────────────────────────
print("\nVmapped JVP:")
for i in range(n_runs):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    di = jnp.vstack((dx0, -dx0)) * (i + 1)
    start = perf_counter()
    sol_batch, dsol_batch = jvp_func(xi, di)
    elapsed = perf_counter() - start
    print(f"  Run {i}: {elapsed:.6f}s")

# ── Verify correctness ──────────────────────────────────────────────
print("\nCorrectness check:")
sol_batch, dsol_batch = jvp_func(x0, perturbations)
sol_plus = solve_mpc(x0 + dx0)
sol_minus = solve_mpc(x0 - dx0)

for i, (label, sol_true) in enumerate([("+dx0", sol_plus), ("-dx0", sol_minus)]):
    x_opt = sol_batch["x"][0]
    dx_jvp = dsol_batch["x"][i] / epsilon
    dx_fd = (sol_true["x"] - x_opt) / epsilon
    rel_err = jnp.linalg.norm(dx_jvp - dx_fd) / jnp.linalg.norm(dx_fd)
    cos_sim = jnp.dot(dx_jvp, dx_fd) / (
        jnp.linalg.norm(dx_jvp) * jnp.linalg.norm(dx_fd)
    )
    print(f"  [{label}] rel_err={rel_err:.6e}  cos_sim={cos_sim:.10f}")

# ── Jacobian computation ─────────────────────────────────────────────
v1 = jnp.array([[1.0, 1.0]])
v2 = jnp.array([[1.0, 0.0]])
e_mat = jnp.array([[1.0, 0.0], [0.0, 1.0]])

jac_x0 = jvp_func(x0, v1)
logging.basicConfig(level=logging.INFO)

start = perf_counter()
jac_x0 = jvp_func(x0, v2)
elapsed = perf_counter() - start
print(f"  Jacobian computation: {elapsed:.6f}s")

start = perf_counter()
jac_x0 = jvp_func(x0, e_mat)
elapsed = perf_counter() - start
print(f"  Jacobian computation: {elapsed:.6f}s")
