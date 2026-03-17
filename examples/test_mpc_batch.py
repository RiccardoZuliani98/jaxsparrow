import jax.numpy as jnp
import jax
from jax import jvp, vmap
from time import perf_counter

import logging
# logging.basicConfig(level=logging.INFO)

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


jax.config.update("jax_enable_x64", True)

horizon = 50

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

P = jnp.diag(
    jnp.hstack((
        jnp.ones((N+1)*nx) * cost_state,
        jnp.ones(N*nu) * cost_input
    ))
)

q = jnp.zeros(nz)

G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))

h = jnp.hstack((
    jnp.ones((N+1)*nx)*xmax,
    jnp.ones(N*nu)*umax,
    -jnp.ones((N+1)*nx)*xmin,
    -jnp.ones(N*nu)*umin
))

S = jnp.diag(jnp.ones(N), -1)
Ax = jnp.kron(jnp.eye(N+1), jnp.eye(nx)) + jnp.kron(S, -A)
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B)
Aeq = jnp.hstack((Ax, Au))

def beq(x_init):
    return jnp.hstack((x_init, jnp.zeros(N*nx)))

neq = Aeq.shape[0]
nineq = G.shape[0]

from src.solver_dense.solver_dense import setup_dense_solver

epsilon = 0.1
x0 = jnp.array([-2.0, -1.0])
dx0 = jnp.array([epsilon, 0.0])

solver = setup_dense_solver(n_var=nz, n_ineq=nineq, n_eq=neq)

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# ── JVP + vmap setup ────────────────────────────────────────────
def jvp_func_base(x0, dx0):
    return jvp(solve_mpc, (x0,), (dx0,))

jvp_func = vmap(jvp_func_base, in_axes=(None, 0))

perturbations = jnp.vstack((dx0, -dx0))  # (2, nx)

# ── Warmup (triggers tracing / compilation) ──────────────────────
solve_mpc(x0)
jvp_func(x0, perturbations)

# ── Timed forward solves ─────────────────────────────────────────
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

# ── Timed vmapped JVP ───────────────────────────────────────────
print("\nVmapped JVP:")
for i in range(n_runs):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    di = jnp.vstack((dx0, -dx0)) * (i + 1)  # different each run
    start = perf_counter()
    sol_batch, dsol_batch = jvp_func(xi, di)
    elapsed = perf_counter() - start
    print(f"  Run {i}: {elapsed:.6f}s")

# ── Verify correctness ──────────────────────────────────────────
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