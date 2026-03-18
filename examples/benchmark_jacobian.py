"""
Benchmark: MPC solve + Jacobian via vmap(jvp) with random initial conditions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit, jvp, vmap
from time import perf_counter

jax.config.update("jax_enable_x64", True)

# ─── Problem setup ───────────────────────────────────────────────────

horizon = 50
N_SAMPLES = 50

A = jnp.array([[1, 1], [0, 1]])
B = jnp.array([[0], [1]])

xmax, xmin = 5, -5
umax, umin = 0.5, -0.5
cost_state, cost_input = 1.0, 0.1

nx, nu = B.shape
N = horizon
nz = (N + 1) * nx + N * nu

P = jnp.diag(jnp.hstack((
    jnp.ones((N + 1) * nx) * cost_state,
    jnp.ones(N * nu) * cost_input,
)))
q = jnp.zeros(nz)

G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
h = jnp.hstack((
    jnp.ones((N + 1) * nx) * xmax,
    jnp.ones(N * nu) * umax,
    -jnp.ones((N + 1) * nx) * xmin,
    -jnp.ones(N * nu) * umin,
))

S = jnp.diag(jnp.ones(N), -1)
Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A)
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B)
Aeq = jnp.hstack((Ax, Au))

neq = Aeq.shape[0]
nineq = G.shape[0]

def beq(x_init):
    return jnp.hstack((x_init, jnp.zeros(N * nx)))

# ─── Solver setup ────────────────────────────────────────────────────

from src.solver_dense.solver_dense import setup_dense_solver

solver = setup_dense_solver(n_var=nz, n_ineq=nineq, n_eq=neq)

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# Jacobian via vmap(jvp) with identity tangent matrix
I_nx = jnp.eye(nx)

def jvp_single(x0, dx0):
    return jvp(solve_mpc, (x0,), (dx0,))

jvp_jacobian = jit(vmap(jvp_single, in_axes=(None, 0)))

# ─── Warmup (JIT compilation) ────────────────────────────────────────

x0_warmup = jnp.array([-1.0, -1.0])
_ = jvp_jacobian(x0_warmup, I_nx)

# ─── Benchmark ───────────────────────────────────────────────────────

rng = np.random.default_rng(42)
x0_samples = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, nx))

# solver.timings.reset()

wall_start = perf_counter()

for i in range(N_SAMPLES):
    x0_i = jnp.array(x0_samples[i])
    sol, dsol = jvp_jacobian(x0_i, I_nx)

wall_elapsed = perf_counter() - wall_start

# ─── Results ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  MPC Jacobian benchmark")
print(f"  horizon = {horizon}, n_var = {nz}, n_eq = {neq}, n_ineq = {nineq}")
print(f"  {N_SAMPLES} random initial conditions, Jacobian via vmap(jvp)")
print(f"{'=' * 60}")
print(f"\n  Wall-clock total : {wall_elapsed:.4f} s")
print(f"  Wall-clock / call: {wall_elapsed / N_SAMPLES * 1e3:.3f} ms\n")
print(solver.timings.summary())