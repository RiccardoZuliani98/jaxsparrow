"""
Benchmark: MPC solve + Jacobian via vmap(vjp) with random initial conditions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit, vjp, vmap
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

from jaxsparrow import setup_dense_solver

solver = setup_dense_solver(
    n_var=nz,
    n_ineq=nineq,
    n_eq=neq,
    options={"differentiator_type": "kkt_rev"},
)

def get_x(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)["x"]

I = jnp.eye(nz)

def get_jacobian_vjp(x0):
    """Compute Jacobian using VJP with identity cotangents."""

    sol, vjp_func = vjp(get_x, x0)
    jac = vmap(vjp_func)(I)  # tuple of length 1, each entry (nz, nx)

    return sol, jac[0]  # unpack the single-arg tuple

jacobian_fn = jit(get_jacobian_vjp)

# ─── Warmup (JIT compilation) ────────────────────────────────────────

x0_warmup = jnp.array([-1.0, -1.0])
_ = jacobian_fn(x0_warmup)

# ─── Benchmark ───────────────────────────────────────────────────────

rng = np.random.default_rng(42)
x0_samples = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, nx))

solver.timings.reset()

wall_elapsed = []

x0_i = jnp.array(x0_samples[0])

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     sol, jac = jacobian_fn(x0_i)
#     sol.block_until_ready()

for i in range(N_SAMPLES):
    x0_i = jnp.array(x0_samples[i])
    start = perf_counter()
    sol, jac = jacobian_fn(x0_i)
    wall_elapsed.append(perf_counter() - start)


# ─── Results ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  MPC Jacobian benchmark (VJP)")
print(f"  horizon = {horizon}, n_var = {nz}, n_eq = {neq}, n_ineq = {nineq}")
print(f"  {N_SAMPLES} random initial conditions, Jacobian via vmap(vjp)")
print(f"{'=' * 60}")
print(f"\n  Wall-clock total : {np.sum(wall_elapsed):.4f} s")
print(f"  Wall-clock / call: {np.sum(wall_elapsed) / N_SAMPLES * 1e3:.3f} ms\n")
print(solver.timings.summary())