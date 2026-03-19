"""
test_mpc_closed_loop_vjp_sparse.py
==================================
Closed-loop MPC comparing VJP vs JVP timing — sparse mode.

Mirrors test_mpc_closed_loop_vjp.py but uses BCOO matrices for P, G, A
and the sparse solver path.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
from jax import vjp, jvp, jit, vmap
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix as sp_csc
from time import perf_counter

jax.config.update("jax_enable_x64", True)

from jaxsparrow import setup_sparse_solver

EPSILON = 0.1
CL_HORIZON = 100
N_RUNS = 10

# ── MPC problem data ────────────────────────────────────────────────
horizon = 30

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

_b_template = jnp.zeros((N+1)*nx)

def beq(x_init):
    return _b_template.at[:nx].set(x_init)


neq = Aeq_dense.shape[0]
nineq = G_dense.shape[0]

# ── Build sparse solvers (VJP + JVP) ────────────────────────────────
sparsity_patterns = {"P": P, "A": Aeq, "G": G}

solver = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
    options={"differentiator_type": "kkt_rev"},
)

solver_jvp = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
    options={"differentiator_type": "kkt_fwd"},
)

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

def solve_mpc_jvp(x_init):
    return solver_jvp(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# ── Closed-loop definitions ─────────────────────────────────────────
x0 = jnp.array([-3.0, -1.0])
dx0 = jnp.array([EPSILON, 0.0])

def closed_loop(x0):
    x_cl = [x0]
    u_cl = []
    for t in range(CL_HORIZON):
        u_cl.append(
            solve_mpc(x_cl[-1])["x"][(horizon + 1) * nx:(horizon + 1) * nx + nu]
        )
        x_cl.append(A_dyn @ x_cl[-1] + B_dyn @ u_cl[-1])
    return jnp.hstack(x_cl), jnp.hstack(u_cl)

@jit
def closed_loop_jvp(x0):
    x_cl = [x0]
    u_cl = []
    for t in range(CL_HORIZON):
        u_cl.append(
            solve_mpc_jvp(x_cl[-1])["x"][(horizon + 1) * nx:(horizon + 1) * nx + nu]
        )
        x_cl.append(A_dyn @ x_cl[-1] + B_dyn @ u_cl[-1])
    return jnp.hstack(x_cl), jnp.hstack(u_cl)

def cost_fun(x_cl, u_cl):
    return jnp.dot(x_cl, x_cl) * cost_state + jnp.dot(u_cl, u_cl) * cost_input

def cl_cost(x0):
    x_cl, u_cl = closed_loop(x0)
    return cost_fun(x_cl, u_cl)

def cl_cost_jvp(x0):
    x_cl, u_cl = closed_loop_jvp(x0)
    return cost_fun(x_cl, u_cl)

# ── VJP path ─────────────────────────────────────────────────────────
@jit
def solve_and_differentiate(x0):
    cost, vjp_func = vjp(cl_cost, x0)
    jac = vjp_func(1.0)
    return cost, jac

# ── JVP path ─────────────────────────────────────────────────────────
def jvp_func_base(x0, dx0):
    return jvp(cl_cost_jvp, (x0,), (dx0,))

jvp_func = jit(vmap(jvp_func_base, in_axes=(None, 0)))

e_mat = jnp.eye(x0.shape[0], dtype=x0.dtype)

# ── Warmup ───────────────────────────────────────────────────────────
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, N_RUNS)

x0 = jax.random.uniform(keys[0], shape=(2,), minval=-2.0, maxval=2.0)
x1 = jax.random.uniform(keys[1], shape=(2,), minval=-2.0, maxval=2.0)
jvp_func(x0, e_mat)
solve_and_differentiate(0.9 * x0)

# ── Timed runs ───────────────────────────────────────────────────────
elapsed_vjp, elapsed_jvp = [], []

solver.timings.reset()
solver_jvp.timings.reset()

for i in range(N_RUNS):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    start = perf_counter()
    solve_and_differentiate(xi)
    elapsed = perf_counter() - start
    elapsed_vjp.append(elapsed)

for i in range(N_RUNS):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    start = perf_counter()
    jvp_func(xi, e_mat)
    elapsed = perf_counter() - start
    elapsed_jvp.append(elapsed)

print(f"VJP: {jnp.sum(jnp.array(elapsed_vjp))}, JVP: {jnp.sum(jnp.array(elapsed_jvp))}")

print(solver.timings.summary())
print(solver_jvp.timings.summary())
