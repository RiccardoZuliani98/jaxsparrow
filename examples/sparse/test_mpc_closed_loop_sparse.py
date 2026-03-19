"""
test_mpc_closed_loop_sparse.py
==============================
Closed-loop MPC with JVP through the loop — sparse mode.

Mirrors test_mpc_closed_loop.py but uses BCOO matrices for P, G, A
and the sparse solver path.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
from jax import jvp, jit, vmap
from jax.experimental.sparse import BCOO
from time import perf_counter

jax.config.update("jax_enable_x64", True)

EPSILON = 0.1
CL_HORIZON = 10

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

def beq(x_init):
    return jnp.hstack((x_init, jnp.zeros(N * nx)))

neq = Aeq_dense.shape[0]
nineq = G_dense.shape[0]

# ── Build sparse solver ─────────────────────────────────────────────
from src.solver_sparse.setup import setup_sparse_solver

sparsity_patterns = {"P": P, "A": Aeq, "G": G}

solver = setup_sparse_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    sparsity_patterns=sparsity_patterns,
)

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# ── Closed-loop simulation ──────────────────────────────────────────
x0 = jnp.array([-3.0, -1.0])
dx0 = jnp.array([EPSILON, 0.0])

@jit
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
def cost_fun(x_cl, u_cl):
    return jnp.dot(x_cl, x_cl) * cost_state + jnp.dot(u_cl, u_cl) * cost_input

@jit
def cl_cost(x0):
    x_cl, u_cl = closed_loop(x0)
    return cost_fun(x_cl, u_cl)

# ── JVP through closed loop ─────────────────────────────────────────
cost_perturbed = cl_cost(x0)

cost, dcost = jvp(cl_cost, (x0,), (dx0,))
cost_perturbed = cl_cost(x0 + dx0)
cost_approx = cost + dcost

print(f"nominal: {cost}, perturbed: {cost_perturbed}, approx: {cost_approx}")

# ── Vmapped JVP for Jacobian ────────────────────────────────────────
def jvp_func_base(x0, dx0):
    return jvp(cl_cost, (x0,), (dx0,))

jvp_func = jit(vmap(jvp_func_base, in_axes=(None, 0)))

e_mat = jnp.array([[1.0, 0.0], [0.0, 1.0]])
d_cl = jvp_func(x0, jnp.array([[0.3, -0.3]]))

start = perf_counter()
d_cl = jvp_func(x0, e_mat)
elapsed = perf_counter() - start
print(f"Elapsed: {elapsed}")
