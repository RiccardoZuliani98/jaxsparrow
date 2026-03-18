"""
Example: MPC closed-loop with finite-difference accuracy checking.

Demonstrates how solver.fd_check automatically validates every JVP
and VJP call against central finite differences in pure NumPy.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
from jax import jvp, vjp, jit, vmap
from time import perf_counter

jax.config.update("jax_enable_x64", True)

# ─── Problem setup (small horizon for speed) ────────────────────────

horizon = 5
CL_HORIZON = 3

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


from src.solver_dense.solver_dense import setup_dense_solver

# ─── Create solvers with fd_check enabled ────────────────────────────

solver_fwd = setup_dense_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    options={"differentiator_type": "kkt_fwd", "fd_check": True},
)

solver_rev = setup_dense_solver(
    n_var=nz, n_ineq=nineq, n_eq=neq,
    options={"differentiator_type": "kkt_rev", "fd_check": True},
)

# ─── Define closed-loop cost functions ───────────────────────────────

def _closed_loop(solve_fn, x0):
    x_cl = [x0]
    u_cl = []
    for t in range(CL_HORIZON):
        sol = solve_fn(P=P, q=q, A=Aeq, b=beq(x_cl[-1]), G=G, h=h)
        u_cl.append(sol["x"][(horizon + 1) * nx:(horizon + 1) * nx + nu])
        x_cl.append(A @ x_cl[-1] + B @ u_cl[-1])
    return jnp.hstack(x_cl), jnp.hstack(u_cl)

def cost_fun(x_cl, u_cl):
    return jnp.dot(x_cl, x_cl) * cost_state + jnp.dot(u_cl, u_cl) * cost_input

@jit
def cl_cost_fwd(x0):
    x_cl, u_cl = _closed_loop(solver_fwd, x0)
    return cost_fun(x_cl, u_cl)

@jit
def cl_cost_rev(x0):
    x_cl, u_cl = _closed_loop(solver_rev, x0)
    return cost_fun(x_cl, u_cl)


# ─── Test points ─────────────────────────────────────────────────────

x0 = jnp.array([-3.0, -1.0])
dx0 = jnp.array([0.1, 0.0])
I_nx = jnp.eye(nx)

print(f"Problem: n_var={nz}, n_eq={neq}, n_ineq={nineq}")
print(f"Horizon={horizon}, CL_HORIZON={CL_HORIZON}")
print()

# ─── JVP: single direction ──────────────────────────────────────────

print("=" * 60)
print("  JVP: single tangent direction")
print("=" * 60)

cost, dcost = jvp(cl_cost_fwd, (x0,), (dx0,))
cost_perturbed = cl_cost_fwd(x0 + dx0)
print(f"  Nominal cost     : {cost:.6f}")
print(f"  Perturbed cost   : {cost_perturbed:.6f}")
print(f"  Linear approx    : {float(cost + dcost):.6f}")
print(f"  Approx error     : {abs(cost_perturbed - cost - dcost):.2e}")

# ─── JVP: vmapped (full Jacobian) ───────────────────────────────────

print()
print("=" * 60)
print("  JVP: vmapped over identity tangents")
print("=" * 60)

def jvp_single(x0, dx0):
    return jvp(cl_cost_fwd, (x0,), (dx0,))

jvp_jacobian = jit(vmap(jvp_single, in_axes=(None, 0)))
costs, grad_fwd = jvp_jacobian(x0, I_nx)
print(f"  Gradient (JVP)   : {grad_fwd}")

# ─── VJP: single cotangent ──────────────────────────────────────────

print()
print("=" * 60)
print("  VJP: single cotangent (grad of scalar cost)")
print("=" * 60)

cost_rev, vjp_fn = vjp(cl_cost_rev, x0)
(grad_rev,) = vjp_fn(1.0)
print(f"  Cost             : {cost_rev:.6f}")
print(f"  Gradient (VJP)   : {grad_rev}")

# ─── Cross-check JVP vs VJP ─────────────────────────────────────────

print()
print("=" * 60)
print("  Cross-check: JVP gradient vs VJP gradient")
print("=" * 60)

grad_diff = jnp.linalg.norm(grad_fwd - grad_rev)
print(f"  JVP grad         : {grad_fwd}")
print(f"  VJP grad         : {grad_rev}")
print(f"  ||JVP - VJP||    : {grad_diff:.2e}")

# ─── Multiple random initial conditions ─────────────────────────────

print()
print("=" * 60)
print("  Running over multiple random initial conditions")
print("=" * 60)

key = jax.random.PRNGKey(42)
N_SAMPLES = 5

for i in range(N_SAMPLES):
    key, subkey = jax.random.split(key)
    xi = jax.random.uniform(subkey, shape=(2,), minval=-2.0, maxval=2.0)

    # JVP
    _, g_jvp = jvp_jacobian(xi, I_nx)

    # VJP
    _, vfn = vjp(cl_cost_rev, xi)
    (g_vjp,) = vfn(1.0)

    diff = float(jnp.linalg.norm(g_jvp - g_vjp))
    print(f"  x0={xi}  |  ||JVP-VJP||={diff:.2e}")

# ─── Print FD accuracy reports ───────────────────────────────────────

print()
print("=" * 60)
print("  Finite-difference accuracy report (JVP solver)")
print("=" * 60)
print(solver_fwd.fd_check.summary())

print()
print("=" * 60)
print("  Finite-difference accuracy report (VJP solver)")
print("=" * 60)
print(solver_rev.fd_check.summary())

# ─── Print timing reports ────────────────────────────────────────────

# print()
# print("=" * 60)
# print("  Timing report (JVP solver)")
# print("=" * 60)
# print(solver_fwd.timings.summary())

# print()
# print("=" * 60)
# print("  Timing report (VJP solver)")
# print("=" * 60)
# print(solver_rev.timings.summary())