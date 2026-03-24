from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
from jax import jit, value_and_grad
from jaxsparrow import qp_value

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

nx,nu = B.shape
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

from jaxsparrow import setup_dense_solver

solver = setup_dense_solver(n_var=nz, n_ineq=nineq, n_eq=neq)

@jit
def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

@jit
def value(x_init, sol):
    return qp_value(P=P, q=q, G=G, h=h, A=Aeq, b=beq(x_init), sol=sol)

@jit
def value_and_dv(x_init, sol):
    return value_and_grad(value, argnums=0)(x_init, sol)


# ============================================================================
# Example
# ============================================================================

key = jax.random.PRNGKey(42)
epsilon = 1e-4
x0_list = jnp.vsplit(jnp.array([
    [-1.0, -1.0],
    [-.5, -.5],
    [1.0, 1.0],
    [.5, .5],
    [-.1, .1],
]),5)

def compute_error(x0):
    # solve and get envelope gradient at x0
    sol0 = solve_mpc(x0)
    v0, grad0 = value_and_dv(x0, sol0)

    d = jax.random.normal(key, shape=x0.shape)
    d = d / jnp.linalg.norm(d)

    # directional derivative from envelope theorem
    dv_analytical = jnp.dot(grad0, d)

    # forward: solve at x0 + eps*d
    sol_fwd = solve_mpc(x0 + epsilon * d)
    v_fwd = value(x0 + epsilon * d, sol_fwd)

    # backward: solve at x0 - epsilon*d
    sol_bwd = solve_mpc(x0 - epsilon * d)
    v_bwd = value(x0 - epsilon * d, sol_bwd)

    dv_fwd = (v_fwd - v0) / epsilon
    dv_ctr = (v_fwd - v_bwd) / (2 * epsilon)

    fwd_err = abs(float(dv_fwd - dv_analytical))
    ctr_err = abs(float(dv_ctr - dv_analytical))

    print(f"Forward error: {fwd_err}, backward error: {ctr_err}")


[compute_error(x0.squeeze()) for x0 in x0_list]