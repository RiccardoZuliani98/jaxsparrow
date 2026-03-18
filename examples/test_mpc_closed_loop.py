import jax.numpy as jnp
import jax
from jax import jvp, jit
jax.config.update("jax_enable_x64", True)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Code
sys.path.insert(0, str(PROJECT_ROOT))

EPSILON = 0.1
CL_HORIZON = 10

horizon = 30

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

from src.solver_dense.setup import setup_dense_solver

solver = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq)

solve_mpc = lambda x_init: solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)



x0 = jnp.array([-3.0,-1.0])
dx0 = jnp.array([EPSILON,0])

@jit
def closed_loop(x0):

    x_cl = [x0]
    u_cl = []

    for t in range(0,CL_HORIZON):

        u_cl.append(solve_mpc(x_cl[-1])["x"][(horizon+1)*nx:(horizon+1)*nx+nu])
        x_cl.append(A@x_cl[-1]+B@u_cl[-1])

    return jnp.hstack(x_cl), jnp.hstack(u_cl)

@jit
def cost_fun(x_cl,u_cl):
    return jnp.dot(x_cl,x_cl)*cost_state +  jnp.dot(u_cl,u_cl)*cost_input

@jit
def cl_cost(x0):
    x_cl, u_cl = closed_loop(x0)
    return cost_fun(x_cl,u_cl)

cost_perturbed = cl_cost(x0)

cost, dcost = jvp(cl_cost,(x0,),(dx0,))
cost_perturbed = cl_cost(x0+dx0)
cost_approx = cost+dcost

print(f"nominal: {cost}, perturbed: {cost_perturbed}, approx: {cost_approx}")

# now vmap
from jax import vmap
from time import perf_counter


def jvp_func_base(x0,dx0):
    return jvp(cl_cost,(x0,),(dx0,))

jvp_func = jit(vmap(jvp_func_base, in_axes=(None,0)))

e_mat = jnp.array([[1.0,0.0],[0.0,1.0]])
d_cl = jvp_func(x0,jnp.array([[0.3,-0.3]]))
start = perf_counter()
d_cl = jvp_func(x0,e_mat)
elapsed = perf_counter()
print(f"Elapsed: {elapsed - start}")