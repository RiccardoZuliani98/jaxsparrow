import jax.numpy as jnp
import jax
from jax import vjp, jvp, jit, vmap
jax.config.update("jax_enable_x64", True)
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.solver_dense.solver_dense import setup_dense_solver

EPSILON = 0.1
CL_HORIZON = 100
N_RUNS = 10

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

solver = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq,options={"differentiator_type":"kkt_rev"})
solver_jvp = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq,options={"differentiator_type":"kkt_fwd"})

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

def solve_mpc_jvp(x_init):
    return solver_jvp(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)["x"][0]

from jax import vjp

_ , vjp_func = vjp(solve_mpc,jnp.array([-1.0,-1.0]))
vjp_func(1.0)

raise Exception



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
def closed_loop_jvp(x0):

    x_cl = [x0]
    u_cl = []

    for t in range(0,CL_HORIZON):

        u_cl.append(solve_mpc_jvp(x_cl[-1])["x"][(horizon+1)*nx:(horizon+1)*nx+nu])
        x_cl.append(A@x_cl[-1]+B@u_cl[-1])

    return jnp.hstack(x_cl), jnp.hstack(u_cl)

@jit
def cost_fun(x_cl,u_cl):
    return jnp.dot(x_cl,x_cl)*cost_state +  jnp.dot(u_cl,u_cl)*cost_input

@jit
def cl_cost(x0):
    x_cl, u_cl = closed_loop(x0)
    return cost_fun(x_cl,u_cl)

@jit
def cl_cost_jvp(x0):
    x_cl, u_cl = closed_loop_jvp(x0)
    return cost_fun(x_cl,u_cl)

@jit
def solve_and_differentiate(x0):
    cost, vjp_func = vjp(cl_cost,x0)
    jac = vjp_func(1.0)
    return cost, jac

def jvp_func_base(x0,dx0):
    return jvp(cl_cost_jvp,(x0,),(dx0,))

jvp_func = jit(vmap(jvp_func_base, in_axes=(None,0)))

solve_and_differentiate(0.9*x0)

elapsed_vjp, elapsed_jvp = [], []

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, N_RUNS)

solver.timings.reset()
solver_jvp.timings.reset()

for i in range(N_RUNS):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    start = perf_counter()
    solve_and_differentiate(xi)
    elapsed = perf_counter()
    elapsed_vjp.append(elapsed - start)

e_mat = jnp.eye(x0.shape[0],dtype=x0.dtype)

for i in range(N_RUNS):
    xi = jax.random.uniform(keys[i], shape=(2,), minval=-2.0, maxval=2.0)
    start = perf_counter()
    jvp_func(xi,e_mat)
    elapsed = perf_counter()
    elapsed_jvp.append(elapsed - start)

print(f"VJP: {jnp.sum(jnp.array(elapsed_vjp))}, JVP: {jnp.sum(jnp.array(elapsed_jvp))}")

print(solver.timings.summary())
print(solver_jvp.timings.summary())