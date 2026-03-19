from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Code
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
import numpy as np

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

epsilon = 0.1

x0 = jnp.array([-3.0,-1.0])
dx0 = jnp.array([epsilon,0])

solver = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq)

from jax import jit, jvp
from time import perf_counter

def solve_mpc_base(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

solve_mpc = jit(solve_mpc_base)
sol1 = solve_mpc(x0)
sol2 = solve_mpc(x0+dx0)

def jvp_function_base(x0, dx0):
    return jvp(solve_mpc,(x0,),(dx0,))

jvp_func = jit(jvp_function_base)
sol, dsol = jvp_func(x0,100*dx0)

start = perf_counter()
sol, dsol = jvp_func(x0,dx0)
print(f"Elapsed: {perf_counter()-start}")

## SECOND SOLVER: some stuff fixed at setup
solver_fixed = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq,fixed_elements={"P":np.array(P),"q":np.array(q)})
sol1_fixed = solver_fixed(P=P, q=q, A=Aeq, b=beq(x0), G=G, h=h)
sol2_fixed = solver_fixed(A=Aeq, b=beq(x0), G=G, h=h)
def solver2(x_init):
    return solver_fixed(A=Aeq, b=beq(x_init), G=G, h=h)

start = perf_counter()
sol, dsol = jvp_func(x0,dx0)
print(f"Elapsed: {perf_counter()-start}")

x_opt = sol["x"][:(horizon+1)*nx].reshape(-1,nx).T
x1_opt, x2_opt = x_opt[0,:].squeeze(), x_opt[1,:].squeeze()
u_opt = sol["x"][(horizon+1)*nx:]

x_opt_approx = x_opt + dsol["x"][:(horizon+1)*nx].reshape(-1,nx).T
x1_opt_approx, x2_opt_approx = x_opt_approx[0,:].squeeze(), x_opt_approx[1,:].squeeze()

start = perf_counter()
sol_perturbed = solve_mpc(x0+dx0)
print(f"Elapsed: {perf_counter()-start}")

x_opt_perturbed = sol_perturbed["x"][:(horizon+1)*nx].reshape(-1,nx).T
x1_opt_perturbed, x2_opt_perturbed = x_opt_perturbed[0,:].squeeze(), x_opt_perturbed[1,:].squeeze()
u_opt = sol_perturbed["x"][(horizon+1)*nx:]

dx = dsol["x"] / epsilon
dx_fd = (sol_perturbed["x"]-sol["x"]) / epsilon

error = jnp.linalg.norm(dx-dx_fd) / jnp.linalg.norm(dx_fd)
cos_sim = jnp.dot(dx,dx_fd) / (jnp.linalg.norm(dx_fd) * jnp.linalg.norm(dx))

print(f"Relative error {error}, cosine similarity: {cos_sim}")

import matplotlib.pyplot as plt
plt.plot(x1_opt,x2_opt,label='Original')
plt.plot(x1_opt_perturbed,x2_opt_perturbed,label='Perturbed')
plt.plot(x1_opt_approx,x2_opt_approx,label='Approx')
plt.legend()
# plt.show()


# # now we compute full jacobian
# from jax import jacfwd
# dmpc = jacfwd(solve_mpc)
# dmpc(x0)

# start = perf_counter()
# dmpc(x0)
# print(f"Elapsed: {perf_counter()-start}")