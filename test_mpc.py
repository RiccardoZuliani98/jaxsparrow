import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

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

from new_solver import setup_dense_solver

solver = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq)

solve_mpc = lambda x_init: solver(P, q, Aeq, beq(x_init), G, h)

x0 = jnp.array([-3,-1])
dx0 = jnp.array([-0.5,0])

sol = solve_mpc(x0)

x_opt = sol["x"][:(horizon+1)*nx].reshape(-1,nx).T
x1_opt, x2_opt = x_opt[0,:].squeeze(), x_opt[1,:].squeeze()
u_opt = sol["x"][(horizon+1)*nx:]

sol = solve_mpc(x0+dx0)

x_opt = sol["x"][:(horizon+1)*nx].reshape(-1,nx).T
x1_opt_perturbed, x2_opt_perturbed = x_opt[0,:].squeeze(), x_opt[1,:].squeeze()
u_opt = sol["x"][(horizon+1)*nx:]

import matplotlib.pyplot as plt
plt.plot(x1_opt,x2_opt,label='Original')
plt.plot(x1_opt_perturbed,x2_opt_perturbed,label='Perturbed')
plt.legend()
plt.show()