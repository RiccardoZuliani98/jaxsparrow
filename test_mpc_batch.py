import jax.numpy as jnp
import jax
from jax import jit, vmap, jvp
jax.config.update("jax_enable_x64", True)

EPSILON = 1.0

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

x0 = jnp.array([-3.0,-1.0])
dx0 = jnp.array([EPSILON,0])

jvp(solve_mpc,(x0,),(dx0,))

dx0 = jnp.array([[EPSILON,0]])

jvp_func_base = lambda x0, dx0: jvp(solve_mpc,(x0,),(dx0,))
jvp_func = vmap(jvp_func_base, in_axes=(None,0))

sol, dsol = jvp_func(x0,jnp.vstack((dx0,dx0)))


# x_opt = sol["x"][:(horizon+1)*nx].reshape(-1,nx).T
# x1_opt, x2_opt = x_opt[0,:].squeeze(), x_opt[1,:].squeeze()
# u_opt = sol["x"][(horizon+1)*nx:]

# x_opt_approx = x_opt + dsol["x"][:(horizon+1)*nx].reshape(-1,nx).T
# x1_opt_approx, x2_opt_approx = x_opt_approx[0,:].squeeze(), x_opt_approx[1,:].squeeze()

# sol_perturbed = solve_mpc(x0+dx0)

# x_opt_perturbed = sol_perturbed["x"][:(horizon+1)*nx].reshape(-1,nx).T
# x1_opt_perturbed, x2_opt_perturbed = x_opt_perturbed[0,:].squeeze(), x_opt_perturbed[1,:].squeeze()
# u_opt = sol_perturbed["x"][(horizon+1)*nx:]

# dx = dsol["x"] / EPSILON
# dx_fd = (sol_perturbed["x"]-sol["x"]) / EPSILON

# error = jnp.linalg.norm(dx-dx_fd) / jnp.linalg.norm(dx_fd)
# cos_sim = jnp.dot(dx,dx_fd) / (jnp.linalg.norm(dx_fd) * jnp.linalg.norm(dx))

# print(f"Relative error {error}, cosine similarity: {cos_sim}")

# import matplotlib.pyplot as plt
# plt.plot(x1_opt,x2_opt,label='Original')
# plt.plot(x1_opt_perturbed,x2_opt_perturbed,label='Perturbed')
# plt.plot(x1_opt_approx,x2_opt_approx,label='Approx')
# plt.legend()
# plt.show()