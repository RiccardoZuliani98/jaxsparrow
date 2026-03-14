import jax
import jax.numpy as jnp
import numpy as np
from qpsolvers import Problem, solve_problem

jax.config.update("jax_enable_x64", True)

# ── paste your solver stack here ──────────────────────────────────────────────

def _qp_solve(P, q, G, h, A, b):
    problem = Problem(P, q, G, h, A, b)
    sol = solve_problem(problem, solver="piqp")
    return (
        sol.x.astype(np.float64),
        sol.z.astype(np.float64),
        sol.y.astype(np.float64),
        np.array([sol.obj], dtype=np.float64),
    )

def _qp_callback(P, q, G, h, A, b):
    n, m, p = P.shape[0], G.shape[0], A.shape[0]
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((p,), jnp.float64),
        jax.ShapeDtypeStruct((1,), jnp.float64),
    )
    return jax.pure_callback(_qp_solve, result_shapes, P, q, G, h, A, b)

@jax.custom_vjp
def solver(P, q, G, h, A, b):
    x, z, y, obj = _qp_callback(P, q, G, h, A, b)
    return x, z, y, obj[0]

def _solver_fwd(P, q, G, h, A, b):
    out = solver(P, q, G, h, A, b)
    x, z, y, _ = out
    return out, (P, A, G, x, z, y)

def _solver_bwd(residuals, g):
    P, A, G, x, z, y = residuals
    dl_x, dl_z, dl_y, _ = g
    tol    = 1e-6
    active = z > tol
    G_a    = G[active]
    z_a    = z[active]
    dl_z_a = dl_z[active]
    n, p, m_a = x.shape[0], y.shape[0], G_a.shape[0]
    KKT = jnp.block([
        [P,   A.T,                 G_a.T               ],
        [A,   jnp.zeros((p, p)),   jnp.zeros((p, m_a)) ],
        [G_a, jnp.zeros((m_a, p)), jnp.zeros((m_a,m_a))],
    ])
    lam    = jnp.linalg.solve(KKT + 1e-9 * jnp.eye(n + p + m_a),
                              jnp.concatenate([dl_x, dl_y, dl_z_a]))
    lam_x  = lam[:n]
    lam_y  = lam[n : n + p]
    lam_za = lam[n + p :]
    dq = -lam_x
    dP = -0.5 * (jnp.outer(lam_x, x) + jnp.outer(x, lam_x))
    db =  lam_y
    dA = -(jnp.outer(lam_y, x) + jnp.outer(y, lam_x))
    dh =  jnp.zeros_like(z).at[active].set(lam_za)
    dG =  jnp.zeros_like(G).at[active].set(
              -(jnp.outer(lam_za, x) + jnp.outer(z_a, lam_x))
          )
    return dP, dq, dG, dh, dA, db

solver.defvjp(_solver_fwd, _solver_bwd)

# ── problem data ──────────────────────────────────────────────────────────────

n = 2                                    # variables

P = jnp.eye(n)                           # ½‖x − p‖² → P = I
G = -jnp.eye(n)                          # −x ≤ 0  (non-negativity)
h = jnp.zeros(n)
A = jnp.ones((1, n))                     # x₁ + x₂ = 1
b = jnp.ones(1)

# ── wrapper: target point p enters via q = −p ─────────────────────────────────

def project(p):
    """Return x*(p) = argmin ½‖x−p‖²  s.t. x∈Δ."""
    q = -p
    x, z, y, obj = solver(P, q, G, h, A, b)
    return x

# ── forward solve ─────────────────────────────────────────────────────────────

p_inside  = jnp.array([0.3, 0.7])   # already on Δ  → projection = p itself
p_outside = jnp.array([1.5, 0.8])   # outside Δ     → clipped

x_in  = project(p_inside)
x_out = project(p_outside)

print("── forward ──────────────────────────────")
print(f"p = {p_inside},  x* = {x_in}")    # → [0.3, 0.7]
print(f"p = {p_outside}, x* = {x_out}")   # → [1.0, 0.0]  (corner)

# ── Jacobian dx*/dp (2×2 matrix) ─────────────────────────────────────────────

J_in  = jax.jacobian(project)(p_inside)
J_out = jax.jacobian(project)(p_outside)

print("\n── Jacobian dx*/dp ──────────────────────")
print("Interior point (p on Δ):")
print(J_in)    # identity projected onto the simplex tangent space
print("Exterior point (p outside Δ, x* at corner):")
print(J_out)   # zeros — x* stuck at corner, insensitive to p

# ── scalar loss: differentiate the objective value w.r.t. p ──────────────────

def loss(p):
    """½‖x*(p) − p‖²  — distance from p to its projection."""
    q = -p
    _, _, _, obj = solver(P, q, G, h, A, b)
    return obj

grad_loss = jax.grad(loss)(p_outside)
print("\n── grad of distance² w.r.t. p ──────────")
print(grad_loss)   # points from x* back toward p