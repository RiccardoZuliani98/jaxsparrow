import jax
import jax.numpy as jnp
import numpy as np
from qpsolvers import Problem, solve_problem

def _qp_solve(P, q, G, h, A, b):
    problem = Problem(P, q, G, h, A, b)
    sol = solve_problem(problem, solver="piqp")
    return (
        sol.x.astype(np.float64),
        sol.z.astype(np.float64),
        sol.y.astype(np.float64),
        np.array([sol.obj], dtype=np.float64),
    )

def builder(n: int, m: int, p: int):

    # result_shapes built once; reused on every pure_callback call
    _result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),   # x
        jax.ShapeDtypeStruct((m,), jnp.float64),   # z
        jax.ShapeDtypeStruct((p,), jnp.float64),   # y
        jax.ShapeDtypeStruct((1,), jnp.float64),   # obj
    )

    # fixed KKT zero block (p×p); the other zero blocks depend on m_a (active
    # set size) which is dynamic, so they are built cheaply at backward time
    _zeros_pp = jnp.zeros((p, p))

    def _callback(P, q, G, h, A, b):
        return jax.pure_callback(_qp_solve, _result_shapes, P, q, G, h, A, b)

    @jax.custom_vjp
    def _solver(P, q, G, h, A, b):
        x, z, y, obj = _callback(P, q, G, h, A, b)
        return x, z, y, obj[0]

    def _fwd(P, q, G, h, A, b):
        out = _solver(P, q, G, h, A, b)
        x, z, y, _ = out
        return out, (P, A, G, x, z, y)

    def _bwd(residuals, g):
        P, A, G, x, z, y = residuals
        dl_x, dl_z, dl_y, _ = g

        # active-set selection (size m_a unknown until runtime)
        active = z > 1e-6
        G_a    = G[active]
        z_a    = z[active]
        dl_z_a = dl_z[active]
        m_a    = G_a.shape[0]

        # KKT system  (n+p+m_a) × (n+p+m_a)
        # _zeros_pp is pre-built; remaining zero blocks are small & cheap
        KKT = jnp.block([
            [P,   A.T,                      G_a.T                    ],
            [A,   _zeros_pp,                jnp.zeros((p,   m_a))    ],
            [G_a, jnp.zeros((m_a, p)),      jnp.zeros((m_a, m_a))    ],
        ])

        lam    = jnp.linalg.solve(
                     KKT + 1e-9 * jnp.eye(n + p + m_a),
                     jnp.concatenate([dl_x, dl_y, dl_z_a]),
                 )
        lam_x  = lam[:n]
        lam_y  = lam[n     : n + p  ]
        lam_za = lam[n + p :         ]

        dP = -0.5 * (jnp.outer(lam_x, x) + jnp.outer(x, lam_x))
        dq = -lam_x
        dG = jnp.zeros((m, n)).at[active].set(
                 -(jnp.outer(lam_za, x) + jnp.outer(z_a, lam_x))
             )
        dh = jnp.zeros(m).at[active].set(lam_za)
        dA = -(jnp.outer(lam_y, x) + jnp.outer(y, lam_x))
        db =  lam_y

        return dP, dq, dG, dh, dA, db

    _solver.defvjp(_fwd, _bwd)

    return _solver

jax.config.update("jax_enable_x64", True)

# ── problem dimensions ────────────────────────────────────────────────────────

n = 2   # variables
m = 2   # inequality constraints  (−x ≤ 0)
p = 1   # equality   constraints  (x₁ + x₂ = 1)

# ── build solver once for this shape ─────────────────────────────────────────

solve_qp = builder(n, m, p)

# ── fixed problem data (not differentiated through) ──────────────────────────

P = jnp.eye(n)           # ½‖x − p‖²
G = -jnp.eye(n)          # −x ≤ 0
h = jnp.zeros(m)
A = jnp.ones((p, n))     # x₁ + x₂ = 1
b = jnp.ones(p)

# ── wrapper: target point enters via q = −p ───────────────────────────────────

def project(p_target):
    """x*(p) = argmin ½‖x − p‖²  s.t. x ∈ Δ."""
    x, _, _, _ = solve_qp(P, -p_target, G, h, A, b)
    return x

def loss(p_target):
    """½‖x*(p) − p‖² — squared distance from p to its projection."""
    _, _, _, obj = solve_qp(P, -p_target, G, h, A, b)
    return obj

# ── forward ───────────────────────────────────────────────────────────────────

p_inside  = jnp.array([0.3, 0.7])   # already on Δ  → projection = p itself
p_outside = jnp.array([1.5, 0.8])   # outside Δ     → clipped to corner

print("── forward ──────────────────────────────────────────")
print(f"p = {p_inside},   x* = {project(p_inside)}")    # → [0.3, 0.7]
print(f"p = {p_outside},  x* = {project(p_outside)}")   # → [1.0, 0.0]

# ── Jacobian dx*/dp ───────────────────────────────────────────────────────────

J_in  = jax.jacobian(project)(p_inside)
J_out = jax.jacobian(project)(p_outside)

print("\n── Jacobian dx*/dp ──────────────────────────────────")
print("Interior point (p on Δ):")
print(J_in)    # rank-1 projection onto simplex tangent: I − ½11ᵀ
print("Exterior point (p outside Δ, x* at corner):")
print(J_out)   # zeros — x* is stuck at the corner

# ── grad of loss w.r.t. p ────────────────────────────────────────────────────

print("\n── grad of distance² w.r.t. p ──────────────────────")
print(jax.grad(loss)(p_outside))   # points from x* back toward p