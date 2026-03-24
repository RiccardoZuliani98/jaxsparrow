"""
Differentiable QP value function via the envelope theorem.

For the standard QP:

    min_x  (1/2) x^T P x + q^T x
    s.t.   G x <= h
           A x  = b

with optimal primal x*, inequality multipliers lam*, and equality multipliers nu*,
the Lagrangian is:

    L(x, lam, nu; P, q, G, h, A, b)
        = (1/2) x^T P x + q^T x
          + lam^T (G x - h)
          + nu^T  (A x - b)

By the envelope theorem, the derivative of the optimal value V* with respect to
any parameter theta is just  dL/d(theta) evaluated at the optimum, treating
(x*, lam*, nu*) as constants.  JAX gives us this for free: we write L in terms
of the parameters, stop-gradient the optimal variables, and let autodiff do the
rest.

Analytical gradients (for verification):

    dV*/dq = x*
    dV*/dP = (1/2) x* x*^T
    dV*/dh = -lam*
    dV*/dG = lam* (x*)^T
    dV*/db = -nu*
    dV*/dA = nu* (x*)^T
"""

from __future__ import annotations
from jaxsparrow._types_common import SolverOutput

import jax
import jax.numpy as jnp


def qp_value(
    P: jnp.ndarray,
    q: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    sol: SolverOutput,
) -> jnp.ndarray:
    """
    Evaluate the QP optimal value as a differentiable function of (P, q, G, h, A, b).

    The primal-dual solution `sol` is treated as a fixed constant (stop-gradiented).
    JAX autodiff through this function yields exactly the envelope-theorem
    sensitivities.

    Parameters
    ----------
    P : (n, n)  quadratic cost
    q : (n,)    linear cost
    G : (m, n)  inequality constraint matrix
    h : (m,)    inequality constraint rhs
    A : (p, n)  equality constraint matrix
    b : (p,)    equality constraint rhs
    sol : SolverOutput  optimal (x*, lam*, nu*) — not differentiated through

    Returns
    -------
    V : scalar  optimal value  (1/2) x*^T P x* + q^T x*
    """
    # ---- stop-gradient the optimal variables --------------------------------
    x   = jax.lax.stop_gradient(sol["x"])
    lam = jax.lax.stop_gradient(sol["lam"])
    mu  = jax.lax.stop_gradient(sol["mu"])

    # ---- Lagrangian evaluated at the optimum --------------------------------
    # objective
    obj = 0.5 * x @ P @ x + q @ x

    # inequality:  lam^T (G x - h)
    ineq = lam @ (G @ x - h)

    # equality:    nu^T (A x - b)
    eq = mu @ (A @ x - b)

    # At the optimum the constraint terms vanish (complementarity + feasibility),
    # so numerically  V ≈ obj.  But we must include them so that autodiff
    # propagates gradients to G, h, A, b correctly.
    return obj + ineq + eq