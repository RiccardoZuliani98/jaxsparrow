"""Tests for the sparse Differentiable-by-Design (DBD) differentiator backend.

Covers:
    - Forward (JVP) differentiation with sparse DBD backend
        - Matches the standard sparse KKT backend on strongly-convex problems
        - Matches finite differences on convex-but-not-strongly-convex problems
    - Backward (VJP) differentiation with sparse DBD backend
        - Matches the standard sparse KKT backend on strongly-convex problems
        - Matches finite differences on convex-but-not-strongly-convex problems
    - vmap over tangent/cotangent directions (JVP and VJP)
    - Cross-check: JVP vs VJP consistency
    - MPC problem with sparse DBD backend

Notes on finite differences with DBD:
    The DBD differentiator may produce different derivatives than the standard
    KKT backend when the problem is not strongly convex. In such cases, we
    validate against finite differences computed with a warmstarted OSQP solver
    to ensure the solver converges to the same local solution under perturbation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix
from typing import cast

jax.config.update("jax_enable_x64", True)

from jaxsparrow._solver_sparse._setup import setup_sparse_solver
from jaxsparrow._solver_sparse._types import SparseIngredientsNP


# =====================================================================
# Helpers — dense→BCOO conversion & fixed-element formatting
# =====================================================================

def _to_bcoo(dense_matrix):
    """Convert a dense JAX array to BCOO."""
    return BCOO.fromdense(dense_matrix)


def _sparsity_dict(**kwargs):
    """Build sparsity_patterns dict from dense matrices."""
    return {k: _to_bcoo(v) for k, v in kwargs.items()}


_MATRIX_KEYS = frozenset({"P", "A", "G"})


def _to_fixed(elements: dict) -> SparseIngredientsNP:
    """Convert a dict of JAX arrays to the format expected by fixed_elements.

    Sparse matrix keys (P, A, G) → scipy.sparse.csc_matrix
    Vector keys (q, b, h)        → numpy ndarray
    """
    out = {}
    for k, v in elements.items():
        if k in _MATRIX_KEYS:
            out[k] = csc_matrix(np.asarray(v))
        else:
            out[k] = np.asarray(v)
    return cast(SparseIngredientsNP, out)


# =====================================================================
# Helpers — finite differences & cosine similarity
# =====================================================================

def _fd_central_warmstarted(solve_fn, param, direction, eps, warmstart):
    """Central FD: (f(p+εd) - f(p-εd)) / 2ε, using warmstart for both solves."""
    sol_plus = solve_fn(param + eps * direction, warmstart)
    sol_minus = solve_fn(param - eps * direction, warmstart)
    return {
        k: (sol_plus[k] - sol_minus[k]) / (2.0 * eps)
        for k in ("x", "lam", "mu")
    }


def _fd_grad_warmstarted(solve_fn, param, cotangent_key, cotangent_vec, eps,
                         warmstart):
    """Compute g^T @ (df/dp) via central FD, element-by-element, warmstarted."""
    flat_param = param.reshape(-1)
    grad = np.zeros_like(flat_param)
    for j in range(flat_param.size):
        e = np.zeros_like(flat_param)
        e[j] = 1.0
        dp = e.reshape(param.shape)
        sol_plus = solve_fn(param + eps * dp, warmstart)
        sol_minus = solve_fn(param - eps * dp, warmstart)
        df = (sol_plus[cotangent_key] - sol_minus[cotangent_key]) / (2.0 * eps)
        grad[j] = jnp.dot(cotangent_vec.reshape(-1), df.reshape(-1))
    return grad.reshape(param.shape)


def _cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _cos_sim_batch(a, b):
    cs_list = [_cos_sim(v1.squeeze(), v2.squeeze())
               for v1, v2 in zip(
                   jnp.hsplit(a, b.shape[0]),
                   jnp.hsplit(b, b.shape[0])
               )]
    return cs_list


# =====================================================================
# Options
# =====================================================================

DBD_FWD_OPTS = {
    "diff_mode": "fwd",
    "differentiator": {
        "backend": "sparse_dbd",
        "rho": 1e-6,
    },
    "solver": {
        "backend": "qpsolvers",
        "solver_name": "osqp",
    },
}

DBD_REV_OPTS = {
    "diff_mode": "rev",
    "differentiator": {
        "backend": "sparse_dbd",
        "rho": 1e-6,
    },
    "solver": {
        "backend": "qpsolvers",
        "solver_name": "osqp",
    },
}

KKT_FWD_OPTS = {
    "diff_mode": "fwd",
    "differentiator": {"backend": "sparse_kkt"},
    "solver": {
        "backend": "qpsolvers",
        "solver_name": "osqp",
    },
}

KKT_REV_OPTS = {
    "diff_mode": "rev",
    "differentiator": {"backend": "sparse_kkt"},
    "solver": {
        "backend": "qpsolvers",
        "solver_name": "osqp",
    },
}

OSQP_OPTS = {
    "solver": {"solver_name": "osqp"},
}


# =====================================================================
# Fixtures — strongly convex problems (P positive definite)
# =====================================================================

@pytest.fixture
def unconstrained_2d():
    """min 0.5 x^T P x + q^T x  →  x* = P^{-1}(-q)."""
    P_dense = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    q = jnp.array([-2.0, -8.0])
    x_expected = jnp.array([1.0, 2.0])
    P = _to_bcoo(P_dense)
    return dict(P=P, q=q, P_dense=P_dense,
                n_var=2, x_expected=x_expected,
                sparsity_patterns=_sparsity_dict(P=P_dense))


@pytest.fixture
def inequality_only():
    """min 0.5 x^T x + q^T x  s.t.  x >= 0 (as Gx <= h with G=-I, h=0)."""
    n = 3
    P_dense = jnp.eye(n)
    q = jnp.array([1.0, -2.0, 1.0])
    G_dense = -jnp.eye(n)
    h = jnp.zeros(n)
    x_expected = jnp.array([0.0, 2.0, 0.0])
    P = _to_bcoo(P_dense)
    G = _to_bcoo(G_dense)
    return dict(P=P, q=q, G=G, h=h,
                P_dense=P_dense, G_dense=G_dense,
                n_var=n, n_ineq=n, x_expected=x_expected,
                sparsity_patterns=_sparsity_dict(P=P_dense, G=G_dense))


@pytest.fixture
def equality_only():
    """min 0.5 x^T x  s.t.  x1+x2=1, x1-x2=0  →  x*=[0.5, 0.5]."""
    P_dense = jnp.eye(2)
    q = jnp.zeros(2)
    A_dense = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    b = jnp.array([1.0, 0.0])
    x_expected = jnp.array([0.5, 0.5])
    P = _to_bcoo(P_dense)
    A = _to_bcoo(A_dense)
    return dict(P=P, q=q, A=A, b=b,
                P_dense=P_dense, A_dense=A_dense,
                n_var=2, n_eq=2, x_expected=x_expected,
                sparsity_patterns=_sparsity_dict(P=P_dense, A=A_dense))


@pytest.fixture
def full_qp():
    """min 0.5 x^T x  s.t.  x1+x2=1, x1-x2=0, x >= 0."""
    n = 2
    P_dense = jnp.eye(n)
    q = jnp.zeros(n)
    A_dense = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    b = jnp.array([1.0, 0.0])
    G_dense = -jnp.eye(n)
    h = jnp.zeros(n)
    x_expected = jnp.array([0.5, 0.5])
    P = _to_bcoo(P_dense)
    A = _to_bcoo(A_dense)
    G = _to_bcoo(G_dense)
    return dict(P=P, q=q, A=A, b=b, G=G, h=h,
                P_dense=P_dense, A_dense=A_dense, G_dense=G_dense,
                n_var=n, n_eq=2, n_ineq=n, x_expected=x_expected,
                sparsity_patterns=_sparsity_dict(P=P_dense, A=A_dense,
                                                 G=G_dense))


@pytest.fixture
def mpc_problem():
    """Small MPC problem with both equality and inequality constraints."""
    N = 3
    A_dyn = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B_dyn = jnp.array([[0.0], [1.0]])
    nx, nu = B_dyn.shape

    nz = (N + 1) * nx + N * nu

    P_dense = jnp.diag(jnp.hstack((
        jnp.ones((N + 1) * nx),
        0.1 * jnp.ones(N * nu),
    )))
    q = jnp.zeros(nz)

    G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
    h = 5.0 * jnp.ones(2 * nz)

    S = jnp.diag(jnp.ones(N), -1)
    Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
    Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
    Au = jnp.kron(Su, -B_dyn)
    Aeq_dense = jnp.hstack((Ax, Au))

    x0 = jnp.array([-2.0, -1.0])
    beq = jnp.hstack((x0, jnp.zeros(N * nx)))

    P = _to_bcoo(P_dense)
    A = _to_bcoo(Aeq_dense)
    G = _to_bcoo(G_dense)

    return dict(
        P=P, q=q, A=A, b=beq, G=G, h=h,
        P_dense=P_dense, A_dense=Aeq_dense, G_dense=G_dense,
        n_var=nz, n_eq=Aeq_dense.shape[0], n_ineq=G_dense.shape[0],
        x0=x0, nx=nx, N=N,
        sparsity_patterns=_sparsity_dict(P=P_dense, A=Aeq_dense,
                                         G=G_dense),
    )


# =====================================================================
# Fixtures — convex but NOT strongly convex (P is PSD, not PD)
# =====================================================================

@pytest.fixture
def psd_unconstrained():
    """P is PSD (rank-deficient): one eigenvalue is zero.

    min 0.5 * x1^2 + q^T x  s.t. nothing
    P = diag(1, 0)  →  x1* = -q1, x2 is free.
    Adding a box constraint to bound x2.
    With q = [-1, 0] and -5 <= x <= 5, x* = [1, 0].
    """
    n = 2
    P_dense = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    q = jnp.array([-1.0, 0.0])
    G_dense = jnp.vstack((jnp.eye(n), -jnp.eye(n)))
    h = 5.0 * jnp.ones(2 * n)
    x_expected = jnp.array([1.0, 0.0])
    P = _to_bcoo(P_dense)
    G = _to_bcoo(G_dense)
    return dict(P=P, q=q, G=G, h=h,
                P_dense=P_dense, G_dense=G_dense,
                n_var=n, n_ineq=2 * n, x_expected=x_expected,
                sparsity_patterns=_sparsity_dict(P=P_dense, G=G_dense))


@pytest.fixture
def psd_equality():
    """P is PSD, with equality constraints pinning down the null space.

    P = [[1, 0], [0, 0]],  q = [-1, 0]
    s.t. x1 + x2 = 2
    Solution: x1=1, x2=1  (equality pins x2).
    """
    n = 2
    P_dense = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    q = jnp.array([-1.0, 0.0])
    A_dense = jnp.array([[1.0, 1.0]])
    b = jnp.array([2.0])
    x_expected = jnp.array([1.0, 1.0])
    P = _to_bcoo(P_dense)
    A = _to_bcoo(A_dense)
    return dict(P=P, q=q, A=A, b=b,
                P_dense=P_dense, A_dense=A_dense,
                n_var=n, n_eq=1, x_expected=x_expected,
                sparsity_patterns=_sparsity_dict(P=P_dense, A=A_dense))


@pytest.fixture
def psd_full():
    """P is PSD with equality + inequality constraints.

    P = [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  q = [-2, 0, 0]
    s.t. x1 + x2 + x3 = 3,  x >= 0
    Solution: x1=2, x2+x3=1 with x2,x3 >= 0.
    The solver should find x = [2, 0.5, 0.5] or similar (depends on
    regularization).  We only check feasibility + FD matching.
    """
    n = 3
    P_dense = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
    q = jnp.array([-2.0, 0.0, 0.0])
    A_dense = jnp.array([[1.0, 1.0, 1.0]])
    b = jnp.array([3.0])
    G_dense = -jnp.eye(n)
    h = jnp.zeros(n)
    P = _to_bcoo(P_dense)
    A = _to_bcoo(A_dense)
    G = _to_bcoo(G_dense)
    return dict(P=P, q=q, A=A, b=b, G=G, h=h,
                P_dense=P_dense, A_dense=A_dense, G_dense=G_dense,
                n_var=n, n_eq=1, n_ineq=n,
                sparsity_patterns=_sparsity_dict(P=P_dense, A=A_dense,
                                                 G=G_dense))


@pytest.fixture
def lp_as_qp():
    """Pure LP (P=0) as a QP — the extreme case of non-strong convexity.

    min  -x1 - x2   s.t. x1 + x2 <= 1, x >= 0
    Solution: any point on x1+x2=1 with x1,x2>=0; e.g. [0.5, 0.5].
    """
    n = 2
    P_dense = jnp.zeros((n, n))
    q = jnp.array([-1.0, -1.0])
    G_dense = jnp.array([
        [1.0, 1.0],   # x1 + x2 <= 1
        [-1.0, 0.0],  # x1 >= 0
        [0.0, -1.0],  # x2 >= 0
    ])
    h = jnp.array([1.0, 0.0, 0.0])
    P = _to_bcoo(P_dense)
    G = _to_bcoo(G_dense)
    return dict(P=P, q=q, G=G, h=h,
                P_dense=P_dense, G_dense=G_dense,
                n_var=n, n_ineq=3,
                sparsity_patterns=_sparsity_dict(P=P_dense, G=G_dense))


@pytest.fixture
def psd_mpc_problem():
    """MPC problem with a PSD (not PD) cost — zero weight on controls.

    This is common in MPC: only state cost, no input regularization.
    The cost matrix has zero eigenvalues in the control block.
    """
    N = 3
    A_dyn = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B_dyn = jnp.array([[0.0], [1.0]])
    nx, nu = B_dyn.shape

    nz = (N + 1) * nx + N * nu

    # Zero weight on controls → P is PSD, not PD
    P_dense = jnp.diag(jnp.hstack((
        jnp.ones((N + 1) * nx),
        jnp.zeros(N * nu),
    )))
    q = jnp.zeros(nz)

    G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
    h = 5.0 * jnp.ones(2 * nz)

    S = jnp.diag(jnp.ones(N), -1)
    Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
    Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
    Au = jnp.kron(Su, -B_dyn)
    Aeq_dense = jnp.hstack((Ax, Au))

    x0 = jnp.array([-2.0, -1.0])
    beq = jnp.hstack((x0, jnp.zeros(N * nx)))

    P = _to_bcoo(P_dense)
    A = _to_bcoo(Aeq_dense)
    G = _to_bcoo(G_dense)

    return dict(
        P=P, q=q, A=A, b=beq, G=G, h=h,
        P_dense=P_dense, A_dense=Aeq_dense, G_dense=G_dense,
        n_var=nz, n_eq=Aeq_dense.shape[0], n_ineq=G_dense.shape[0],
        x0=x0, nx=nx, N=N,
        sparsity_patterns=_sparsity_dict(P=P_dense, A=Aeq_dense,
                                         G=G_dense),
    )


# =====================================================================
# JVP: DBD matches standard KKT on strongly-convex problems
# =====================================================================

class TestDBDJVPMatchesKKT:
    """On strongly-convex problems, sparse DBD JVP should match sparse KKT JVP."""

    ATOL_X = 1e-5
    ATOL_DUAL = 1e-4

    def test_jvp_dq_unconstrained(self, unconstrained_2d):
        d = unconstrained_2d
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        dq = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val)

        def solve_kkt(q_val):
            return solver_kkt(P=d["P"], q=q_val)

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["q"],), (dq,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)

    def test_jvp_dq_equality(self, equality_only):
        d = equality_only
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        dq = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        def solve_kkt(q_val):
            return solver_kkt(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["q"],), (dq,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)
        np.testing.assert_allclose(tang_dbd["mu"], tang_kkt["mu"],
                                   atol=self.ATOL_DUAL)

    def test_jvp_db_equality(self, equality_only):
        d = equality_only
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        db = jnp.array([1.0, 0.0])

        def solve_dbd(b_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        def solve_kkt(b_val):
            return solver_kkt(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        _, tang_dbd = jax.jvp(solve_dbd, (d["b"],), (db,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["b"],), (db,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)
        np.testing.assert_allclose(tang_dbd["mu"], tang_kkt["mu"],
                                   atol=self.ATOL_DUAL)

    def test_jvp_dq_inequality(self, inequality_only):
        d = inequality_only
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        dq = jnp.array([0.0, 1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def solve_kkt(q_val):
            return solver_kkt(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["q"],), (dq,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)
        np.testing.assert_allclose(tang_dbd["lam"], tang_kkt["lam"],
                                   atol=self.ATOL_DUAL)

    def test_jvp_dq_full(self, full_qp):
        d = full_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        dq = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_kkt(q_val):
            return solver_kkt(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["q"],), (dq,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)
        np.testing.assert_allclose(tang_dbd["mu"], tang_kkt["mu"],
                                   atol=self.ATOL_DUAL)
        np.testing.assert_allclose(tang_dbd["lam"], tang_kkt["lam"],
                                   atol=self.ATOL_DUAL)

    def test_jvp_db_full(self, full_qp):
        d = full_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        db = jnp.array([1.0, 0.0])

        def solve_dbd(b_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                              G=d["G"], h=d["h"])

        def solve_kkt(b_val):
            return solver_kkt(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                              G=d["G"], h=d["h"])

        _, tang_dbd = jax.jvp(solve_dbd, (d["b"],), (db,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["b"],), (db,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)
        np.testing.assert_allclose(tang_dbd["mu"], tang_kkt["mu"],
                                   atol=self.ATOL_DUAL)

    def test_jvp_dh_full(self, full_qp):
        d = full_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=KKT_FWD_OPTS)

        dh = jnp.array([0.1, 0.0])

        def solve_dbd(h_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                              G=d["G"], h=h_val)

        def solve_kkt(h_val):
            return solver_kkt(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                              G=d["G"], h=h_val)

        _, tang_dbd = jax.jvp(solve_dbd, (d["h"],), (dh,))
        _, tang_kkt = jax.jvp(solve_kkt, (d["h"],), (dh,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)
        np.testing.assert_allclose(tang_dbd["lam"], tang_kkt["lam"],
                                   atol=self.ATOL_DUAL)

    def test_jvp_db_mpc(self, mpc_problem):
        d = mpc_problem
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_FWD_OPTS)
        solver_kkt = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=KKT_FWD_OPTS)

        db = jnp.zeros(d["n_eq"]).at[0].set(1.0)

        def solve_dbd_b(b_val):
            return solver_dbd(b=b_val)

        def solve_kkt_b(b_val):
            return solver_kkt(b=b_val)

        _, tang_dbd = jax.jvp(solve_dbd_b, (d["b"],), (db,))
        _, tang_kkt = jax.jvp(solve_kkt_b, (d["b"],), (db,))

        np.testing.assert_allclose(tang_dbd["x"], tang_kkt["x"],
                                   atol=self.ATOL_X)


# =====================================================================
# VJP: DBD matches standard KKT on strongly-convex problems
# =====================================================================

class TestDBDVJPMatchesKKT:
    """On strongly-convex problems, sparse DBD VJP should match sparse KKT VJP."""

    ATOL_X = 1e-5
    ATOL_DUAL = 1e-4

    @staticmethod
    def _vjp_grad(solver_opts, d, make_solve_fn, param_name, cotangent_key,
                  cotangent_vec):
        """Generic VJP gradient helper."""
        solve_fn = make_solve_fn(solver_opts, d)
        sol0 = solve_fn(d[param_name])
        _, vjp_fn = jax.vjp(solve_fn, d[param_name])
        cotangent = {
            "x": jnp.zeros_like(sol0["x"]),
            "lam": jnp.zeros_like(sol0["lam"]),
            "mu": jnp.zeros_like(sol0["mu"]),
        }
        cotangent[cotangent_key] = cotangent_vec
        (grad,) = vjp_fn(cotangent)
        return grad

    def test_vjp_dq_unconstrained(self, unconstrained_2d):
        d = unconstrained_2d
        g_x = jnp.array([1.0, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda q_val: solver(P=d["P"], q=q_val)

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "q", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "q", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)

    def test_vjp_dq_equality(self, equality_only):
        d = equality_only
        g_x = jnp.array([1.0, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_eq=d["n_eq"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda q_val: solver(P=d["P"], q=q_val, A=d["A"],
                                        b=d["b"])

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "q", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "q", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)

    def test_vjp_db_equality(self, equality_only):
        d = equality_only
        g_x = jnp.array([1.0, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_eq=d["n_eq"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda b_val: solver(P=d["P"], q=d["q"], A=d["A"],
                                        b=b_val)

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "b", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "b", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)

    def test_vjp_dq_inequality(self, inequality_only):
        d = inequality_only
        g_x = jnp.array([0.0, 1.0, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_ineq=d["n_ineq"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda q_val: solver(P=d["P"], q=q_val, G=d["G"],
                                        h=d["h"])

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "q", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "q", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)

    def test_vjp_dq_full(self, full_qp):
        d = full_qp
        g_x = jnp.array([1.0, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda q_val: solver(P=d["P"], q=q_val, A=d["A"],
                                        b=d["b"], G=d["G"], h=d["h"])

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "q", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "q", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)

    def test_vjp_db_full(self, full_qp):
        d = full_qp
        g_x = jnp.array([1.0, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda b_val: solver(P=d["P"], q=d["q"], A=d["A"],
                                        b=b_val, G=d["G"], h=d["h"])

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "b", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "b", "x", g_x)

        np.testing.assert_allclose(
            _cos_sim(grad_dbd, grad_kkt), 1.0, atol=self.ATOL_X)

    def test_vjp_dh_full(self, full_qp):
        d = full_qp
        g_x = jnp.array([0.1, 0.0])

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
                sparsity_patterns=d["sparsity_patterns"],
                options=opts)
            return lambda h_val: solver(P=d["P"], q=d["q"], A=d["A"],
                                        b=d["b"], G=d["G"], h=h_val)

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "h", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "h", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)

    def test_vjp_db_mpc(self, mpc_problem):
        d = mpc_problem

        key = jax.random.PRNGKey(99)
        g_x = jax.random.normal(key, (d["n_var"],))

        def make_solve(opts, d):
            solver = setup_sparse_solver(
                n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
                sparsity_patterns=d["sparsity_patterns"],
                fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                          "A": d["A_dense"],
                                          "G": d["G_dense"], "h": d["h"]}),
                options=opts)
            return lambda b_val: solver(b=b_val)

        grad_dbd = self._vjp_grad(DBD_REV_OPTS, d, make_solve, "b", "x", g_x)
        grad_kkt = self._vjp_grad(KKT_REV_OPTS, d, make_solve, "b", "x", g_x)

        np.testing.assert_allclose(grad_dbd, grad_kkt, atol=self.ATOL_X)


# =====================================================================
# JVP: DBD matches FD on PSD (non-strongly-convex) problems
# =====================================================================

class TestDBDJVPFiniteDiffPSD:
    """Verify sparse DBD JVP tangents against warmstarted FD on PSD problems."""

    EPS = 1e-4
    ATOL_X = 1e-3

    def test_jvp_dq_psd_unconstrained(self, psd_unconstrained):
        d = psd_unconstrained
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        dq = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, G=d["G"], h=d["h"],
                             warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        fd = _fd_central_warmstarted(solve_fd, d["q"], dq, self.EPS,
                                     sol0["x"])

        np.testing.assert_allclose(tang_dbd["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_dq_psd_equality(self, psd_equality):
        d = psd_equality
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"])

        dq = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        fd = _fd_central_warmstarted(solve_fd, d["q"], dq, self.EPS,
                                     sol0["x"])

        np.testing.assert_allclose(tang_dbd["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_db_psd_equality(self, psd_equality):
        d = psd_equality
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"])

        db = jnp.array([1.0])

        def solve_dbd(b_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        def solve_fd(b_val, ws):
            return solver_fd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                             warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["b"],), (db,))
        fd = _fd_central_warmstarted(solve_fd, d["b"], db, self.EPS,
                                     sol0["x"])

        np.testing.assert_allclose(tang_dbd["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_dq_psd_full(self, psd_full):
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(200)
        dq = jax.random.normal(key, (d["n_var"],))

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             G=d["G"], h=d["h"], warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        fd = _fd_central_warmstarted(solve_fd, d["q"], dq, self.EPS,
                                     sol0["x"])

        np.testing.assert_allclose(
            _cos_sim(tang_dbd["x"], fd["x"]), 1.0, atol=self.ATOL_X)

    def test_jvp_db_psd_full(self, psd_full):
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        db = jnp.array([1.0])

        def solve_dbd(b_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                              G=d["G"], h=d["h"])

        def solve_fd(b_val, ws):
            return solver_fd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                             G=d["G"], h=d["h"], warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["b"],), (db,))
        fd = _fd_central_warmstarted(solve_fd, d["b"], db, self.EPS,
                                     sol0["x"] + tang_dbd["x"])

        np.testing.assert_allclose(tang_dbd["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_dh_psd_full(self, psd_full):
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(201)
        dh = jax.random.normal(key, (d["n_ineq"],))

        def solve_dbd(h_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                              G=d["G"], h=h_val)

        def solve_fd(h_val, ws):
            return solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                             G=d["G"], h=h_val, warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["h"],), (dh,))
        fd = _fd_central_warmstarted(solve_fd, d["h"], dh, self.EPS,
                                     sol0["x"] + tang_dbd["x"])

        np.testing.assert_allclose(
            _cos_sim(tang_dbd["x"], fd["x"]), 1.0, atol=self.ATOL_X)

    def test_jvp_dq_lp(self, lp_as_qp):
        """JVP on a pure LP (P=0) — the hardest case for DBD."""
        d = lp_as_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        dq = jnp.array([0.1, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, G=d["G"], h=d["h"],
                             warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd, (d["q"],), (dq,))
        fd = _fd_central_warmstarted(solve_fd, d["q"], dq, self.EPS,
                                     sol0["x"])

        np.testing.assert_allclose(
            _cos_sim(tang_dbd["x"], fd["x"]), 1.0, atol=self.ATOL_X)

    def test_jvp_db_psd_mpc(self, psd_mpc_problem):
        """JVP on MPC with PSD cost (zero control weight)."""
        d = psd_mpc_problem
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=OSQP_OPTS)

        sol0 = solver_fd(b=d["b"])

        db = jnp.zeros(d["n_eq"]).at[0].set(1.0)

        def solve_dbd_b(b_val):
            return solver_dbd(b=b_val)

        def solve_fd_b(b_val, ws):
            return solver_fd(b=b_val, warmstart=ws)

        _, tang_dbd = jax.jvp(solve_dbd_b, (d["b"],), (db,))
        fd = _fd_central_warmstarted(solve_fd_b, d["b"], db, self.EPS,
                                     sol0["x"])

        np.testing.assert_allclose(tang_dbd["x"], fd["x"], atol=self.ATOL_X)


# =====================================================================
# VJP: DBD matches FD on PSD (non-strongly-convex) problems
# =====================================================================

class TestDBDVJPFiniteDiffPSD:
    """Verify sparse DBD VJP cotangents against warmstarted FD on PSD problems."""

    EPS = 1e-6
    ATOL_X = 1e-3

    def test_vjp_dq_psd_unconstrained(self, psd_unconstrained):
        d = psd_unconstrained
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        g_x = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, G=d["G"], h=d["h"],
                             warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(0),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["q"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_dq_psd_equality(self, psd_equality):
        d = psd_equality
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        g_x = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(0),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["q"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_db_psd_equality(self, psd_equality):
        d = psd_equality
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        g_x = jnp.array([1.0, 0.0])

        def solve_dbd(b_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        def solve_fd(b_val, ws):
            return solver_fd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                             warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["b"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(0),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["b"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_dq_psd_full(self, psd_full):
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(300)
        g_x = jax.random.normal(key, (d["n_var"],))

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             G=d["G"], h=d["h"], warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["q"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(_cos_sim(grad, fd), 1.0, atol=1e-3)

    def test_vjp_db_psd_full(self, psd_full):
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        g_x = jnp.array([1.0, 0.0, 0.0])

        def solve_dbd(b_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                              G=d["G"], h=d["h"])

        def solve_fd(b_val, ws):
            return solver_fd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                             G=d["G"], h=d["h"], warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["b"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["b"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_dh_psd_full(self, psd_full):
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(301)
        g_x = jax.random.normal(key, (d["n_var"],))

        def solve_dbd(h_val):
            return solver_dbd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                              G=d["G"], h=h_val)

        def solve_fd(h_val, ws):
            return solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                             G=d["G"], h=h_val, warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["h"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["h"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(_cos_sim(grad, fd), 1.0, atol=1e-4)

    def test_vjp_dq_lp(self, lp_as_qp):
        """VJP on a pure LP (P=0)."""
        d = lp_as_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        g_x = jnp.array([1.0, 0.0])

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, G=d["G"], h=d["h"],
                             warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd, d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(0),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd, d["q"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(_cos_sim(grad, fd), 1.0, atol=1e-4)

    def test_vjp_db_psd_mpc(self, psd_mpc_problem):
        """VJP on MPC with PSD cost (zero control weight)."""
        d = psd_mpc_problem
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=OSQP_OPTS)

        sol0 = solver_fd(b=d["b"])

        key = jax.random.PRNGKey(302)
        g_x = jax.random.normal(key, (d["n_var"],))

        def solve_dbd_b(b_val):
            return solver_dbd(b=b_val)

        def solve_fd_b(b_val, ws):
            return solver_fd(b=b_val, warmstart=ws)

        _, vjp_fn = jax.vjp(solve_dbd_b, d["b"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)

        fd = _fd_grad_warmstarted(solve_fd_b, d["b"], "x", g_x, self.EPS,
                                  sol0["x"])

        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)


# =====================================================================
# vmap JVP: DBD with batched tangent directions
# =====================================================================

class TestDBDJVPVmap:
    """Verify that vmapped sparse DBD JVP tangents match per-direction FD."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_X_PSD = 1e-3
    N_DIRS = 3

    @staticmethod
    def _fd_batch_warmstarted(solve_fn, param, directions, eps, warmstart):
        results = [
            _fd_central_warmstarted(solve_fn, param, d, eps, warmstart)
            for d in directions
        ]
        return {
            k: jnp.stack([r[k] for r in results])
            for k in ("x", "lam", "mu")
        }

    def test_vmap_jvp_dq_full(self, full_qp):
        """Batched JVP dq on strongly-convex full QP."""
        d = full_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(400)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             G=d["G"], h=d["h"], warmstart=ws)

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_dbd, (d["q"],), (dq,))
            return tangents

        batched = jax.vmap(jvp_one)(dqs)
        fd = self._fd_batch_warmstarted(solve_fd, d["q"], dqs, self.EPS,
                                        sol0["x"])

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)

    def test_vmap_jvp_dq_psd_full(self, psd_full):
        """Batched JVP dq on PSD full QP."""
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(401)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             G=d["G"], h=d["h"], warmstart=ws)

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_dbd, (d["q"],), (dq,))
            return tangents

        batched = jax.vmap(jvp_one)(dqs)
        fd = self._fd_batch_warmstarted(solve_fd, d["q"], dqs, self.EPS,
                                        sol0["x"])

        cs_list = _cos_sim_batch(batched["x"], fd["x"])

        np.testing.assert_allclose(cs_list, 1.0, atol=1e-4)

    def test_vmap_jvp_db_psd_mpc(self, psd_mpc_problem):
        """Batched JVP db on MPC with PSD cost."""
        d = psd_mpc_problem
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_FWD_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=OSQP_OPTS)

        sol0 = solver_fd(b=d["b"])

        key = jax.random.PRNGKey(402)
        dbs = jax.random.normal(key, (self.N_DIRS, d["n_eq"]))

        def solve_dbd_b(b_val):
            return solver_dbd(b=b_val)

        def solve_fd_b(b_val, ws):
            return solver_fd(b=b_val, warmstart=ws)

        def jvp_one(db):
            _, tangents = jax.jvp(solve_dbd_b, (d["b"],), (db,))
            return tangents

        batched = jax.vmap(jvp_one)(dbs)
        fd = self._fd_batch_warmstarted(solve_fd_b, d["b"], dbs, self.EPS,
                                        sol0["x"])

        np.testing.assert_allclose(batched["x"], fd["x"],
                                   atol=self.ATOL_X_PSD)


# =====================================================================
# vmap VJP: DBD with batched cotangent directions
# =====================================================================

class TestDBDVJPVmap:
    """Verify that vmapped sparse DBD VJP cotangents match per-direction FD."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_X_PSD = 1e-3
    N_DIRS = 3

    @staticmethod
    def _fd_batch_vjp_warmstarted(solve_fn, param, cotangent_key, g_xs,
                                  eps, warmstart):
        results = []
        for g_x in g_xs:
            grad = _fd_grad_warmstarted(solve_fn, param, cotangent_key,
                                        g_x, eps, warmstart)
            results.append(grad)
        return jnp.stack(results)

    def test_vmap_vjp_dq_full(self, full_qp):
        """Batched VJP dq on strongly-convex full QP."""
        d = full_qp
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(500)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             G=d["G"], h=d["h"], warmstart=ws)

        def vjp_one(g_x):
            _, vjp_fn = jax.vjp(solve_dbd, d["q"])
            cotangent = {
                "x": g_x,
                "lam": jnp.zeros(d["n_ineq"]),
                "mu": jnp.zeros(d["n_eq"]),
            }
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp_warmstarted(
            solve_fd, d["q"], "x", g_xs, self.EPS, sol0["x"])

        np.testing.assert_allclose(batched, fd, atol=self.ATOL_X)

    def test_vmap_vjp_dq_psd_full(self, psd_full):
        """Batched VJP dq on PSD full QP."""
        d = psd_full
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=OSQP_OPTS)

        sol0 = solver_fd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                         G=d["G"], h=d["h"])

        key = jax.random.PRNGKey(501)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_dbd(q_val):
            return solver_dbd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_fd(q_val, ws):
            return solver_fd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                             G=d["G"], h=d["h"], warmstart=ws)

        def vjp_one(g_x):
            _, vjp_fn = jax.vjp(solve_dbd, d["q"])
            cotangent = {
                "x": g_x,
                "lam": jnp.zeros(d["n_ineq"]),
                "mu": jnp.zeros(d["n_eq"]),
            }
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp_warmstarted(
            solve_fd, d["q"], "x", g_xs, self.EPS, sol0["x"])

        cs_list = _cos_sim_batch(batched, fd)

        np.testing.assert_allclose(cs_list, 1.0, atol=1e-4)

    def test_vmap_vjp_db_psd_mpc(self, psd_mpc_problem):
        """Batched VJP db on MPC with PSD cost."""
        d = psd_mpc_problem
        solver_dbd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_REV_OPTS)
        solver_fd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=OSQP_OPTS)

        sol0 = solver_fd(b=d["b"])

        key = jax.random.PRNGKey(502)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_dbd_b(b_val):
            return solver_dbd(b=b_val)

        def solve_fd_b(b_val, ws):
            return solver_fd(b=b_val, warmstart=ws)

        def vjp_one(g_x):
            _, vjp_fn = jax.vjp(solve_dbd_b, d["b"])
            cotangent = {
                "x": g_x,
                "lam": jnp.zeros(d["n_ineq"]),
                "mu": jnp.zeros(d["n_eq"]),
            }
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp_warmstarted(
            solve_fd_b, d["b"], "x", g_xs, self.EPS, sol0["x"])

        np.testing.assert_allclose(batched, fd, atol=self.ATOL_X_PSD)


# =====================================================================
# Cross-check: DBD JVP vs VJP consistency
# =====================================================================

class TestDBDJVPVJPConsistency:
    """g^T @ J @ d should be the same computed via JVP or VJP with sparse DBD."""

    ATOL = 1e-7

    def test_fwd_rev_consistency_full(self, full_qp):
        d = full_qp

        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)

        key = jax.random.PRNGKey(600)
        k1, k2 = jax.random.split(key)
        dq = jax.random.normal(k1, (d["n_var"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_fwd(q_val):
            return solver_fwd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_rev(q_val):
            return solver_rev(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        # JVP: J @ dq, then dot with g_x
        _, tangents = jax.jvp(solve_fwd, (d["q"],), (dq,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        # VJP: J^T @ g_x, then dot with dq
        _, vjp_fn = jax.vjp(solve_rev, d["q"])
        sol0 = solve_rev(d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, dq)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=self.ATOL)

    def test_fwd_rev_consistency_psd_full(self, psd_full):
        d = psd_full

        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)

        key = jax.random.PRNGKey(601)
        k1, k2 = jax.random.split(key)
        dq = jax.random.normal(k1, (d["n_var"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_fwd(q_val):
            return solver_fwd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_rev(q_val):
            return solver_rev(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_fwd, (d["q"],), (dq,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        _, vjp_fn = jax.vjp(solve_rev, d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, dq)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=self.ATOL)

    def test_fwd_rev_consistency_psd_mpc(self, psd_mpc_problem):
        d = psd_mpc_problem

        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_FWD_OPTS)
        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                                      "A": d["A_dense"], "G": d["G_dense"],
                                      "h": d["h"]}),
            options=DBD_REV_OPTS)

        key = jax.random.PRNGKey(602)
        k1, k2 = jax.random.split(key)
        db = jax.random.normal(k1, (d["n_eq"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_fwd_b(b_val):
            return solver_fwd(b=b_val)

        def solve_rev_b(b_val):
            return solver_rev(b=b_val)

        _, tangents = jax.jvp(solve_fwd_b, (d["b"],), (db,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        _, vjp_fn = jax.vjp(solve_rev_b, d["b"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(d["n_eq"]),
        }
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, db)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=self.ATOL)

    def test_fwd_rev_consistency_lp(self, lp_as_qp):
        """JVP vs VJP consistency on a pure LP."""
        d = lp_as_qp

        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_FWD_OPTS)
        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options=DBD_REV_OPTS)

        key = jax.random.PRNGKey(603)
        k1, k2 = jax.random.split(key)
        dq = jax.random.normal(k1, (d["n_var"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_fwd(q_val):
            return solver_fwd(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def solve_rev(q_val):
            return solver_rev(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_fwd, (d["q"],), (dq,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        _, vjp_fn = jax.vjp(solve_rev, d["q"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros(d["n_ineq"]),
            "mu": jnp.zeros(0),
        }
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, dq)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=self.ATOL)