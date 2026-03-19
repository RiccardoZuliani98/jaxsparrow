"""Tests for solver_sparse.setup.setup_sparse_solver.

Covers:
    - Inequality-only QP (sparse)
    - Equality-only QP (sparse)
    - Fully constrained QP (equality + inequality, sparse)
    - Fixed elements at setup
    - Output shapes and dtypes
    - KKT optimality conditions
    - Multiple solves with the same solver
    - MPC problem (parametric initial condition)
    - JVP vs finite differences (single and vmap)
    - VJP vs finite differences (single and vmap)
    - VJP ↔ JVP consistency
    - Sparse gradients: verify new nnz-only grad format
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import logging
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix

jax.config.update("jax_enable_x64", True)

from src.solver_sparse.setup import setup_sparse_solver


# =====================================================================
# Helpers — dense→BCOO conversion
# =====================================================================

def _to_bcoo(dense_matrix):
    """Convert a dense JAX array to BCOO."""
    return BCOO.fromdense(dense_matrix)


def _sparsity_dict(**kwargs):
    """Build sparsity_patterns dict from dense matrices."""
    return {k: _to_bcoo(v) for k, v in kwargs.items()}


# Keys that are matrices (need scipy CSC) vs vectors (need numpy array)
_MATRIX_KEYS = frozenset({"P", "A", "G"})


def _to_fixed(elements: dict) -> dict:
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
    return out


# =====================================================================
# Fixtures
# =====================================================================

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

    return dict(
        P=P, q=q, G=G, h=h,
        P_dense=P_dense, G_dense=G_dense,
        n_var=n, n_ineq=n,
        x_expected=x_expected,
        sparsity_patterns=_sparsity_dict(P=P_dense, G=G_dense),
    )


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

    return dict(
        P=P, q=q, A=A, b=b,
        P_dense=P_dense, A_dense=A_dense,
        n_var=2, n_eq=2,
        x_expected=x_expected,
        sparsity_patterns=_sparsity_dict(P=P_dense, A=A_dense),
    )


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

    return dict(
        P=P, q=q, A=A, b=b, G=G, h=h,
        P_dense=P_dense, A_dense=A_dense, G_dense=G_dense,
        n_var=n, n_eq=2, n_ineq=n,
        x_expected=x_expected,
        sparsity_patterns=_sparsity_dict(P=P_dense, A=A_dense, G=G_dense),
    )


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
        sparsity_patterns=_sparsity_dict(P=P_dense, A=Aeq_dense, G=G_dense),
    )


# =====================================================================
# Basic solve tests
# =====================================================================

class TestInequalityOnlyQP:
    """QP with P, q, G, h (no equality constraints) — sparse."""

    def test_solve(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-6)

    def test_output_shapes(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (d["n_ineq"],)
        assert sol["mu"].shape == (0,)

    def test_output_dtypes(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        assert sol["x"].dtype == jnp.float64
        assert sol["lam"].dtype == jnp.float64
        assert sol["mu"].dtype == jnp.float64

    def test_dual_nonnegative(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        lam = np.array(sol["lam"])
        assert np.all(lam >= -1e-8)


class TestEqualityOnlyQP:
    """QP with P, q, A, b (no inequality constraints) — sparse."""

    def test_solve(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_output_shapes(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (0,)
        assert sol["mu"].shape == (d["n_eq"],)

    def test_equality_satisfied(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        residual = d["A_dense"] @ sol["x"] - d["b"]
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)


class TestFullQP:
    """QP with all constraints (equality + inequality) — sparse."""

    def test_solve(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_output_shapes(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (d["n_ineq"],)
        assert sol["mu"].shape == (d["n_eq"],)

    def test_equality_satisfied(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        residual = d["A_dense"] @ sol["x"] - d["b"]
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)

    def test_inequality_satisfied(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        slack = np.array(d["h"] - d["G_dense"] @ sol["x"])
        assert np.all(slack >= -1e-8)


class TestMPC:
    """MPC problem — sparse."""

    def test_solve_feasible(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])

        residual = d["A_dense"] @ sol["x"] - d["b"]
        np.testing.assert_allclose(residual, 0.0, atol=1e-7)

        slack = np.array(d["h"] - d["G_dense"] @ sol["x"])
        assert np.all(slack >= -1e-7)

    def test_initial_condition_embedded(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        x_first = sol["x"][:d["nx"]]
        np.testing.assert_allclose(x_first, d["x0"], atol=1e-8)


# =====================================================================
# KKT optimality checks
# =====================================================================

class TestKKTConditions:
    """Verify that solutions satisfy KKT conditions (sparse)."""

    def test_stationarity_equality(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        grad = d["P_dense"] @ sol["x"] + d["q"] + d["A_dense"].T @ sol["mu"]
        np.testing.assert_allclose(grad, 0.0, atol=1e-7)

    def test_stationarity_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        grad = (d["P_dense"] @ sol["x"] + d["q"]
                + d["A_dense"].T @ sol["mu"]
                + d["G_dense"].T @ sol["lam"])
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_complementary_slackness(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        slack = d["G_dense"] @ sol["x"] - d["h"]
        cs = np.array(sol["lam"] * slack)
        np.testing.assert_allclose(cs, 0.0, atol=1e-6)


# =====================================================================
# Fixed elements
# =====================================================================

class TestFixedElements:
    """Test fixing elements at setup time (sparse)."""

    def test_fix_cost(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"]}),
        )
        sol = solver(A=d["A"], b=d["b"], G=d["G"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fix_constraints(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"A": d["A_dense"], "b": d["b"],
                            "G": d["G_dense"], "h": d["h"]}),
        )
        sol = solver(P=d["P"], q=d["q"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fix_everything(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                            "A": d["A_dense"], "b": d["b"],
                            "G": d["G_dense"], "h": d["h"]}),
        )
        sol = solver()
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fixed_matches_dynamic(self, full_qp):
        d = full_qp

        solver_all_dynamic = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )
        sol_dynamic = solver_all_dynamic(
            P=d["P"], q=d["q"], A=d["A"], b=d["b"], G=d["G"], h=d["h"]
        )

        solver_some_fixed = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"]}),
        )
        sol_fixed = solver_some_fixed(
            A=d["A"], b=d["b"], G=d["G"], h=d["h"]
        )

        np.testing.assert_allclose(sol_dynamic["x"], sol_fixed["x"], atol=1e-12)
        np.testing.assert_allclose(sol_dynamic["mu"], sol_fixed["mu"], atol=1e-10)
        np.testing.assert_allclose(sol_dynamic["lam"], sol_fixed["lam"], atol=1e-10)


# =====================================================================
# Multiple solves with same solver
# =====================================================================

class TestMultipleSolves:
    """Same sparse solver instance, different RHS."""

    def test_different_b(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        sol1 = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        np.testing.assert_allclose(sol1["x"], jnp.array([0.5, 0.5]), atol=1e-8)

        b2 = jnp.array([2.0, 0.0])
        sol2 = solver(P=d["P"], q=d["q"], A=d["A"], b=b2)
        np.testing.assert_allclose(sol2["x"], jnp.array([1.0, 1.0]), atol=1e-8)

    def test_parametric_mpc(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                            "A": d["A_dense"], "G": d["G_dense"], "h": d["h"]}),
        )

        key = jax.random.PRNGKey(0)
        for i in range(5):
            key, subkey = jax.random.split(key)
            x0_i = jax.random.uniform(subkey, shape=(d["nx"],),
                                       minval=-2.0, maxval=2.0)
            b_i = jnp.hstack((x0_i, jnp.zeros(d["n_eq"] - d["nx"])))
            sol = solver(b=b_i)

            residual = d["A_dense"] @ sol["x"] - b_i
            np.testing.assert_allclose(residual, 0.0, atol=1e-7,
                                        err_msg=f"Failed for x0={x0_i}")
            np.testing.assert_allclose(sol["x"][:d["nx"]], x0_i, atol=1e-8)


# =====================================================================
# JVP vs Finite Differences
# =====================================================================

class TestJVPFiniteDifferences:
    """Verify JVP tangents against central finite differences (sparse)."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_DUAL = 1e-3

    @staticmethod
    def _fd_central(solve_fn, param, direction, eps):
        sol_plus = solve_fn(param + eps * direction)
        sol_minus = solve_fn(param - eps * direction)
        return {
            k: (sol_plus[k] - sol_minus[k]) / (2.0 * eps)
            for k in ("x", "lam", "mu")
        }

    # ── d/dq ─────────────────────────────────────────────────────────

    def test_jvp_dq_inequality(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        dq = jnp.array([0.0, 1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_dq_equality(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        dq = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_jvp_dq_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        dq = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    # ── d/db ─────────────────────────────────────────────────────────

    def test_jvp_db_equality(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        db = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_db_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        db = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                          G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    # ── d/dh ─────────────────────────────────────────────────────────

    def test_jvp_dh_inequality(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        dh = jnp.array([1.0, 0.0, 0.0])

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], G=d["G"], h=h_val)

        _, tangents = jax.jvp(solve_h, (d["h"],), (dh,))
        fd = self._fd_central(solve_h, d["h"], dh, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_dh_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        dh = jnp.array([0.1, 0.0])

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=h_val)

        _, tangents = jax.jvp(solve_h, (d["h"],), (dh,))
        fd = self._fd_central(solve_h, d["h"], dh, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    # ── MPC: d/db (parametric initial condition) ─────────────────────

    def test_jvp_db_mpc(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                            "A": d["A_dense"], "G": d["G_dense"], "h": d["h"]}),
        )

        db = jnp.zeros(d["n_eq"])
        db = db.at[0].set(1.0)

        def solve_b(b_val):
            return solver(b=b_val)

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    # ── Fixed elements: JVP only flows through dynamic keys ──────────

    def test_jvp_fixed_P_q_diff_through_b(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"]}),
        )

        db = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(A=d["A"], b=b_val, G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)
        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)


# =====================================================================
# JVP under vmap vs Finite Differences
# =====================================================================

class TestJVPVmapFiniteDifferences:
    """Verify vmapped JVP tangents match per-direction FD (sparse)."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_DUAL = 1e-3
    N_DIRS = 3

    @staticmethod
    def _fd_central(solve_fn, param, direction, eps):
        sol_plus = solve_fn(param + eps * direction)
        sol_minus = solve_fn(param - eps * direction)
        return {
            k: (sol_plus[k] - sol_minus[k]) / (2.0 * eps)
            for k in ("x", "lam", "mu")
        }

    def _fd_batch(self, solve_fn, param, directions, eps):
        results = [self._fd_central(solve_fn, param, d, eps) for d in directions]
        return {
            k: jnp.stack([r[k] for r in results])
            for k in ("x", "lam", "mu")
        }

    def test_vmap_jvp_dq_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        key = jax.random.PRNGKey(40)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
            return tangents

        batched = jax.vmap(jvp_one)(dqs)
        fd = self._fd_batch(solve_q, d["q"], dqs, self.EPS)
        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)

    def test_vmap_jvp_db_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        key = jax.random.PRNGKey(41)
        dbs = jax.random.normal(key, (self.N_DIRS, d["n_eq"]))

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                          G=d["G"], h=d["h"])

        def jvp_one(db):
            _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
            return tangents

        batched = jax.vmap(jvp_one)(dbs)
        fd = self._fd_batch(solve_b, d["b"], dbs, self.EPS)
        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)

    def test_vmap_jvp_dh_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        key = jax.random.PRNGKey(42)
        dhs = jax.random.normal(key, (self.N_DIRS, d["n_ineq"]))

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=h_val)

        def jvp_one(dh):
            _, tangents = jax.jvp(solve_h, (d["h"],), (dh,))
            return tangents

        batched = jax.vmap(jvp_one)(dhs)
        fd = self._fd_batch(solve_h, d["h"], dhs, self.EPS)
        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)

    def test_vmap_jvp_db_mpc(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                            "A": d["A_dense"], "G": d["G_dense"], "h": d["h"]}),
        )

        key = jax.random.PRNGKey(50)
        dbs = jax.random.normal(key, (self.N_DIRS, d["n_eq"]))

        def solve_b(b_val):
            return solver(b=b_val)

        def jvp_one(db):
            _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
            return tangents

        batched = jax.vmap(jvp_one)(dbs)
        fd = self._fd_batch(solve_b, d["b"], dbs, self.EPS)
        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)

    def test_vmap_matches_sequential(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
        )

        key = jax.random.PRNGKey(70)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
            return tangents

        batched = jax.vmap(jvp_one)(dqs)

        seq_xs = []
        for i in range(self.N_DIRS):
            _, t = jax.jvp(solve_q, (d["q"],), (dqs[i],))
            seq_xs.append(t["x"])

        np.testing.assert_allclose(
            batched["x"], jnp.stack(seq_xs), atol=1e-12
        )


# =====================================================================
# VJP vs Finite Differences
# =====================================================================

class TestVJPFiniteDifferences:
    """Verify VJP cotangents against central finite differences (sparse)."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_DUAL = 1e-3

    @staticmethod
    def _fd_grad(solve_fn, param, cotangent_key, cotangent_vec, eps):
        flat_param = param.reshape(-1)
        grad = np.zeros_like(flat_param)
        for j in range(flat_param.size):
            e = np.zeros_like(flat_param)
            e[j] = 1.0
            dp = e.reshape(param.shape)
            sol_plus = solve_fn(param + eps * dp)
            sol_minus = solve_fn(param - eps * dp)
            df = (sol_plus[cotangent_key] - sol_minus[cotangent_key]) / (2.0 * eps)
            grad[j] = jnp.dot(cotangent_vec.reshape(-1), df.reshape(-1))
        return grad.reshape(param.shape)

    @staticmethod
    def _vjp_grad(solve_fn, param, cotangent_key, cotangent_vec):
        _, vjp_fn = jax.vjp(solve_fn, param)
        cotangent = {
            "x": jnp.zeros_like(solve_fn(param)["x"]),
            "lam": jnp.zeros_like(solve_fn(param)["lam"]),
            "mu": jnp.zeros_like(solve_fn(param)["mu"]),
        }
        cotangent[cotangent_key] = cotangent_vec
        (grad,) = vjp_fn(cotangent)
        return grad

    # ── d/dq ─────────────────────────────────────────────────────────

    def test_vjp_dq_inequality(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([0.0, 1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        grad = self._vjp_grad(solve_q, d["q"], "x", g_x)
        fd = self._fd_grad(solve_q, d["q"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_dq_equality(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        grad = self._vjp_grad(solve_q, d["q"], "x", g_x)
        fd = self._fd_grad(solve_q, d["q"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_dq_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        grad = self._vjp_grad(solve_q, d["q"], "x", g_x)
        fd = self._fd_grad(solve_q, d["q"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    # ── d/db ─────────────────────────────────────────────────────────

    def test_vjp_db_equality(self, equality_only):
        d = equality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        grad = self._vjp_grad(solve_b, d["b"], "x", g_x)
        fd = self._fd_grad(solve_b, d["b"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_db_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                          G=d["G"], h=d["h"])

        grad = self._vjp_grad(solve_b, d["b"], "x", g_x)
        fd = self._fd_grad(solve_b, d["b"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    # ── d/dh ─────────────────────────────────────────────────────────

    def test_vjp_dh_inequality(self, inequality_only):
        d = inequality_only
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([0.0, 1.0, 0.0])

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], G=d["G"], h=h_val)

        grad = self._vjp_grad(solve_h, d["h"], "x", g_x)
        fd = self._fd_grad(solve_h, d["h"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    def test_vjp_dh_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([0.1, 0.0])

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=h_val)

        grad = self._vjp_grad(solve_h, d["h"], "x", g_x)
        fd = self._fd_grad(solve_h, d["h"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    # ── MPC: d/db ────────────────────────────────────────────────────

    def test_vjp_db_mpc(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                            "A": d["A_dense"], "G": d["G_dense"], "h": d["h"]}),
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(99)
        g_x = jax.random.normal(key, (d["n_var"],))

        def solve_b(b_val):
            return solver(b=b_val)

        grad = self._vjp_grad(solve_b, d["b"], "x", g_x)
        fd = self._fd_grad(solve_b, d["b"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)

    # ── Multi-parameter: VJP w.r.t. (q, b, h) jointly ───────────────

    def test_vjp_random_cotangent_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(7)
        g_x = jax.random.normal(key, (d["n_var"],))

        def solve_qbh(q_val, b_val, h_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=b_val,
                          G=d["G"], h=h_val)

        sol0 = solve_qbh(d["q"], d["b"], d["h"])
        _, vjp_fn = jax.vjp(solve_qbh, d["q"], d["b"], d["h"])
        cotangent = {
            "x": g_x,
            "lam": jnp.zeros_like(sol0["lam"]),
            "mu": jnp.zeros_like(sol0["mu"]),
        }
        grad_q, grad_b, grad_h = vjp_fn(cotangent)

        fd_q = self._fd_grad(
            lambda q: solve_qbh(q, d["b"], d["h"]),
            d["q"], "x", g_x, self.EPS
        )
        fd_b = self._fd_grad(
            lambda b: solve_qbh(d["q"], b, d["h"]),
            d["b"], "x", g_x, self.EPS
        )
        fd_h = self._fd_grad(
            lambda h: solve_qbh(d["q"], d["b"], h),
            d["h"], "x", g_x, self.EPS
        )

        np.testing.assert_allclose(grad_q, fd_q, atol=self.ATOL_X)
        np.testing.assert_allclose(grad_b, fd_b, atol=self.ATOL_X)
        np.testing.assert_allclose(grad_h, fd_h, atol=self.ATOL_X)

    # ── Fixed elements: VJP only flows through dynamic keys ──────────

    def test_vjp_fixed_P_q_diff_through_b(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"]}),
            options={"differentiator_type": "kkt_rev"},
        )

        g_x = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(A=d["A"], b=b_val, G=d["G"], h=d["h"])

        grad = self._vjp_grad(solve_b, d["b"], "x", g_x)
        fd = self._fd_grad(solve_b, d["b"], "x", g_x, self.EPS)
        np.testing.assert_allclose(grad, fd, atol=self.ATOL_X)


# =====================================================================
# VJP under vmap (batched cotangents) vs Finite Differences
# =====================================================================

class TestVJPVmapFiniteDifferences:
    """Verify vmapped VJP cotangents match per-direction FD (sparse)."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_DUAL = 1e-3
    N_DIRS = 3

    @staticmethod
    def _fd_grad(solve_fn, param, cotangent_key, cotangent_vec, eps):
        flat_param = param.reshape(-1)
        grad = np.zeros_like(flat_param)
        for j in range(flat_param.size):
            e = np.zeros_like(flat_param)
            e[j] = 1.0
            dp = e.reshape(param.shape)
            sol_plus = solve_fn(param + eps * dp)
            sol_minus = solve_fn(param - eps * dp)
            df = (sol_plus[cotangent_key] - sol_minus[cotangent_key]) / (2.0 * eps)
            grad[j] = jnp.dot(cotangent_vec.reshape(-1), df.reshape(-1))
        return grad.reshape(param.shape)

    @staticmethod
    def _fd_batch_vjp(solve_fn, param, cotangent_key, cotangent_vecs, eps):
        results = []
        flat_param = param.reshape(-1)
        for j in range(flat_param.size):
            e = np.zeros_like(flat_param)
            e[j] = 1.0
            dp = e.reshape(param.shape)
            sol_plus = solve_fn(param + eps * dp)
            sol_minus = solve_fn(param - eps * dp)
            df = (sol_plus[cotangent_key] - sol_minus[cotangent_key]) / (2.0 * eps)
            results.append(df)
        # jac shape: (n_param, n_output)
        jac = jnp.stack(results)
        # cotangent_vecs: (n_dirs, n_output), result: (n_dirs, n_param)
        return cotangent_vecs @ jac

    # ── vmap VJP w.r.t. q ────────────────────────────────────────────

    def test_vmap_vjp_dq_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(130)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        def vjp_one(g_x):
            sol, vjp_fn = jax.vjp(solve_q, d["q"])
            cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                         "mu": jnp.zeros(d["n_eq"])}
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp(solve_q, d["q"], "x", g_xs, self.EPS)
        np.testing.assert_allclose(batched, fd, atol=self.ATOL_X)

    # ── vmap VJP w.r.t. b ────────────────────────────────────────────

    def test_vmap_vjp_db_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(131)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                          G=d["G"], h=d["h"])

        def vjp_one(g_x):
            sol, vjp_fn = jax.vjp(solve_b, d["b"])
            cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                         "mu": jnp.zeros(d["n_eq"])}
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp(solve_b, d["b"], "x", g_xs, self.EPS)
        np.testing.assert_allclose(batched, fd, atol=self.ATOL_X)

    # ── vmap VJP w.r.t. h ────────────────────────────────────────────

    def test_vmap_vjp_dh_full(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(132)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=h_val)

        def vjp_one(g_x):
            sol, vjp_fn = jax.vjp(solve_h, d["h"])
            cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                         "mu": jnp.zeros(d["n_eq"])}
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp(solve_h, d["h"], "x", g_xs, self.EPS)
        np.testing.assert_allclose(batched, fd, atol=self.ATOL_X)

    # ── MPC: vmap VJP w.r.t. b ───────────────────────────────────────

    def test_vmap_vjp_db_mpc(self, mpc_problem):
        d = mpc_problem
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            fixed_elements=_to_fixed({"P": d["P_dense"], "q": d["q"],
                            "A": d["A_dense"], "G": d["G_dense"], "h": d["h"]}),
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(140)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_b(b_val):
            return solver(b=b_val)

        def vjp_one(g_x):
            sol, vjp_fn = jax.vjp(solve_b, d["b"])
            cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                         "mu": jnp.zeros(d["n_eq"])}
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)
        fd = self._fd_batch_vjp(solve_b, d["b"], "x", g_xs, self.EPS)
        np.testing.assert_allclose(batched, fd, atol=self.ATOL_X)

    # ── Consistency: vmap VJP matches sequential VJP ──────────────────

    def test_vmap_vjp_matches_sequential(self, full_qp):
        d = full_qp
        solver = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )

        key = jax.random.PRNGKey(160)
        g_xs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        def vjp_one(g_x):
            sol, vjp_fn = jax.vjp(solve_q, d["q"])
            cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                         "mu": jnp.zeros(d["n_eq"])}
            (grad,) = vjp_fn(cotangent)
            return grad

        batched = jax.vmap(vjp_one)(g_xs)

        seq = []
        for i in range(self.N_DIRS):
            seq.append(vjp_one(g_xs[i]))

        np.testing.assert_allclose(batched, jnp.stack(seq), atol=1e-12)


# =====================================================================
# Cross-check: VJP vs JVP consistency
# =====================================================================

class TestVJPJVPConsistency:
    """g^T @ J @ d should be the same computed via JVP or VJP (sparse)."""

    def test_vjp_jvp_consistency_dq(self, full_qp):
        d = full_qp

        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )
        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_fwd"},
        )

        key = jax.random.PRNGKey(170)
        k1, k2 = jax.random.split(key)
        dq = jax.random.normal(k1, (d["n_var"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_q_rev(q_val):
            return solver_rev(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        def solve_q_fwd(q_val):
            return solver_fwd(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                              G=d["G"], h=d["h"])

        # JVP: J @ dq, then dot with g_x
        _, tangents = jax.jvp(solve_q_fwd, (d["q"],), (dq,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        # VJP: J^T @ g_x, then dot with dq
        _, vjp_fn = jax.vjp(solve_q_rev, d["q"])
        sol0 = solve_q_rev(d["q"])
        cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                     "mu": jnp.zeros(d["n_eq"])}
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, dq)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=1e-8)

    def test_vjp_jvp_consistency_db(self, full_qp):
        d = full_qp

        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )
        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_fwd"},
        )

        key = jax.random.PRNGKey(171)
        k1, k2 = jax.random.split(key)
        db = jax.random.normal(k1, (d["n_eq"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_b_rev(b_val):
            return solver_rev(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                              G=d["G"], h=d["h"])

        def solve_b_fwd(b_val):
            return solver_fwd(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                              G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_b_fwd, (d["b"],), (db,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        _, vjp_fn = jax.vjp(solve_b_rev, d["b"])
        sol0 = solve_b_rev(d["b"])
        cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                     "mu": jnp.zeros(d["n_eq"])}
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, db)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=1e-8)

    def test_vjp_jvp_consistency_dh(self, full_qp):
        d = full_qp

        solver_rev = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_rev"},
        )
        solver_fwd = setup_sparse_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            sparsity_patterns=d["sparsity_patterns"],
            options={"differentiator_type": "kkt_fwd"},
        )

        key = jax.random.PRNGKey(172)
        k1, k2 = jax.random.split(key)
        dh = jax.random.normal(k1, (d["n_ineq"],))
        g_x = jax.random.normal(k2, (d["n_var"],))

        def solve_h_rev(h_val):
            return solver_rev(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                              G=d["G"], h=h_val)

        def solve_h_fwd(h_val):
            return solver_fwd(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                              G=d["G"], h=h_val)

        _, tangents = jax.jvp(solve_h_fwd, (d["h"],), (dh,))
        jvp_val = jnp.dot(g_x, tangents["x"])

        _, vjp_fn = jax.vjp(solve_h_rev, d["h"])
        sol0 = solve_h_rev(d["h"])
        cotangent = {"x": g_x, "lam": jnp.zeros(d["n_ineq"]),
                     "mu": jnp.zeros(d["n_eq"])}
        (grad,) = vjp_fn(cotangent)
        vjp_val = jnp.dot(grad, dh)

        np.testing.assert_allclose(jvp_val, vjp_val, atol=1e-8)