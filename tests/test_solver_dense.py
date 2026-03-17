"""Tests for solver_dense.solver_dense.setup_dense_solver.

Covers:
    - Unconstrained QP (P, q only)
    - Inequality-only QP
    - Equality-only QP
    - Fully constrained QP (equality + inequality)
    - Fixed elements at setup
    - Overriding fixed elements at runtime (warning)
    - Missing dynamic keys (ValueError)
    - Output shapes and dtypes
    - KKT optimality conditions
    - Multiple solves with the same solver
    - Known bug: squeeze destroys shape when n_eq=1 or n_ineq=1
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import logging

jax.config.update("jax_enable_x64", True)

from src.solver_dense.solver_dense import setup_dense_solver


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def unconstrained_2d():
    """min 0.5 x^T P x + q^T x  →  x* = P^{-1}(-q)."""
    P = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    q = jnp.array([-2.0, -8.0])
    x_expected = jnp.array([1.0, 2.0])
    return dict(P=P, q=q, n_var=2, x_expected=x_expected)


@pytest.fixture
def inequality_only():
    """min 0.5 x^T x + q^T x  s.t.  x >= 0 (as Gx <= h with G=-I, h=0)."""
    n = 3
    P = jnp.eye(n)
    q = jnp.array([1.0, -2.0, 1.0])
    G = -jnp.eye(n)
    h = jnp.zeros(n)
    # x* = max(0, -q) = [0, 2, 0]
    x_expected = jnp.array([0.0, 2.0, 0.0])
    return dict(P=P, q=q, G=G, h=h, n_var=n, n_ineq=n,
                x_expected=x_expected)


@pytest.fixture
def equality_only():
    """min 0.5 x^T x  s.t.  x1+x2=1, x1-x2=0  →  x*=[0.5, 0.5]."""
    P = jnp.eye(2)
    q = jnp.zeros(2)
    A = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    b = jnp.array([1.0, 0.0])
    x_expected = jnp.array([0.5, 0.5])
    return dict(P=P, q=q, A=A, b=b, n_var=2, n_eq=2,
                x_expected=x_expected)


@pytest.fixture
def full_qp():
    """min 0.5 x^T x  s.t.  x1+x2=1, x >= 0."""
    n = 2
    P = jnp.eye(n)
    q = jnp.zeros(n)
    A = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    b = jnp.array([1.0, 0.0])
    G = -jnp.eye(n)
    h = jnp.zeros(n)
    x_expected = jnp.array([0.5, 0.5])
    return dict(P=P, q=q, A=A, b=b, G=G, h=h,
                n_var=n, n_eq=2, n_ineq=n, x_expected=x_expected)


@pytest.fixture
def mpc_problem():
    """Small MPC problem with both equality and inequality constraints."""
    N = 3
    A_dyn = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B_dyn = jnp.array([[0.0], [1.0]])
    nx, nu = B_dyn.shape

    nz = (N + 1) * nx + N * nu

    P = jnp.diag(jnp.hstack((
        jnp.ones((N + 1) * nx),
        0.1 * jnp.ones(N * nu),
    )))
    q = jnp.zeros(nz)

    G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
    h = 5.0 * jnp.ones(2 * nz)

    S = jnp.diag(jnp.ones(N), -1)
    Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
    Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
    Au = jnp.kron(Su, -B_dyn)
    Aeq = jnp.hstack((Ax, Au))

    x0 = jnp.array([-2.0, -1.0])
    beq = jnp.hstack((x0, jnp.zeros(N * nx)))

    return dict(
        P=P, q=q, A=Aeq, b=beq, G=G, h=h,
        n_var=nz, n_eq=Aeq.shape[0], n_ineq=G.shape[0],
        x0=x0, nx=nx, N=N,
    )


# =====================================================================
# Basic solve tests
# =====================================================================

class TestUnconstrainedQP:
    """QP with only P and q (no constraints)."""

    def test_solve(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])
        sol = solver(P=d["P"], q=d["q"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_output_shapes(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])
        sol = solver(P=d["P"], q=d["q"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (0,)
        assert sol["mu"].shape == (0,)

    def test_output_dtypes(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])
        sol = solver(P=d["P"], q=d["q"])

        assert sol["x"].dtype == jnp.float64
        assert sol["lam"].dtype == jnp.float64
        assert sol["mu"].dtype == jnp.float64


class TestInequalityOnlyQP:
    """QP with P, q, G, h (no equality constraints)."""

    def test_solve(self, inequality_only):
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-6)

    def test_output_shapes(self, inequality_only):
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (d["n_ineq"],)
        assert sol["mu"].shape == (0,)

    def test_dual_nonnegative(self, inequality_only):
        """Inequality duals should be >= 0 for active constraints."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])
        lam = np.array(sol["lam"])
        assert np.all(lam >= -1e-8)


class TestEqualityOnlyQP:
    """QP with P, q, A, b (no inequality constraints)."""

    def test_solve(self, equality_only):
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_output_shapes(self, equality_only):
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (0,)
        assert sol["mu"].shape == (d["n_eq"],)

    def test_equality_satisfied(self, equality_only):
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        residual = d["A"] @ sol["x"] - d["b"]
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)


class TestFullQP:
    """QP with all constraints (equality + inequality)."""

    def test_solve(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_output_shapes(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])

        assert sol["x"].shape == (d["n_var"],)
        assert sol["lam"].shape == (d["n_ineq"],)
        assert sol["mu"].shape == (d["n_eq"],)

    def test_equality_satisfied(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        residual = d["A"] @ sol["x"] - d["b"]
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)

    def test_inequality_satisfied(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        slack = np.array(d["h"] - d["G"] @ sol["x"])
        assert np.all(slack >= -1e-8)


class TestMPC:
    """MPC problem — larger, more realistic."""

    def test_solve_feasible(self, mpc_problem):
        d = mpc_problem
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])

        # Check equality constraints (dynamics)
        residual = d["A"] @ sol["x"] - d["b"]
        np.testing.assert_allclose(residual, 0.0, atol=1e-7)

        # Check inequality constraints
        slack = np.array(d["h"] - d["G"] @ sol["x"])
        assert np.all(slack >= -1e-7)

    def test_initial_condition_embedded(self, mpc_problem):
        """First nx components of x should match x0."""
        d = mpc_problem
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])
        x_first = sol["x"][:d["nx"]]
        np.testing.assert_allclose(x_first, d["x0"], atol=1e-8)


# =====================================================================
# KKT optimality checks
# =====================================================================

class TestKKTConditions:
    """Verify that solutions satisfy KKT conditions."""

    def test_stationarity_unconstrained(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])
        sol = solver(P=d["P"], q=d["q"])

        # Stationarity: Px + q = 0
        grad = d["P"] @ sol["x"] + d["q"]
        np.testing.assert_allclose(grad, 0.0, atol=1e-8)

    def test_stationarity_equality(self, equality_only):
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])

        # Stationarity: Px + q + A^T mu = 0
        grad = d["P"] @ sol["x"] + d["q"] + d["A"].T @ sol["mu"]
        np.testing.assert_allclose(grad, 0.0, atol=1e-7)

    def test_stationarity_full(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"]
        )
        sol = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                     G=d["G"], h=d["h"])

        # Stationarity: Px + q + A^T mu + G^T lam = 0
        grad = (d["P"] @ sol["x"] + d["q"]
                + d["A"].T @ sol["mu"]
                + d["G"].T @ sol["lam"])
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_complementary_slackness(self, inequality_only):
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        # lam_i * (G_i x - h_i) = 0 for all i
        slack = d["G"] @ sol["x"] - d["h"]
        cs = np.array(sol["lam"] * slack)
        np.testing.assert_allclose(cs, 0.0, atol=1e-6)


# =====================================================================
# Fixed elements
# =====================================================================

class TestFixedElements:
    """Test fixing elements at setup time."""

    def test_fix_cost(self, full_qp):
        """Fix P and q, pass constraints at runtime."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"]},
        )
        sol = solver(A=d["A"], b=d["b"], G=d["G"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fix_constraints(self, full_qp):
        """Fix A, b, G, h, pass cost at runtime."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"A": d["A"], "b": d["b"],
                            "G": d["G"], "h": d["h"]},
        )
        sol = solver(P=d["P"], q=d["q"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fix_everything(self, full_qp):
        """Fix all elements — solver takes no runtime args."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"],
                            "A": d["A"], "b": d["b"],
                            "G": d["G"], "h": d["h"]},
        )
        sol = solver()
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fix_partial_mixed(self, full_qp):
        """Fix P and G, pass the rest at runtime."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "G": d["G"]},
        )
        sol = solver(q=d["q"], A=d["A"], b=d["b"], h=d["h"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_fixed_matches_dynamic(self, full_qp):
        """Solutions should be identical whether elements are fixed or not."""
        d = full_qp

        solver_all_dynamic = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )
        sol_dynamic = solver_all_dynamic(
            P=d["P"], q=d["q"], A=d["A"], b=d["b"], G=d["G"], h=d["h"]
        )

        solver_some_fixed = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"]},
        )
        sol_fixed = solver_some_fixed(
            A=d["A"], b=d["b"], G=d["G"], h=d["h"]
        )

        np.testing.assert_allclose(sol_dynamic["x"], sol_fixed["x"], atol=1e-12)
        np.testing.assert_allclose(sol_dynamic["mu"], sol_fixed["mu"], atol=1e-10)
        np.testing.assert_allclose(sol_dynamic["lam"], sol_fixed["lam"], atol=1e-10)


# =====================================================================
# Override fixed elements warning
# =====================================================================

class TestOverrideWarning:
    """Passing a fixed key at runtime should log a warning."""

    def test_override_warns(self, full_qp, caplog):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"]},
        )
        with caplog.at_level(logging.WARNING):
            sol = solver(
                P=d["P"], q=d["q"],  # these are fixed — should warn
                A=d["A"], b=d["b"], G=d["G"], h=d["h"],
            )
        assert "Ignoring runtime values for fixed keys" in caplog.text

    def test_override_still_uses_fixed(self, full_qp):
        """The runtime value should be IGNORED; the fixed value is used."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"]},
        )
        # Pass a different q at runtime — should be ignored
        q_wrong = jnp.ones(d["n_var"]) * 999.0
        sol = solver(
            P=d["P"], q=q_wrong,
            A=d["A"], b=d["b"], G=d["G"], h=d["h"],
        )
        # Solution should match the fixed q, not the runtime q
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)


# =====================================================================
# Missing keys
# =====================================================================

class TestMissingKeys:
    """Missing dynamic keys should raise ValueError."""

    def test_missing_dynamic_key(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )
        with pytest.raises((ValueError, KeyError)):
            # Missing G, h
            solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])

    def test_missing_single_key(self, full_qp):
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )
        with pytest.raises((ValueError, KeyError)):
            # Missing just h
            solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"], G=d["G"])


# =====================================================================
# Multiple solves with same solver
# =====================================================================

class TestMultipleSolves:
    """Same solver instance, different RHS."""

    def test_different_b(self, equality_only):
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        # First solve: x1+x2=1, x1-x2=0 → [0.5, 0.5]
        sol1 = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"])
        np.testing.assert_allclose(sol1["x"], jnp.array([0.5, 0.5]), atol=1e-8)

        # Second solve: x1+x2=2, x1-x2=0 → [1.0, 1.0]
        b2 = jnp.array([2.0, 0.0])
        sol2 = solver(P=d["P"], q=d["q"], A=d["A"], b=b2)
        np.testing.assert_allclose(sol2["x"], jnp.array([1.0, 1.0]), atol=1e-8)

    def test_different_q(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        sol1 = solver(P=d["P"], q=d["q"])
        np.testing.assert_allclose(sol1["x"], d["x_expected"], atol=1e-8)

        # Different q → different solution
        q2 = jnp.array([-4.0, -8.0])
        sol2 = solver(P=d["P"], q=q2)
        np.testing.assert_allclose(sol2["x"], jnp.array([2.0, 2.0]), atol=1e-8)

    def test_parametric_mpc(self, mpc_problem):
        """Multiple initial conditions through the same solver."""
        d = mpc_problem
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"],
                            "A": d["A"], "G": d["G"], "h": d["h"]},
        )

        key = jax.random.PRNGKey(0)
        for i in range(5):
            key, subkey = jax.random.split(key)
            x0_i = jax.random.uniform(subkey, shape=(d["nx"],),
                                       minval=-2.0, maxval=2.0)
            b_i = jnp.hstack((x0_i, jnp.zeros(d["n_eq"] - d["nx"])))
            sol = solver(b=b_i)

            # Check equality constraints
            residual = d["A"] @ sol["x"] - b_i
            np.testing.assert_allclose(residual, 0.0, atol=1e-7,
                                        err_msg=f"Failed for x0={x0_i}")

            # Check initial condition embedded in solution
            np.testing.assert_allclose(sol["x"][:d["nx"]], x0_i, atol=1e-8)


# =====================================================================
# Warmstart
# =====================================================================

class TestWarmstart:
    """Test warmstart functionality."""

    def test_warmstart_same_solution(self, full_qp):
        """Warmstart should not change the solution, only speed it up."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )
        sol_cold = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])
        sol_warm = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"],
                          warmstart=jnp.array([0.4, 0.6]))
        np.testing.assert_allclose(sol_cold["x"], sol_warm["x"], atol=1e-10)

    def test_warmstart_with_previous_solution(self, mpc_problem):
        """Use a previous solution as warmstart for the next call."""
        d = mpc_problem
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )
        sol1 = solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                       G=d["G"], h=d["h"])

        # Slightly perturbed b
        b2 = d["b"].at[0].add(0.01)
        sol2 = solver(P=d["P"], q=d["q"], A=d["A"], b=b2,
                       G=d["G"], h=d["h"],
                       warmstart=sol1["x"])

        # Should still satisfy constraints
        residual = d["A"] @ sol2["x"] - b2
        np.testing.assert_allclose(residual, 0.0, atol=1e-7)

    def test_warmstart_none_default(self, unconstrained_2d):
        """warmstart=None should behave identically to no warmstart."""
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])
        sol1 = solver(P=d["P"], q=d["q"])
        sol2 = solver(P=d["P"], q=d["q"], warmstart=None)
        np.testing.assert_allclose(sol1["x"], sol2["x"], atol=1e-12)

    def test_warmstart_cleared_after_use(self, unconstrained_2d):
        """Warmstart should not persist across calls."""
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        # Call with warmstart
        solver(P=d["P"], q=d["q"], warmstart=jnp.array([0.5, 1.0]))

        # Next call without warmstart should still work
        sol = solver(P=d["P"], q=d["q"])
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_warmstart_with_fixed_elements(self, full_qp):
        """Warmstart should work when some elements are fixed."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"]},
        )
        sol = solver(A=d["A"], b=d["b"], G=d["G"], h=d["h"],
                     warmstart=jnp.array([0.4, 0.6]))
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-8)

    def test_warmstart_with_jvp(self, unconstrained_2d):
        """Warmstart should work through the JVP path."""
        from jax import jvp
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        def solve_warm(q_val):
            return solver(P=d["P"], q=q_val,
                          warmstart=jnp.array([0.9, 1.9]))

        def solve_cold(q_val):
            return solver(P=d["P"], q=q_val)

        dq = jnp.ones(d["n_var"])

        sol_w, dsol_w = jvp(solve_warm, (d["q"],), (dq,))
        sol_c, dsol_c = jvp(solve_cold, (d["q"],), (dq,))

        # Same primal and tangent regardless of warmstart
        np.testing.assert_allclose(sol_w["x"], sol_c["x"], atol=1e-10)
        np.testing.assert_allclose(dsol_w["x"], dsol_c["x"], atol=1e-10)

    def test_warmstart_good_vs_bad_speed(self):
        """A good warmstart should converge faster than a bad one."""
        import time

        n_var, n_eq, n_ineq = 200, 50, 100

        # Random well-conditioned QP
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        M = jax.random.normal(k1, (n_var, n_var))
        P = M.T @ M + 0.1 * jnp.eye(n_var)  # positive definite
        q = jax.random.normal(k2, (n_var,))
        A = jax.random.normal(k3, (n_eq, n_var))
        b = A @ jnp.ones(n_var)  # feasible at x=1
        G = jax.random.normal(k4, (n_ineq, n_var))
        h = G @ jnp.ones(n_var) + 1.0  # feasible with slack

        solver = setup_dense_solver(n_var=n_var, n_eq=n_eq, n_ineq=n_ineq, options={"solver":"osqp"})

        # Get the true solution for a "good" warmstart
        sol = solver(P=P, q=q, A=A, b=b, G=G, h=h)
        good_ws = sol["x"]

        # solver = setup_dense_solver(n_var=n_var, n_eq=n_eq, n_ineq=n_ineq, options={"solver":"piqp"})

        # Bad warmstart: far from solution
        bad_ws = jnp.ones(n_var) * 1e3

        # JIT warmup — run each path once so compilation is not measured
        solver(P=P, q=q, A=A, b=b, G=G, h=h,
            warmstart=good_ws)["x"].block_until_ready()
        solver(P=P, q=q, A=A, b=b, G=G, h=h,
            warmstart=bad_ws)["x"].block_until_ready()

        n_runs = 20

        # Time bad warmstart
        t0 = time.perf_counter()
        for _ in range(n_runs):
            solver(P=P, q=q, A=A, b=b, G=G, h=h,
                warmstart=bad_ws)["x"].block_until_ready()
        t_bad = (time.perf_counter() - t0) / n_runs

        # Time good warmstart
        t0 = time.perf_counter()
        for _ in range(n_runs):
            solver(P=P, q=q, A=A, b=b, G=G, h=h,
                warmstart=good_ws)["x"].block_until_ready()
        t_good = (time.perf_counter() - t0) / n_runs

        logging.info(f"Avg time — good warmstart: {t_good:.6f}s, "
                    f"bad warmstart: {t_bad:.6f}s")

        assert t_good < t_bad, (
            f"Good warmstart ({t_good:.6f}s) should be faster than "
            f"bad warmstart ({t_bad:.6f}s)"
        )


# # =====================================================================
# # Known bug: squeeze destroys shape for n_eq=1 or n_ineq=1
# # =====================================================================

# class TestSqueezeBug:
#     """np.squeeze() in _solve_qp removes size-1 constraint dimensions.

#     When n_eq=1, A has shape (1, n_var). After squeeze it becomes
#     (n_var,), which fails the shape assertion. Same for n_ineq=1.
#     """

#     @pytest.mark.xfail(reason="squeeze bug: n_eq=1 collapses A shape")
#     def test_single_equality(self):
#         P = jnp.eye(2)
#         q = jnp.zeros(2)
#         A = jnp.array([[1.0, 1.0]])     # shape (1, 2)
#         b = jnp.array([1.0])
#         solver = setup_dense_solver(n_var=2, n_eq=1)
#         sol = solver(P=P, q=q, A=A, b=b)
#         np.testing.assert_allclose(sol["x"].sum(), 1.0, atol=1e-8)

#     @pytest.mark.xfail(reason="squeeze bug: n_ineq=1 collapses G shape")
#     def test_single_inequality(self):
#         P = jnp.eye(2)
#         q = jnp.array([-2.0, -2.0])
#         G = jnp.array([[1.0, 1.0]])     # shape (1, 2)
#         h = jnp.array([1.0])            # x1 + x2 <= 1
#         solver = setup_dense_solver(n_var=2, n_ineq=1)
#         sol = solver(P=P, q=q, G=G, h=h)
#         assert float(G @ sol["x"]) <= 1.0 + 1e-8


# =====================================================================
# Solver options
# =====================================================================

class TestOptions:
    """Test custom solver options."""

    def test_float32_dtype(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(
            n_var=d["n_var"],
            options={"dtype": jnp.float32},
        )
        sol = solver(P=d["P"], q=d["q"])
        assert sol["x"].dtype == jnp.float32
        np.testing.assert_allclose(sol["x"], d["x_expected"], atol=1e-4)


# =====================================================================
# JVP vs Finite Differences
# =====================================================================

class TestJVPFiniteDifferences:
    """Verify JVP tangents against central finite differences."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_DUAL = 1e-3

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _fd_central(solve_fn, param, direction, eps):
        """Central finite difference: (f(p+εd) - f(p-εd)) / 2ε."""
        sol_plus = solve_fn(param + eps * direction)
        sol_minus = solve_fn(param - eps * direction)
        return {
            k: (sol_plus[k] - sol_minus[k]) / (2.0 * eps)
            for k in ("x", "lam", "mu")
        }

    # -----------------------------------------------------------------
    # Unconstrained: d/dq, d/dP
    # -----------------------------------------------------------------

    def test_jvp_dq_unconstrained(self, unconstrained_2d):
        """dx/dq via JVP vs finite differences (unconstrained)."""
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        dq = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val)

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    def test_jvp_dP_unconstrained(self, unconstrained_2d):
        """dx/dP via JVP vs finite differences (unconstrained)."""
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        dP = jnp.array([[1.0, 0.0], [0.0, 0.0]])

        def solve_P(P_val):
            return solver(P=P_val, q=d["q"])

        _, tangents = jax.jvp(solve_P, (d["P"],), (dP,))
        fd = self._fd_central(solve_P, d["P"], dP, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    # -----------------------------------------------------------------
    # Equality only: d/dq, d/db, d/dA
    # -----------------------------------------------------------------

    def test_jvp_dq_equality(self, equality_only):
        """dx/dq via JVP vs finite differences (equality constrained)."""
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        dq = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_jvp_db_equality(self, equality_only):
        """dx/db via JVP vs finite differences (equality constrained)."""
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        db = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_jvp_dA_equality(self, equality_only):
        """dx/dA via JVP vs finite differences (equality constrained)."""
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        dA = jnp.array([[1.0, 0.0], [0.0, 0.0]])

        def solve_A(A_val):
            return solver(P=d["P"], q=d["q"], A=A_val, b=d["b"])

        _, tangents = jax.jvp(solve_A, (d["A"],), (dA,))
        fd = self._fd_central(solve_A, d["A"], dA, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)

    # -----------------------------------------------------------------
    # Inequality only: d/dq, d/dh, d/dG
    # -----------------------------------------------------------------

    def test_jvp_dq_inequality(self, inequality_only):
        """dx/dq via JVP vs finite differences (inequality only)."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])

        dq = jnp.array([0.0, 1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["lam"], fd["lam"], atol=self.ATOL_DUAL)

    def test_jvp_dh_inequality(self, inequality_only):
        """dx/dh via JVP vs finite differences (inequality only)."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])

        dh = jnp.array([1.0, 0.0, 0.0])

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], G=d["G"], h=h_val)

        _, tangents = jax.jvp(solve_h, (d["h"],), (dh,))
        fd = self._fd_central(solve_h, d["h"], dh, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["lam"], fd["lam"], atol=self.ATOL_DUAL)

    def test_jvp_dG_inequality(self, inequality_only):
        """dx/dG via JVP vs finite differences (inequality only)."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])

        key = jax.random.PRNGKey(42)
        dG = jax.random.normal(key, d["G"].shape)

        def solve_G(G_val):
            return solver(P=d["P"], q=d["q"], G=G_val, h=d["h"])

        _, tangents = jax.jvp(solve_G, (d["G"],), (dG,))
        fd = self._fd_central(solve_G, d["G"], dG, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["lam"], fd["lam"], atol=self.ATOL_DUAL)

    # -----------------------------------------------------------------
    # Full QP: d/dq, d/db, d/dh (all constraints active)
    # -----------------------------------------------------------------

    def test_jvp_dq_full(self, full_qp):
        """dx/dq via JVP vs finite differences (full QP)."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )

        dq = jnp.array([1.0, 0.0])

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
        fd = self._fd_central(solve_q, d["q"], dq, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)
        np.testing.assert_allclose(tangents["lam"], fd["lam"], atol=self.ATOL_DUAL)

    def test_jvp_db_full(self, full_qp):
        """dx/db via JVP vs finite differences (full QP)."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )

        db = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val,
                          G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_jvp_dh_full(self, full_qp):
        """dx/dh via JVP vs finite differences (full QP)."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )

        dh = jnp.array([0.1, 0.0])

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=d["b"],
                          G=d["G"], h=h_val)

        _, tangents = jax.jvp(solve_h, (d["h"],), (dh,))
        fd = self._fd_central(solve_h, d["h"], dh, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["lam"], fd["lam"], atol=self.ATOL_DUAL)

    # -----------------------------------------------------------------
    # MPC: d/db (parametric initial condition)
    # -----------------------------------------------------------------

    def test_jvp_db_mpc(self, mpc_problem):
        """dx/db via JVP vs finite differences on the MPC problem."""
        d = mpc_problem
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"],
                            "A": d["A"], "G": d["G"], "h": d["h"]},
        )

        # Perturb only the initial condition slots of b
        db = jnp.zeros(d["n_eq"])
        db = db.at[0].set(1.0)

        def solve_b(b_val):
            return solver(b=b_val)

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)

    # -----------------------------------------------------------------
    # Random direction (checks linearity of JVP in the tangent)
    # -----------------------------------------------------------------

    def test_jvp_random_direction_full(self, full_qp):
        """JVP with a random tangent direction matches FD on full QP."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )

        key = jax.random.PRNGKey(7)
        k1, k2, k3 = jax.random.split(key, 3)
        dq = jax.random.normal(k1, d["q"].shape)
        db = jax.random.normal(k2, d["b"].shape)
        dh = jax.random.normal(k3, d["h"].shape)

        def solve_qbh(q_val, b_val, h_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=b_val,
                          G=d["G"], h=h_val)

        _, tangents = jax.jvp(
            solve_qbh,
            (d["q"], d["b"], d["h"]),
            (dq, db, dh),
        )

        # FD over the combined perturbation
        def solve_eps(eps):
            return solve_qbh(
                d["q"] + eps * dq,
                d["b"] + eps * db,
                d["h"] + eps * dh,
            )

        sol_p = solve_eps(self.EPS)
        sol_m = solve_eps(-self.EPS)
        fd_x = (sol_p["x"] - sol_m["x"]) / (2.0 * self.EPS)

        np.testing.assert_allclose(tangents["x"], fd_x, atol=self.ATOL_X)

    # -----------------------------------------------------------------
    # Fixed elements: JVP only flows through dynamic keys
    # -----------------------------------------------------------------

    def test_jvp_fixed_P_q_diff_through_b(self, full_qp):
        """With P, q fixed, JVP w.r.t. b still matches FD."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"]},
        )

        db = jnp.array([1.0, 0.0])

        def solve_b(b_val):
            return solver(A=d["A"], b=b_val, G=d["G"], h=d["h"])

        _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
        fd = self._fd_central(solve_b, d["b"], db, self.EPS)

        np.testing.assert_allclose(tangents["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(tangents["mu"], fd["mu"], atol=self.ATOL_DUAL)

# =====================================================================
# JVP under vmap vs Finite Differences
# =====================================================================

class TestJVPVmapFiniteDifferences:
    """Verify that vmapped JVP tangents match per-direction finite differences."""

    EPS = 1e-6
    ATOL_X = 1e-4
    ATOL_DUAL = 1e-3
    N_DIRS = 3  # number of random tangent directions per test

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _fd_central(solve_fn, param, direction, eps):
        """Central finite difference for a single direction."""
        sol_plus = solve_fn(param + eps * direction)
        sol_minus = solve_fn(param - eps * direction)
        return {
            k: (sol_plus[k] - sol_minus[k]) / (2.0 * eps)
            for k in ("x", "lam", "mu")
        }

    def _fd_batch(self, solve_fn, param, directions, eps):
        """Run central FD for each direction, stack results."""
        results = [self._fd_central(solve_fn, param, d, eps) for d in directions]
        return {
            k: jnp.stack([r[k] for r in results])
            for k in ("x", "lam", "mu")
        }

    # -----------------------------------------------------------------
    # Unconstrained: vmap d/dq, vmap d/dP
    # -----------------------------------------------------------------

    def test_vmap_jvp_dq_unconstrained(self, unconstrained_2d):
        """Batched dx/dq via vmap(jvp) vs finite differences."""
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        key = jax.random.PRNGKey(10)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val)

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
            return tangents

        batched_tangents = jax.vmap(jvp_one)(dqs)
        fd = self._fd_batch(solve_q, d["q"], dqs, self.EPS)

        np.testing.assert_allclose(batched_tangents["x"], fd["x"], atol=self.ATOL_X)

    def test_vmap_jvp_dP_unconstrained(self, unconstrained_2d):
        """Batched dx/dP via vmap(jvp) vs finite differences."""
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])

        key = jax.random.PRNGKey(11)
        dPs = jax.random.normal(key, (self.N_DIRS, d["n_var"], d["n_var"]))
        # Symmetrize tangent directions (P is symmetric)
        dPs = (dPs + jnp.swapaxes(dPs, -2, -1)) / 2.0

        def solve_P(P_val):
            return solver(P=P_val, q=d["q"])

        def jvp_one(dP):
            _, tangents = jax.jvp(solve_P, (d["P"],), (dP,))
            return tangents

        batched_tangents = jax.vmap(jvp_one)(dPs)
        fd = self._fd_batch(solve_P, d["P"], dPs, self.EPS)

        np.testing.assert_allclose(batched_tangents["x"], fd["x"], atol=self.ATOL_X)

    # -----------------------------------------------------------------
    # Equality only: vmap d/dq, vmap d/db, vmap d/dA
    # -----------------------------------------------------------------

    def test_vmap_jvp_dq_equality(self, equality_only):
        """Batched dx/dq via vmap(jvp) vs FD (equality constrained)."""
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        key = jax.random.PRNGKey(20)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"])

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
            return tangents

        batched = jax.vmap(jvp_one)(dqs)
        fd = self._fd_batch(solve_q, d["q"], dqs, self.EPS)

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(batched["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_vmap_jvp_db_equality(self, equality_only):
        """Batched dx/db via vmap(jvp) vs FD (equality constrained)."""
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        key = jax.random.PRNGKey(21)
        dbs = jax.random.normal(key, (self.N_DIRS, d["n_eq"]))

        def solve_b(b_val):
            return solver(P=d["P"], q=d["q"], A=d["A"], b=b_val)

        def jvp_one(db):
            _, tangents = jax.jvp(solve_b, (d["b"],), (db,))
            return tangents

        batched = jax.vmap(jvp_one)(dbs)
        fd = self._fd_batch(solve_b, d["b"], dbs, self.EPS)

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(batched["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_vmap_jvp_dA_equality(self, equality_only):
        """Batched dx/dA via vmap(jvp) vs FD (equality constrained)."""
        d = equality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_eq=d["n_eq"])

        key = jax.random.PRNGKey(22)
        dAs = jax.random.normal(key, (self.N_DIRS, d["n_eq"], d["n_var"]))

        def solve_A(A_val):
            return solver(P=d["P"], q=d["q"], A=A_val, b=d["b"])

        def jvp_one(dA):
            _, tangents = jax.jvp(solve_A, (d["A"],), (dA,))
            return tangents

        batched = jax.vmap(jvp_one)(dAs)
        fd = self._fd_batch(solve_A, d["A"], dAs, self.EPS)

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(batched["mu"], fd["mu"], atol=self.ATOL_DUAL)

    # -----------------------------------------------------------------
    # Inequality only: vmap d/dq, vmap d/dh, vmap d/dG
    # -----------------------------------------------------------------

    def test_vmap_jvp_dq_inequality(self, inequality_only):
        """Batched dx/dq via vmap(jvp) vs FD (inequality only)."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])

        key = jax.random.PRNGKey(30)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, G=d["G"], h=d["h"])

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
            return tangents

        batched = jax.vmap(jvp_one)(dqs)
        fd = self._fd_batch(solve_q, d["q"], dqs, self.EPS)

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(batched["lam"], fd["lam"], atol=self.ATOL_DUAL)

    def test_vmap_jvp_dh_inequality(self, inequality_only):
        """Batched dx/dh via vmap(jvp) vs FD (inequality only)."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])

        key = jax.random.PRNGKey(31)
        dhs = jax.random.normal(key, (self.N_DIRS, d["n_ineq"]))

        def solve_h(h_val):
            return solver(P=d["P"], q=d["q"], G=d["G"], h=h_val)

        def jvp_one(dh):
            _, tangents = jax.jvp(solve_h, (d["h"],), (dh,))
            return tangents

        batched = jax.vmap(jvp_one)(dhs)
        fd = self._fd_batch(solve_h, d["h"], dhs, self.EPS)

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(batched["lam"], fd["lam"], atol=self.ATOL_DUAL)

    def test_vmap_jvp_dG_inequality(self, inequality_only):
        """Batched dx/dG via vmap(jvp) vs FD (inequality only)."""
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])

        key = jax.random.PRNGKey(32)
        dGs = jax.random.normal(key, (self.N_DIRS, d["n_ineq"], d["n_var"]))

        def solve_G(G_val):
            return solver(P=d["P"], q=d["q"], G=G_val, h=d["h"])

        def jvp_one(dG):
            _, tangents = jax.jvp(solve_G, (d["G"],), (dG,))
            return tangents

        batched = jax.vmap(jvp_one)(dGs)
        fd = self._fd_batch(solve_G, d["G"], dGs, self.EPS)

        np.testing.assert_allclose(batched["x"], fd["x"], atol=self.ATOL_X)
        np.testing.assert_allclose(batched["lam"], fd["lam"], atol=self.ATOL_DUAL)

    # -----------------------------------------------------------------
    # Full QP: vmap d/dq, vmap d/db, vmap d/dh
    # -----------------------------------------------------------------

    def test_vmap_jvp_dq_full(self, full_qp):
        """Batched dx/dq via vmap(jvp) vs FD (full QP)."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
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
        np.testing.assert_allclose(batched["mu"], fd["mu"], atol=self.ATOL_DUAL)
        np.testing.assert_allclose(batched["lam"], fd["lam"], atol=self.ATOL_DUAL)

    def test_vmap_jvp_db_full(self, full_qp):
        """Batched dx/db via vmap(jvp) vs FD (full QP)."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
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
        np.testing.assert_allclose(batched["mu"], fd["mu"], atol=self.ATOL_DUAL)

    def test_vmap_jvp_dh_full(self, full_qp):
        """Batched dx/dh via vmap(jvp) vs FD (full QP)."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
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
        np.testing.assert_allclose(batched["lam"], fd["lam"], atol=self.ATOL_DUAL)

    # -----------------------------------------------------------------
    # MPC: vmap d/db (parametric initial condition)
    # -----------------------------------------------------------------

    def test_vmap_jvp_db_mpc(self, mpc_problem):
        """Batched dx/db via vmap(jvp) vs FD on MPC problem."""
        d = mpc_problem
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
            fixed_elements={"P": d["P"], "q": d["q"],
                            "A": d["A"], "G": d["G"], "h": d["h"]},
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

    # -----------------------------------------------------------------
    # Multi-parameter vmap: simultaneous perturbation of q, b, h
    # -----------------------------------------------------------------

    def test_vmap_jvp_multi_param_full(self, full_qp):
        """Batched JVP over joint (dq, db, dh) directions matches FD."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )

        key = jax.random.PRNGKey(60)
        k1, k2, k3 = jax.random.split(key, 3)
        dqs = jax.random.normal(k1, (self.N_DIRS, d["n_var"]))
        dbs = jax.random.normal(k2, (self.N_DIRS, d["n_eq"]))
        dhs = jax.random.normal(k3, (self.N_DIRS, d["n_ineq"]))

        def solve_qbh(q_val, b_val, h_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=b_val,
                          G=d["G"], h=h_val)

        def jvp_one(dq, db, dh):
            _, tangents = jax.jvp(
                solve_qbh,
                (d["q"], d["b"], d["h"]),
                (dq, db, dh),
            )
            return tangents

        batched = jax.vmap(jvp_one)(dqs, dbs, dhs)

        # FD per direction
        fd_xs = []
        for i in range(self.N_DIRS):
            sp = solve_qbh(
                d["q"] + self.EPS * dqs[i],
                d["b"] + self.EPS * dbs[i],
                d["h"] + self.EPS * dhs[i],
            )
            sm = solve_qbh(
                d["q"] - self.EPS * dqs[i],
                d["b"] - self.EPS * dbs[i],
                d["h"] - self.EPS * dhs[i],
            )
            fd_xs.append((sp["x"] - sm["x"]) / (2.0 * self.EPS))

        fd_x = jnp.stack(fd_xs)
        np.testing.assert_allclose(batched["x"], fd_x, atol=self.ATOL_X)

    # -----------------------------------------------------------------
    # Consistency: vmap result matches sequential JVP calls
    # -----------------------------------------------------------------

    def test_vmap_matches_sequential(self, full_qp):
        """vmap(jvp) should produce identical results to sequential jvp calls."""
        d = full_qp
        solver = setup_dense_solver(
            n_var=d["n_var"], n_eq=d["n_eq"], n_ineq=d["n_ineq"],
        )

        key = jax.random.PRNGKey(70)
        dqs = jax.random.normal(key, (self.N_DIRS, d["n_var"]))

        def solve_q(q_val):
            return solver(P=d["P"], q=q_val, A=d["A"], b=d["b"],
                          G=d["G"], h=d["h"])

        def jvp_one(dq):
            _, tangents = jax.jvp(solve_q, (d["q"],), (dq,))
            return tangents

        # Batched path
        batched = jax.vmap(jvp_one)(dqs)

        # Sequential path
        seq_xs = []
        seq_lams = []
        seq_mus = []
        for i in range(self.N_DIRS):
            _, t = jax.jvp(solve_q, (d["q"],), (dqs[i],))
            seq_xs.append(t["x"])
            seq_lams.append(t["lam"])
            seq_mus.append(t["mu"])

        np.testing.assert_allclose(
            batched["x"], jnp.stack(seq_xs), atol=1e-12
        )
        np.testing.assert_allclose(
            batched["lam"], jnp.stack(seq_lams), atol=1e-12
        )
        np.testing.assert_allclose(
            batched["mu"], jnp.stack(seq_mus), atol=1e-12
        )