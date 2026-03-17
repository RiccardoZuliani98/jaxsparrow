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

from solver_dense.solver_dense import setup_dense_solver


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
        assert sol["active"].shape == (0,)

    def test_output_dtypes(self, unconstrained_2d):
        d = unconstrained_2d
        solver = setup_dense_solver(n_var=d["n_var"])
        sol = solver(P=d["P"], q=d["q"])

        assert sol["x"].dtype == jnp.float64
        assert sol["lam"].dtype == jnp.float64
        assert sol["mu"].dtype == jnp.float64
        assert sol["active"].dtype == jnp.bool_


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
        assert sol["active"].shape == (d["n_ineq"],)

    def test_active_set(self, inequality_only):
        d = inequality_only
        solver = setup_dense_solver(n_var=d["n_var"], n_ineq=d["n_ineq"])
        sol = solver(P=d["P"], q=d["q"], G=d["G"], h=d["h"])

        # Constraints 0 and 2 should be active (x1=0, x3=0)
        active = np.array(sol["active"])
        assert active[0] == True   # x1 = 0 is active
        assert active[1] == False  # x2 = 2 is not active
        assert active[2] == True   # x3 = 0 is active

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
        assert sol["active"].shape == (0,)

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
        assert sol["active"].shape == (d["n_ineq"],)

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
# Known bug: squeeze destroys shape for n_eq=1 or n_ineq=1
# =====================================================================

class TestSqueezeBug:
    """np.squeeze() in _solve_qp removes size-1 constraint dimensions.

    When n_eq=1, A has shape (1, n_var). After squeeze it becomes
    (n_var,), which fails the shape assertion. Same for n_ineq=1.
    """

    @pytest.mark.xfail(reason="squeeze bug: n_eq=1 collapses A shape")
    def test_single_equality(self):
        P = jnp.eye(2)
        q = jnp.zeros(2)
        A = jnp.array([[1.0, 1.0]])     # shape (1, 2)
        b = jnp.array([1.0])
        solver = setup_dense_solver(n_var=2, n_eq=1)
        sol = solver(P=P, q=q, A=A, b=b)
        np.testing.assert_allclose(sol["x"].sum(), 1.0, atol=1e-8)

    @pytest.mark.xfail(reason="squeeze bug: n_ineq=1 collapses G shape")
    def test_single_inequality(self):
        P = jnp.eye(2)
        q = jnp.array([-2.0, -2.0])
        G = jnp.array([[1.0, 1.0]])     # shape (1, 2)
        h = jnp.array([1.0])            # x1 + x2 <= 1
        solver = setup_dense_solver(n_var=2, n_ineq=1)
        sol = solver(P=P, q=q, G=G, h=h)
        assert float(G @ sol["x"]) <= 1.0 + 1e-8


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