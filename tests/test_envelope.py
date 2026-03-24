"""Tests for qp_value: value computation and envelope-theorem gradients.

Covers:
    - Forward value equals objective at optimum
    - Analytical envelope gradients: dV/dP, dV/dq, dV/dG, dV/dh, dV/dA, dV/db
    - Finite-difference validation (central differences, multiple directions)
    - JIT compatibility (value, grad, value_and_grad)
    - Dense vs sparse solver: value and gradients agree
    - Constraint regimes: inequality-only, equality-only, full, MPC
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

jax.config.update("jax_enable_x64", True)

from jaxsparrow import qp_value
from jaxsparrow._solver_dense._setup import setup_dense_solver
from jaxsparrow._solver_sparse._setup import setup_sparse_solver


# =====================================================================
# Helpers
# =====================================================================

def _to_bcoo(dense_matrix):
    return BCOO.fromdense(dense_matrix)


def _sparsity_dict(**kwargs):
    return {k: _to_bcoo(v) for k, v in kwargs.items()}


def _solve_dense(P, q, G, h, A, b):
    """Solve with dense backend, return SolverOutput dict."""
    n_var = q.shape[0]
    n_ineq = h.shape[0]
    n_eq = b.shape[0]
    solver = setup_dense_solver(n_var=n_var, n_ineq=n_ineq, n_eq=n_eq)
    return solver(P=P, q=q, G=G, h=h, A=A, b=b)


def _solve_sparse(P, q, G, h, A, b):
    """Solve with sparse backend, return SolverOutput dict."""
    n_var = q.shape[0]
    n_ineq = h.shape[0]
    n_eq = b.shape[0]
    patterns = _sparsity_dict(P=P, G=G, A=A)
    solver = setup_sparse_solver(
        n_var=n_var, n_ineq=n_ineq, n_eq=n_eq,
        sparsity_patterns=patterns,
    )
    return solver(P=_to_bcoo(P), q=q, G=_to_bcoo(G), h=h, A=_to_bcoo(A), b=b)


def _objective(P, q, x):
    return 0.5 * x @ P @ x + q @ x


def _all_grads(P, q, G, h, A, b, sol):
    """Autodiff of qp_value w.r.t. all six ingredients."""
    _, grads = jax.value_and_grad(qp_value, argnums=(0, 1, 2, 3, 4, 5))(
        P, q, G, h, A, b, sol,
    )
    return dict(zip(("P", "q", "G", "h", "A", "b"), grads))


# =====================================================================
# Fixtures — each provides (P, q, G, h, A, b) as dense arrays
# =====================================================================

@pytest.fixture
def inequality_only():
    """min 0.5 x^T x + q^T x  s.t.  x >= 0."""
    n = 3
    P = jnp.eye(n)
    q = jnp.array([1.0, -2.0, 1.0])
    G = -jnp.eye(n)
    h = jnp.zeros(n)
    A = jnp.zeros((0, n))
    b = jnp.zeros(0)
    return dict(P=P, q=q, G=G, h=h, A=A, b=b)


@pytest.fixture
def equality_only():
    """min 0.5 x^T x  s.t.  x1+x2=1, x1-x2=0."""
    n = 2
    P = jnp.eye(n)
    q = jnp.zeros(n)
    G = jnp.zeros((0, n))
    h = jnp.zeros(0)
    A = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    b = jnp.array([1.0, 0.0])
    return dict(P=P, q=q, G=G, h=h, A=A, b=b)


@pytest.fixture
def full_qp():
    """min 0.5 x^T x  s.t.  x1+x2=1, x1-x2=0, x >= 0."""
    n = 2
    P = jnp.eye(n)
    q = jnp.zeros(n)
    G = -jnp.eye(n)
    h = jnp.zeros(n)
    A = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    b = jnp.array([1.0, 0.0])
    return dict(P=P, q=q, G=G, h=h, A=A, b=b)


@pytest.fixture
def mpc_problem():
    """Small MPC with dynamics equality + box inequality constraints."""
    N = 5
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

    return dict(P=P, q=q, G=G, h=h, A=Aeq, b=beq, nx=nx, N=N, x0=x0)


ALL_FIXTURES = ["inequality_only", "equality_only", "full_qp", "mpc_problem"]


# =====================================================================
# 1. Value equals objective at optimum
# =====================================================================

class TestValueEqualsObjective:
    """qp_value must return 0.5 x*^T P x* + q^T x* at the optimum."""

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    def test_dense(self, fixture_name, request):
        d = request.getfixturevalue(fixture_name)
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        V = qp_value(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        obj = _objective(d["P"], d["q"], sol["x"])
        np.testing.assert_allclose(V, obj, atol=1e-10)

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    def test_sparse(self, fixture_name, request):
        d = request.getfixturevalue(fixture_name)
        sol = _solve_sparse(**{k: d[k] for k in "P q G h A b".split()})
        V = qp_value(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        obj = _objective(d["P"], d["q"], sol["x"])
        np.testing.assert_allclose(V, obj, atol=1e-10)


# =====================================================================
# 2. Analytical envelope gradients
# =====================================================================

class TestAnalyticalGradients:
    """
    Autodiff of qp_value must match closed-form envelope formulas:
        dV/dq = x*           dV/dP = 0.5 x* x*^T
        dV/dh = -lam*        dV/dG = lam* x*^T
        dV/db = -mu*         dV/dA = mu* x*^T
    """
    ATOL = 1e-10

    # -- individual gradient checks on full_qp --

    def test_dq(self, full_qp):
        d = full_qp
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        g = _all_grads(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        np.testing.assert_allclose(g["q"], sol["x"], atol=self.ATOL)

    def test_dP(self, full_qp):
        d = full_qp
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        g = _all_grads(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        np.testing.assert_allclose(g["P"], 0.5 * jnp.outer(sol["x"], sol["x"]),
                                   atol=self.ATOL)

    def test_dh(self, full_qp):
        d = full_qp
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        g = _all_grads(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        np.testing.assert_allclose(g["h"], -sol["lam"], atol=self.ATOL)

    def test_dG(self, full_qp):
        d = full_qp
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        g = _all_grads(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        np.testing.assert_allclose(g["G"], jnp.outer(sol["lam"], sol["x"]),
                                   atol=self.ATOL)

    def test_db(self, full_qp):
        d = full_qp
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        g = _all_grads(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        np.testing.assert_allclose(g["b"], -sol["mu"], atol=self.ATOL)

    def test_dA(self, full_qp):
        d = full_qp
        sol = _solve_dense(**{k: d[k] for k in "P q G h A b".split()})
        g = _all_grads(**{k: d[k] for k in "P q G h A b".split()}, sol=sol)
        np.testing.assert_allclose(g["A"], jnp.outer(sol["mu"], sol["x"]),
                                   atol=self.ATOL)

    # -- all six gradients, parametrized over every fixture and solver --

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    @pytest.mark.parametrize("solver_fn", [_solve_dense, _solve_sparse],
                             ids=["dense", "sparse"])
    def test_all_gradients(self, fixture_name, solver_fn, request):
        d = request.getfixturevalue(fixture_name)
        ingredients = {k: d[k] for k in "P q G h A b".split()}
        sol = solver_fn(**ingredients)
        g = _all_grads(**ingredients, sol=sol)

        np.testing.assert_allclose(g["q"], sol["x"], atol=self.ATOL)
        np.testing.assert_allclose(g["P"],
                                   0.5 * jnp.outer(sol["x"], sol["x"]),
                                   atol=self.ATOL)
        np.testing.assert_allclose(g["h"], -sol["lam"], atol=self.ATOL)
        np.testing.assert_allclose(g["G"],
                                   jnp.outer(sol["lam"], sol["x"]),
                                   atol=self.ATOL)
        np.testing.assert_allclose(g["b"], -sol["mu"], atol=self.ATOL)
        np.testing.assert_allclose(g["A"],
                                   jnp.outer(sol["mu"], sol["x"]),
                                   atol=self.ATOL)


# =====================================================================
# 3. Finite-difference validation
# =====================================================================

class TestFiniteDifferences:
    """
    Central finite-difference check: for each ingredient theta, verify
        [V*(theta + eps*d) - V*(theta - eps*d)] / (2 eps)  ≈  grad_theta . d
    where V*(theta) re-solves the QP at the perturbed point.
    """
    EPS = 1e-5
    ATOL = 1e-4
    N_DIRS = 5

    @staticmethod
    def _fd_directional(solve_and_eval, param, direction, eps):
        v_fwd = solve_and_eval(param + eps * direction)
        v_bwd = solve_and_eval(param - eps * direction)
        return float((v_fwd - v_bwd) / (2 * eps))

    def _check_gradient(self, ingredients, param_name, solver_fn, seed):
        """Generic FD check for one ingredient."""
        base = ingredients[param_name]
        sol = solver_fn(**ingredients)
        idx = "P q G h A b".split().index(param_name)
        grad_ad = jax.grad(qp_value, argnums=idx)(
            *[ingredients[k] for k in "P q G h A b".split()], sol,
        )

        def solve_and_eval(param_val):
            ing = {**ingredients, param_name: param_val}
            s = solver_fn(**ing)
            return float(qp_value(**ing, sol=s))

        key = jax.random.PRNGKey(seed)
        for i in range(self.N_DIRS):
            key, subkey = jax.random.split(key)
            d = jax.random.normal(subkey, shape=base.shape)
            d = d / jnp.linalg.norm(d)

            ad_dd = float(jnp.sum(grad_ad * d))
            fd_dd = self._fd_directional(solve_and_eval, base, d, self.EPS)
            np.testing.assert_allclose(
                ad_dd, fd_dd, atol=self.ATOL,
                err_msg=f"dV/d{param_name} direction {i}",
            )

    # -- vector ingredients, dense solver --

    @pytest.mark.parametrize("param_name", ["q", "h", "b"])
    def test_fd_vector_full_qp_dense(self, full_qp, param_name):
        ingredients = {k: full_qp[k] for k in "P q G h A b".split()}
        if ingredients[param_name].shape[0] == 0:
            pytest.skip(f"{param_name} is empty for this fixture")
        self._check_gradient(ingredients, param_name, _solve_dense, seed=10)

    @pytest.mark.parametrize("param_name", ["q", "h", "b"])
    def test_fd_vector_mpc_dense(self, mpc_problem, param_name):
        ingredients = {k: mpc_problem[k] for k in "P q G h A b".split()}
        self._check_gradient(ingredients, param_name, _solve_dense, seed=20)

    @pytest.mark.parametrize("param_name", ["q", "h"])
    def test_fd_vector_inequality_only_dense(self, inequality_only, param_name):
        ingredients = {k: inequality_only[k] for k in "P q G h A b".split()}
        if ingredients[param_name].shape[0] == 0:
            pytest.skip(f"{param_name} is empty for this fixture")
        self._check_gradient(ingredients, param_name, _solve_dense, seed=30)

    @pytest.mark.parametrize("param_name", ["q", "b"])
    def test_fd_vector_equality_only_dense(self, equality_only, param_name):
        ingredients = {k: equality_only[k] for k in "P q G h A b".split()}
        if ingredients[param_name].shape[0] == 0:
            pytest.skip(f"{param_name} is empty for this fixture")
        self._check_gradient(ingredients, param_name, _solve_dense, seed=40)

    # -- matrix ingredients, dense solver --

    @pytest.mark.parametrize("param_name", ["P", "G", "A"])
    def test_fd_matrix_full_qp_dense(self, full_qp, param_name):
        ingredients = {k: full_qp[k] for k in "P q G h A b".split()}
        if ingredients[param_name].shape[0] == 0:
            pytest.skip(f"{param_name} is empty for this fixture")
        self._check_gradient(ingredients, param_name, _solve_dense, seed=50)

    @pytest.mark.parametrize("param_name", ["P", "A"])
    def test_fd_matrix_mpc_dense(self, mpc_problem, param_name):
        ingredients = {k: mpc_problem[k] for k in "P q G h A b".split()}
        self._check_gradient(ingredients, param_name, _solve_dense, seed=60)

    # -- vector ingredients, sparse solver --

    @pytest.mark.parametrize("param_name", ["q", "h", "b"])
    def test_fd_vector_full_qp_sparse(self, full_qp, param_name):
        ingredients = {k: full_qp[k] for k in "P q G h A b".split()}
        if ingredients[param_name].shape[0] == 0:
            pytest.skip(f"{param_name} is empty for this fixture")
        self._check_gradient(ingredients, param_name, _solve_sparse, seed=70)

    @pytest.mark.parametrize("param_name", ["q", "h", "b"])
    def test_fd_vector_mpc_sparse(self, mpc_problem, param_name):
        ingredients = {k: mpc_problem[k] for k in "P q G h A b".split()}
        self._check_gradient(ingredients, param_name, _solve_sparse, seed=80)

    # -- matrix ingredients, sparse solver --

    @pytest.mark.parametrize("param_name", ["P", "G", "A"])
    def test_fd_matrix_full_qp_sparse(self, full_qp, param_name):
        ingredients = {k: full_qp[k] for k in "P q G h A b".split()}
        if ingredients[param_name].shape[0] == 0:
            pytest.skip(f"{param_name} is empty for this fixture")
        self._check_gradient(ingredients, param_name, _solve_sparse, seed=90)


# =====================================================================
# 4. JIT compatibility
# =====================================================================

class TestJIT:
    """qp_value and its gradients behave identically under jax.jit."""

    def _ingredients(self, d):
        return {k: d[k] for k in "P q G h A b".split()}

    def test_jit_value(self, full_qp):
        ing = self._ingredients(full_qp)
        sol = _solve_dense(**ing)
        v_eager = qp_value(**ing, sol=sol)
        v_jit = jax.jit(qp_value)(**ing, sol=sol)
        np.testing.assert_allclose(v_eager, v_jit, atol=1e-12)

    def test_jit_grad(self, full_qp):
        ing = self._ingredients(full_qp)
        sol = _solve_dense(**ing)

        def val(q):
            return qp_value(P=ing["P"], q=q, G=ing["G"], h=ing["h"],
                            A=ing["A"], b=ing["b"], sol=sol)

        g_eager = jax.grad(val)(ing["q"])
        g_jit = jax.jit(jax.grad(val))(ing["q"])
        np.testing.assert_allclose(g_eager, g_jit, atol=1e-12)

    def test_jit_value_and_grad(self, mpc_problem):
        ing = self._ingredients(mpc_problem)
        sol = _solve_dense(**ing)

        def val(b):
            return qp_value(P=ing["P"], q=ing["q"], G=ing["G"], h=ing["h"],
                            A=ing["A"], b=b, sol=sol)

        v1, g1 = jax.value_and_grad(val)(ing["b"])
        v2, g2 = jax.jit(jax.value_and_grad(val))(ing["b"])
        np.testing.assert_allclose(v1, v2, atol=1e-12)
        np.testing.assert_allclose(g1, g2, atol=1e-12)

    def test_jit_all_grads(self, full_qp):
        ing = self._ingredients(full_qp)
        sol = _solve_dense(**ing)

        def val(*args):
            return qp_value(*args, sol=sol)

        args = tuple(ing[k] for k in "P q G h A b".split())
        g_eager = jax.grad(val, argnums=(0, 1, 2, 3, 4, 5))(*args)
        g_jit = jax.jit(jax.grad(val, argnums=(0, 1, 2, 3, 4, 5)))(*args)
        for ge, gj in zip(g_eager, g_jit):
            np.testing.assert_allclose(ge, gj, atol=1e-12)


# =====================================================================
# 5. Dense vs sparse solver agreement
# =====================================================================

class TestDenseSparseAgreement:
    """Value and all gradients must agree when using either solver backend."""

    ATOL_VAL = 1e-8
    ATOL_GRAD = 1e-7

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    def test_value(self, fixture_name, request):
        d = request.getfixturevalue(fixture_name)
        ing = {k: d[k] for k in "P q G h A b".split()}
        sol_d = _solve_dense(**ing)
        sol_s = _solve_sparse(**ing)
        v_d = float(qp_value(**ing, sol=sol_d))
        v_s = float(qp_value(**ing, sol=sol_s))
        np.testing.assert_allclose(v_d, v_s, atol=self.ATOL_VAL)

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    def test_all_gradients(self, fixture_name, request):
        d = request.getfixturevalue(fixture_name)
        ing = {k: d[k] for k in "P q G h A b".split()}
        sol_d = _solve_dense(**ing)
        sol_s = _solve_sparse(**ing)
        g_d = _all_grads(**ing, sol=sol_d)
        g_s = _all_grads(**ing, sol=sol_s)
        for name in "P q G h A b".split():
            np.testing.assert_allclose(
                g_d[name], g_s[name], atol=self.ATOL_GRAD,
                err_msg=f"dV/d{name} disagrees between dense and sparse",
            )