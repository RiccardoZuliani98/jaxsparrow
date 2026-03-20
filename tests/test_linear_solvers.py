"""
tests/test_linear_solvers.py
============================
Tests for the dense and sparse linear solver backends.

Strategy:
  - Generate random well-conditioned linear systems.
  - Solve with every registered backend (native + wrapped).
  - Assert solutions match a reference (numpy.linalg.solve).
  - Cover 1-D (single RHS) and 2-D (multi-RHS) cases.
  - Cover symmetric positive-definite and general non-singular systems.
  - Cover the cross-registry wrappers (dense↔sparse).
"""

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose
from scipy.sparse import csc_matrix, random as sp_random

# ── Import the module under test ─────────────────────────────────────
# We inline the module here so the tests are self-contained and don't
# require the full jaxsparrow package to be installed.

from typing import Callable
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu, spsolve, lsqr, spilu

from jaxsparrow._utils._linear_solvers import (
    _DENSE_NATIVE, _SPARSE_NATIVE, 
    get_dense_linear_solver, get_sparse_linear_solver
)


# =====================================================================
# Test fixtures
# =====================================================================

SEED = 42
SIZES = [5, 20, 50]
N_RHS = 4  # number of columns for multi-RHS tests

# Tolerances: spilu is approximate, so it gets a wider tolerance
TOL_EXACT = 1e-10
TOL_APPROX = 1e-4  # for spilu


def _make_spd(n: int, rng: np.random.Generator) -> ndarray:
    """Generate a random symmetric positive-definite matrix."""
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _make_nonsing(n: int, rng: np.random.Generator) -> ndarray:
    """Generate a random non-singular (but not necessarily symmetric) matrix."""
    A = rng.standard_normal((n, n))
    # Ensure non-singularity by adding a diagonal shift
    return A + n * np.eye(n)


def _make_sparse_spd(n: int, rng: np.random.Generator, density: float = 0.3) -> csc_matrix:
    """Generate a random sparse SPD matrix in CSC format."""
    A = sp_random(n, n, density=density, random_state=rng, format="csc")
    return (A @ A.T + n * csc_matrix(np.eye(n))).tocsc()


def _make_sparse_nonsing(n: int, rng: np.random.Generator, density: float = 0.3) -> csc_matrix:
    """Generate a random sparse non-singular matrix in CSC format."""
    A = sp_random(n, n, density=density, random_state=rng, format="csc")
    return (A + n * csc_matrix(np.eye(n))).tocsc()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# =====================================================================
# Dense solver tests
# =====================================================================

ALL_DENSE_NAMES = list(_DENSE_NATIVE.keys()) + list(_SPARSE_NATIVE.keys())
# spilu and sp_lstsq are approximate (iterative) — they get wider tolerances
APPROX_SOLVERS = {"spilu", "sp_lstsq"}
EXACT_DENSE_NAMES = [n for n in ALL_DENSE_NAMES if n not in APPROX_SOLVERS]


class TestDenseSolversSingleRHS:
    """Dense solvers with a single RHS vector (1-D)."""

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_spd_single_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_spd(n, rng)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(A, b)

        solver = get_dense_linear_solver(name)
        x = solver(A, b)

        assert x.shape == ref.shape
        assert_allclose(x, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_nonsing_single_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_nonsing(n, rng)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(A, b)

        solver = get_dense_linear_solver(name)
        x = solver(A, b)

        assert x.shape == ref.shape
        assert_allclose(x, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", sorted(APPROX_SOLVERS))
    @pytest.mark.parametrize("n", SIZES)
    def test_approximate_single_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_spd(n, rng)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(A, b)

        solver = get_dense_linear_solver(name)
        x = solver(A, b)

        assert x.shape == ref.shape
        assert_allclose(x, ref, atol=TOL_APPROX, rtol=TOL_APPROX)


class TestDenseSolversMultiRHS:
    """Dense solvers with multiple RHS columns (2-D)."""

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_spd_multi_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_spd(n, rng)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(A, B)

        solver = get_dense_linear_solver(name)
        X = solver(A, B)

        assert X.shape == ref.shape
        assert_allclose(X, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_nonsing_multi_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_nonsing(n, rng)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(A, B)

        solver = get_dense_linear_solver(name)
        X = solver(A, B)

        assert X.shape == ref.shape
        assert_allclose(X, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", sorted(APPROX_SOLVERS))
    @pytest.mark.parametrize("n", SIZES)
    def test_approximate_multi_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_spd(n, rng)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(A, B)

        solver = get_dense_linear_solver(name)
        X = solver(A, B)

        assert X.shape == ref.shape
        assert_allclose(X, ref, atol=TOL_APPROX, rtol=TOL_APPROX)


# =====================================================================
# Sparse solver tests
# =====================================================================

ALL_SPARSE_NAMES = list(_SPARSE_NATIVE.keys()) + list(_DENSE_NATIVE.keys())
EXACT_SPARSE_NAMES = [n for n in ALL_SPARSE_NAMES if n not in APPROX_SOLVERS]


class TestSparseSolversSingleRHS:
    """Sparse solvers with a single RHS vector (1-D)."""

    @pytest.mark.parametrize("name", EXACT_SPARSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_sparse_spd_single_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_spd(n, rng)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(K.toarray(), b)

        solver = get_sparse_linear_solver(name)
        x = solver(K, b)

        assert x.shape == ref.shape
        assert_allclose(x, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", EXACT_SPARSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_sparse_nonsing_single_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_nonsing(n, rng)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(K.toarray(), b)

        solver = get_sparse_linear_solver(name)
        x = solver(K, b)

        assert x.shape == ref.shape
        assert_allclose(x, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", sorted(APPROX_SOLVERS))
    @pytest.mark.parametrize("n", SIZES)
    def test_approximate_sparse_single_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_spd(n, rng)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(K.toarray(), b)

        solver = get_sparse_linear_solver(name)
        x = solver(K, b)

        assert x.shape == ref.shape
        assert_allclose(x, ref, atol=TOL_APPROX, rtol=TOL_APPROX)


class TestSparseSolversMultiRHS:
    """Sparse solvers with multiple RHS columns (2-D)."""

    @pytest.mark.parametrize("name", EXACT_SPARSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_sparse_spd_multi_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_spd(n, rng)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(K.toarray(), B)

        solver = get_sparse_linear_solver(name)
        X = solver(K, B)

        assert X.shape == ref.shape
        assert_allclose(X, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", EXACT_SPARSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_sparse_nonsing_multi_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_nonsing(n, rng)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(K.toarray(), B)

        solver = get_sparse_linear_solver(name)
        X = solver(K, B)

        assert X.shape == ref.shape
        assert_allclose(X, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", sorted(APPROX_SOLVERS))
    @pytest.mark.parametrize("n", SIZES)
    def test_approximate_sparse_multi_rhs(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_spd(n, rng)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(K.toarray(), B)

        solver = get_sparse_linear_solver(name)
        X = solver(K, B)

        assert X.shape == ref.shape
        assert_allclose(X, ref, atol=TOL_APPROX, rtol=TOL_APPROX)


# =====================================================================
# Cross-consistency: dense and sparse solvers agree on the same system
# =====================================================================

class TestCrossConsistency:
    """Verify that all solvers (dense and sparse, native and wrapped)
    produce the same answer for an identical linear system."""

    @pytest.mark.parametrize("n", SIZES)
    def test_all_solvers_agree_single_rhs(self, n: int, rng: np.random.Generator) -> None:
        A_dense = _make_spd(n, rng)
        A_sparse = csc_matrix(A_dense)
        b = rng.standard_normal(n)
        ref = np.linalg.solve(A_dense, b)

        for name in ALL_DENSE_NAMES:
            solver = get_dense_linear_solver(name)
            x = solver(A_dense, b)
            tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
            assert_allclose(x, ref, atol=tol, rtol=tol,
                            err_msg=f"dense solver {name!r} disagrees")

        for name in ALL_SPARSE_NAMES:
            solver = get_sparse_linear_solver(name)
            x = solver(A_sparse, b)
            tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
            assert_allclose(x, ref, atol=tol, rtol=tol,
                            err_msg=f"sparse solver {name!r} disagrees")

    @pytest.mark.parametrize("n", SIZES)
    def test_all_solvers_agree_multi_rhs(self, n: int, rng: np.random.Generator) -> None:
        A_dense = _make_spd(n, rng)
        A_sparse = csc_matrix(A_dense)
        B = rng.standard_normal((n, N_RHS))
        ref = np.linalg.solve(A_dense, B)

        for name in ALL_DENSE_NAMES:
            solver = get_dense_linear_solver(name)
            X = solver(A_dense, B)
            tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
            assert_allclose(X, ref, atol=tol, rtol=tol,
                            err_msg=f"dense solver {name!r} disagrees")

        for name in ALL_SPARSE_NAMES:
            solver = get_sparse_linear_solver(name)
            X = solver(A_sparse, B)
            tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
            assert_allclose(X, ref, atol=tol, rtol=tol,
                            err_msg=f"sparse solver {name!r} disagrees")


# =====================================================================
# KKT-like saddle-point systems
# =====================================================================

class TestKKTLikeSystems:
    """Test on symmetric indefinite systems resembling KKT matrices:

        K = [[ P,  A^T ],
             [ A,   0  ]]

    These are the actual system shapes seen by the differentiators.
    """

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    def test_dense_kkt_system(self, name: str, rng: np.random.Generator) -> None:
        n_var, n_con = 10, 3
        P = _make_spd(n_var, rng)
        A = rng.standard_normal((n_con, n_var))
        K = np.block([
            [P, A.T],
            [A, np.zeros((n_con, n_con))],
        ])
        b = rng.standard_normal(n_var + n_con)
        ref = np.linalg.solve(K, b)

        solver = get_dense_linear_solver(name)
        x = solver(K, b)

        assert_allclose(x, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", EXACT_SPARSE_NAMES)
    def test_sparse_kkt_system(self, name: str, rng: np.random.Generator) -> None:
        n_var, n_con = 10, 3
        P = _make_spd(n_var, rng)
        A = rng.standard_normal((n_con, n_var))
        K_dense = np.block([
            [P, A.T],
            [A, np.zeros((n_con, n_con))],
        ])
        K_sparse = csc_matrix(K_dense)
        b = rng.standard_normal(n_var + n_con)
        ref = np.linalg.solve(K_dense, b)

        solver = get_sparse_linear_solver(name)
        x = solver(K_sparse, b)

        assert_allclose(x, ref, atol=TOL_EXACT, rtol=TOL_EXACT)

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    def test_dense_kkt_multi_rhs(self, name: str, rng: np.random.Generator) -> None:
        n_var, n_con = 15, 5
        P = _make_spd(n_var, rng)
        A = rng.standard_normal((n_con, n_var))
        K = np.block([
            [P, A.T],
            [A, np.zeros((n_con, n_con))],
        ])
        B = rng.standard_normal((n_var + n_con, N_RHS))
        ref = np.linalg.solve(K, B)

        solver = get_dense_linear_solver(name)
        X = solver(K, B)

        assert_allclose(X, ref, atol=TOL_EXACT, rtol=TOL_EXACT)


# =====================================================================
# Edge cases and error handling
# =====================================================================

class TestEdgeCases:
    """Edge cases: 1×1 systems, identity matrices, error paths."""

    @pytest.mark.parametrize("name", ALL_DENSE_NAMES)
    def test_1x1_dense(self, name: str) -> None:
        A = np.array([[3.0]])
        b = np.array([6.0])
        solver = get_dense_linear_solver(name)
        x = solver(A, b)
        tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
        assert_allclose(x, [2.0], atol=tol)

    @pytest.mark.parametrize("name", ALL_SPARSE_NAMES)
    def test_1x1_sparse(self, name: str) -> None:
        K = csc_matrix(np.array([[3.0]]))
        b = np.array([6.0])
        solver = get_sparse_linear_solver(name)
        x = solver(K, b)
        tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
        assert_allclose(x, [2.0], atol=tol)

    @pytest.mark.parametrize("name", ALL_DENSE_NAMES)
    def test_identity_dense(self, name: str, rng: np.random.Generator) -> None:
        n = 10
        A = np.eye(n)
        b = rng.standard_normal(n)
        solver = get_dense_linear_solver(name)
        x = solver(A, b)
        tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
        assert_allclose(x, b, atol=tol)

    @pytest.mark.parametrize("name", ALL_SPARSE_NAMES)
    def test_identity_sparse(self, name: str, rng: np.random.Generator) -> None:
        n = 10
        K = csc_matrix(np.eye(n))
        b = rng.standard_normal(n)
        solver = get_sparse_linear_solver(name)
        x = solver(K, b)
        tol = TOL_APPROX if name in APPROX_SOLVERS else TOL_EXACT
        assert_allclose(x, b, atol=tol)

    def test_unknown_dense_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown linear solver"):
            get_dense_linear_solver("bogus")

    def test_unknown_sparse_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown sparse linear solver"):
            get_sparse_linear_solver("bogus")


# =====================================================================
# Residual checks: ||Ax - b|| should be small
# =====================================================================

class TestResiduals:
    """Verify that the residual ||A x - b|| is small, as a
    solver-agnostic correctness check."""

    @pytest.mark.parametrize("name", EXACT_DENSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_dense_residual(self, name: str, n: int, rng: np.random.Generator) -> None:
        A = _make_nonsing(n, rng)
        b = rng.standard_normal(n)
        solver = get_dense_linear_solver(name)
        x = solver(A, b)
        residual = np.linalg.norm(A @ x - b)
        assert residual < TOL_EXACT * n

    @pytest.mark.parametrize("name", EXACT_SPARSE_NAMES)
    @pytest.mark.parametrize("n", SIZES)
    def test_sparse_residual(self, name: str, n: int, rng: np.random.Generator) -> None:
        K = _make_sparse_nonsing(n, rng)
        b = rng.standard_normal(n)
        solver = get_sparse_linear_solver(name)
        x = solver(K, b)
        residual = np.linalg.norm(K.toarray() @ x - b)
        assert residual < TOL_EXACT * n