"""
utils/_linear_solvers.py
========================
Dense and sparse linear solver backends for KKT differentiators.

Each path (dense / sparse) has its own registry and lookup function.
Solvers from the *other* registry are available via automatic
conversion wrappers, so the user can request e.g. ``"splu"`` from
the dense path — the wrapper will convert the dense LHS to CSC
before calling the sparse solver.

Public API
----------
- ``get_dense_linear_solver(name)``  → ``(ndarray,  ndarray) -> ndarray``
- ``get_sparse_linear_solver(name)`` → ``(csc_matrix, ndarray) -> ndarray``
"""

from typing import Callable
from numpy import ndarray
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve, lsqr, spilu

# ── Type aliases ─────────────────────────────────────────────────────

DenseLinearSolver  = Callable[[ndarray,     ndarray], ndarray]
SparseLinearSolver = Callable[[csc_matrix,  ndarray], ndarray]


# =====================================================================
# Native dense solvers
# =====================================================================

def _dense_lstsq(a: ndarray, b: ndarray) -> ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0]

def _dense_solve(a: ndarray, b: ndarray) -> ndarray:
    return np.linalg.solve(a, b)

def _dense_lu(a: ndarray, b: ndarray) -> ndarray:
    lu, piv = lu_factor(a)
    return lu_solve((lu, piv), b)


# =====================================================================
# Native sparse solvers
# =====================================================================

def _sparse_splu(K: csc_matrix, rhs: ndarray) -> ndarray:
    """Direct solve via SuperLU factorization."""
    lu = splu(K)
    if rhs.ndim == 1:
        return lu.solve(rhs)
    return np.column_stack([lu.solve(rhs[:, i]) for i in range(rhs.shape[1])])

def _sparse_spilu(K: csc_matrix, rhs: ndarray) -> ndarray:
    """Incomplete LU factorization (approximate)."""
    lu = spilu(K)
    if rhs.ndim == 1:
        return lu.solve(rhs)
    return np.column_stack([lu.solve(rhs[:, i]) for i in range(rhs.shape[1])])

def _sparse_spsolve(K: csc_matrix, rhs: ndarray) -> ndarray:
    """Sparse direct solve via ``scipy.sparse.linalg.spsolve``."""
    if rhs.ndim == 1:
        return spsolve(K, rhs)
    return np.column_stack([spsolve(K, rhs[:, i]) for i in range(rhs.shape[1])])

def _sparse_lstsq(K: csc_matrix, rhs: ndarray) -> ndarray:
    """Least-squares solve via ``scipy.sparse.linalg.lsqr``."""
    if rhs.ndim == 1:
        return lsqr(K, rhs)[0]
    return np.column_stack([lsqr(K, rhs[:, i])[0] for i in range(rhs.shape[1])])


# =====================================================================
# Registries
# =====================================================================

_DENSE_NATIVE: dict[str, DenseLinearSolver] = {
    "lstsq": _dense_lstsq,
    "solve": _dense_solve,
    "lu":    _dense_lu,
}

_SPARSE_NATIVE: dict[str, SparseLinearSolver] = {
    "splu":    _sparse_splu,
    "spilu":   _sparse_spilu,
    "spsolve": _sparse_spsolve,
    "sp_lstsq": _sparse_lstsq,
}


# =====================================================================
# Conversion wrappers
# =====================================================================

def _wrap_sparse_for_dense(sp_solver: SparseLinearSolver) -> DenseLinearSolver:
    """Wrap a sparse solver so it accepts a dense LHS (ndarray → CSC)."""
    def wrapper(a: ndarray, b: ndarray) -> ndarray:
        return sp_solver(csc_matrix(a), b)
    return wrapper

def _wrap_dense_for_sparse(d_solver: DenseLinearSolver) -> SparseLinearSolver:
    """Wrap a dense solver so it accepts a sparse LHS (CSC → ndarray)."""
    def wrapper(K: csc_matrix, b: ndarray) -> ndarray:
        return d_solver(K.toarray(), b)
    return wrapper


# =====================================================================
# Public lookup functions
# =====================================================================

def get_dense_linear_solver(name: str) -> DenseLinearSolver:
    """Get a linear solver that operates on dense LHS.

    Looks up *name* in the native dense registry first, then falls
    back to wrapping a sparse solver (dense → CSC conversion).

    Available native:  ``"solve"``, ``"lstsq"``, ``"lu"``
    Available wrapped: ``"splu"``, ``"spilu"``, ``"spsolve"``, ``"sp_lstsq"``
    """
    if name in _DENSE_NATIVE:
        return _DENSE_NATIVE[name]
    if name in _SPARSE_NATIVE:
        return _wrap_sparse_for_dense(_SPARSE_NATIVE[name])
    raise ValueError(
        f"Unknown linear solver: {name!r}. "
        f"Available: {sorted({**_DENSE_NATIVE, **_SPARSE_NATIVE})}."
    )


def get_sparse_linear_solver(name: str) -> SparseLinearSolver:
    """Get a linear solver that operates on sparse CSC LHS.

    Looks up *name* in the native sparse registry first, then falls
    back to wrapping a dense solver (CSC → dense conversion).

    Available native:  ``"splu"``, ``"spilu"``, ``"spsolve"``, ``"sp_lstsq"``
    Available wrapped: ``"solve"``, ``"lstsq"``, ``"lu"``
    """
    if name in _SPARSE_NATIVE:
        return _SPARSE_NATIVE[name]
    if name in _DENSE_NATIVE:
        return _wrap_dense_for_sparse(_DENSE_NATIVE[name])
    raise ValueError(
        f"Unknown sparse linear solver: {name!r}. "
        f"Available: {sorted({**_SPARSE_NATIVE, **_DENSE_NATIVE})}."
    )