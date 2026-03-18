# In solver_dense/linear_solvers.py (or even just at the top of differentiators.py)

from typing import Callable
from numpy import ndarray
import numpy as np
from scipy.linalg import lu_factor, lu_solve

LinearSolver = Callable[[ndarray, ndarray], ndarray]

def _lstsq(a: ndarray, b: ndarray) -> ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0]

def _solve(a: ndarray, b: ndarray) -> ndarray:
    return np.linalg.solve(a, b)

def _lu_solve(a: ndarray, b: ndarray) -> ndarray:
    lu, piv = lu_factor(a)
    return lu_solve((lu, piv), b)

LINEAR_SOLVERS: dict[str, LinearSolver] = {
    "lstsq": _lstsq,
    "solve": _solve,
    "lu": _lu_solve,
}

def get_linear_solver(name: str) -> LinearSolver:
    if name not in LINEAR_SOLVERS:
        raise ValueError(
            f"Unknown linear solver: {name!r}. "
            f"Available: {sorted(LINEAR_SOLVERS)}."
        )
    return LINEAR_SOLVERS[name]