"""
solver_sparse/options.py
========================
Differentiator and solver options specific to the sparse path.
"""

from jaxsparrow._options_common import DifferentiatorOptions, SolverOptions
import numpy as np
from typing import Literal


# ── Differentiator options ───────────────────────────────────────────

class SparseKKTOptions(DifferentiatorOptions):
    """Partial differentiator options for the sparse KKT path.

    All keys are optional; missing keys are filled from
    ``DEFAULT_DIFF_OPTIONS`` via :func:`parse_options`.
    """
    backend:        str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

class SparseKKTOptionsFull(DifferentiatorOptions, total=True):
    """Complete differentiator options for the sparse KKT path.

    All keys are required. This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Differentiator backend name. Controls which
            :class:`DifferentiatorBackend` implementation is used.
            Default is ``"kkt"`` (sparse KKT via SuperLU).
        dtype: NumPy floating-point dtype for all computations.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
        linear_solver: Name of the sparse linear solver backend.
    """
    backend:        str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]

DEFAULT_DIFF_OPTIONS: SparseKKTOptionsFull = {
    "backend":       "kkt",
    "dtype":         np.float64,
    "bool_dtype":    np.bool_,
    "cst_tol":       1e-8,
    "linear_solver": "splu",
}


# ── Solver options ───────────────────────────────────────────────────

class SparseSolverOptions(SolverOptions):
    solver_name:    str
    dtype:          type[np.floating]

class SparseSolverOptionsFull(SolverOptions, total=True):
    solver_name:    str
    dtype:          type[np.floating]

DEFAULT_SOLVER_OPTIONS: SparseSolverOptionsFull = {
    "solver_name":  "piqp",
    "dtype":        np.float64,
}