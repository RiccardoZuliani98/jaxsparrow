"""
solver_sparse/options.py
========================
Differentiator and solver options specific to the sparse path.
"""

from jaxsparrow._options_common import DifferentiatorOptions, SolverOptions
import numpy as np


# ── Differentiator options ───────────────────────────────────────────

class SparseKKTOptions(DifferentiatorOptions):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

class SparseKKTOptionsFull(DifferentiatorOptions, total=True):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

DEFAULT_DIFF_OPTIONS: SparseKKTOptionsFull = {
    "dtype":         np.float64,
    "bool_dtype":    np.bool_,
    "cst_tol":       1e-8,
    "linear_solver": "solve",
}


# ── Solver options ───────────────────────────────────────────────────

class SparseQPSolverOptions(SolverOptions):
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float

class SparseQPSolverOptionsFull(SolverOptions, total=True):
    solver_name:    str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float

DEFAULT_SOLVER_OPTIONS: SparseQPSolverOptionsFull = {
    "solver_name":  "piqp",
    "dtype":        np.float64,
    "bool_dtype":   np.bool_,
    "cst_tol":      1e-8,
}
