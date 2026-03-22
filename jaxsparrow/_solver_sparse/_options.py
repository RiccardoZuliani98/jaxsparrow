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
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

class SparseKKTOptionsFull(DifferentiatorOptions, total=True):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]

DEFAULT_DIFF_OPTIONS: SparseKKTOptionsFull = {
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