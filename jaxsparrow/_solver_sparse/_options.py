"""
solver_sparse/_options.py
=========================
Differentiator and solver options specific to the sparse path.

Each differentiator backend has its own options class and defaults:

- ``"sparse_kkt"`` → :class:`SparseKKTDiffOptions` /
  ``DEFAULT_SPARSE_KKT_DIFF_OPTIONS``
- ``"sparse_dbd"`` → :class:`SparseDBDDiffOptions` /
  ``DEFAULT_SPARSE_DBD_DIFF_OPTIONS``

Each solver backend has its own options class and defaults:

- ``"qpsolvers"`` → :class:`SparseQpSolverOptions` /
  ``DEFAULT_SPARSE_QPSOLVERS_OPTIONS``
- ``"qoco"`` → :class:`SparseQOCOSolverOptions` /
  ``DEFAULT_SPARSE_QOCO_OPTIONS``

The ``"backend"`` field is declared on the common base classes
:class:`~jaxsparrow._options_common.DifferentiatorOptions` and
:class:`~jaxsparrow._options_common.SolverOptions` and selects
which backend implementation to use.  The ``"dtype"`` field is
common to all solver backends via :class:`SolverOptions`.
"""

from jaxsparrow._options_common import DifferentiatorOptions, SolverOptions
import numpy as np
from typing import Literal


# ======================================================================
# Differentiator options
# ======================================================================

class SparseKKTDiffOptions(DifferentiatorOptions):
    """Partial differentiator options for the ``sparse_kkt`` backend.

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_KKT_DIFF_OPTIONS`` via :func:`parse_options`.

    The ``backend`` field is inherited from
    :class:`DifferentiatorOptions`.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str


class SparseKKTDiffOptionsFull(DifferentiatorOptions, total=True):
    """Complete differentiator options for the ``sparse_kkt`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Differentiator backend name (``"sparse_kkt"``).
            Redeclared here to make it required in the resolved form.
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


DEFAULT_SPARSE_KKT_DIFF_OPTIONS: SparseKKTDiffOptionsFull = {
    "backend":       "sparse_kkt",
    "dtype":         np.float64,
    "bool_dtype":    np.bool_,
    "cst_tol":       1e-8,
    "linear_solver": "splu",
}


# ----------------------------------------------------------------------
# Sparse DBD backend options
# ----------------------------------------------------------------------

class SparseDBDDiffOptions(DifferentiatorOptions):
    """Partial differentiator options for the ``sparse_dbd`` backend.

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_DBD_DIFF_OPTIONS`` via :func:`parse_options`.

    The ``backend`` field is inherited from
    :class:`DifferentiatorOptions`.

    Attributes:
        rho: Regularisation strength for the DBD perturbation.
            Must be ``> 0``.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str
    rho:            float


class SparseDBDDiffOptionsFull(DifferentiatorOptions, total=True):
    """Complete differentiator options for the ``sparse_dbd`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Differentiator backend name (``"sparse_dbd"``).
            Redeclared here to make it required in the resolved form.
        dtype: NumPy floating-point dtype for all computations.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
        linear_solver: Name of the sparse linear solver backend.
        rho: Regularisation strength for the DBD perturbation
            (scalar ``> 0``).
    """
    backend:        str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]
    rho:            float


DEFAULT_SPARSE_DBD_DIFF_OPTIONS: SparseDBDDiffOptionsFull = {
    "backend":       "sparse_dbd",
    "dtype":         np.float64,
    "bool_dtype":    np.bool_,
    "cst_tol":       1e-8,
    "linear_solver": "splu",
    "rho":           1e-5,
}


# ----------------------------------------------------------------------
# Differentiator defaults registry
# ----------------------------------------------------------------------

DIFF_OPTIONS_DEFAULTS: dict[str, DifferentiatorOptions] = {
    "sparse_kkt": DEFAULT_SPARSE_KKT_DIFF_OPTIONS, #type: ignore
    "sparse_dbd": DEFAULT_SPARSE_DBD_DIFF_OPTIONS,
}
"""Look-up table used by the factory functions in
``_differentiators.py`` to select the correct defaults for the
chosen differentiator backend."""

DEFAULT_DIFF_BACKEND = "sparse_kkt"
"""Default differentiator backend when no ``"backend"`` key is
supplied in the differentiator options."""


# ======================================================================
# Solver options
# ======================================================================

# ----------------------------------------------------------------------
# qpsolvers backend options
# ----------------------------------------------------------------------

class SparseQpSolverOptions(SolverOptions):
    """Partial solver options for the ``qpsolvers`` backend (sparse).

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_QPSOLVERS_OPTIONS`` via :func:`parse_options`.

    The ``backend`` and ``dtype`` fields are inherited from
    :class:`SolverOptions`.

    Attributes:
        solver_name: Backend solver name passed to ``qpsolvers``
            (e.g. ``"piqp"``, ``"osqp"``, ``"clarabel"``).
    """
    solver_name:    str


class SparseQpSolverOptionsFull(SolverOptions, total=True):
    """Complete solver options for the ``qpsolvers`` backend (sparse).

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Solver backend protocol name (``"qpsolvers"``).
            Redeclared here to make it required in the resolved form.
        dtype: NumPy floating-point dtype for all arrays.
            Redeclared here to make it required in the resolved form.
        solver_name: Backend solver name passed to ``qpsolvers``.
    """
    backend:        str
    dtype:          type[np.floating]
    solver_name:    str


DEFAULT_SPARSE_QPSOLVERS_OPTIONS: SparseQpSolverOptionsFull = {
    "backend":      "qpsolvers",
    "dtype":        np.float64,
    "solver_name":  "piqp",
}


# ----------------------------------------------------------------------
# QOCO backend options
# ----------------------------------------------------------------------

class SparseQOCOSolverOptions(SolverOptions):
    """Partial solver options for the ``qoco`` backend (sparse).

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_QOCO_OPTIONS`` via :func:`parse_options`.

    The ``backend`` and ``dtype`` fields are inherited from
    :class:`SolverOptions`.

    Attributes:
        verbose: Verbosity level passed to QOCO (0 = silent).
        abstol: Absolute feasibility tolerance.
        reltol: Relative feasibility tolerance.
    """
    verbose:    int
    abstol:     float
    reltol:     float


class SparseQOCOSolverOptionsFull(SolverOptions, total=True):
    """Complete solver options for the ``qoco`` backend (sparse).

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Solver backend protocol name (``"qoco"``).
            Redeclared here to make it required in the resolved form.
        dtype: NumPy floating-point dtype for all arrays.
            Redeclared here to make it required in the resolved form.
        verbose: Verbosity level passed to QOCO (0 = silent).
        abstol: Absolute feasibility tolerance.
        reltol: Relative feasibility tolerance.
    """
    backend:    str
    dtype:      type[np.floating]
    verbose:    int
    abstol:     float
    reltol:     float


DEFAULT_SPARSE_QOCO_OPTIONS: SparseQOCOSolverOptionsFull = {
    "backend":  "qoco",
    "dtype":    np.float64,
    "verbose":  0,
    "abstol":   1e-7,
    "reltol":   1e-7,
}


# ----------------------------------------------------------------------
# Solver defaults registry
# ----------------------------------------------------------------------

SOLVER_OPTIONS_DEFAULTS: dict[str, SolverOptions] = {
    "qpsolvers": DEFAULT_SPARSE_QPSOLVERS_OPTIONS, #type: ignore
    "qoco":      DEFAULT_SPARSE_QOCO_OPTIONS,      #type: ignore
}
"""Look-up table used by the factory functions in ``_solvers.py``
to select the correct defaults for the chosen solver backend."""

DEFAULT_SOLVER_BACKEND = "qpsolvers"
"""Default solver backend when no ``"backend"`` key is supplied
in the solver options."""



# ======================================================================
# Full description of options to be passed to user through utility
# ======================================================================

# all options to be passed to the user
ALL_SPARSE_DIFF_OPTIONS = {
    "sparse_kkt": {
        "option": SparseKKTDiffOptions,
        "default": DEFAULT_SPARSE_KKT_DIFF_OPTIONS,
        "description": {
            "backend": "Differentiator backend name (fixed to 'sparse_kkt' in resolved form).",
            "dtype": "NumPy floating-point dtype for all computations.",
            "bool_dtype": "NumPy boolean dtype for active-set masks.",
            "cst_tol": "Tolerance for determining active inequality constraints.",
            "linear_solver": "Name of the sparse linear solver backend.",
        },
    },
    "sparse_dbd": {
        "option": SparseDBDDiffOptions,
        "default": DEFAULT_SPARSE_DBD_DIFF_OPTIONS,
        "description": {
            "backend": "Differentiator backend name (fixed to 'sparse_dbd' in resolved form).",
            "dtype": "NumPy floating-point dtype for all computations.",
            "bool_dtype": "NumPy boolean dtype for active-set masks.",
            "cst_tol": "Tolerance for determining active inequality constraints.",
            "linear_solver": "Name of the sparse linear solver backend.",
            "rho": "Regularisation strength for the DBD perturbation (> 0).",
        },
    },
}

# all options to be passed to user
ALL_SPARSE_SOLVER_OPTIONS = {
    "qpsolvers": {
        "option": SparseQpSolverOptions,
        "default": DEFAULT_SPARSE_QPSOLVERS_OPTIONS,
        "description": {
            "backend": "Solver backend protocol name (fixed to 'qpsolvers' in resolved form).",
            "dtype": "NumPy floating-point dtype for all arrays.",
            "solver_name": "Backend solver name passed to qpsolvers (e.g., 'piqp', 'osqp', 'clarabel').",
        },
    },
    "qoco": {
        "option": SparseQOCOSolverOptions,
        "default": DEFAULT_SPARSE_QOCO_OPTIONS,
        "description": {
            "backend": "Solver backend protocol name (fixed to 'qoco' in resolved form).",
            "dtype": "NumPy floating-point dtype for all arrays.",
            "verbose": "Verbosity level passed to QOCO (0 = silent).",
            "abstol": "Absolute feasibility tolerance for QOCO.",
            "reltol": "Relative feasibility tolerance for QOCO.",
        },
    },
}