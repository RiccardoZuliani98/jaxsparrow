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
from typing import Literal, TypedDict
import os

# ======================================================================
# Differentiator options
# ======================================================================

class SparseKKTDiffOptions(DifferentiatorOptions):
    """Partial differentiator options for the ``sparse_kkt`` backend.

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_KKT_DIFF_OPTIONS`` via :func:`parse_options`.

    backend: Differentiator backend name (``"sparse_kkt"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the sparse linear solver backend.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str


class SparseKKTDiffOptionsFull(TypedDict, total=True):
    """Complete differentiator options for the ``sparse_kkt`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Differentiator backend name (``"sparse_kkt"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the sparse linear solver backend.
    """
    backend:        str
    dump_failed:    bool
    dump_dir:       str | os.PathLike
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]


DEFAULT_SPARSE_KKT_DIFF_OPTIONS: SparseKKTDiffOptionsFull = {
    "backend":       "sparse_kkt",
    "dump_failed":  False,
    "dump_dir":     "",
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

    Attributes
    ----------
    backend: Differentiator backend name (``"sparse_dbd"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the sparse linear solver backend.
    rho: Regularisation strength for the DBD perturbation
        (scalar ``> 0``).
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str
    rho:            float


class SparseDBDDiffOptionsFull(TypedDict, total=True):
    """Complete differentiator options for the ``sparse_dbd`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Differentiator backend name (``"sparse_dbd"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the sparse linear solver backend.
    rho: Regularisation strength for the DBD perturbation
        (scalar ``> 0``).
    """
    backend:        str
    dump_failed:    bool
    dump_dir:       str | os.PathLike
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
    "dump_failed":  False,
    "dump_dir":     "",
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

    The ``backend``, ``dtype``, ``dump_failed``, and ``dump_dir``
    fields are inherited from :class:`SolverOptions`.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qpsolvers"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    solver_name: Backend solver name passed to ``qpsolvers``.
    """
    solver_name:    str


class SparseQpSolverOptionsFull(TypedDict, total=True):
    """Complete solver options for the ``qpsolvers`` backend (sparse).

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qpsolvers"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    solver_name: Backend solver name passed to ``qpsolvers``.
    """
    backend:        str
    dtype:          type[np.floating]
    dump_failed:    bool
    dump_dir:       str | os.PathLike
    solver_name:    str


DEFAULT_SPARSE_QPSOLVERS_OPTIONS: SparseQpSolverOptionsFull = {
    "backend":      "qpsolvers",
    "dtype":        np.float64,
    "dump_failed":  False,
    "dump_dir":     "",
    "solver_name":  "piqp",
}

# ----------------------------------------------------------------------
# PIQP backend options
# ----------------------------------------------------------------------

class SparsePIQPSolverOptions(SolverOptions):
    """Partial solver options for the ``piqp`` backend (sparse).

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_PIQP_OPTIONS`` via :func:`parse_options`.

    The ``backend``, ``dtype``, ``dump_failed``, and ``dump_dir``
    fields are inherited from :class:`SolverOptions`.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qpsolvers"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    verbose: Enable solver output.
    sparse: Whether to use the SparseSolver (True) or DenseSolver (False).
    """
    verbose:    bool
    sparse:     bool


class SparsePIQPSolverOptionsFull(TypedDict, total=True):
    """Complete solver options for the ``piqp`` backend (sparse).

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Solver backend protocol name (``"piqp"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False).
    dump_dir: directory where failed QPs are stored.
    verbose: Enable solver output.
    sparse: Whether to use the SparseSolver (True) or DenseSolver (False).
    """
    backend:        str
    dtype:          type[np.floating]
    dump_failed:    bool
    dump_dir:       str | os.PathLike
    verbose:        bool
    sparse:         bool


DEFAULT_SPARSE_PIQP_OPTIONS: SparsePIQPSolverOptionsFull = {
    "backend":  "piqp",
    "dtype":    np.float64,
    "dump_failed": False,
    "dump_dir":     "",
    "verbose":  False,
    "sparse":   True,
}

# ----------------------------------------------------------------------
# QOCO backend options
# ----------------------------------------------------------------------

class SparseQOCOSolverOptions(SolverOptions):
    """Partial solver options for the ``qoco`` backend (sparse).

    All keys are optional; missing keys are filled from
    ``DEFAULT_SPARSE_QOCO_OPTIONS`` via :func:`parse_options`.

    The ``backend``, ``dtype``, ``dump_failed``, and ``dump_dir``
    fields are inherited from :class:`SolverOptions`.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qoco"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False).
    dump_dir: directory where failed QPs are stored.
    verbose: Verbosity level passed to QOCO (0 = silent).
    """
    verbose:    int


class SparseQOCOSolverOptionsFull(TypedDict, total=True):
    """Complete solver options for the ``qoco`` backend (sparse).

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qoco"``).
    dtype: NumPy floating-point dtype for all arrays.
    verbose: Verbosity level passed to QOCO (0 = silent).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False).
    dump_dir: directory where failed QPs are stored.
    """
    backend:        str
    dtype:          type[np.floating]
    dump_failed:    bool
    dump_dir:       str | os.PathLike
    verbose:        int


DEFAULT_SPARSE_QOCO_OPTIONS: SparseQOCOSolverOptionsFull = {
    "backend":      "qoco",
    "dtype":        np.float64,
    "dump_failed":  False,
    "dump_dir":     "",
    "verbose":      0,
}

# ----------------------------------------------------------------------
# Solver defaults registry
# ----------------------------------------------------------------------

SOLVER_OPTIONS_DEFAULTS: dict[str, SolverOptions] = {
    "qpsolvers": DEFAULT_SPARSE_QPSOLVERS_OPTIONS, #type: ignore
    "qoco":      DEFAULT_SPARSE_QOCO_OPTIONS,      #type: ignore
    "piqp":      DEFAULT_SPARSE_PIQP_OPTIONS,      #type: ignore
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
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
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
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
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
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        },
    },
    "qoco": {
        "option": SparseQOCOSolverOptions,
        "default": DEFAULT_SPARSE_QOCO_OPTIONS,
        "description": {
            "backend": "Solver backend protocol name (fixed to 'qoco' in resolved form).",
            "dtype": "NumPy floating-point dtype for all arrays.",
            "verbose": "Verbosity level passed to QOCO (0 = silent).",
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        },
    },
    "piqp": {
        "option": SparsePIQPSolverOptions,
        "default": DEFAULT_SPARSE_PIQP_OPTIONS,
        "description": {
            "backend": "Solver backend protocol name (fixed to 'piqp' in resolved form).",
            "dtype": "NumPy floating-point dtype for all arrays.",
            "verbose": "Whether to enable solver output.",
            "sparse": "Flag to use PIQP's sparse solver (True) or dense solver (False).",
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        },
    },
}