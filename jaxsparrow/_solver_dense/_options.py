"""
solver_dense/_options.py
========================
Differentiator and solver options for the dense path.

Defines backend-specific option TypedDicts and default values used by
the dense solver and differentiator factory functions.

Each differentiator backend has its own options class and defaults:

- ``"dense_kkt"`` → :class:`DenseKKTDiffOptions` /
  ``DEFAULT_DENSE_KKT_DIFF_OPTIONS``
- ``"dense_dbd"`` → :class:`DenseDBDDiffOptions` /
  ``DEFAULT_DENSE_DBD_DIFF_OPTIONS``

Each solver backend has its own options class and defaults:

- ``"qpsolvers"`` → :class:`DenseQpSolverOptions` /
  ``DEFAULT_DENSE_QPSOLVERS_OPTIONS``
- ``"qoco"`` → :class:`DenseQOCOSolverOptions` /
  ``DEFAULT_DENSE_QOCO_OPTIONS``
- ``"piqp"`` → :class:`DensePIQPSolverOptions` /
  ``DEFAULT_DENSE_PIQP_OPTIONS``

The ``"backend"`` and ``"dtype"`` fields are declared on the common
base classes :class:`~jaxsparrow._options_common.SolverOptions` and
:class:`~jaxsparrow._options_common.DifferentiatorOptions`.

The ``"linear_solver"`` field (differentiator) accepts any solver
name registered in :func:`get_dense_linear_solver`, including native
dense backends (``"solve"``, ``"lstsq"``, ``"lu"``) and sparse
backends available via automatic conversion (``"splu"``, ``"spilu"``,
``"spsolve"``, ``"sp_lstsq"``).
"""

from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._options_common import SolverOptions
import numpy as np
from typing import Literal, TypedDict
import os


# ======================================================================
# Differentiator options
# ======================================================================

# ----------------------------------------------------------------------
# Dense KKT backend options
# ----------------------------------------------------------------------

class DenseKKTDiffOptions(DifferentiatorOptions):
    """Partial differentiator options for the ``dense_kkt`` backend.

    All keys are optional; missing keys are filled from
    ``DEFAULT_DENSE_KKT_DIFF_OPTIONS`` via :func:`parse_options`.

    Attributes
    ----------
    backend: Differentiator backend name (``"dense_kkt"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the dense linear solver backend.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str


class DenseKKTDiffOptionsFull(TypedDict, total=True):
    """Complete differentiator options for the ``dense_kkt`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Differentiator backend name (``"dense_kkt"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the linear solver backend.  Accepts
        any key from the dense or sparse solver registries.
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


DEFAULT_DENSE_KKT_DIFF_OPTIONS: DenseKKTDiffOptionsFull = {
    "backend": "dense_kkt",
    "dump_failed":  False,
    "dump_dir":     "",
    "dtype": np.float64,
    "bool_dtype": np.bool_,
    "cst_tol": 1e-8,
    "linear_solver": "solve",
}


# ----------------------------------------------------------------------
# Dense DBD backend options
# ----------------------------------------------------------------------

class DenseDBDDiffOptions(DifferentiatorOptions):
    """Partial differentiator options for the ``dense_dbd`` backend.

    All keys are optional; missing keys are filled from
    ``DEFAULT_DENSE_DBD_DIFF_OPTIONS`` via :func:`parse_options`.

    The ``backend`` field is inherited from
    :class:`DifferentiatorOptions`.

    Attributes
    ----------
    backend: Differentiator backend name (``"dense_dbd"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the linear solver backend.
    rho: Regularisation strength for the DBD perturbation
        (scalar ``> 0``).
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str
    rho:            float


class DenseDBDDiffOptionsFull(TypedDict, total=True):
    """Complete differentiator options for the ``dense_dbd`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Differentiator backend name (``"dense_dbd"``).
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    dtype: NumPy floating-point dtype for all computations.
    bool_dtype: NumPy boolean dtype for active-set masks.
    cst_tol: Tolerance for determining active inequality
        constraints (``|G x - h| <= cst_tol``).
    linear_solver: Name of the linear solver backend.
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


DEFAULT_DENSE_DBD_DIFF_OPTIONS: DenseDBDDiffOptionsFull = {
    "backend": "dense_dbd",
    "dump_failed":  False,
    "dump_dir":     "",
    "dtype": np.float64,
    "bool_dtype": np.bool_,
    "cst_tol": 1e-8,
    "linear_solver": "solve",
    "rho": 1e-5,
}


# ----------------------------------------------------------------------
# Differentiator defaults registry
# ----------------------------------------------------------------------
DIFF_OPTIONS_DEFAULTS: dict[str, DifferentiatorOptions] = {
    "dense_kkt": DEFAULT_DENSE_KKT_DIFF_OPTIONS, #type: ignore
    "dense_dbd": DEFAULT_DENSE_DBD_DIFF_OPTIONS,
}
"""Look-up table used by the factory functions in
``_differentiators.py`` to select the correct defaults for the
chosen differentiator backend."""

DEFAULT_DIFF_BACKEND = "dense_kkt"
"""Default differentiator backend when no ``"backend"`` key is
supplied in the differentiator options."""


# ======================================================================
# Solver options
# ======================================================================

# ----------------------------------------------------------------------
# qpsolvers backend options
# ----------------------------------------------------------------------

class DenseQpSolverOptions(SolverOptions):
    """Partial solver options for the ``qpsolvers`` backend.

    All keys are optional; missing keys are filled from
    ``DEFAULT_DENSE_QPSOLVERS_OPTIONS`` via :func:`parse_options`.

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


class DenseQpSolverOptionsFull(TypedDict, total=True):
    """Complete solver options for the ``qpsolvers`` backend.

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


DEFAULT_DENSE_QPSOLVERS_OPTIONS: DenseQpSolverOptionsFull = {
    "backend": "qpsolvers",
    "dtype": np.float64,
    "dump_failed": False,
    "dump_dir":     "",
    "solver_name": "piqp",
}

# ----------------------------------------------------------------------
# PIQP backend options
# ----------------------------------------------------------------------

class DensePIQPSolverOptions(SolverOptions):
    """Partial solver options for the ``piqp`` backend (dense).

    All keys are optional; missing keys are filled from
    ``DEFAULT_DENSE_PIQP_OPTIONS`` via :func:`parse_options`.

    The ``backend``, ``dtype``, ``dump_failed``, and ``dump_dir``
    fields are inherited from :class:`SolverOptions`.

    Attributes
    ----------
    backend: Solver backend protocol name (``"piqp"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False). 
    dump_dir: directory where failed QPs are stored.
    verbose: Enable solver output.
    sparse: Whether to use the SparseSolver (True) or DenseSolver (False).
    """
    verbose:        bool
    sparse:         bool


class DensePIQPSolverOptionsFull(TypedDict, total=True):
    """Complete solver options for the ``piqp`` backend (dense).

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


DEFAULT_DENSE_PIQP_OPTIONS: DensePIQPSolverOptionsFull = {
    "backend":      "piqp",
    "dtype":        np.float64,
    "dump_failed":  False,
    "dump_dir":     "",
    "verbose":      False,
    "sparse":       False,
}


# ----------------------------------------------------------------------
# QOCO backend options
# ----------------------------------------------------------------------

class DenseQOCOSolverOptions(SolverOptions):
    """Partial solver options for the ``qoco`` backend (dense).

    All keys are optional; missing keys are filled from
    ``DEFAULT_DENSE_QOCO_OPTIONS`` via :func:`parse_options`.

    The ``backend``, ``dtype``, ``dump_failed``, and ``dump_dir``
    fields are inherited from :class:`SolverOptions`.

    Note: QOCO operates on sparse (CSC) matrices internally.
    When used with the dense path, the backend automatically
    converts dense matrices to CSC format before passing them
    to the solver.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qoco"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False).
    dump_dir: directory where failed QPs are stored.
    verbose: Verbosity level passed to QOCO (0 = silent).
    """
    verbose:        int


class DenseQOCOSolverOptionsFull(TypedDict, total=True):
    """Complete solver options for the ``qoco`` backend (dense).

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes
    ----------
    backend: Solver backend protocol name (``"qoco"``).
    dtype: NumPy floating-point dtype for all arrays.
    dump_failed: Ingredients of failed QPs are dumped to be analyzed (False).
    dump_dir: directory where failed QPs are stored.
    verbose: Verbosity level passed to QOCO (0 = silent).
    """
    backend:        str
    dtype:          type[np.floating]
    dump_failed:    bool
    dump_dir:       str | os.PathLike
    verbose:        int


DEFAULT_DENSE_QOCO_OPTIONS: DenseQOCOSolverOptionsFull = {
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
    "qpsolvers": DEFAULT_DENSE_QPSOLVERS_OPTIONS, #type: ignore
    "qoco":      DEFAULT_DENSE_QOCO_OPTIONS,      #type: ignore
    "piqp":      DEFAULT_DENSE_PIQP_OPTIONS,      #type: ignore
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
ALL_DENSE_DIFF_OPTIONS = {
    "dense_kkt": {
        "option": DenseKKTDiffOptions,
        "default": DEFAULT_DENSE_KKT_DIFF_OPTIONS,
        "description": {
            "backend": "Differentiator backend name (fixed to 'dense_kkt' in resolved form).",
            "dtype": "NumPy floating-point dtype for all computations.",
            "bool_dtype": "NumPy boolean dtype for active-set masks.",
            "cst_tol": "Tolerance for determining active inequality constraints (|G x - h| <= cst_tol).",
            "linear_solver": "Name of the linear solver backend. Accepts any key from the dense or sparse solver registries.",
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        }
    },
    "dense_dbd": {
        "option": DenseDBDDiffOptions,
        "default": DEFAULT_DENSE_DBD_DIFF_OPTIONS,
        "description": {
            "backend": "Differentiator backend name (fixed to 'dense_dbd' in resolved form).",
            "dtype": "NumPy floating-point dtype for all computations.",
            "bool_dtype": "NumPy boolean dtype for active-set masks.",
            "cst_tol": "Tolerance for determining active inequality constraints (|G x - h| <= cst_tol).",
            "linear_solver": "Name of the linear solver backend.",
            "rho": "Regularisation strength for the DBD perturbation (scalar > 0).",
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        }
    }
}

# all options to be passed to user
ALL_DENSE_SOLVER_OPTIONS = {
    "qpsolvers": {
        "option": DenseQpSolverOptions,
        "default": DEFAULT_DENSE_QPSOLVERS_OPTIONS,
        "description": {
            "backend": "Solver backend protocol name (fixed to 'qpsolvers' in resolved form).",
            "dtype": "NumPy floating-point dtype for all arrays.",
            "solver_name": "Backend solver name passed to qpsolvers (e.g. 'piqp', 'osqp', 'clarabel').",
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        }
    },
    "qoco": {
        "option": DenseQOCOSolverOptions,
        "default": DEFAULT_DENSE_QOCO_OPTIONS,
        "description": {
            "backend": "Solver backend protocol name (fixed to 'qoco' in resolved form).",
            "dtype": "NumPy floating-point dtype for all arrays.",
            "verbose": "Verbosity level passed to QOCO (0 = silent).",
            "dump_failed": "Ingredients of failed QPs are dumped to be analyzed (False).",
            "dump_dir": "Directory where dumped QPs are stored ('')."
        }
    },
    "piqp": {
        "option": DensePIQPSolverOptions,
        "default": DEFAULT_DENSE_PIQP_OPTIONS,
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