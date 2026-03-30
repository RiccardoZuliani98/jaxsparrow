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
from typing import Literal


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

    The ``backend`` field is inherited from
    :class:`DifferentiatorOptions`.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str


class DenseKKTDiffOptionsFull(DifferentiatorOptions, total=True):
    """Complete differentiator options for the ``dense_kkt`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Differentiator backend name (``"dense_kkt"``).
            Redeclared here to make it required in the resolved form.
        dtype: NumPy floating-point dtype for all computations.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
        linear_solver: Name of the linear solver backend.  Accepts
            any key from the dense or sparse solver registries.
    """
    backend:        str
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]


DEFAULT_DENSE_KKT_DIFF_OPTIONS: DenseKKTDiffOptionsFull = {
    "backend": "dense_kkt",
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

    Attributes:
        rho: Regularisation strength for the DBD perturbation.
            Must be ``> 0``.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str
    rho:            float


class DenseDBDDiffOptionsFull(DifferentiatorOptions, total=True):
    """Complete differentiator options for the ``dense_dbd`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Differentiator backend name (``"dense_dbd"``).
            Redeclared here to make it required in the resolved form.
        dtype: NumPy floating-point dtype for all computations.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
        linear_solver: Name of the linear solver backend.
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


DEFAULT_DENSE_DBD_DIFF_OPTIONS: DenseDBDDiffOptionsFull = {
    "backend": "dense_dbd",
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
    "dense_kkt": DEFAULT_DENSE_KKT_DIFF_OPTIONS,
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

    The ``backend`` and ``dtype`` fields are inherited from
    :class:`SolverOptions`.

    Attributes:
        solver_name: Backend solver name passed to ``qpsolvers``
            (e.g. ``"piqp"``, ``"osqp"``, ``"clarabel"``).
    """
    solver_name:    str


class DenseQpSolverOptionsFull(SolverOptions, total=True):
    """Complete solver options for the ``qpsolvers`` backend.

    All keys are required.  This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        backend: Solver backend protocol name (``"qpsolvers"``).
            Redeclared here to make it required in the resolved form.
        dtype: NumPy floating-point dtype for all arrays.
            Redeclared here to make it required in the resolved form.
        solver_name: Backend solver name passed to ``qpsolvers``
            (e.g. ``"piqp"``, ``"osqp"``, ``"clarabel"``).
    """
    backend:        str
    dtype:          type[np.floating]
    solver_name:    str


DEFAULT_DENSE_QPSOLVERS_OPTIONS: DenseQpSolverOptionsFull = {
    "backend": "qpsolvers",
    "dtype": np.float64,
    "solver_name": "piqp",
}


# ----------------------------------------------------------------------
# Solver defaults registry
# ----------------------------------------------------------------------

SOLVER_OPTIONS_DEFAULTS: dict[str, SolverOptions] = {
    "qpsolvers": DEFAULT_DENSE_QPSOLVERS_OPTIONS,
}
"""Look-up table used by the factory functions in ``_solvers.py``
to select the correct defaults for the chosen solver backend."""

DEFAULT_SOLVER_BACKEND = "qpsolvers"
"""Default solver backend when no ``"backend"`` key is supplied
in the solver options."""