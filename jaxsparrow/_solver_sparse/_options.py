"""
solver_sparse/_options.py
=========================
Differentiator and solver options specific to the sparse path.

Each differentiator backend has its own options class and defaults:

- ``"sparse_kkt"`` → :class:`SparseKKTDiffOptions` /
  ``DEFAULT_SPARSE_KKT_DIFF_OPTIONS``

Each solver backend has its own options class and defaults:

- ``"qpsolvers"`` → :class:`SparseQpSolverOptions` /
  ``DEFAULT_SPARSE_QPSOLVERS_OPTIONS``

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
# Differentiator defaults registry
# ----------------------------------------------------------------------

DIFF_OPTIONS_DEFAULTS: dict[str, DifferentiatorOptions] = {
    "sparse_kkt": DEFAULT_SPARSE_KKT_DIFF_OPTIONS,
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
# Solver defaults registry
# ----------------------------------------------------------------------

SOLVER_OPTIONS_DEFAULTS: dict[str, SolverOptions] = {
    "qpsolvers": DEFAULT_SPARSE_QPSOLVERS_OPTIONS,
}
"""Look-up table used by the factory functions in ``_solvers.py``
to select the correct defaults for the chosen solver backend."""

DEFAULT_SOLVER_BACKEND = "qpsolvers"
"""Default solver backend when no ``"backend"`` key is supplied
in the solver options."""


# ----------------------------------------------------------------------
# Legacy aliases
# ----------------------------------------------------------------------

SparseSolverOptions = SparseQpSolverOptions
SparseSolverOptionsFull = SparseQpSolverOptionsFull
DEFAULT_DIFF_OPTIONS = DEFAULT_SPARSE_KKT_DIFF_OPTIONS
DEFAULT_SOLVER_OPTIONS = DEFAULT_SPARSE_QPSOLVERS_OPTIONS
