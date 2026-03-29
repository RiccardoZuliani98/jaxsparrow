"""
solver_dense/_options.py
========================
Differentiator and solver options for the dense path.

Defines backend-specific option TypedDicts and default values used by
:func:`create_dense_kkt_differentiator_fwd` and
:func:`create_dense_kkt_differentiator_rev`.

Each differentiator backend has its own options class and defaults:

- ``"dense_kkt"`` → :class:`DenseKKTDiffOptions` /
  ``DEFAULT_DENSE_KKT_DIFF_OPTIONS``
- ``"dense_dbd"`` → :class:`DenseDBDDiffOptions` /
  ``DEFAULT_DENSE_DBD_DIFF_OPTIONS``

The ``"backend"`` field is declared on the common base class
:class:`~jaxsparrow._options_common.DifferentiatorOptions` and
selects which backend implementation to use.

The ``"linear_solver"`` field accepts any solver name registered in
:func:`get_dense_linear_solver`, including native dense backends
(``"solve"``, ``"lstsq"``, ``"lu"``) and sparse backends available
via automatic conversion (``"splu"``, ``"spilu"``, ``"spsolve"``,
``"sp_lstsq"``).
"""

from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._options_common import SolverOptions
import numpy as np
from typing import Literal



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

    The ``backend`` field is inherited from
    :class:`DifferentiatorOptions`.

    Attributes:
        dtype: NumPy floating-point dtype for all computations.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
        linear_solver: Name of the linear solver backend.  Accepts
            any key from the dense or sparse solver registries.
    """
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

    The ``backend`` field is inherited from
    :class:`DifferentiatorOptions`.

    Attributes:
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
# Mapping from backend name to its defaults
# ----------------------------------------------------------------------

"""Look-up table used by the factory functions in
``_differentiators.py`` to select the correct defaults for the
chosen backend."""
DIFF_OPTIONS_DEFAULTS: dict[str, DifferentiatorOptions] = {
    "dense_kkt": DEFAULT_DENSE_KKT_DIFF_OPTIONS,
    "dense_dbd": DEFAULT_DENSE_DBD_DIFF_OPTIONS,
}

# Default backend when no "backend" key is supplied in options
DEFAULT_DIFF_BACKEND = "dense_kkt"



# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------

class DenseSolverOptions(SolverOptions):
    """Partial solver options for the dense path.

    All keys are optional; missing keys are filled from
    ``DEFAULT_SOLVER_OPTIONS`` via :func:`parse_options`.

    Attributes:
        solver_name: Backend solver name passed to ``qpsolvers``
            (e.g. ``"piqp"``, ``"osqp"``, ``"clarabel"``).
        dtype: NumPy floating-point dtype for all arrays.
        backend: Solver backend protocol name. Currently only
            ``"qpsolvers"`` is supported.
    """
    solver_name:    str
    dtype:          type[np.floating]
    backend:        Literal["qpsolvers"]


class DenseSolverOptionsFull(SolverOptions, total=True):
    """Complete solver options for the dense path.

    Attributes:
        solver_name: Backend solver name passed to ``qpsolvers``
            (e.g. ``"piqp"``, ``"osqp"``, ``"clarabel"``).
        dtype: NumPy floating-point dtype for all arrays.
        backend: Solver backend protocol name (``"qpsolvers"``).
    """
    solver_name:    str
    dtype:          type[np.floating]
    backend:        Literal["qpsolvers"]


DEFAULT_SOLVER_OPTIONS: DenseSolverOptionsFull = {
    "solver_name": "piqp",
    "dtype": np.float64,
    "backend": "qpsolvers",
}