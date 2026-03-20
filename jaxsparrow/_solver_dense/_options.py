"""
solver_dense/_options.py
========================
Differentiator options for the dense QP path.

Defines the option TypedDicts and default values used by
:func:`create_dense_kkt_differentiator_fwd` and
:func:`create_dense_kkt_differentiator_rev`.

The ``"linear_solver"`` field accepts any solver name registered in
:func:`get_dense_linear_solver`, including native dense backends
(``"solve"``, ``"lstsq"``, ``"lu"``) and sparse backends available
via automatic conversion (``"splu"``, ``"spilu"``, ``"spsolve"``,
``"sp_lstsq"``).
"""

from jaxsparrow._options_common import DifferentiatorOptions
import numpy as np
from typing import Literal


class DenseKKTfwdOptions(DifferentiatorOptions):
    """Partial differentiator options for the dense KKT path.

    All keys are optional; missing keys are filled from
    ``DEFAULT_DIFF_OPTIONS`` via :func:`parse_options`.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str


class DenseKKTfwdOptionsFull(DifferentiatorOptions, total=True):
    """Complete differentiator options for the dense KKT path.

    All keys are required. This is the resolved form after merging
    user-supplied options with defaults.

    Attributes:
        dtype: NumPy floating-point dtype for all computations.
        bool_dtype: NumPy boolean dtype for active-set masks.
        cst_tol: Tolerance for determining active inequality
            constraints (``|G x - h| <= cst_tol``).
        linear_solver: Name of the linear solver backend. Accepts
            any key from the dense or sparse solver registries.
    """
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]


DEFAULT_DIFF_OPTIONS: DenseKKTfwdOptionsFull = {
    "dtype": np.float64,
    "bool_dtype": np.bool_,
    "cst_tol": 1e-8,
    "linear_solver": "solve",
}
