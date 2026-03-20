#TODO: docstring

from jaxsparrow._options_common import DifferentiatorOptions
import numpy as np
from typing import Literal

class DenseKKTfwdOptions(DifferentiatorOptions):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

class DenseKKTfwdOptionsFull(DifferentiatorOptions,total=True):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  Literal[
                        "splu", "spilu", "spsolve", "lu",
                        "sp_lstsq", "lstsq", "solve"
                    ]

DEFAULT_DIFF_OPTIONS : DenseKKTfwdOptionsFull = {
    "dtype": np.float64,
    "bool_dtype":np.bool_,
    "cst_tol": 1e-8,
    "linear_solver": "solve"
}