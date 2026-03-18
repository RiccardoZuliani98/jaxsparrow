#TODO: docstring

from src.options_common import DifferentiatorOptions
import numpy as np

class DenseKKTfwdOptions(DifferentiatorOptions):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

class DenseKKTfwdOptionsFull(DifferentiatorOptions,total=True):
    dtype:          type[np.floating]
    bool_dtype:     type[np.bool]
    cst_tol:        float
    linear_solver:  str

DEFAULT_DIFF_OPTIONS : DenseKKTfwdOptionsFull = {
    "dtype": np.float64,
    "bool_dtype":np.bool_,
    "cst_tol": 1e-8,
    "linear_solver": "lu"
}