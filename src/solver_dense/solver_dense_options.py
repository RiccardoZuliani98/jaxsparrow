#TODO: docstring

from typing import TypedDict, Final
import jax.numpy as jnp

class SolverOptions(TypedDict, total=False):
    pass
class DifferentiatorOptions(TypedDict, total=False):
    pass

class ConstructorOptions(TypedDict, total=False):
    differentiator_type: str
    solver_type: str
    solver:SolverOptions
    differentiator:DifferentiatorOptions
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool
    fd_check: bool

class ConstructorOptionsFull(TypedDict):
    differentiator_type: str
    solver_type: str
    solver:SolverOptions
    differentiator:DifferentiatorOptions
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool
    fd_check: bool

DEFAULT_CONSTRUCTOR_OPTIONS: Final[ConstructorOptionsFull] = {
    "differentiator_type": "kkt_fwd",
    "solver_type": "qp_solvers",
    "solver":{},
    "differentiator":{},
    "dtype": jnp.float64,
    "bool_dtype": jnp.bool_,
    "verbose": True,
    "debug": True,
    "fd_check": False
}