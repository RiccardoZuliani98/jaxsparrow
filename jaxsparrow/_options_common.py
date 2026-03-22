from typing import TypedDict, Final, Literal
import jax.numpy as jnp

class SolverOptions(TypedDict, total=False):
    pass
class DifferentiatorOptions(TypedDict, total=False):
    pass

class ConstructorOptions(TypedDict, total=False):
    diff_mode: Literal["fwd","rev"]
    solver: SolverOptions | dict
    differentiator: DifferentiatorOptions | dict
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool
    fd_check: bool
    fd_eps: float

class ConstructorOptionsFull(TypedDict):
    diff_mode: Literal["fwd","rev"]
    solver:SolverOptions
    differentiator:DifferentiatorOptions
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool
    fd_check: bool
    fd_eps: float

DEFAULT_CONSTRUCTOR_OPTIONS: Final[ConstructorOptionsFull] = {
    "diff_mode": "fwd",
    "solver":{},
    "differentiator":{},
    "dtype": jnp.float64,
    "bool_dtype": jnp.bool_,
    "verbose": True,
    "debug": True,
    "fd_check": False,
    "fd_eps": 1e-6
}