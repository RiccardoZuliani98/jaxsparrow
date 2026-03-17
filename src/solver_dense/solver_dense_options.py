"""
Every solver will have its own options and default options,
so it makes sense to store options separately from the rest.
Should we include differentiator options here?
This will lead to very different architectures:

If we choose to have solvers and differentiators as a unique thing,
then every time we choose a different solver / differentiator, 
everything has to change.

In my opinion we should have a skeleton that works e.g. sparse scipy/numpy, 
dense numpy, dense jax, sparse jax. That's it. Then the differentiator and 
the solver should be added on top and not treated as fixed.

This means that it's ok to have a "dense_numpy_solver" folder, but then the
actual solver can be retrieved from a list.

We should have "suboptions", meaning dictionaries inside the dictionary 
below that will be passed to the solver to set up its options.

Similar for the differentiator.
"""

#TODO: docstring

from typing import TypedDict, Final
import jax.numpy as jnp
import numpy as np

class SolverOptions(TypedDict, total=False):
    differentiator_type: str
    solver_type: str
    solver:dict
    differentiator:dict
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool

class SolverOptionsFull(TypedDict):
    differentiator_type: str
    solver_type: str
    solver:dict
    differentiator:dict
    dtype: jnp.dtype
    bool_dtype: jnp.dtype
    verbose: bool
    debug: bool

DEFAULT_SOLVER_OPTIONS: Final[SolverOptionsFull] = {
    "differentiator_type": "kkt_fwd",
    "solver_type": "qp_solvers",
    "solver":{},
    "differentiator":{"dtype":np.float64},
    "dtype": jnp.float64,
    "bool_dtype": jnp.bool_,
    "verbose": True,
    "debug": True
}