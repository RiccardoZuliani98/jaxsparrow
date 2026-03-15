from __future__ import annotations

from typing import Final, TypedDict, Dict, Tuple, TypeAlias
from jax.experimental.sparse import BCOO
from jax import Array
from numpy import ndarray
import jax.numpy as jnp


class SolverOptions(TypedDict, total=False):
    jac_tol: float
    sparse: bool
    solver: str
    dtype: jnp.dtype


class SolverOptionsFull(TypedDict):
    jac_tol: float
    sparse: bool
    solver: str
    dtype: jnp.dtype


DEFAULT_SOLVER_OPTIONS: Final[SolverOptionsFull] = {
    "jac_tol": 1e-8,
    "sparse": True,
    "solver": "daqp",
    "dtype": jnp.float64
}

#: Global sparsity pattern in NumPy indexing format (row indices, column indices).
SparsityPattern: TypeAlias = Dict[str, Tuple[ndarray, ndarray]]

#: Input dictionary mapping variable names to JAX arrays.
ArrayIn: TypeAlias = Dict[str, Array]

#: Dense QP ingredients:
#:
#:   A_eq   : (n_eq, n_var) equality lhs (Array)
#:   rhs_eq : (n_eq,)       equality rhs (Array)
#:   A_in   : (n_in, n_var) inequality lhs (Array)
#:   rhs_in : (n_in,)       inequality rhs (Array)
#:   Q      : (n_var, n_var) Hessian (Array)
#:   q      : (n_var,)       linear term (Array)
#:   c      : () or (1,)     constant scalar (Array)
DenseProblemIngredients: TypeAlias = Tuple[
    Array,  # A_eq
    Array,  # rhs_eq
    Array,  # A_in
    Array,  # rhs_in
    Array,  # Q
    Array,  # q
    Array,  # c
]

#: Sparse QP ingredients:
#:
#:   A_eq   : (n_eq, n_var) equality lhs (BCOO)
#:   rhs_eq : (n_eq,)       equality rhs (Array)
#:   A_in   : (n_in, n_var) inequality lhs (BCOO)
#:   rhs_in : (n_in,)       inequality rhs (Array)
#:   Q      : (n_var, n_var) Hessian (BCOO)
#:   q      : (n_var,)       linear term (Array)
#:   c      : () or (1,)     constant scalar (Array)
SparseProblemIngredients: TypeAlias = Tuple[
    BCOO,   # A_eq
    Array,  # rhs_eq
    BCOO,   # A_in
    Array,  # rhs_in
    BCOO,   # Q
    Array,  # q
    Array,  # c
]