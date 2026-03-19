# solver_dense/converters.py

import numpy as np
from numpy import ndarray
from jax import Array
import jax.numpy as jnp

from jaxsparrow._solver_common import EXPECTED_NDIM


def dense_primal_converter(key: str, val, dtype) -> ndarray:
    """JAX array → dense numpy, squeezing any batch-1 leading dim."""
    arr = np.asarray(val, dtype=dtype)
    if arr.ndim > EXPECTED_NDIM[key]:
        arr = arr[0]
    return arr


def dense_tangent_converter(key: str, val, dtype) -> ndarray:
    """JAX tangent → dense numpy, keeping batch dim if present."""
    return np.asarray(val, dtype=dtype)


def dense_grad_to_jax(key: str, val, dtype) -> Array:
    """Numpy gradient → JAX array."""
    return jnp.asarray(val, dtype=dtype)