"""
solver_dense/_converters.py
===========================
Conversion functions between JAX arrays and NumPy arrays for the
dense QP pipeline.

These are called by the generic solver/differentiator framework to
move data across the JAX ↔ NumPy boundary:

- **primal converter**: JAX → NumPy, squeezing spurious batch-1 dims.
- **tangent converter**: JAX → NumPy, preserving the batch dimension.
- **grad-to-JAX converter**: NumPy → JAX, for returning gradients.
"""

import numpy as np
from numpy import ndarray
from jax import Array
import jax.numpy as jnp

from jaxsparrow._solver_common import EXPECTED_NDIM


def dense_primal_converter(key: str, val: Array, dtype: type[np.floating]) -> ndarray:
    """Convert a JAX primal array to a dense NumPy array.

    Squeezes a leading batch-1 dimension if the resulting ``ndim``
    exceeds the expected rank for *key* (defined in
    ``EXPECTED_NDIM``). This handles the case where ``vmap`` adds a
    unit batch dimension to a non-batched parameter.

    Args:
        key: Parameter name (``"P"``, ``"q"``, ``"A"``, etc.).
        val: JAX array to convert.
        dtype: Target NumPy dtype.

    Returns:
        A dense NumPy array with the canonical shape for *key*.
    """
    arr: ndarray = np.asarray(val, dtype=dtype)
    if arr.ndim > EXPECTED_NDIM[key]:
        arr = arr[0]
    return arr


def dense_tangent_converter(key: str, val: Array, dtype: type[np.floating]) -> ndarray:
    """Convert a JAX tangent array to a dense NumPy array.

    Unlike the primal converter, the batch dimension (if present)
    is preserved — tangents from ``vmap`` are ``(batch, ...)`` and
    need to stay that way for batched RHS assembly.

    Args:
        key: Parameter name (``"P"``, ``"q"``, ``"A"``, etc.).
        val: JAX tangent array to convert.
        dtype: Target NumPy dtype.

    Returns:
        A dense NumPy array, keeping any leading batch dimension.
    """
    return np.asarray(val, dtype=dtype)


def dense_grad_to_jax(key: str, val: ndarray, dtype: type[np.floating]) -> Array:
    """Convert a NumPy gradient array back to a JAX array.

    Used when returning parameter gradients from the NumPy
    differentiator back to the JAX computation graph.

    Args:
        key: Parameter name (``"P"``, ``"q"``, ``"A"``, etc.).
        val: NumPy gradient array.
        dtype: Target JAX dtype.

    Returns:
        A JAX array suitable for further AD composition.
    """
    return jnp.asarray(val, dtype=dtype)
