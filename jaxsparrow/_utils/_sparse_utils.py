from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix
import numpy as np

def bcoo_to_csc(mat: BCOO, dtype: np.floating) -> csc_matrix:
    """Convert a JAX BCOO matrix to a SciPy CSC matrix."""
    idx = np.asarray(mat.indices)
    data = np.asarray(mat.data, dtype=dtype)
    rows = idx[:, 0]
    cols = idx[:, 1]
    return csc_matrix((data, (rows, cols)), shape=mat.shape)