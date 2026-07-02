import numpy as np

def fmt_times(t: dict[str, float]) -> str:
    """Format a timing dict into a single-line summary string.

    Args:
        t: Dict mapping stage names to elapsed seconds.

    Returns:
        Formatted string, e.g. ``"solve=1.2e-03  active=4.5e-05"``.
    """
    return "  ".join(f"{k}={v:.3e}s" for k, v in t.items())

import numpy as np

def save_array_to_csv(data, filepath, index_names=None, column_names=None, is_vector=False):
    """
    Converts array-like or sparse matrix data to a dense NumPy array, wraps it 
    in a pandas DataFrame, and saves it as a structured CSV file.

    Parameters
    ----------
    data : array_like, sparse matrix, or None
        The input matrix or vector to format. Can be a JAX array, SciPy sparse 
        matrix, JAX BCOO matrix, or standard NumPy array. If None, the function 
        returns without creating a file.
    filepath : str
        The full destination path and filename for the resulting CSV file.
    index_names : list of str, optional
        Names for the rows (DataFrame index). If None, numeric indices are used.
    column_names : list of str, optional
        Names for the columns. If None, numeric indices are used.
    is_vector : bool, default False
        If True, the input data is flattened to a strictly 1D array using 
        `np.ravel()`. This ensures vectors are saved as a single column rather 
        than a single row.

    Returns
    -------
    None
        Writes the DataFrame directly to the specified `filepath` in CSV format 
        using scientific notation (4 decimal places).
    """

    import pandas as pd

    if data is None:
        return
        
    # Safely convert various array/matrix formats to dense numpy arrays
    if hasattr(data, 'toarray'):
        # SciPy sparse matrices
        dense_data = np.asarray(data.toarray())
    elif hasattr(data, 'todense'):
        # JAX BCOO matrices or older SciPy formats
        dense_data = np.asarray(data.todense())
    else:
        # Fallback for standard JAX arrays, NumPy arrays, or lists
        dense_data = np.asarray(data)

    # Flatten if specified to ensure proper vector formatting in the DataFrame
    if is_vector:
        dense_data = np.ravel(dense_data)

    # Create the DataFrame
    df = pd.DataFrame(dense_data, index=index_names, columns=column_names)
    
    # Save directly to CSV using pandas, applying the exponential float format
    df.to_csv(filepath, float_format="%.4e")


def export_qp_ingredients_csv(
    P=None, q=None, A=None, b=None, G=None, h=None, 
    var_names=None, eq_names=None, ineq_names=None, 
    file_prefix="qp_data"
):
    """
    Extracts Quadratic Program (QP) ingredients, safely converts them to dense 
    NumPy arrays, and writes each provided matrix/vector to its own CSV file.
    
    This function handles mixed array types gracefully, extracting data from 
    JAX arrays, JAX BCOO sparse matrices, SciPy sparse matrices, and standard 
    NumPy arrays. Only the explicitly provided matrices/vectors are exported.

    Parameters
    ----------
    P : array_like or sparse matrix, optional
        The Hessian matrix of the quadratic cost. Shape (n, n).
    q : array_like, optional
        The linear cost vector. Shape (n,).
    A : array_like or sparse matrix, optional
        The equality constraint matrix. Shape (p, n).
    b : array_like, optional
        The equality constraint bound vector. Shape (p,).
    G : array_like or sparse matrix, optional
        The inequality constraint matrix. Shape (m, n).
    h : array_like, optional
        The inequality constraint bound vector. Shape (m,).
    var_names : list of str, optional
        Names of the decision variables. Used for labeling columns in P, A, G 
        and rows in P, q. If None, numeric indices are used.
    eq_names : list of str, optional
        Names of the equality constraints. Used for labeling rows in A and b. 
        If None, numeric indices are used.
    ineq_names : list of str, optional
        Names of the inequality constraints. Used for labeling rows in G and h. 
        If None, numeric indices are used.
    file_prefix : str, default "qp_data"
        The prefix used for the generated CSV files. For example, if "qp_data" 
        is provided, the Hessian will be saved as "qp_data_P_hessian.csv".

    Returns
    -------
    None
        Writes the provided matrices and vectors to individual CSV files in the 
        current working directory.
    """
    
    if P is not None:
        save_array_to_csv(
            P, f"{file_prefix}_P_hessian.csv", 
            index_names=var_names, column_names=var_names
        )
        
    if A is not None:
        save_array_to_csv(
            A, f"{file_prefix}_A_equalities.csv", 
            index_names=eq_names, column_names=var_names
        )
        
    if G is not None:
        save_array_to_csv(
            G, f"{file_prefix}_G_inequalities.csv", 
            index_names=ineq_names, column_names=var_names
        )

    if q is not None:
        save_array_to_csv(
            q, f"{file_prefix}_q_cost.csv", 
            index_names=var_names, column_names=["Cost_q"], is_vector=True
        )
        
    if b is not None:
        save_array_to_csv(
            b, f"{file_prefix}_b_eq_bounds.csv", 
            index_names=eq_names, column_names=["Eq_Bound_b"], is_vector=True
        )
        
    if h is not None:
        save_array_to_csv(
            h, f"{file_prefix}_h_ineq_bounds.csv", 
            index_names=ineq_names, column_names=["Ineq_Bound_h"], is_vector=True
        )