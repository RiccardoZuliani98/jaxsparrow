import numpy as np

def fmt_times(t: dict[str, float]) -> str:
    """Format a timing dict into a single-line summary string.

    Args:
        t: Dict mapping stage names to elapsed seconds.

    Returns:
        Formatted string, e.g. ``"solve=1.2e-03  active=4.5e-05"``.
    """
    return "  ".join(f"{k}={v:.3e}s" for k, v in t.items())

def format_array_to_string(data, index_names=None, column_names=None, is_vector=False):
    """
    Converts array-like or sparse matrix data to a dense NumPy array, wraps it 
    in a pandas DataFrame, and formats it as a vertically aligned string.

    Parameters
    ----------
    data : array_like, sparse matrix, or None
        The input matrix or vector to format. Can be a JAX array, SciPy sparse 
        matrix, JAX BCOO matrix, or standard NumPy array. If None, an empty 
        string is returned.
    index_names : list of str, optional
        Names for the rows (DataFrame index). If None, numeric indices are used.
    column_names : list of str, optional
        Names for the columns. If None, numeric indices are used.
    is_vector : bool, default False
        If True, the input data is flattened to a strictly 1D array using 
        `np.ravel()`. This is useful for column vectors.

    Returns
    -------
    str
        A formatted string representation of the data using scientific notation 
        (15 characters wide, 4 decimal places). Returns an empty string if 
        `data` is None.
    """

    import pandas as pd

    if data is None:
        return ""
        
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
    
    # Define the exponential formatter
    formatter = lambda x: f"{x:15.4e}"
    
    # Apply the formatter to all columns and return the string
    return df.to_string(formatters=[formatter] * df.shape[1])


def print_qp_ingredients(
    P=None, q=None, A=None, b=None, G=None, h=None, 
    var_names=None, eq_names=None, ineq_names=None, 
    filename="qp_elements.txt"
):
    """
    Extracts Quadratic Program (QP) ingredients, safely converts them to dense 
    NumPy arrays, and writes them to a formatted, vertically aligned text file.
    
    This function handles mixed array types gracefully, extracting data from 
    JAX arrays, JAX BCOO sparse matrices, SciPy sparse matrices, and standard 
    NumPy arrays. Only the explicitly provided matrices/vectors are printed.

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
    filename : str
        Name of the txt file where the ingredients are printed.

    Returns
    -------
    None
        Writes directly to a file named `qp_elements.txt` (or provided filename) 
        in the current working directory.
    """
    with open(filename, "w") as f:
        
        if P is not None:
            f.write("=== Matrix P (Hessian) ===\n")
            f.write(format_array_to_string(P, index_names=var_names, column_names=var_names))
            f.write("\n\n")
            
        if A is not None:
            f.write("=== Matrix A (Equalities) ===\n")
            f.write(format_array_to_string(A, index_names=eq_names, column_names=var_names))
            f.write("\n\n")
            
        if G is not None:
            f.write("=== Matrix G (Inequalities) ===\n")
            f.write(format_array_to_string(G, index_names=ineq_names, column_names=var_names))
            f.write("\n\n")

        if q is not None:
            f.write("=== Vector q (Linear Cost) ===\n")
            f.write(format_array_to_string(q, index_names=var_names, column_names=["Cost_q"], is_vector=True))
            f.write("\n\n")
            
        if b is not None:
            f.write("=== Vector b (Equality Bounds) ===\n")
            f.write(format_array_to_string(b, index_names=eq_names, column_names=["Eq_Bound_b"], is_vector=True))
            f.write("\n\n")
            
        if h is not None:
            f.write("=== Vector h (Inequality Bounds) ===\n")
            f.write(format_array_to_string(h, index_names=ineq_names, column_names=["Ineq_Bound_h"], is_vector=True))
            f.write("\n")