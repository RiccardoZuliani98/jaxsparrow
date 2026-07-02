"""
Quadratic Programming (QP) Diagnostic and Debugging Toolkit.
Identifies numerical issues, infeasibilities, and bad scaling in QP matrices.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cvxpy as cp
import scipy.linalg as la
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from jaxsparrow._utils._printing_utils import save_array_to_csv

# =============================================================================
# SECTION 0: DATA STRUCTURE FOR OUTPUT
# =============================================================================

def _get_name(registries, reg_key, idx, prefix):
    """Helper to setup strings in constraint determination."""
    reg = registries.get(reg_key, {})
    if isinstance(reg, dict): return reg.get(idx, f"{prefix} {idx}")
    if isinstance(reg, list) and idx < len(reg): return reg[idx]
    return f"{prefix} {idx}"

@dataclass
class QPSolution:
    """
    Container for the results of a single Quadratic Program solver run.

    Attributes
    ----------
    status : str
        The final status returned by the solver (e.g., 'optimal', 'infeasible').
    cost : float, optional
        The optimal objective function value. Default is None.
    x_value : numpy.ndarray, optional
        The optimal primal variables. Default is None.
    dual_eq : numpy.ndarray, optional
        The dual variables associated with equality constraints. Default is None.
    dual_ineq : numpy.ndarray, optional
        The dual variables associated with inequality constraints. Default is None.
    message : str, optional
        Error message or solver log if an exception occurred. Default is None.
    """
    status: str
    cost: Optional[float] = None
    x_value: Optional[np.ndarray] = None
    dual_eq: Optional[np.ndarray] = None
    dual_ineq: Optional[np.ndarray] = None
    message: Optional[str] = None

    def summary(self):
        """Print summary of the QP solution"""
        print(f"  Solver Status: {self.status.upper() if self.status else 'UNKNOWN'}")
        if self.status == 'optimal' and self.cost is not None:
            print(f"  Optimal Cost: {self.cost:.4e}")
        elif self.message:
            print(f"  Solver Message: {self.message}")

@dataclass
class DependentSubmatrix:
    """
    Container for a submatrix alongside its semantic row and column labels.
    
    Attributes
    ----------
    matrix : numpy.ndarray
        The 2D dense numerical data.
    row_names : list of str
        The semantic names for each row.
    col_names : list of str
        The semantic names for each column.
    """
    matrix: np.ndarray
    row_names: List[str]
    col_names: List[str]

@dataclass
class QPLicqResult:
    """
    Container for Linear Independence Constraint Qualification (LICQ) results.

    Attributes
    ----------
    evaluated : bool
        Whether the LICQ evaluation was successfully executed.
    licq_satisfied : bool
        True if the active constraints are linearly independent, False otherwise.
    num_active_total : int
        The total number of active constraints (equalities + active inequalities).
    rank : int
        The computed rank of the active constraint Jacobian.
    dependent_eq_indices : list of int
        Indices of the equality constraints that are linearly dependent.
    dependent_ineq_indices : list of int
        Indices of the inequality constraints that are linearly dependent.
    error : str, optional
        Error message if the evaluation could not be performed (e.g., missing solution).
    redundancy_details : list of dict
        A detailed breakdown of each redundant constraint. Each dictionary contains:
        - 'dependent_constraint': dict with 'type' ('eq' or 'ineq') and its original 'index'.
        - 'involved_variables': list of column indices corresponding to non-zero entries in the constraint.
        - 'spanned_by': list of dicts detailing the independent constraints ('type', 'index') 
          and their 'coefficient' in the linear combination that reconstructs the dependent row.
    dependent_submatrix : DependentSubmatrix, optional
        A structured container holding the dense 2D submatrix of the dependent constraints,
        filtered to remove globally zero columns, along with semantic row and column labels.
    """
    evaluated:              bool = False
    licq_satisfied:         bool = True
    num_active_total:       int = 0
    rank:                   int = 0
    dependent_eq_indices:   List[int] = field(default_factory=list)
    dependent_ineq_indices: List[int] = field(default_factory=list)
    error:                  Optional[str] = None
    redundancy_details:     List[Dict[str, Any]] = field(default_factory=list)
    dependent_submatrix:    Optional[DependentSubmatrix] = None

    def summary(self, registries=None):
        """Print summary of LICQ information."""

        if registries is None:
            registries = {}

        if not self.evaluated:
            err_msg = self.error if self.error else "No valid primal solution to evaluate active set"
            print(f"  Status: SKIPPED ({err_msg})")
            return

        print(f"  Active Constraints Evaluated: {self.num_active_total}")
        print(f"  Jacobian Rank: {self.rank}")
        
        if self.licq_satisfied:
            print("  Status: PASS (Constraints are linearly independent)")
        else:
            print("  Status: FAIL (Redundant/Dependent active constraints detected)")
            for idx in self.dependent_eq_indices:
                print(f"    -> Redundant Equality: '{_get_name(registries, 'eq_names', idx, 'Eq Row')}'")
            for idx in self.dependent_ineq_indices:
                print(f"    -> Redundant Active Inequality: '{_get_name(registries, 'ineq_names', idx, 'Ineq Row')}'")
                
            if self.redundancy_details:
                print("\n  --- Detailed Redundancy Breakdown ---")
                for detail in self.redundancy_details:
                    # Dependent constraint name resolution
                    dep = detail["dependent_constraint"]
                    dep_key = 'eq_names' if dep['type'] == 'eq' else 'ineq_names'
                    dep_prefix = 'Eq Row' if dep['type'] == 'eq' else 'Ineq Row'
                    dep_name = _get_name(registries, dep_key, dep['index'], dep_prefix)
                    
                    print(f"\n  Dependent Constraint: '{dep_name}'")
                    
                    # Variable name resolution
                    var_names = [_get_name(registries, 'var_names', v, 'Var') for v in detail['involved_variables']]
                    print(f"    Involved Variables: [{', '.join(var_names)}]")
                    
                    # Spanning constraint name resolution
                    print("    Spanned by Linear Combination:")
                    if not detail["spanned_by"]:
                        print("      (Zero row / No spanning constraints)")
                    else:
                        for span in detail["spanned_by"]:
                            span_key = 'eq_names' if span['type'] == 'eq' else 'ineq_names'
                            span_prefix = 'Eq Row' if span['type'] == 'eq' else 'Ineq Row'
                            span_name = _get_name(registries, span_key, span['index'], span_prefix)
                            
                            coeff_str = f"{span['coefficient']:>8.5f}"
                            print(f"      {coeff_str} * '{span_name}'")

@dataclass
class InfeasibilityReport:
    """
    Container for the results of an elastic Phase 1 conflict analysis.

    Attributes
    ----------
    solution : QPSolution
        Solver results for the full problem. Default is None.
    problem_solved : bool
        True if feasibility problem was solved.
    is_feasible : bool
        True if the problem is strictly feasible (zero violation cost).
    total_violation : float
        The sum of all constraint violations from the elastic phase.
    conflicting_eq_indices : list of int
        Indices of the equality constraints that must be violated.
    conflicting_ineq_indices : list of int
        Indices of the inequality constraints that must be violated.
    """
    solution: QPSolution
    problem_solved: bool
    is_feasible: bool
    total_violation: float
    conflicting_eq_indices: List[int] = field(default_factory=list)
    conflicting_ineq_indices: List[int] = field(default_factory=list)

    def summary(self, registries=None):
        """Print summary of infeasibility information."""

        if registries is None:
            registries = {}

        if self.solution:
            print(f"  Phase 1 solution report  ")
            self.solution.summary()

        if not self.is_feasible:
            print(f"  Phase 1 Violation Cost (Elastic IIS): {self.total_violation:.4e}")
            for idx in self.conflicting_eq_indices:
                print(f"    -> Conflicting Equality: '{_get_name(registries, 'eq_names', idx, 'Eq Row')}'")
            for idx in self.conflicting_ineq_indices:
                print(f"    -> Conflicting Inequality: '{_get_name(registries, 'ineq_names', idx, 'Ineq Row')}'")

@dataclass
class ConvexityReport:
    """
    Container for Hessian convexity and conditioning diagnostics.

    Attributes
    ----------
    is_psd : bool
        True if the Hessian matrix is Positive Semi-Definite (convex), False otherwise.
    min_eigenvalue : float
        The minimum eigenvalue computed for the Hessian matrix. Used to determine convexity.
    condition_number : float
        The condition number (kappa) of the Hessian matrix, reflecting its numerical stability.
    """
    is_psd: bool
    min_eigenvalue: float
    condition_number: float

    def summary(self):
        """
        Formats and prints the convexity check results and condition number to the terminal.
        """
        psd_status = "PASS (Convex)" if self.is_psd else "FAIL (Non-Convex)"
        print(f"  Hessian PSD Check: {psd_status} (Min Eigenvalue: {self.min_eigenvalue:.2e})")
        print(f"  Hessian Condition Number (kappa): {self.condition_number:.2e}")

@dataclass
class ScalingReport:
    """
    Container for the numerical statistics and scaling diagnostics of a single matrix or vector.

    Attributes
    ----------
    name : str
        A human-readable label identifying the matrix or vector (e.g., 'Hessian P', 'Equality A').
    has_nan : bool, optional
        True if the matrix contains any NaN values. Default is False.
    has_inf : bool, optional
        True if the matrix contains any infinite values (Inf). Default is False.
    nan_indices : list
        List of entries that evaluate to NaN.
    inf_indices : list
        List of entries that evaluate to infinity.
    min_abs_val : float, optional
        The minimum absolute value among all non-zero elements in the matrix. Default is 0.0.
    max_abs_val : float, optional
        The maximum absolute value among all elements in the matrix. Default is 0.0.
    row_norms_inf : numpy.ndarray, optional
        An array containing the infinity-norm (maximum absolute row sum) of each row. Default is None.
    col_norms_inf : numpy.ndarray, optional
        An array containing the infinity-norm (maximum absolute row sum) of each col. Default is None.
    scaling_threshold : float, optional
        The ratio threshold (max/min) above which a warning for poor structural scaling 
        is flagged. Default is 1e5.
    """
    name: str
    has_nan: bool = False
    has_inf: bool = False
    nan_indices: List[Any] = field(default_factory=list)
    inf_indices: List[Any] = field(default_factory=list)
    min_abs_val: float = 0.0
    max_abs_val: float = 0.0
    row_norms_inf: Optional[np.ndarray] = None
    col_norms_inf: Optional[np.ndarray] = None
    scaling_threshold: float = 1e5

    def summary(self, matrix_key: str, registries=None):
        """
        Evaluates the numerical properties and prints scaling warnings or row-level anomalies.

        Parameters
        ----------
        matrix_key : str
            The short string identifier of the matrix (e.g., 'A', 'b', 'G', 'h') used to
            determine the correct semantic names for poorly scaled rows.
        registries : dict, optional
            A dictionary containing 'eq_names', 'ineq_names', and 'var_names' to map raw 
            indices into human-readable row or variable names.
        """
        if registries is None:
            registries = {}
            
        print(f"\n  Matrix: {self.name}")
        
        if self.has_nan or self.has_inf:
            print("    CRITICAL: Contains NaN or Inf values!")
            
        print(f"    Min absolute non-zero: {self.min_abs_val:.2e}")
        print(f"    Max absolute value:    {self.max_abs_val:.2e}")
        
        if self.min_abs_val > 0:
            ratio = self.max_abs_val / self.min_abs_val
            if ratio > self.scaling_threshold:
                print(f"    WARNING: High internal scaling ratio ({ratio:.2e})")
        
        if self.row_norms_inf is not None and len(self.row_norms_inf) > 0:
            bad_rows = np.where(self.row_norms_inf > 1e4)[0] 
            
            reg_key, prefix = 'var_names', 'Index'
            if matrix_key in ['A', 'b']:
                reg_key, prefix = 'eq_names', 'Eq Row'
            elif matrix_key in ['G', 'h']:
                reg_key, prefix = 'ineq_names', 'Ineq Row'

            for row_idx in bad_rows:
                semantic_name = _get_name(registries, reg_key, row_idx, prefix)
                norm_val = self.row_norms_inf[row_idx]
                print(f"      -> Badly scaled row detected: '{semantic_name}' (Norm: {norm_val:.2e})")

@dataclass
class QPData:
    """
    Container for a Quadratic Program formulation, its diagnostics, and solve results.

    Attributes
    ----------
    P : scipy.sparse matrix or numpy.ndarray
        Hessian matrix of the objective.
    q : numpy.ndarray
        Linear objective vector.
    scaling_reports : dict of str to ScalingReport
        Dictionary mapping matrix identifiers (e.g., 'A', 'P') to their scaling reports.
    convexity_report : ConvexityReport, optional
        Dataclass containing PSD flag, condition number, and min/max eigenvalues.
    A : scipy.sparse matrix or numpy.ndarray, optional
        Equality constraint matrix. Default is None.
    b : numpy.ndarray, optional
        Equality constraint bounds. Default is None.
    G : scipy.sparse matrix or numpy.ndarray, optional
        Inequality constraint matrix. Default is None.
    h : numpy.ndarray, optional
        Inequality constraint bounds. Default is None.
    solution_full : QPSolution, optional
        Solver results for the full problem. Default is None.
    licq_info : QPLicqResult, optional
        Results of the LICQ analysis evaluated at the full problem's optimal solution.
    infeasibility_info : InfeasibilityReport, optional
        Results of the elastic Phase 1 conflict analysis if the problem is infeasible.
    registries : Optional[dict[str,List[str]]]
        Dictionary with keys "var_names", "eq_names", and "ineq_names", containing a list
        of strings providing descriptive names to each variable / constraint.
    """
    P: Any
    q: Any
    scaling_reports: Dict[str, ScalingReport] = field(default_factory=dict)
    convexity_report: Optional[ConvexityReport] = None
    A: Optional[Any] = None
    b: Optional[Any] = None
    G: Optional[Any] = None
    h: Optional[Any] = None
    solution_full: Optional[QPSolution] = None
    licq_info: Optional[QPLicqResult] = None
    infeasibility_info: Optional[InfeasibilityReport] = None
    registries: Optional[dict[str,List[str]]] = None

    def summary(self, title="QP DIAGNOSTIC REPORT"):
        """
        Formats the analysis dataclasses into a human-readable terminal report.
        Delegates segment printing to the respective sub-dataclasses.
        """
        if self.registries is None:
            registries = {}
        else:
            registries = self.registries

        print("\n" + "=" * 60)
        print(f" {title} ".center(60, "="))
        print("=" * 60)
        
        # ---------------------------------------------------------
        # 1. Feasibility Report
        # ---------------------------------------------------------
        print("\n[1. Feasibility Analysis]")
        infeas_info = self.infeasibility_info
        
        # if infeas_info is not passed, don't print anything
        if infeas_info is None:
            print("  Status: SKIPPED (No feasibility solution provided)")
        # if infeasible, print summary
        elif not infeas_info.is_feasible:
            infeas_info.summary(registries)
        # otherwise, just print solver status
        else:
            print(f"  Status: PASS (Geometrically Feasible) - Solver: {infeas_info.solution.status}")

        # ---------------------------------------------------------
        # 2. Robust Solver Report
        # ---------------------------------------------------------
        print("\n[2. Robust Solve Attempt (Full Problem)]")
        if self.solution_full:
            self.solution_full.summary()
        else:
            print("  Status: SKIPPED")

        # ---------------------------------------------------------
        # 3. Convexity Report
        # ---------------------------------------------------------
        print("\n[3. Convexity & Conditioning]")
        if self.convexity_report:
            self.convexity_report.summary()
        else:
            print("  Status: NO DATA")
       
        # ---------------------------------------------------------
        # 4. Matrix Scaling & Sanity Checks
        # ---------------------------------------------------------
        print("\n[4. Matrix Scaling & Sanity Checks]")
        if not self.scaling_reports:
            print("  Status: NO DATA")
        else:
            for matrix_key, report in self.scaling_reports.items():
                report.summary(matrix_key=matrix_key, registries=registries)

        # ---------------------------------------------------------
        # 5. LICQ Report
        # ---------------------------------------------------------
        print("\n[5. Linear Independence Constraint Qualification (LICQ)]")
        if self.licq_info:
            self.licq_info.summary(registries)
        else:
            print("  Status: SKIPPED (No LICQ data available)")
            
        print("\n" + "=" * 60 + "\n")

    def export_dependent_submatrix(self, filepath: str = "./dependent_submatrix.csv"):
        """
        Exports the dependent constraints submatrix to a CSV file if it exists.
        
        Parameters
        ----------
        filepath : str
            The full destination path and filename for the resulting CSV file.
        """
        if self.licq_info is None or getattr(self.licq_info, 'dependent_submatrix', None) is None:
            print("  Status: SKIPPED (No dependent submatrix available to export)")
            return

        submatrix = self.licq_info.dependent_submatrix

        # Calls the imported function
        if submatrix is not None:
            save_array_to_csv(
                data=submatrix.matrix,
                filepath=filepath,
                index_names=submatrix.row_names,
                column_names=submatrix.col_names,
                is_vector=False
            )
            print(f"  Exported dependent submatrix to: {filepath}")

@dataclass
class QPScaling:
    """
    Container for the transformation operators used to scale a Quadratic Program.

    Parameters
    ----------
    T : Any
        Dense transformation matrix for the variables (from Hessian eigen-scaling).
    D_A : Any, optional
        Row equilibration diagonal matrix for Equality Constraints. Default is None.
    D_G : Any, optional
        Row equilibration diagonal matrix for Inequality Constraints. Default is None.
    """
    T: Any
    D_A: Optional[Any] = None
    D_G: Optional[Any] = None

@dataclass
class QPDiagnosticResults:
    """
    A comprehensive container for Quadratic Program diagnostics, preserving 
    the original problem, the scaled problem, and the scaling operators.

    Attributes
    ----------
    qp_original : QPData
        The original QP matrices, diagnostics, and solve solutions.
    qp_scaled : QPData
        The scaled QP matrices, diagnostics, and solve solutions.
    scaling : QPScaling
        The transformation operators used to scale the problem.
    """
    qp_original: QPData
    qp_scaled: QPData
    scaling: QPScaling

# =============================================================================
# SECTION 1: DATA SANITY & SCALING ANALYZERS
# =============================================================================
def analyze_matrix_sanity(name, mat):
    """
    Analyzes a single matrix or vector for NaNs, Infs, and extreme scaling.
    Records the exact indices of corrupted data and computes row/col norms.
    
    Parameters
    ----------
    name (str): Human-readable name of the matrix (e.g., 'Hessian (P)').
    mat (np.ndarray or scipy.sparse matrix): The data to analyze.
        
    Returns
    -------
    dict: A dictionary containing min/max absolute values, NaN/Inf flags, 
          and row infinity norms.
    """

    has_nan = False
    has_inf = False
    nan_indices = []
    inf_indices = []
    min_abs_val = 0.0
    max_abs_val = 0.0
    row_norms_inf = np.array([])
    col_norms_inf = np.array([])

    is_sparse = sp.issparse(mat)
    
    # 1. Detect NaNs and Infs and capture their exact locations
    if is_sparse:
        mat_coo = mat.tocoo()
        nan_mask = np.isnan(mat_coo.data)
        inf_mask = np.isinf(mat_coo.data)
        
        if nan_mask.any():
            has_nan = True
            nan_indices = list(zip(mat_coo.row[nan_mask], mat_coo.col[nan_mask]))
            
        if inf_mask.any():
            has_inf = True
            inf_indices = list(zip(mat_coo.row[inf_mask], mat_coo.col[inf_mask]))
            
        flat_data = mat_coo.data
    else:
        mat_dense = np.asarray(mat)
        nan_mask = np.isnan(mat_dense)
        inf_mask = np.isinf(mat_dense)
        
        if nan_mask.any():
            has_nan = True
            if mat_dense.ndim == 1:
                nan_indices = np.where(nan_mask)[0].tolist()
            else:
                rows, cols = np.where(nan_mask)
                nan_indices = list(zip(rows, cols))
                
        if inf_mask.any():
            has_inf = True
            if mat_dense.ndim == 1:
                inf_indices = np.where(inf_mask)[0].tolist()
            else:
                rows, cols = np.where(inf_mask)
                inf_indices = list(zip(rows, cols))
                
        flat_data = mat_dense.flatten()

    # 2. Compute overall min/max scaling
    valid_data = np.abs(flat_data[np.isfinite(flat_data) & (flat_data != 0)])
    if len(valid_data) > 0:
        min_abs_val = float(np.min(valid_data))
        max_abs_val = float(np.max(valid_data))

    # 3. Compute norms to identify badly scaled constraints (rows) and variables (columns)
    if is_sparse and mat.ndim == 2:
        row_norms_inf = np.array(np.abs(mat).max(axis=1).todense()).flatten()
        col_norms_inf = np.array(np.abs(mat).max(axis=0).todense()).flatten()
    elif not is_sparse and mat.ndim == 1:
        # A 1D array represents either a bound (rows) or an objective (variables)
        row_norms_inf = np.abs(flat_data)

    return ScalingReport(
        name=name,
        has_nan=has_nan,
        has_inf=has_inf,
        nan_indices=nan_indices,
        inf_indices=inf_indices,
        min_abs_val=min_abs_val,
        max_abs_val=max_abs_val,
        row_norms_inf=row_norms_inf,
        col_norms_inf=col_norms_inf
    )

def analyze_qp_data(P, q, A, b, G, h):
    """
    Runs sanity checks on all standard QP matrices.
    
    Parameters
    ----------
        P, G, A: scipy.sparse matrices (Hessian, Ineq, Eq constraints).
        q, h, b: numpy 1D arrays (Linear objective, Ineq bounds, Eq bounds).
        
    Returns
    -------
    dict: A dictionary mapping matrix variable names to their sanity stats.
    """
    return {
        "P": analyze_matrix_sanity("Hessian (P)", P),
        "q": analyze_matrix_sanity("Linear Obj (q)", q),
        "A": analyze_matrix_sanity("Eq Constraints (A)", A),
        "b": analyze_matrix_sanity("Eq Bounds (b)", b),
        "G": analyze_matrix_sanity("Ineq Constraints (G)", G),
        "h": analyze_matrix_sanity("Ineq Bounds (h)", h)
    }

def analyze_convexity_and_conditioning(P, psd_tol=1e-10, cond_tol=1e-12):
    """
    Estimates the condition number and convexity (PSD status) of the Hessian.
    
    Parameters
    ----------
    P (scipy.sparse matrix): The Hessian matrix.
    psd_tol (float, optional): Tolerance for numerical noise when checking PSD. Default is 1e-10.
    cond_tol (float, optional): Tolerance for treating the minimum eigenvalue as zero. Default is 1e-12.
        
    Returns
    -------
    ConvexityReport: A dataclass containing the PSD flag, min eigenvalue, 
                     and estimated condition number.
    """
    result = {}

    # Find the smallest algebraic eigenvalue to check for strict/weak convexity
    min_eval = spla.eigsh(P, k=1, which='SA', return_eigenvectors=False)[0]
    is_psd = min_eval >= -psd_tol  # Tolerance for numerical noise
        
    # Find the largest magnitude eigenvalue for condition number
    max_eval = spla.eigsh(P, k=1, which='LM', return_eigenvectors=False)[0]
        
    if abs(min_eval) > cond_tol:
        condition_number = float(abs(max_eval / min_eval))
    else:
        condition_number = float('inf')
        
    return ConvexityReport(
        is_psd=bool(is_psd),
        min_eigenvalue=float(min_eval),
        condition_number=condition_number
    )

def compute_scaling(P, q, A=None, b=None, G=None, h=None, eig_tol=1e-10, row_tol=1e-10):
    """
    Applies Hessian Eigen-Scaling (whitening) followed by constraint Row Equilibration.
    
    WARNING: This destroys matrix sparsity and requires an O(N^3) eigendecomposition.
    It is highly effective for ill-conditioned, small-to-medium QPs, but will likely 
    cause Out-Of-Memory errors or massive slowdowns on large-scale sparse problems.
    
    Parameters
    ----------
    P (scipy.sparse matrix): The Hessian matrix.
    q (numpy.ndarray): Linear objective vector.
    A (scipy.sparse matrix, optional): Equality constraint matrix.
    b (numpy.ndarray, optional): Equality constraint bounds.
    G (scipy.sparse matrix, optional): Inequality constraint matrix.
    h (numpy.ndarray, optional): Inequality constraint bounds.
    eig_tol (float, optional): Tolerance for ignoring null-space eigenvalues. Default is 1e-10.
    row_tol (float, optional): Tolerance for preventing division by zero in row scaling. Default is 1e-10.
        
    Returns
    -------
    P_t, q_t: Transformed objective
    A_s, b_s: Transformed equality constraints (if provided)
    G_s, h_s: Transformed inequality constraints (if provided)
    T: The dense transformation matrix needed to recover original variables
        (i.e., x_original = T @ x_solved)
    D_A, D_G: Scaling matrices for equality and inequality constraints.
    """
    # ---------------------------------------------------------
    # 1. Hessian Eigen-Scaling (Whitening)
    # ---------------------------------------------------------
    # Ensure P is dense and perfectly symmetric for la.eigh
    P_dense = P.toarray() if sp.issparse(P) else np.asarray(P)
    P_sym = (P_dense + P_dense.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    evals, evecs = la.eigh(P_sym)
    T_dense = np.zeros_like(P_sym)
    
    # Construct transformation matrix T
    for i in range(len(evals)):
        if abs(evals[i]) > eig_tol:
            T_dense += (1.0 / np.sqrt(abs(evals[i]))) * np.outer(evecs[:, i], evecs[:, i])
        else:
            # Leave null-space alone to avoid dividing by zero
            T_dense += np.outer(evecs[:, i], evecs[:, i])
            
    # Apply transformation to objective
    T = sp.csr_matrix(T_dense) if sp.issparse(P) else T_dense
    P_t = T @ P_sym @ T
    q_t = T @ q
    
    # Apply transformation to constraints (if they exist)
    A_t = A @ T if A is not None and A.shape[0] > 0 else A
    G_t = G @ T if G is not None and G.shape[0] > 0 else G

    # ---------------------------------------------------------
    # 2. Row Equilibration
    # ---------------------------------------------------------
    def scale_rows(Mat, vec):
        if Mat is None or Mat.shape[0] == 0:
            return Mat, vec, None
            
        # Find maximum absolute value in each row
        if sp.issparse(Mat):
            row_max = np.array(np.abs(Mat).max(axis=1).todense()).flatten()
        else:
            row_max = np.max(np.abs(Mat), axis=1)
            
        # Prevent division by zero or amplifying extreme noise
        row_max[row_max < row_tol] = 1.0
        
        # Scale the matrix and the right-hand-side vector
        D = sp.diags(1.0 / row_max)
        return D @ Mat, D @ vec, D

    # Equilibrate constraints
    A_s, b_s, D_A = scale_rows(A_t, b)
    G_s, h_s, D_G = scale_rows(G_t, h)
    
    return P_t, q_t, A_s, b_s, G_s, h_s, T, D_A, D_G

def compute_jacobi_scaling(P, q, A=None, b=None, G=None, h=None, tol=1e-10):
    """
    Applies a simple single-pass Jacobi-style diagonal scaling to the QP.

    This strategy scales the variables based on the inverse square root of the
    diagonal of the Hessian (P). It then scales the equality and inequality
    constraints by their respective inverse row infinity-norms.

    This requires no matrix factorizations and operates in a single pass,
    making it extremely fast and simple.

    Parameters
    ----------
    P : scipy.sparse matrix or ndarray
        The Hessian matrix.
    q : numpy.ndarray
        Linear objective vector.
    A : scipy.sparse matrix or ndarray, optional
        Equality constraint matrix.
    b : numpy.ndarray, optional
        Equality constraint bounds.
    G : scipy.sparse matrix or ndarray, optional
        Inequality constraint matrix.
    h : numpy.ndarray, optional
        Inequality constraint bounds.
    tol : float, optional
        Tolerance to prevent division by zero. Default is 1e-10.

    Returns
    -------
    P_t, q_t: Transformed objective
    A_s, b_s: Transformed equality constraints (if provided)
    G_s, h_s: Transformed inequality constraints (if provided)
    T: Diagonal transformation matrix for the variables.
    D_A, D_G: Diagonal scaling matrices for equality and inequality constraints.
    """
    n = P.shape[0]

    # 1. Variable Scaling (T) based on the diagonal of P
    if sp.issparse(P):
        diag_P = np.abs(P.diagonal())
    else:
        diag_P = np.abs(np.diag(P))

    # Prevent division by zero for zero-diagonal elements (e.g. slack variables)
    diag_P[diag_P < tol] = 1.0

    dx = 1.0 / np.sqrt(diag_P)
    T_mat = sp.diags(dx) if sp.issparse(P) else np.diag(dx)

    # Scale Objective symmetrically
    P_t = T_mat @ P @ T_mat
    q_t = dx * q

    # 2. Equality Constraint Scaling (D_A)
    A_s, b_s, D_A_mat = None, None, None
    if A is not None and A.shape[0] > 0:
        # First, apply the variable transformation to the constraint matrix
        A_t = A @ T_mat

        # Find maximum absolute value in each row (infinity norm)
        if sp.issparse(A_t):
            row_norm_A = np.array(np.abs(A_t).max(axis=1).todense()).flatten()
        else:
            row_norm_A = np.max(np.abs(A_t), axis=1)

        row_norm_A[row_norm_A < tol] = 1.0
        da = 1.0 / row_norm_A

        D_A_mat = sp.diags(da) if sp.issparse(A) else np.diag(da)
        A_s = D_A_mat @ A_t
        b_s = da * b

    # 3. Inequality Constraint Scaling (D_G)
    G_s, h_s, D_G_mat = None, None, None
    if G is not None and G.shape[0] > 0:
        # First, apply the variable transformation to the constraint matrix
        G_t = G @ T_mat

        # Find maximum absolute value in each row (infinity norm)
        if sp.issparse(G_t):
            row_norm_G = np.array(np.abs(G_t).max(axis=1).todense()).flatten()
        else:
            row_norm_G = np.max(np.abs(G_t), axis=1)

        row_norm_G[row_norm_G < tol] = 1.0
        dg = 1.0 / row_norm_G

        D_G_mat = sp.diags(dg) if sp.issparse(G) else np.diag(dg)
        G_s = D_G_mat @ G_t
        h_s = dg * h

    return P_t, q_t, A_s, b_s, G_s, h_s, T_mat, D_A_mat, D_G_mat

def compute_diagonal_scaling(P, q, A=None, b=None, G=None, h=None, max_iter=10, tol=1e-10):
    """
    Applies Ruiz Equilibration (Iterative Diagonal Scaling) to the QP.
    
    This strictly preserves matrix sparsity and operates in O(N_nnz) time per iteration.
    It is highly effective for large-scale, sparse QPs to improve conditioning 
    without the O(N^3) memory/compute explosion of eigen-scaling.
    
    Parameters
    ----------
    P (scipy.sparse matrix or ndarray): The Hessian matrix.
    q (numpy.ndarray): Linear objective vector.
    A (scipy.sparse matrix or ndarray, optional): Equality constraint matrix.
    b (numpy.ndarray, optional): Equality constraint bounds.
    G (scipy.sparse matrix or ndarray, optional): Inequality constraint matrix.
    h (numpy.ndarray, optional): Inequality constraint bounds.
    max_iter (int, optional): Number of Ruiz equilibration steps. Default is 10.
    tol (float, optional): Tolerance to prevent division by zero. Default is 1e-10.
        
    Returns
    -------
    P_t, q_t: Transformed objective
    A_s, b_s: Transformed equality constraints (if provided)
    G_s, h_s: Transformed inequality constraints (if provided)
    T: Diagonal transformation matrix needed to recover original variables
        (i.e., x_original = T @ x_solved)
    D_A, D_G: Diagonal scaling matrices for equality and inequality constraints.
    """
    n = P.shape[0]

    # 1. Initialize accumulated scaling arrays (1D arrays to save memory)
    T_vec = np.ones(n)
    D_A_vec = np.ones(A.shape[0]) if A is not None else None
    D_G_vec = np.ones(G.shape[0]) if G is not None else None

    # Work on copies to avoid modifying originals in-place
    P_t = P.copy()
    q_t = q.copy()
    A_s = A.copy() if A is not None else None
    b_s = b.copy() if b is not None else None
    G_s = G.copy() if G is not None else None
    h_s = h.copy() if h is not None else None

    # Helper function to get infinite norms of columns
    def get_col_norms(mat):
        if mat is None or mat.shape[0] == 0:
            return np.zeros(n)
        if sp.issparse(mat):
            return np.array(np.abs(mat).max(axis=0).todense()).flatten()
        return np.max(np.abs(mat), axis=0)

    # Helper function to get infinite norms of rows
    def get_row_norms(mat):
        if mat is None or mat.shape[0] == 0:
            return np.array([])
        if sp.issparse(mat):
            return np.array(np.abs(mat).max(axis=1).todense()).flatten()
        return np.max(np.abs(mat), axis=1)

    # 2. Iterative Equilibration
    for _ in range(max_iter):
        # --- Column Norms (Variables) ---
        col_norm_P = get_col_norms(P_t)
        col_norm_A = get_col_norms(A_s)
        col_norm_G = get_col_norms(G_s)

        # Maximum element across P, A, and G for each column
        col_norm = np.maximum.reduce([col_norm_P, col_norm_A, col_norm_G])
        col_norm[col_norm < tol] = 1.0 # Prevent division by zero
        
        # Inverse square root for variable scaling
        dx = 1.0 / np.sqrt(col_norm)

        # --- Row Norms (Constraints) ---
        row_norm_A = get_row_norms(A_s)
        if row_norm_A.size > 0:
            row_norm_A[row_norm_A < tol] = 1.0
            da = 1.0 / np.sqrt(row_norm_A)
        else:
            da = np.array([])

        row_norm_G = get_row_norms(G_s)
        if row_norm_G.size > 0:
            row_norm_G[row_norm_G < tol] = 1.0
            dg = 1.0 / np.sqrt(row_norm_G)
        else:
            dg = np.array([])

        # --- Apply Scaling ---
        Dx_mat = sp.diags(dx) if sp.issparse(P_t) else np.diag(dx)
        
        # Scale Objective (Symmetrically for P)
        P_t = Dx_mat @ P_t @ Dx_mat
        q_t = dx * q_t

        # Scale Equality Constraints
        if A_s is not None and A_s.shape[0] > 0:
            Da_mat = sp.diags(da) if sp.issparse(A_s) else np.diag(da)
            A_s = Da_mat @ A_s @ Dx_mat
            b_s = da * b_s

        # Scale Inequality Constraints
        if G_s is not None and G_s.shape[0] > 0:
            Dg_mat = sp.diags(dg) if sp.issparse(G_s) else np.diag(dg)
            G_s = Dg_mat @ G_s @ Dx_mat
            h_s = dg * h_s

        # --- Accumulate Scalers ---
        T_vec *= dx
        if D_A_vec is not None and da.size > 0:
            D_A_vec *= da
        if D_G_vec is not None and dg.size > 0:
            D_G_vec *= dg

    # 3. Convert accumulated 1D scalers to diagonal matrices for output
    T = sp.diags(T_vec) if sp.issparse(P) else np.diag(T_vec)
    D_A = None
    if A is not None:
        D_A = sp.diags(D_A_vec) if sp.issparse(A) else np.diag(D_A_vec)
    D_G = None
    if G is not None:
        D_G = sp.diags(D_G_vec) if sp.issparse(G) else np.diag(D_G_vec)

    return P_t, q_t, A_s, b_s, G_s, h_s, T, D_A, D_G

# =============================================================================
# SECTION 2: CVXPY QP SOLVER
# =============================================================================
def qp_solve(P, q, A=None, b=None, G=None, h=None, x_warmstart=None, solver=cp.GUROBI):
    """
    Attempts to solve the true QP using a highly robust interior-point method.
    Returns a structured QPSolution dataclass.
    """
    n = P.shape[0]
    x = cp.Variable(n)
    
    if x_warmstart is not None:
        x.value = x_warmstart
        
    # Ensure P is perfectly symmetric to avoid CVXPY DCP errors
    P_sym = (P + P.T) / 2.0
    
    objective = cp.Minimize(0.5 * cp.quad_form(x, P_sym) + q.T @ x)
    
    constraints = []
    has_eq = A is not None and A.shape[0] > 0
    has_ineq = G is not None and G.shape[0] > 0
    
    if has_eq:
        constraints.append(A @ x == b)
    if has_ineq:
        constraints.append(G @ x <= h)
        
    prob = cp.Problem(objective, constraints)
    
    try:
        if solver == "nlp":
            prob.solve(warm_start=(x_warmstart is not None), nlp=True)
        else:
            prob.solve(solver=solver, warm_start=(x_warmstart is not None))
    except Exception as e:
        # If the solver crashes midway, x.value may contain the last recorded iterate.
        return QPSolution(
            status=prob.status,
            message=str(e),
            x_value=x.value,
            cost=None,
            dual_eq=np.array([]),
            dual_ineq=np.array([])
        )
        
    # Safely handle prob.value in case the status is infeasible/unbounded 
    # where prob.value might be None.
    cost_val = float(np.array(prob.value)) if prob.value is not None else None
        
    result = QPSolution(
        status=prob.status,
        cost=cost_val,
        x_value=x.value,  # Captures last iterate for non-exception failures
        dual_eq=np.array([]),
        dual_ineq=np.array([])
    )
    
    if prob.status not in ["infeasible", "unbounded", None]:
        if has_eq and constraints[0].dual_value is not None:
            result.dual_eq = constraints[0].dual_value
        if has_ineq:
            ineq_idx = 1 if has_eq else 0
            if constraints[ineq_idx].dual_value is not None:
                result.dual_ineq = constraints[ineq_idx].dual_value
            
    return result

def analyze_conflicting_constraints(A=None, b=None, G=None, h=None, tol=1e-5, solver=cp.GUROBI):
    """
    Identifies specific incompatible constraints using an elastic penalty formulation.

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray, optional
        Equality constraint matrix.
    b : numpy.ndarray, optional
        Equality constraint bounds.
    G : scipy.sparse matrix or numpy.ndarray, optional
        Inequality constraint matrix.
    h : numpy.ndarray, optional
        Inequality constraint bounds.
    tol : float, optional
        Numerical tolerance to consider a slack variable non-zero. Default is 1e-5.
    solver : str, optional
        The CVXPY solver to use. Default is cp.GUROBI.

    Returns
    -------
    InfeasibilityReport
        A dataclass detailing the feasibility status and the specific indices 
        of conflicting constraints.
    """
    has_eq = A is not None and A.shape[0] > 0
    has_ineq = G is not None and G.shape[0] > 0

    if has_eq:
        n_vars = A.shape[1] #type: ignore
    elif has_ineq:
        n_vars = G.shape[1] #type: ignore
    else:
        return InfeasibilityReport(
            solution=QPSolution(status="no_constraints", cost=0.0),
            problem_solved=True, 
            is_feasible=True, 
            total_violation=0.0
        )
            
    x = cp.Variable(n_vars)
    objective_terms = []
    constraints = []
    
    if has_eq:
        v_eq = cp.Variable(A.shape[0]) #type: ignore
        constraints.append(A @ x + v_eq == b)
        objective_terms.append(cp.norm1(v_eq))
        
    if has_ineq:
        v_ineq = cp.Variable(G.shape[0], nonneg=True) #type: ignore
        constraints.append(G @ x <= h + v_ineq)
        objective_terms.append(cp.sum(v_ineq))
        
    prob = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
    
    try:
        if solver == "nlp":
            prob.solve(nlp=True)
        else:
            prob.solve(solver=solver)
    except Exception as e:
        return InfeasibilityReport(
            solution=QPSolution(status="SolverError", message=str(e), x_value=x.value),
            problem_solved=False, 
            is_feasible=False, 
            total_violation=np.inf
        )
        
    total_violation = float(np.array(prob.value)) if prob.value is not None else np.inf
    is_feasible = total_violation <= tol
    
    # Extract duals safely 
    dual_eq_val = np.array([])
    dual_ineq_val = np.array([])
    
    if prob.status not in ["infeasible", "unbounded", None]:
        idx = 0
        if has_eq:
            if constraints[idx].dual_value is not None:
                dual_eq_val = constraints[idx].dual_value
            idx += 1
        if has_ineq:
            if constraints[idx].dual_value is not None:
                dual_ineq_val = constraints[idx].dual_value

    # Build the full QPSolution object for the Phase 1 problem
    phase_1_sol = QPSolution(
        status=prob.status,
        cost=total_violation if prob.value is not None else None,
        x_value=x.value,
        dual_eq=dual_eq_val,
        dual_ineq=dual_ineq_val
    )
    
    report = InfeasibilityReport(
        solution=phase_1_sol,
        problem_solved=True,
        is_feasible=is_feasible,
        total_violation=total_violation if not is_feasible else 0.0
    )
    
    if not is_feasible:
        if has_eq and v_eq.value is not None:
            bad_eqs = np.where(np.abs(v_eq.value) > tol)[0]
            report.conflicting_eq_indices = bad_eqs.tolist()
            
        if has_ineq and v_ineq.value is not None:
            bad_ineqs = np.where(v_ineq.value > tol)[0]
            report.conflicting_ineq_indices = bad_ineqs.tolist()
            
    return report

# =============================================================================
# SECTION 3: LICQ HELPER
# =============================================================================
def analyze_licq(A, G, h, x_val, tol=1e-5):
    """
    Evaluates Linear Independence Constraint Qualification (LICQ) at a given point.
    Identifies specific redundant (linearly dependent) constraints using QR pivoting.

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray, optional
        Equality constraint matrix.
    G : scipy.sparse matrix or numpy.ndarray, optional
        Inequality constraint matrix.
    h : numpy.ndarray, optional
        Inequality constraint bounds.
    x_val : numpy.ndarray
        Primal solution vector to evaluate the active set at.
    tol : float, optional
        Tolerance for active constraint detection. Default is 1e-5.

    Returns
    -------
    QPLicqResult
        A dataclass containing the LICQ status, rank, and indices of any 
        dependent constraints.
    """
    if x_val is None:
        return QPLicqResult(error="No primal solution provided.")
        
    result = QPLicqResult(evaluated=True)
    
    # Normalize inputs for safe processing
    A_safe = A if A is not None else np.array([])
    G_safe = G if G is not None else np.array([])
    h_safe = h if h is not None else np.array([])
    
    # 1. Identify active inequalities
    if G_safe.shape[0] > 0:
        slack = h_safe - (G_safe @ x_val)
        active_ineq_indices = np.where(np.abs(slack) <= tol)[0]
        G_active = G_safe[active_ineq_indices]
    else:
        active_ineq_indices = np.array([], dtype=int)
        G_active = np.empty((0, x_val.shape[0]))
        
    # 2. Construct the Active Jacobian (J)
    A_dense = A_safe.toarray() if sp.issparse(A_safe) else A_safe #type: ignore
    G_active_dense = G_active.toarray() if sp.issparse(G_active) else G_active #type: ignore
    
    if A_dense.shape[0] == 0 and G_active_dense.shape[0] == 0:
        return result  # Unconstrained or no active constraints; LICQ trivially satisfied
        
    if A_dense.shape[0] > 0 and G_active_dense.shape[0] > 0:
        J = np.vstack((A_dense, G_active_dense))
    elif A_dense.shape[0] > 0:
        J = A_dense
    else:
        J = G_active_dense
        
    num_active_total = J.shape[0]
    num_eq = A_dense.shape[0] if A_dense.shape[0] > 0 else 0
    
    result.num_active_total = num_active_total
    
    # 3. Compute Rank
    rank_J = np.linalg.matrix_rank(J, tol=tol)
    result.rank = rank_J
    
    if rank_J < num_active_total:
        result.licq_satisfied = False
        
        # 4. Identify dependent and independent rows using QR pivoting on J^T
        _, _, P_idx = la.qr(J.T, pivoting=True)
        indep_indices = P_idx[:rank_J]
        dep_indices = P_idx[rank_J:]
        
        J_indep = J[indep_indices, :]
        
        # Helper to map J row indices back to A or G
        def map_to_original(idx):
            if idx < num_eq:
                return "eq", int(idx)
            return "ineq", int(active_ineq_indices[idx - num_eq])

        # 5. Extract dependencies
        for idx in dep_indices:
            orig_type, orig_idx = map_to_original(idx)
            
            if orig_type == "eq":
                result.dependent_eq_indices.append(orig_idx)
            else:
                result.dependent_ineq_indices.append(orig_idx)
            
            # --- NEW CAPABILITY ---
            
            # Variables involved in this specific redundant constraint
            involved_vars = np.where(np.abs(J[idx]) > tol)[0].tolist()
            
            spanning_constraints = []
            
            # Find the linear combination of independent rows that span this dependent row.
            # We solve: J_indep^T * c = J[idx]^T
            if rank_J > 0:
                c, _, _, _ = la.lstsq(J_indep.T, J[idx].T)
                
                for i, coeff in enumerate(c):
                    if np.abs(coeff) > tol:
                        span_type, span_idx = map_to_original(indep_indices[i])
                        spanning_constraints.append({
                            "type": span_type,
                            "index": span_idx,
                            "coefficient": round(float(coeff), 5)
                        })
            
            result.redundancy_details.append({
                "dependent_constraint": {"type": orig_type, "index": orig_idx},
                "involved_variables": involved_vars,
                "spanned_by": spanning_constraints
            })
                
    return result

def build_dependent_submatrix(
    licq_result: QPLicqResult, 
    A, 
    G, 
    registries=None, 
    tol=1e-5
) -> Optional[DependentSubmatrix]:
    """
    Extracts dependent rows based on LICQ analysis, removes globally zero columns, 
    assigns semantic names, and returns a DependentSubmatrix object.
    
    Parameters
    ----------
    licq_result : QPLicqResult
        The result object returned by `analyze_licq`.
    A : scipy.sparse matrix or numpy.ndarray, optional
        Equality constraint matrix.
    G : scipy.sparse matrix or numpy.ndarray, optional
        Inequality constraint matrix.
    registries : dict, optional
        Dictionary containing 'eq_names', 'ineq_names', and 'var_names'.
    tol : float, optional
        Tolerance for identifying zero columns.
        
    Returns
    -------
    DependentSubmatrix or None
        The structured submatrix container, or None if there are no dependent constraints.
    """
    if not licq_result or not licq_result.redundancy_details:
        return None
        
    if registries is None:
        registries = {}

    dependent_rows = []
    row_names = []

    # 1. Extract the raw rows and construct row labels
    for detail in licq_result.redundancy_details:
        for dep in [detail["dependent_constraint"]] + detail["spanned_by"]:
            idx = dep["index"]
            if dep["type"] == "eq" and A is not None:
                row_data = A[[idx], :] if sp.issparse(A) else A[idx, :]
                name = _get_name(registries, 'eq_names', idx, 'Eq Row')
            elif dep["type"] == "ineq" and G is not None:
                row_data = G[[idx], :] if sp.issparse(G) else G[idx, :]
                name = _get_name(registries, 'ineq_names', idx, 'Ineq Row')
            else:
                continue
            dependent_rows.append(row_data)
            row_names.append(name)

    if not dependent_rows:
        return None

    # 2. Stack into a single 2D matrix safely
    if sp.issparse(dependent_rows[0]):
        D = sp.vstack(dependent_rows).toarray()
    else:
        D = np.vstack(dependent_rows)

    # 3. Identify and filter out globally zero columns
    non_zero_cols = np.where(np.max(np.abs(D), axis=0) > tol)[0]
    D_filtered = D[:, non_zero_cols]

    # 4. Generate column names for the kept variables
    col_names = [_get_name(registries, 'var_names', c, 'Var') for c in non_zero_cols]

    # 5. Return the structured container
    return DependentSubmatrix(
        matrix=D_filtered,
        row_names=row_names,
        col_names=col_names
    )

# =============================================================================
# SECTION 4: REPORTING & ORCHESTRATION
# =============================================================================
def run_qp_diagnostics(
    P, q, A = None, b = None, G = None, h = None, solver = cp.GUROBI,
    psd_tol = 1e-10, cond_tol = 1e-12, 
    conflict_tol = 1e-5, licq_tol = 1e-5,
    scaling_mode : str = "diagonal", 
    scaling_options : Optional[dict] = None,
    registries : Optional[dict[str,List[str]]] = None
):
    """
    Executes a full diagnostic, scaling, and testing pipeline on a Quadratic Program.

    This function performs data validation, applies coordinate scaling, solves 
    both feasibility and full optimization problems, and finally evaluates LICQ 
    at the optimal solutions for both the original and scaled spaces.

    Parameters
    ----------
    P : scipy.sparse matrix or numpy.ndarray
        Hessian matrix of the objective.
    q : numpy.ndarray
        Linear objective vector.
    A : scipy.sparse matrix or numpy.ndarray, optional
        Equality constraint matrix. Default is None.
    b : numpy.ndarray, optional
        Equality constraint bounds. Default is None.
    G : scipy.sparse matrix or numpy.ndarray, optional
        Inequality constraint matrix. Default is None.
    h : numpy.ndarray, optional
        Inequality constraint bounds. Default is None.
    solver : str, optional
        The CVXPY solver constant to use for the evaluations. Default is cp.GUROBI.
    psd_tol : float, optional
        Tolerance for numerical noise when checking PSD status. Default is 1e-10.
    cond_tol : float, optional
        Tolerance for treating the minimum eigenvalue as zero. Default is 1e-12.
    eig_tol : float, optional
        Tolerance for ignoring null-space eigenvalues in scaling. Default is 1e-10.
    row_tol : float, optional
        Tolerance for preventing division by zero during row equilibration. Default is 1e-10.
    conflict_tol : float, optional
        Tolerance used when analyzing conflicting constraints. Default is 1e-5.
    licq_tol : float, optional
        Tolerance used when evaluating LICQ. Default is 1e-5.
    scaling_mode : str, optional
        Type of scaling ("diagonal" or "full").
    registries : Optional[dict[str,List[str]]]
        Dictionary with keys "var_names", "eq_names", and "ineq_names", containing a list
        of strings providing descriptive names to each variable / constraint.

    Returns
    -------
    QPDiagnosticResults
        A nested dataclass containing the original and scaled problem states, 
        their corresponding solver results, LICQ analysis, and the computed 
        scaling operators.
    """
    
    # 1. Normalize empty inputs to prevent crashes in the analysis functions
    n_vars = q.shape[0] if q is not None else (P.shape[0] if P is not None else 0)
    A_safe = A if A is not None else np.array([])
    b_safe = b if b is not None else np.array([])
    G_safe = G if G is not None else np.array([])
    h_safe = h if h is not None else np.array([])
    
    # 2. Run Diagnostics on Original Data
    sanity_results = analyze_qp_data(P, q, A_safe, b_safe, G_safe, h_safe)
    convexity_results = analyze_convexity_and_conditioning(P, psd_tol=psd_tol, cond_tol=cond_tol)
    
    # 3. Compute Scaled Matrices and Capture Operators
    scaling_options = scaling_options or {}
    if scaling_mode == "full":
        scaling_options = {"eig_tol":1e-10, "row_tol":1e-10} | scaling_options
        P_s, q_s, A_s, b_s, G_s, h_s, T, D_A, D_G = compute_scaling(
            P, q, A, b, G, h, 
            eig_tol=scaling_options["eig_tol"],
            row_tol=scaling_options["row_tol"]
        )
    elif scaling_mode == "diagonal":
        scaling_options = {"max_iter":10, "tol":1e-10} | scaling_options
        P_s, q_s, A_s, b_s, G_s, h_s, T, D_A, D_G = compute_diagonal_scaling(
            P, q, A, b, G, h,
            max_iter=scaling_options["max_iter"],
            tol=scaling_options["tol"]
        )
    elif scaling_mode == "jacobi":
        scaling_options = {"tol": 1e-10} | scaling_options
        P_s, q_s, A_s, b_s, G_s, h_s, T, D_A, D_G = compute_jacobi_scaling(
            P, q, A, b, G, h,
            tol=scaling_options["tol"]
        )
    else:
        raise ValueError("Unknown scaling mode.")
    
    # 4. Normalize scaled empty inputs
    A_s_safe = A_s if A_s is not None else np.array([])
    b_s_safe = b_s if b_s is not None else np.array([])
    G_s_safe = G_s if G_s is not None else np.array([])
    h_s_safe = h_s if h_s is not None else np.array([])
    
    # 5. Run Diagnostics on Scaled Data
    sanity_results_scaled = analyze_qp_data(P_s, q_s, A_s_safe, b_s_safe, G_s_safe, h_s_safe)
    convexity_results_scaled = analyze_convexity_and_conditioning(P_s, psd_tol=psd_tol, cond_tol=cond_tol)
    
    # 6. Executes Solves & Feasibility Triggers
    infeas_orig = analyze_conflicting_constraints(
        A_safe, b_safe, G_safe, h_safe, tol=conflict_tol, solver=solver
    )

    # extract warmstart
    x_value_phase_one = infeas_orig.solution.x_value
    
    # solve original problem with warmstart
    sol_orig_full = qp_solve(P, q, A_safe, b_safe, G_safe, h_safe, solver=solver, x_warmstart=x_value_phase_one)
    
    # Scaled Problem Solves
    infeas_scaled = analyze_conflicting_constraints(
        A_s_safe, b_s_safe, G_s_safe, h_s_safe, tol=conflict_tol, solver=solver
    )

    # extract warmstart
    x_value_phase_one_scaled = infeas_scaled.solution.x_value
        
    sol_scaled_full = qp_solve(P_s, q_s, A_s_safe, b_s_safe, G_s_safe, h_s_safe, solver=solver, x_warmstart=x_value_phase_one_scaled)
    
    # 7. Evaluate LICQ at the Optimal Solutions
    licq_orig = analyze_licq(A_safe, G_safe, h_safe, sol_orig_full.x_value, tol=licq_tol)
    licq_scaled = analyze_licq(A_s_safe, G_s_safe, h_s_safe, sol_scaled_full.x_value, tol=licq_tol)

    # 8. Extract the Dependent Submatrix directly into the LICQ result objects
    licq_orig.dependent_submatrix = build_dependent_submatrix(
        licq_result=licq_orig, 
        A=A_safe, 
        G=G_safe, 
        tol=licq_tol,
        registries=registries
    )
    
    licq_scaled.dependent_submatrix = build_dependent_submatrix(
        licq_result=licq_scaled, 
        A=A_s_safe, 
        G=G_s_safe, 
        tol=licq_tol,
        registries=registries
    )

    qp_original = QPData(
        P=P, q=q, A=A, b=b, G=G, h=h,
        scaling_reports=sanity_results,
        convexity_report=convexity_results,
        solution_full=sol_orig_full,
        licq_info=licq_orig,
        infeasibility_info=infeas_orig,
        registries=registries
    )
    
    qp_scaled = QPData(
        P=P_s, q=q_s, A=A_s, b=b_s, G=G_s, h=h_s,
        scaling_reports=sanity_results_scaled,
        convexity_report=convexity_results_scaled,
        solution_full=sol_scaled_full,
        licq_info=licq_scaled,
        infeasibility_info=infeas_scaled,
        registries=registries
    )
    
    scaling = QPScaling(T=T, D_A=D_A, D_G=D_G)
    
    return QPDiagnosticResults(
        qp_original=qp_original,
        qp_scaled=qp_scaled,
        scaling=scaling
    )