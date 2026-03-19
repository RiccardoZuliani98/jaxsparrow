"""
solver_sparse/differentiators.py
================================
KKT-based forward and reverse differentiators for the sparse QP path.

Key design decisions:
  - Primal matrices (P, A, G) are stored and manipulated as
    ``scipy.sparse.csc_matrix`` throughout — no densification.
  - The KKT LHS is assembled as a sparse matrix using
    ``scipy.sparse.bmat`` / ``scipy.sparse.vstack``.
  - The RHS is dense (vectors or thin batched matrices) — this is
    appropriate because the RHS dimension is small relative to the
    system size.
  - The linear solve uses ``scipy.sparse.linalg.splu`` for both
    single and multiple RHS (factorize once, solve per column).
  - Reverse-mode gradients for P, A, G are returned as 1-D arrays
    of nonzero values (matching the sparsity pattern of each matrix),
    not as dense matrices. The converter can wrap these directly into
    BCOO without any dense round-trip.
"""

from __future__ import annotations

from numpy import ndarray
from time import perf_counter
from jaxtyping import Bool
import numpy as np
from typing import cast, Optional, Sequence, Union
from scipy.sparse import (
    csc_matrix,
    issparse,
    bmat as sp_bmat,
    vstack as sp_vstack,
)
from scipy.sparse.linalg import splu

from src.solver_sparse.types import (
    SparseQPIngredientsNP,
    SparseQPIngredientsTangentsNP,
)
from src.types_common import QPOutputNP, QPDiffOutNP
from src.options_common import DifferentiatorOptions
from src.utils.parsing_utils import parse_options
from src.solver_sparse.options import DEFAULT_DIFF_OPTIONS


# ── Sparse helpers ───────────────────────────────────────────────────

def _ensure_csc(v, dtype) -> csc_matrix:
    """Ensure a value is a CSC matrix."""
    if isinstance(v, csc_matrix) and v.dtype == dtype:
        return v
    if issparse(v):
        return csc_matrix(v, dtype=dtype)
    return csc_matrix(np.asarray(v, dtype=dtype))

def _sparse_zeros(m: int, n: int, dtype) -> csc_matrix:
    """Create a sparse (m, n) zero matrix."""
    return csc_matrix((m, n), dtype=dtype)

def _build_sparse_kkt(
    P: csc_matrix,
    H: csc_matrix,
    n_h: int,
    dtype,
) -> csc_matrix:
    """
    Build the KKT matrix in sparse form::

        K = [[P,   H^T],
             [H,   0  ]]

    All inputs and outputs are CSC.
    """
    if n_h == 0:
        return P.tocsc()

    zeros_block = _sparse_zeros(n_h, n_h, dtype)
    # sp_bmat returns COO → convert to CSC for splu
    return cast(csc_matrix,sp_bmat([
        [P, H.T],
        [H, zeros_block],
    ], format="csc"))

def _sparse_solve(K: csc_matrix, rhs: ndarray) -> ndarray:
    """
    Solve K @ x = rhs where K is sparse CSC and rhs is dense.

    Handles both vector rhs (1-D) and matrix rhs (2-D, multiple RHS).
    Factorizes K once via SuperLU, then solves per column.
    """
    lu = splu(K)
    if rhs.ndim == 1:
        return lu.solve(rhs)
    else:
        # Multiple RHS: solve each column with the same factorization
        return np.column_stack([
            lu.solve(rhs[:, i]) for i in range(rhs.shape[1])
        ])


# =====================================================================
# FORWARD (JVP) DIFFERENTIATOR
# =====================================================================

def create_sparse_kkt_differentiator_fwd(
    n_var: int,
    n_eq: int,
    n_ineq: int,
    options: Optional[DifferentiatorOptions] = None,
    fixed_elements: Optional[SparseQPIngredientsNP] = None,
):
    options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
    _dtype = options_parsed["dtype"]

    # ── Fixed elements: keep matrices as CSC, vectors as ndarray ─────
    #
    # We never densify fixed matrices. They stay sparse for LHS
    # assembly. Zero tangents for fixed elements are stored dense
    # (tangents are always dense for RHS computation since they
    # represent perturbations, not the matrices themselves).
    _fixed: dict[str, Union[csc_matrix, ndarray]] = {}
    _d_fixed: dict[str, ndarray] = {}

    if fixed_elements is not None:
        for k, v in fixed_elements.items():
            if issparse(v):
                _fixed[k] = csc_matrix(v, dtype=_dtype)
                _d_fixed[k] = np.zeros(v.shape, dtype=_dtype) #type: ignore
            else:
                _fixed[k] = np.asarray(v, dtype=_dtype).squeeze()
                _d_fixed[k] = np.zeros_like(np.asarray(v, dtype=_dtype).squeeze())

    # Zero placeholders for absent constraints
    if n_eq == 0:
        _fixed["A"]   = _sparse_zeros(0, n_var, _dtype)
        _fixed["b"]   = np.zeros((0,), dtype=_dtype)
        _d_fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
        _d_fixed["b"] = np.zeros((0,), dtype=_dtype)
    if n_ineq == 0:
        _fixed["G"]   = _sparse_zeros(0, n_var, _dtype)
        _fixed["h"]   = np.zeros((0,), dtype=_dtype)
        _d_fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
        _d_fixed["h"] = np.zeros((0,), dtype=_dtype)

    _d_fixed_batched = {k: np.expand_dims(v, 0) for k, v in _d_fixed.items()}

    # ─────────────────────────────────────────────────────────────────

    def kkt_differentiator_fwd(
        sol_np: QPOutputNP,
        dyn_primals_np: SparseQPIngredientsNP,
        dyn_tangents_np: SparseQPIngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[QPDiffOutNP, dict[str, float]]:

        t: dict[str, float] = {}
        start = perf_counter()
        batched = batch_size > 0

        x_np, lam_np, mu_np, active_np = sol_np

        # ── Merge primals (keep sparse matrices as-is) ───────────────
        # Dynamic primals arrive as CSC (from the converter).
        # Fixed primals are already stored as CSC / ndarray.
        prob = {**_fixed, **dyn_primals_np}

        P_sp = _ensure_csc(prob["P"], _dtype)

        # ── Merge tangents (always dense) ────────────────────────────
        # The tangent converter already provides dense ndarray from
        # BCOO. Fixed tangents are zero dense arrays.
        if batched:
            d_np = cast(dict[str, ndarray], {**_d_fixed_batched, **dyn_tangents_np})
        else:
            d_np = cast(dict[str, ndarray], {**_d_fixed, **dyn_tangents_np})

        n_active = int(np.sum(active_np))

        # ── Build sparse LHS ─────────────────────────────────────────
        # H = [A; G_active], stacked as sparse CSC.
        # Then K = [[P, H^T], [H, 0]] assembled via sp_bmat.
        H_parts = []
        if n_eq > 0:
            H_parts.append(_ensure_csc(prob["A"], _dtype))
        if n_ineq > 0 and n_active > 0:
            G_sp = _ensure_csc(prob["G"], _dtype)
            H_parts.append(G_sp[active_np, :])

        if H_parts:
            H_sp = cast(csc_matrix, sp_vstack(H_parts, format="csc"))
        else:
            H_sp = _sparse_zeros(0, n_var, _dtype)

        n_h = H_sp.shape[0] #type: ignore
        lhs = _build_sparse_kkt(P_sp, H_sp, n_h, _dtype)

        # ── Build dense RHS ──────────────────────────────────────────
        # Tangents are dense, so all RHS operations are standard numpy.
        # This is cheap — the RHS is (n_system,) or (n_system, batch).
        dL_np = d_np["P"] @ x_np + d_np["q"]
        rhs_pieces = []

        if batched:
            if n_eq > 0:
                dL_np = dL_np + d_np["A"].transpose(0, 2, 1) @ mu_np
                rhs_pieces.append(d_np["A"] @ x_np - d_np["b"])
            if n_ineq > 0 and n_active > 0:
                dG_active = d_np["G"][:, active_np, :]
                dL_np = dL_np + dG_active.transpose(0, 2, 1) @ lam_np[active_np]
                rhs_pieces.append(dG_active @ x_np - d_np["h"][:, active_np])

            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in [dL_np] + rhs_pieces
            ], axis=1).T
        else:
            if n_eq > 0:
                dL_np += d_np["A"].T @ mu_np
                rhs_pieces.append(d_np["A"] @ x_np - d_np["b"])
            if n_ineq > 0 and n_active > 0:
                dG_active = d_np["G"][active_np, :]
                dL_np += dG_active.T @ lam_np[active_np]
                rhs_pieces.append(dG_active @ x_np - d_np["h"][active_np])

            rhs = np.hstack([dL_np] + rhs_pieces)

        t["build_system"] = perf_counter() - start

        # ── Sparse solve ─────────────────────────────────────────────
        start = perf_counter()
        sol = _sparse_solve(lhs, -rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract dx, dlam, dmu ────────────────────────────────────
        if batch_size > 0:
            dx_np = sol[:n_var, :].T
            dmu_np = (
                sol[n_var:n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=_dtype)
            )
            dlam_np = np.zeros((batch_size, n_ineq), dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[:, active_np] = sol[n_var + n_eq:, :].T
        else:
            dx_np = sol[:n_var]
            dmu_np = (
                sol[n_var:n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            dlam_np = np.zeros(n_ineq, dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[active_np] = sol[n_var + n_eq:]

        return cast(QPDiffOutNP, (dx_np, dlam_np, dmu_np)), t

    return kkt_differentiator_fwd


# =====================================================================
# REVERSE (VJP) DIFFERENTIATOR
# =====================================================================

def create_sparse_kkt_differentiator_rev(
    n_var: int,
    n_eq: int,
    n_ineq: int,
    options: Optional[DifferentiatorOptions] = None,
    fixed_elements: Optional[SparseQPIngredientsNP] = None,
    dynamic_keys: Optional[Sequence[str]] = None,
):
    options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
    _dtype = options_parsed["dtype"]
    _bool_dtype = options_parsed["bool_dtype"]

    # ── Fixed elements: keep matrices as CSC ─────────────────────────
    _fixed: dict[str, Union[csc_matrix, ndarray]] = {}

    if fixed_elements is not None:
        for k, v in fixed_elements.items():
            if issparse(v):
                _fixed[k] = csc_matrix(v, dtype=_dtype)
            else:
                _fixed[k] = np.asarray(v, dtype=_dtype).squeeze()

    if n_eq == 0:
        _fixed["A"] = _sparse_zeros(0, n_var, _dtype)
        _fixed["b"] = np.zeros((0,), dtype=_dtype)
    if n_ineq == 0:
        _fixed["G"] = _sparse_zeros(0, n_var, _dtype)
        _fixed["h"] = np.zeros((0,), dtype=_dtype)

    # Pre-compute which keys need gradients
    if dynamic_keys is not None:
        _dyn_set = frozenset(dynamic_keys)
    else:
        _dyn_set = None

    def _need(key: str) -> bool:
        return _dyn_set is None or key in _dyn_set

    # ── Pre-extract sparsity patterns (row, col indices) ─────────────
    # These are fixed for the lifetime of the solver — the sparsity
    # structure never changes, only the values do.
    # We store them once so the inner function can compute gradients
    # at only the nonzero positions via fancy indexing: O(nnz) instead
    # of O(n^2) dense outer products.

    _sp_indices: dict[str, tuple[ndarray, ndarray]] = {}

    if fixed_elements is not None:
        for key in ("P", "A", "G"):
            val = fixed_elements.get(key)
            if val is not None and issparse(val):
                mat = csc_matrix(val)
                rows, cols = mat.nonzero()
                _sp_indices[key] = (np.asarray(rows), np.asarray(cols))

    # ─────────────────────────────────────────────────────────────────

    def kkt_differentiator_rev(
        dyn_primals_np: SparseQPIngredientsNP,
        x_np: ndarray,
        lam_np: ndarray,
        mu_np: ndarray,
        g_x: ndarray,
        g_lam: ndarray,
        g_mu: ndarray,
        batch_size: int,
    ) -> tuple[dict[str, ndarray], dict[str, float]]:

        t: dict[str, float] = {}
        batched = batch_size > 0

        # ── Merge primals (keep sparse) ──────────────────────────────
        prob = {**_fixed, **dyn_primals_np}
        P_sp = _ensure_csc(prob["P"], _dtype)

        # ── Cache sparsity indices for dynamic keys on first call ────
        # (Dynamic matrices may not be in fixed_elements, so we lazily
        #  extract their patterns here. The pattern is stable across
        #  calls because BCOO sparsity structure is fixed.)
        #TODO: this is not a pure function! Might be worth avoiding, but it works.
        for k in ("P", "A", "G"):
            if k not in _sp_indices and k in prob and issparse(prob[k]):
                mat = csc_matrix(prob[k])
                rows, cols = mat.nonzero()
                _sp_indices[k] = (np.asarray(rows), np.asarray(cols))

        # ── Active set ───────────────────────────────────────────────
        # G @ x works directly on CSC; result is a dense vector.
        start = perf_counter()
        if n_ineq > 0:
            G_sp = _ensure_csc(prob["G"], _dtype)
            Gx = np.asarray(G_sp @ x_np).ravel()
            h_vec = np.asarray(prob["h"], dtype=_dtype).ravel()
            active_np: Bool[ndarray, "n_ineq"] = np.asarray(
                np.abs(Gx - h_vec) <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            G_sp = _sparse_zeros(0, n_var, _dtype)
            active_np = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start

        n_active = int(np.sum(active_np))

        # ── Build sparse LHS ─────────────────────────────────────────
        start = perf_counter()

        H_parts = []
        if n_eq > 0:
            H_parts.append(_ensure_csc(prob["A"], _dtype))
        if n_ineq > 0 and n_active > 0:
            H_parts.append(G_sp[active_np, :])

        if H_parts:
            H_sp = sp_vstack(H_parts, format="csc")
        else:
            H_sp = _sparse_zeros(0, n_var, _dtype)

        
        n_h : int = H_sp.shape[0] #type: ignore
        lhs = _build_sparse_kkt(P_sp, H_sp, n_h, _dtype) #type: ignore

        # ── Build dense RHS from cotangent vectors ───────────────────
        if batched:
            if g_x.shape[0] == 1 and g_x.ndim > 1:
                g_x = np.broadcast_to(g_x, (batch_size, *g_x.shape[1:]))
            if g_lam.shape[0] == 1 and g_lam.ndim > 1:
                g_lam = np.broadcast_to(g_lam, (batch_size, *g_lam.shape[1:]))
            if g_mu.shape[0] == 1 and g_mu.ndim > 1:
                g_mu = np.broadcast_to(g_mu, (batch_size, *g_mu.shape[1:]))

            rhs_parts = [g_x.T]
            if n_eq > 0:
                rhs_parts.append(g_mu.T)
            if n_ineq > 0 and n_active > 0:
                rhs_parts.append(g_lam[:, active_np].T)
            rhs = np.vstack(rhs_parts)
        else:
            rhs_parts_list: list[ndarray] = [g_x]
            if n_eq > 0:
                rhs_parts_list.append(g_mu)
            if n_ineq > 0 and n_active > 0:
                rhs_parts_list.append(g_lam[active_np])
            rhs = np.hstack(rhs_parts_list)

        t["build_system"] = perf_counter() - start

        # ── Sparse solve ─────────────────────────────────────────────
        start = perf_counter()
        v = _sparse_solve(lhs, rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract adjoint variables ────────────────────────────────
        start = perf_counter()
        if batched:
            v_x = v[:n_var, :].T
            v_mu = (
                v[n_var:n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=_dtype)
            )
            v_lam_a = (
                v[n_var + n_eq:, :].T
                if (n_ineq > 0 and n_active > 0)
                else np.empty((batch_size, 0), dtype=_dtype)
            )
        else:
            v_x = v[:n_var]
            v_mu = (
                v[n_var:n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            v_lam_a = (
                v[n_var + n_eq:]
                if (n_ineq > 0 and n_active > 0)
                else np.empty(0, dtype=_dtype)
            )
        t["extract_adjoint"] = perf_counter() - start

        # ── Compute parameter cotangents (sparse) ────────────────────
        #
        # Given adjoint variables (v_x, v_mu, v_lam_a), compute
        # cotangents θ̄ = -v^T (∂F/∂θ) for each dynamic parameter θ.
        #
        # For matrix parameters (P, A, G), we exploit the known
        # sparsity pattern: instead of forming a full dense outer
        # product O(n²) and extracting nonzeros afterwards, we
        # compute gradient values *only* at the nonzero positions
        # via fancy indexing — O(nnz).
        #
        # The returned arrays for P, A, G are 1-D (length nnz) in
        # the unbatched case, or 2-D (batch, nnz) in the batched
        # case. The converter can wrap these directly into BCOO
        # using the stored indices, with no dense round-trip.
        #
        # Only dynamic keys are computed; fixed keys are skipped.

        start = perf_counter()
        grads: dict[str, ndarray] = {}

        #TODO: check this
        if batched:

            # P̄: grad only at nonzero positions
            # Dense equivalent: -(v_x[:, :, None] * x_np[None, None, :])
            if _need("P") and "P" in _sp_indices:
                P_rows, P_cols = _sp_indices["P"]
                # v_x is (batch, n_var), x_np is (n_var,)
                grads["P"] = -(v_x[:, P_rows] * x_np[P_cols])

            # q̄ = -v_x,  shape: (batch, n_var)
            if _need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                # Ā: grad only at nonzero positions
                # Dense equivalent: -(mu ⊗ v_x + v_mu ⊗ x)
                if _need("A") and "A" in _sp_indices:
                    A_rows, A_cols = _sp_indices["A"]
                    # mu_np is (n_eq,), v_x is (batch, n_var)
                    # v_mu is (batch, n_eq), x_np is (n_var,)
                    grads["A"] = -(
                        mu_np[A_rows] * v_x[:, A_cols]
                        + v_mu[:, A_rows] * x_np[A_cols]
                    )

                # b̄ = v_μ,  shape: (batch, n_eq)
                if _need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                # Ḡ: grad only at nonzero positions, only active rows contribute
                if _need("G") and "G" in _sp_indices:
                    G_rows, G_cols = _sp_indices["G"]
                    g_G = np.zeros((batch_size, len(G_rows)), dtype=_dtype)
                    if n_active > 0:
                        # Build map from active compressed index → full inequality index
                        # active_np is a bool mask of shape (n_ineq,)
                        # v_lam_a is (batch, n_active) — compressed to active rows only
                        active_indices = np.where(active_np)[0]
                        # For each nnz entry, check if its row is active
                        nnz_row_is_active = active_np[G_rows]
                        # Map G_rows to compressed active index for v_lam_a lookup
                        # full_to_active[i] gives the compressed index for inequality i
                        full_to_active = np.empty(n_ineq, dtype=np.intp)
                        full_to_active[active_indices] = np.arange(n_active)

                        active_nnz = np.where(nnz_row_is_active)[0]
                        compressed_rows = full_to_active[G_rows[active_nnz]]

                        g_G[:, active_nnz] = -(
                            lam_np[G_rows[active_nnz]] * v_x[:, G_cols[active_nnz]]
                            + v_lam_a[:, compressed_rows] * x_np[G_cols[active_nnz]]
                        )
                    grads["G"] = g_G

                # h̄: only active entries nonzero
                if _need("h"):
                    g_h_full = np.zeros((batch_size, n_ineq), dtype=_dtype)
                    if n_active > 0:
                        g_h_full[:, active_np] = v_lam_a
                    grads["h"] = g_h_full

        else:

            # P̄: grad only at nonzero positions
            # Dense equivalent: -np.outer(v_x, x_np)
            if _need("P") and "P" in _sp_indices:
                P_rows, P_cols = _sp_indices["P"]
                grads["P"] = -(v_x[P_rows] * x_np[P_cols])

            # q̄ = -v_x,  shape: (n_var,)
            if _need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                # Ā: grad only at nonzero positions
                # Dense equivalent: -(np.outer(mu_np, v_x) + np.outer(v_mu, x_np))
                if _need("A") and "A" in _sp_indices:
                    A_rows, A_cols = _sp_indices["A"]
                    grads["A"] = -(
                        mu_np[A_rows] * v_x[A_cols]
                        + v_mu[A_rows] * x_np[A_cols]
                    )

                # b̄ = v_μ,  shape: (n_eq,)
                if _need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                # Ḡ: grad only at nonzero positions, only active rows
                if _need("G") and "G" in _sp_indices:
                    G_rows, G_cols = _sp_indices["G"]
                    g_G = np.zeros(len(G_rows), dtype=_dtype)
                    if n_active > 0:
                        active_indices = np.where(active_np)[0]
                        nnz_row_is_active = active_np[G_rows]
                        full_to_active = np.empty(n_ineq, dtype=np.intp)
                        full_to_active[active_indices] = np.arange(n_active)

                        active_nnz = np.where(nnz_row_is_active)[0]
                        compressed_rows = full_to_active[G_rows[active_nnz]]

                        g_G[active_nnz] = -(
                            lam_np[G_rows[active_nnz]] * v_x[G_cols[active_nnz]]
                            + v_lam_a[compressed_rows] * x_np[G_cols[active_nnz]]
                        )
                    grads["G"] = g_G

                if _need("h"):
                    g_h_full = np.zeros(n_ineq, dtype=_dtype)
                    if n_active > 0:
                        g_h_full[active_np] = v_lam_a
                    grads["h"] = g_h_full

        t["compute_grads"] = perf_counter() - start

        return grads, t

    return kkt_differentiator_rev