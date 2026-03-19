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
  - Reverse-mode gradients for P, A, G are returned as dense ndarray;
    the converter extracts nonzero entries before returning to JAX.
"""

from __future__ import annotations

from numpy import ndarray
from time import perf_counter
from jaxtyping import Float, Bool
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
    SparseQPIngredientsNPFull,
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

        n_h = H_sp.shape[0]
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

        n_h = H_sp.shape[0]
        lhs = _build_sparse_kkt(P_sp, H_sp, n_h, _dtype)

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

        # ── Compute parameter cotangents ─────────────────────────────
        #
        # Given adjoint variables (v_x, v_mu, v_lam_a), compute
        # cotangents θ̄ = -v^T (∂F/∂θ) for each dynamic parameter θ.
        #
        # Gradients for P, A, G are returned as *dense* matrices.
        # The converter (make_sparse_grad_to_jax) extracts the nonzero
        # entries before returning to JAX.
        #
        # Only dynamic keys are computed; fixed keys are skipped.

        start = perf_counter()
        grads: dict[str, ndarray] = {}

        if batched:

            # P̄ = -v_x ⊗ x,  shape: (batch, n_var, n_var)
            if _need("P"):
                grads["P"] = -(v_x[:, :, None] * x_np[None, None, :])

            # q̄ = -v_x,  shape: (batch, n_var)
            if _need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                # Ā = -(μ ⊗ v_x + v_μ ⊗ x),  shape: (batch, n_eq, n_var)
                if _need("A"):
                    term1 = mu_np[None, :, None] * v_x[:, None, :]
                    term2 = v_mu[:, :, None] * x_np[None, None, :]
                    grads["A"] = -(term1 + term2)

                # b̄ = v_μ,  shape: (batch, n_eq)
                if _need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                # Ḡ: only active rows nonzero
                if _need("G"):
                    g_G = np.zeros((batch_size, n_ineq, n_var), dtype=_dtype)
                    if n_active > 0:
                        lam_active = lam_np[active_np]
                        term1 = lam_active[None, :, None] * v_x[:, None, :]
                        term2 = v_lam_a[:, :, None] * x_np[None, None, :]
                        g_G[:, active_np, :] = -(term1 + term2)
                    grads["G"] = g_G

                # h̄: only active entries nonzero
                if _need("h"):
                    g_h_full = np.zeros((batch_size, n_ineq), dtype=_dtype)
                    if n_active > 0:
                        g_h_full[:, active_np] = v_lam_a
                    grads["h"] = g_h_full

        else:

            # P̄ = -v_x ⊗ x,  shape: (n_var, n_var)
            if _need("P"):
                grads["P"] = -np.outer(v_x, x_np)

            # q̄ = -v_x,  shape: (n_var,)
            if _need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                # Ā = -(μ ⊗ v_x + v_μ ⊗ x),  shape: (n_eq, n_var)
                if _need("A"):
                    grads["A"] = -(np.outer(mu_np, v_x) + np.outer(v_mu, x_np))

                # b̄ = v_μ,  shape: (n_eq,)
                if _need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if _need("G"):
                    g_G = np.zeros((n_ineq, n_var), dtype=_dtype)
                    if n_active > 0:
                        g_G[active_np, :] = -(
                            np.outer(lam_np[active_np], v_x)
                            + np.outer(v_lam_a, x_np)
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