"""
solver_sparse/differentiators.py
================================
KKT-based forward and reverse differentiators for the sparse QP path.

The KKT linear system is always dense (it's reduced to active
constraints), so the math is identical to the dense path.  The only
differences are:

1. Primal matrices (P, A, G) arrive as ``scipy.sparse.csc_matrix`` and
   must be densified for the block assembly.
2. Tangent matrices arrive as dense ndarray (the converter already
   handles BCOO → dense).
3. Reverse-mode gradients for P, A, G are returned as dense ndarray;
   the converter in ``solver_sparse/converters.py`` then extracts the
   nonzero-pattern entries before handing them back to JAX.
"""

from __future__ import annotations

from numpy import ndarray
from time import perf_counter
from jaxtyping import Float, Bool
import numpy as np
from typing import cast, Optional
from scipy.sparse import issparse

from src.solver_sparse.types import (
    SparseQPIngredientsNP,
    SparseQPIngredientsNPFull,
    SparseQPIngredientsTangentsNP,
)
from src.types_common import QPOutputNP, QPDiffOutNP
from src.options_common import DifferentiatorOptions
from src.utils.parsing_utils import parse_options
from src.solver_sparse.options import DEFAULT_DIFF_OPTIONS
from src.utils.linear_solvers import get_linear_solver


def _to_dense(v) -> ndarray:
    """Convert a value to a dense ndarray (handles csc_matrix and ndarray)."""
    if issparse(v):
        return v.toarray()
    return np.asarray(v)


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

    # ── Fixed elements (densified for KKT assembly) ──────────────────
    _fixed: dict[str, ndarray] = {}
    _d_fixed: dict[str, ndarray] = {}

    #TODO: this does not work, we need to maintain the sparsity pattern
    # when passing to differentiator. Should we store _d_fixed as sparse
    # too?
    if fixed_elements is not None:
        for k, v in fixed_elements.items():
            _fixed[k] = np.asarray(_to_dense(v), dtype=_dtype).squeeze()
            _d_fixed[k] = np.zeros_like(_fixed[k])

    # Zero matrices for absent constraints
    if n_eq == 0:
        _fixed["A"]   = np.zeros((0, n_var), dtype=_dtype)
        _fixed["b"]   = np.zeros((0,),       dtype=_dtype)
        _d_fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
        _d_fixed["b"] = np.zeros((0,),       dtype=_dtype)
    if n_ineq == 0:
        _fixed["G"]   = np.zeros((0, n_var), dtype=_dtype)
        _fixed["h"]   = np.zeros((0,),       dtype=_dtype)
        _d_fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
        _d_fixed["h"] = np.zeros((0,),       dtype=_dtype)

    _d_fixed_batched = {k: np.expand_dims(v, 0) for k, v in _d_fixed.items()}

    _solve_linear_system = get_linear_solver(options_parsed["linear_solver"])

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

        # ── Merge & densify primals ──────────────────────────────────
        #TODO: also no, we don't want to densify online as this takes
        # a lot of time
        dyn_dense = {
            k: np.asarray(_to_dense(v), dtype=_dtype).squeeze()
            for k, v in dyn_primals_np.items()
        }
        prob_np = cast(dict[str, ndarray], {**_fixed, **dyn_dense})

        # ── Merge tangents ───────────────────────────────────────────
        if batched:
            d_np = cast(dict[str, ndarray], {**_d_fixed_batched, **dyn_tangents_np})
        else:
            d_np = cast(dict[str, ndarray], {**_d_fixed, **dyn_tangents_np})

        n_active = int(np.sum(active_np))

        # ── Build LHS (identical to dense) ───────────────────────────
        #TODO: not sure if numpy is the correct choice here, since we want to 
        # preserve the sparsity
        H_parts: list[ndarray] = []
        if n_eq > 0:
            H_parts.append(prob_np["A"])
        if n_ineq > 0 and n_active > 0:
            H_parts.append(prob_np["G"][active_np, :])

        H_np = np.vstack(H_parts) if H_parts else np.empty((0, n_var), dtype=_dtype)
        n_h = H_np.shape[0]

        lhs = np.block([
            [prob_np["P"], H_np.T],
            [H_np,         np.zeros((n_h, n_h), dtype=_dtype)],
        ])

        # ── Build RHS ────────────────────────────────────────────────
        #TODO: probably ok to keep numpy for RHSs as this is usually a low amount
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

        # ── Solve ────────────────────────────────────────────────────
        start = perf_counter()
        sol = _solve_linear_system(lhs, -rhs)
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
):
    options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
    _dtype = options_parsed["dtype"]
    _bool_dtype = options_parsed["bool_dtype"]

    # ── Fixed elements (densified) ───────────────────────────────────
    _fixed: dict[str, ndarray] = {}

    if fixed_elements is not None:
        for k, v in fixed_elements.items():
            _fixed[k] = np.asarray(_to_dense(v), dtype=_dtype).squeeze()

    if n_eq == 0:
        _fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
        _fixed["b"] = np.zeros((0,),       dtype=_dtype)
    if n_ineq == 0:
        _fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
        _fixed["h"] = np.zeros((0,),       dtype=_dtype)

    _solve_linear_system = get_linear_solver(options_parsed["linear_solver"])

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
        start = perf_counter()
        batched = batch_size > 0

        # ── Merge & densify primals ──────────────────────────────────
        dyn_dense = {
            k: np.asarray(_to_dense(v), dtype=_dtype).squeeze()
            for k, v in dyn_primals_np.items()
        }
        prob_np = cast(dict[str, ndarray], {**_fixed, **dyn_dense})

        # ── Active set ───────────────────────────────────────────────
        start_a = perf_counter()
        if n_ineq > 0:
            active_np: Bool[ndarray, "n_ineq"] = np.asarray(
                np.abs(prob_np["G"] @ x_np - prob_np["h"])
                <= options_parsed["cst_tol"],
                dtype=_bool_dtype,
            ).reshape(-1)
        else:
            active_np = np.empty(0, dtype=_bool_dtype)
        t["active_set"] = perf_counter() - start_a

        n_active = int(np.sum(active_np))

        # ── Build LHS ────────────────────────────────────────────────
        H_parts: list[ndarray] = []
        if n_eq > 0:
            H_parts.append(prob_np["A"])
        if n_ineq > 0 and n_active > 0:
            H_parts.append(prob_np["G"][active_np, :])

        H_np = np.vstack(H_parts) if H_parts else np.empty((0, n_var), dtype=_dtype)
        n_h = H_np.shape[0]

        lhs = np.block([
            [prob_np["P"], H_np.T],
            [H_np,         np.zeros((n_h, n_h), dtype=_dtype)],
        ])

        # ── Build RHS ────────────────────────────────────────────────
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

        # ── Solve adjoint system ─────────────────────────────────────
        start = perf_counter()
        v = _solve_linear_system(lhs, rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract adjoint variables ────────────────────────────────
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

        # ── Compute parameter cotangents ─────────────────────────────
        #
        # NOTE: gradients for P, A, G are returned as *dense* matrices.
        # The converter (make_sparse_grad_to_jax) extracts the nonzero
        # entries before returning to JAX.

        start = perf_counter()
        grads: dict[str, ndarray] = {}

        if batched:
            grads["P"] = -np.einsum("bi,j->bij", v_x, x_np)
            grads["q"] = -v_x

            if n_eq > 0:
                term1 = np.einsum("i,bj->bij", mu_np, v_x)
                term2 = np.einsum("bi,j->bij", v_mu, x_np)
                grads["A"] = -(term1 + term2)
                grads["b"] = v_mu

            if n_ineq > 0:
                g_G = np.zeros((batch_size, n_ineq, n_var), dtype=_dtype)
                g_h_full = np.zeros((batch_size, n_ineq), dtype=_dtype)
                if n_active > 0:
                    term1 = np.einsum("i,bj->bij", lam_np[active_np], v_x)
                    term2 = np.einsum("bi,j->bij", v_lam_a, x_np)
                    g_G[:, active_np, :] = -(term1 + term2)
                    g_h_full[:, active_np] = v_lam_a
                grads["G"] = g_G
                grads["h"] = g_h_full
        else:
            grads["P"] = -np.outer(v_x, x_np)
            grads["q"] = -v_x

            if n_eq > 0:
                grads["A"] = -(np.outer(mu_np, v_x) + np.outer(v_mu, x_np))
                grads["b"] = v_mu

            if n_ineq > 0:
                g_G = np.zeros((n_ineq, n_var), dtype=_dtype)
                g_h_full = np.zeros(n_ineq, dtype=_dtype)
                if n_active > 0:
                    g_G[active_np, :] = -(
                        np.outer(lam_np[active_np], v_x)
                        + np.outer(v_lam_a, x_np)
                    )
                    g_h_full[active_np] = v_lam_a
                grads["G"] = g_G
                grads["h"] = g_h_full

        t["compute_grads"] = perf_counter() - start

        return grads, t

    return kkt_differentiator_rev
