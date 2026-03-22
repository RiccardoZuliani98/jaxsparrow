"""
utils/_differentiator_backends.py
=================================
Abstract differentiator backend protocol and concrete implementations
for the sparse path.

The two-phase protocol separates differentiator lifecycle into:

1. **setup** — one-time initialization: receive the fixed QP
   ingredients, cast and store them, pre-compute zero tangents,
   extract sparsity indices, select the linear solver.

2. **differentiate_fwd / differentiate_rev** — per-call: receive
   the solution and dynamic ingredients, compute the active set,
   assemble the KKT system, solve it, and extract output
   tangents (fwd) or parameter cotangents (rev).

Backends
--------
- ``SparseKKTDifferentiatorBackend``: assembles the KKT system in
  sparse CSC form and solves with a configurable sparse linear
  solver (default: ``splu``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Optional, Sequence, Union, cast

import numpy as np
from numpy import ndarray
from jaxtyping import Float, Bool
from scipy.sparse import (
    csc_matrix,
    issparse,
    bmat as sp_bmat,
    vstack as sp_vstack,
)

from jaxsparrow._solver_sparse._types import (
    SparseIngredientsNP,
    SparseIngredientsTangentsNP,
)
from jaxsparrow._solver_sparse._converters import SparsityInfo
from jaxsparrow._types_common import SolverOutputNP, SolverDiffOutNP
from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._solver_sparse._options import DEFAULT_DIFF_OPTIONS
from jaxsparrow._utils._linear_solvers import SparseLinearSolver, get_sparse_linear_solver


# ── Sparse helpers ───────────────────────────────────────────────────

def _sparse_zeros(m: int, n: int, dtype) -> csc_matrix:
    """Create an empty sparse matrix with no stored entries."""
    return csc_matrix((m, n), dtype=dtype)


def _build_sparse_kkt(
    P: csc_matrix,
    H: csc_matrix,
    n_h: int,
    dtype,
) -> csc_matrix:
    """Assemble the KKT matrix in sparse CSC form.

    Builds::

        K = [[ P,  H^T ],
             [ H,   0  ]]

    Returns ``P`` directly when ``n_h == 0``.
    """
    if n_h == 0:
        return P.tocsc()

    zeros_block = _sparse_zeros(n_h, n_h, dtype)
    return cast(csc_matrix, sp_bmat([
        [P, H.T],
        [H, zeros_block],
    ], format="csc"))


# =====================================================================
# Sparse KKT backend
# =====================================================================

class SparseKKTDifferentiatorBackend(DifferentiatorBackend):
    """KKT-based differentiator for sparse problems.

    Assembles the KKT system in sparse CSC form and solves with a
    configurable sparse linear solver. Fixed elements, zero tangents,
    sparsity indices, and the linear solver are all set up once at
    construction time.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options (dtype, bool_dtype, cst_tol,
            linear_solver).
    """

    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        options: Optional[DifferentiatorOptions] = None,
    ) -> None:
        self._n_var = n_var
        self._n_eq = n_eq
        self._n_ineq = n_ineq

        options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
        self._dtype = options_parsed["dtype"]
        self._bool_dtype = options_parsed["bool_dtype"]
        self._cst_tol = options_parsed["cst_tol"]

        self._solve_linear_system: SparseLinearSolver = get_sparse_linear_solver(
            options_parsed["linear_solver"]
        )

        # Populated by setup()
        self._fixed: dict[str, Union[csc_matrix, ndarray]] = {}
        self._d_fixed: dict[str, ndarray] = {}
        self._d_fixed_batched: dict[str, ndarray] = {}
        self._sp_indices: dict[str, tuple[ndarray, ndarray]] = {}
        self._dyn_set: Optional[frozenset[str]] = None

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[SparseIngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """One-time initialization: store fixed elements, zero
        tangents, sparsity indices, and dynamic key set.
        """
        start = perf_counter()

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        # ── Store fixed elements and zero tangents ───────────────────
        self._fixed = {}
        self._d_fixed = {}

        if fixed_elements is not None:
            for k, v in fixed_elements.items():
                if issparse(v):
                    self._fixed[k] = csc_matrix(v, dtype=_dtype)
                    self._d_fixed[k] = np.zeros(v.shape, dtype=_dtype)  # type: ignore
                else:
                    self._fixed[k] = np.asarray(v, dtype=_dtype).squeeze()
                    self._d_fixed[k] = np.zeros_like(
                        np.asarray(v, dtype=_dtype).squeeze()
                    )

        # Zero placeholders for absent constraints
        if n_eq == 0:
            self._fixed["A"] = _sparse_zeros(0, n_var, _dtype)
            self._fixed["b"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
            self._d_fixed["b"] = np.zeros((0,), dtype=_dtype)
        if n_ineq == 0:
            self._fixed["G"] = _sparse_zeros(0, n_var, _dtype)
            self._fixed["h"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
            self._d_fixed["h"] = np.zeros((0,), dtype=_dtype)

        self._d_fixed_batched = {
            k: np.expand_dims(v, 0) for k, v in self._d_fixed.items()
        }

        # ── Dynamic key set ──────────────────────────────────────────
        if dynamic_keys is not None:
            self._dyn_set = frozenset(dynamic_keys)
        else:
            self._dyn_set = None

        # ── Sparsity indices ─────────────────────────────────────────
        self._sp_indices = {}

        # From sparsity_info (dynamic sparse keys)
        if sparsity_info is not None:
            for key in ("P", "A", "G"):
                if key in sparsity_info:
                    si = sparsity_info[key]
                    self._sp_indices[key] = (si["rows"], si["cols"])

        # From fixed_elements (fixed sparse keys not already covered)
        if fixed_elements is not None:
            for key in ("P", "A", "G"):
                if key not in self._sp_indices:
                    val = fixed_elements.get(key)
                    if val is not None and issparse(val):
                        mat = csc_matrix(val)
                        rows, cols = mat.nonzero()
                        self._sp_indices[key] = (
                            np.asarray(rows), np.asarray(cols),
                        )

        return {"setup": perf_counter() - start}

    # ── Helpers ──────────────────────────────────────────────────────

    def _need(self, key: str) -> bool:
        """Return True if gradient is needed for *key*."""
        return self._dyn_set is None or key in self._dyn_set

    def _compute_active_set(
        self, prob: dict, x_np: ndarray,
    ) -> tuple[Bool[ndarray, "n_ineq"], csc_matrix]:
        """Compute the active inequality constraint mask.

        Returns ``(active_np, G_sp)`` where ``G_sp`` is the CSC
        inequality matrix (reused downstream to avoid re-extraction).
        """
        if self._n_ineq > 0:
            assert "G" in prob and "h" in prob, (
                "G and h are required when n_ineq > 0. "
                "Provide them via fixed_elements or as dynamic arguments."
            )
            G_sp = prob["G"]
            Gx = np.asarray(G_sp @ x_np).ravel()
            h_vec = np.asarray(prob["h"], dtype=self._dtype).ravel()
            active_np = np.asarray(
                np.abs(Gx - h_vec) <= self._cst_tol,
                dtype=self._bool_dtype,
            ).reshape(-1)
        else:
            G_sp = _sparse_zeros(0, self._n_var, self._dtype)
            active_np = np.empty(0, dtype=self._bool_dtype)
        return active_np, G_sp

    def _build_kkt_lhs(
        self, prob: dict, P_sp: csc_matrix, G_sp: csc_matrix,
        active_np: ndarray, n_active: int,
    ) -> tuple[csc_matrix, int]:
        """Build the KKT LHS matrix. Returns ``(lhs, n_h)``."""
        H_parts: list[csc_matrix] = []
        if self._n_eq > 0:
            assert "A" in prob, (
                "A is required when n_eq > 0. "
                "Provide it via fixed_elements or as a dynamic argument."
            )
            H_parts.append(prob["A"])
        if self._n_ineq > 0 and n_active > 0:
            H_parts.append(G_sp[active_np, :])

        if H_parts:
            H_sp = cast(csc_matrix, sp_vstack(H_parts, format="csc"))
        else:
            H_sp = _sparse_zeros(0, self._n_var, self._dtype)

        n_h: int = H_sp.shape[0]  # type: ignore
        lhs = _build_sparse_kkt(P_sp, H_sp, n_h, self._dtype)
        return lhs, n_h

    # ── Forward differentiation ──────────────────────────────────────

    def differentiate_fwd(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: SparseIngredientsNP,
        dyn_tangents_np: SparseIngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[SolverDiffOutNP, dict[str, float]]:
        """Forward-mode (JVP) through KKT conditions."""

        t: dict[str, float] = {}
        start = perf_counter()
        batched = batch_size > 0

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        x_np, lam_np, mu_np = sol_np

        # ── Merge primals ────────────────────────────────────────────
        prob = cast(SparseIngredientsNP, {**self._fixed, **dyn_primals_np})

        assert "P" in prob and "q" in prob, (
            "P and q are required. "
            "Provide them via fixed_elements or as dynamic arguments."
        )
        P_sp = prob["P"]

        # ── Active set ───────────────────────────────────────────────
        active_np, G_sp = self._compute_active_set(prob, x_np)
        n_active = int(np.sum(active_np))

        # ── Merge tangents ───────────────────────────────────────────
        if batched:
            d_np = cast(dict[str, ndarray], {**self._d_fixed_batched, **dyn_tangents_np})
        else:
            d_np = cast(dict[str, ndarray], {**self._d_fixed, **dyn_tangents_np})

        # ── Build KKT LHS ────────────────────────────────────────────
        lhs, n_h = self._build_kkt_lhs(prob, P_sp, G_sp, active_np, n_active)

        # ── Build dense RHS ──────────────────────────────────────────
        dL_np = d_np["P"] @ x_np + d_np["q"]
        rhs_pieces: list[ndarray] = []

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
        sol = self._solve_linear_system(lhs, -rhs)
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

        return cast(SolverDiffOutNP, (dx_np, dlam_np, dmu_np)), t

    # ── Reverse differentiation ──────────────────────────────────────

    def differentiate_rev(
        self,
        dyn_primals_np: SparseIngredientsNP,
        x_np: ndarray,
        lam_np: ndarray,
        mu_np: ndarray,
        g_x: ndarray,
        g_lam: ndarray,
        g_mu: ndarray,
        batch_size: int,
    ) -> tuple[dict[str, ndarray], dict[str, float]]:
        """Reverse-mode (VJP) through KKT conditions."""

        t: dict[str, float] = {}
        batched = batch_size > 0

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        # ── Merge primals ────────────────────────────────────────────
        prob = cast(SparseIngredientsNP, {**self._fixed, **dyn_primals_np})

        assert "P" in prob and "q" in prob, (
            "P and q are required. "
            "Provide them via fixed_elements or as dynamic arguments."
        )
        P_sp = prob["P"]

        # ── Active set ───────────────────────────────────────────────
        start = perf_counter()
        active_np, G_sp = self._compute_active_set(prob, x_np)
        n_active = int(np.sum(active_np))
        t["active_set"] = perf_counter() - start

        # ── Build KKT LHS ────────────────────────────────────────────
        start = perf_counter()
        lhs, n_h = self._build_kkt_lhs(prob, P_sp, G_sp, active_np, n_active)

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
        v = self._solve_linear_system(lhs, rhs)
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
        start = perf_counter()
        grads: dict[str, ndarray] = {}
        _sp = self._sp_indices

        if batched:
            if self._need("P") and "P" in _sp:
                P_rows, P_cols = _sp["P"]
                grads["P"] = -(v_x[:, P_rows] * x_np[P_cols])

            if self._need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                if self._need("A") and "A" in _sp:
                    A_rows, A_cols = _sp["A"]
                    grads["A"] = -(
                        mu_np[A_rows] * v_x[:, A_cols]
                        + v_mu[:, A_rows] * x_np[A_cols]
                    )
                if self._need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if self._need("G") and "G" in _sp:
                    G_rows, G_cols = _sp["G"]
                    g_G = np.zeros((batch_size, len(G_rows)), dtype=_dtype)
                    if n_active > 0:
                        active_indices = np.where(active_np)[0]
                        nnz_row_is_active = active_np[G_rows]
                        full_to_active = np.empty(n_ineq, dtype=np.intp)
                        full_to_active[active_indices] = np.arange(n_active)
                        active_nnz = np.where(nnz_row_is_active)[0]
                        compressed_rows = full_to_active[G_rows[active_nnz]]
                        g_G[:, active_nnz] = -(
                            lam_np[G_rows[active_nnz]] * v_x[:, G_cols[active_nnz]]
                            + v_lam_a[:, compressed_rows] * x_np[G_cols[active_nnz]]
                        )
                    grads["G"] = g_G

                if self._need("h"):
                    g_h_full = np.zeros((batch_size, n_ineq), dtype=_dtype)
                    if n_active > 0:
                        g_h_full[:, active_np] = v_lam_a
                    grads["h"] = g_h_full

        else:
            if self._need("P") and "P" in _sp:
                P_rows, P_cols = _sp["P"]
                grads["P"] = -(v_x[P_rows] * x_np[P_cols])

            if self._need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                if self._need("A") and "A" in _sp:
                    A_rows, A_cols = _sp["A"]
                    grads["A"] = -(
                        mu_np[A_rows] * v_x[A_cols]
                        + v_mu[A_rows] * x_np[A_cols]
                    )
                if self._need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if self._need("G") and "G" in _sp:
                    G_rows, G_cols = _sp["G"]
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

                if self._need("h"):
                    g_h_full = np.zeros(n_ineq, dtype=_dtype)
                    if n_active > 0:
                        g_h_full[active_np] = v_lam_a
                    grads["h"] = g_h_full

        t["compute_grads"] = perf_counter() - start

        return grads, t


# =====================================================================
# Registry and factory
# =====================================================================

_DIFF_BACKEND_REGISTRY: dict[str, type[DifferentiatorBackend]] = {
    "kkt": SparseKKTDifferentiatorBackend,
}


def register_differentiator_backend(
    name: str, cls: type[DifferentiatorBackend],
) -> None:
    """Register a new differentiator backend."""
    if not (isinstance(cls, type) and issubclass(cls, DifferentiatorBackend)):
        raise TypeError(
            f"Expected a DifferentiatorBackend subclass, got {cls!r}"
        )
    _DIFF_BACKEND_REGISTRY[name] = cls


def get_differentiator_backend(name: str, **kwargs: Any) -> DifferentiatorBackend:
    """Instantiate a differentiator backend by name.

    Args:
        name: Registered backend name (e.g. ``"kkt"``).
        **kwargs: Passed to the backend constructor.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _DIFF_BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown differentiator backend: {name!r}. "
            f"Available: {sorted(_DIFF_BACKEND_REGISTRY)}."
        )
    return _DIFF_BACKEND_REGISTRY[name](**kwargs)