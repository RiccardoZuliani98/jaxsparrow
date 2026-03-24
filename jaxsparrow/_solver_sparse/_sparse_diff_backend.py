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

from time import perf_counter
from typing import Optional, Sequence, Union, cast

import numpy as np
from numpy import ndarray
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
from jaxsparrow._types_common import (
    SolverOutputNP,
    SolverDiffOutFwdNP,
    SolverDiffOutRevNP
)
from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._solver_sparse._options import DEFAULT_DIFF_OPTIONS
from jaxsparrow._utils._linear_solvers import (
    SparseLinearSolver, 
    get_sparse_linear_solver
)
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    register_differentiator_backend,
)

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
        options: Differentiator options. Supported keys:
            - dtype: Floating point dtype (default: np.float64)
            - bool_dtype: Boolean dtype (default: np.bool_)
            - cst_tol: Constraint tolerance (default: 1e-8)
            - linear_solver: Solver name (default: "splu")

    Raises:
        ValueError: If dimensions are negative or inconsistent.
        TypeError: If options contain invalid values.
    """

    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        options: Optional[DifferentiatorOptions] = None,
    ) -> None:
        if n_var < 0 or n_eq < 0 or n_ineq < 0:
            raise ValueError(
                f"Dimensions must be non-negative: "
                f"n_var={n_var}, n_eq={n_eq}, n_ineq={n_ineq}"
            )
        
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
        # Keys whose tangents are structurally zero (fixed and not
        # in dynamic_keys).  Forward-mode skips all arithmetic
        # involving these tangents instead of multiplying by zero.
        self._zero_tangent_keys: frozenset[str] = frozenset()

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[SparseIngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """One-time initialization: store fixed elements, zero
        tangents, sparsity indices, and dynamic key set.

        Args:
            fixed_elements: Ingredients constant across calls.
                Must have correct dimensions for the problem.
            dynamic_keys: Keys for which gradients are needed.
                None means gradients for all keys.
            sparsity_info: Per-key sparsity info from BCOO patterns
                for dynamic sparse keys.

        Returns:
            Timing dictionary with "setup" key.

        Raises:
            ValueError: If fixed_elements dimensions don't match
                n_var, n_eq, n_ineq.
            KeyError: If required keys are missing when constraints exist.
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

        # ── Determine which fixed keys have structurally zero tangents ──
        # A key has a zero tangent when:
        #   - it appears only in _d_fixed (not overridden by a dynamic
        #     tangent at call time), AND
        #   - dynamic_keys is not None (if None, all keys are
        #     potentially dynamic so nothing can be skipped).
        #
        # At differentiate_fwd time, any key in _zero_tangent_keys
        # will NOT appear in dyn_tangents_np (by contract), so we
        # can skip the corresponding matvec / addition.
        if dynamic_keys is not None:
            all_fixed_keys = frozenset(self._d_fixed.keys())
            dyn = frozenset(dynamic_keys)
            self._zero_tangent_keys = all_fixed_keys - dyn
        else:
            # All keys might be dynamic — nothing is guaranteed zero.
            self._zero_tangent_keys = frozenset()

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

    def _has_nonzero_tangent(self, key: str) -> bool:
        """Return True if the tangent for *key* is potentially nonzero.

        A tangent is structurally zero when the key is fixed and not
        listed among the dynamic keys.
        """
        return key not in self._zero_tangent_keys

    def _compute_active_set(
        self, 
        prob: SparseIngredientsNP, 
        x_np: ndarray,
    ) -> tuple[ndarray, csc_matrix]:
        """Compute the active inequality constraint mask.

        Returns:
            Tuple of:
                - active_np: Boolean array of shape (n_ineq,) where True
                  indicates an active inequality constraint.
                - G_sp: CSC inequality matrix (reused downstream to avoid
                  re-extraction).

        Raises:
            KeyError: If G or h are missing when n_ineq > 0.
        """
        if self._n_ineq > 0:
            if "G" not in prob or "h" not in prob:
                raise KeyError(
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
        self, 
        prob: SparseIngredientsNP, 
        P_sp: csc_matrix, 
        G_sp: csc_matrix,
        active_np: ndarray, 
        n_active: int,
    ) -> tuple[csc_matrix, int]:
        """Build the KKT LHS matrix.

        Returns:
            Tuple of (lhs_matrix, n_h) where n_h is the number of
            active constraints (equalities + active inequalities).

        Raises:
            KeyError: If A is missing when n_eq > 0.
        """
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
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]:
        """Forward-mode (JVP) through KKT conditions.

        Args:
            sol_np: Tuple of (x, lam, mu) solution arrays.
            dyn_primals_np: Dynamic ingredients (primal values).
            dyn_tangents_np: Tangents for dynamic ingredients.
            batch_size: Number of problems (0 for single, >0 for batched).

        Returns:
            Tuple of (dx, dlam, dmu) tangents and timing dict.

        Raises:
            KeyError: If required ingredients are missing.
            ValueError: If batch_size doesn't match array shapes.
            RuntimeError: If linear solver fails.
        """

        t: dict[str, float] = {}
        start = perf_counter()
        batched = batch_size > 0

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype
        _nz = self._has_nonzero_tangent

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

        # ── Merge tangents (only for keys with nonzero tangents) ─────
        # For keys in _zero_tangent_keys the tangent is structurally
        # zero, so we never look them up.  We merge only keys whose
        # tangent can actually be nonzero: dynamic tangents override
        # fixed ones as before.
        if batched:
            d_np: dict[str, ndarray] = {
                k: v for k, v in self._d_fixed_batched.items()
                if _nz(k)
            }
            d_np.update(dyn_tangents_np) #type: ignore
        else:
            d_np = {k: v for k, v in self._d_fixed.items() if _nz(k)}
            d_np.update(dyn_tangents_np) #type: ignore

        # ── Build KKT LHS ────────────────────────────────────────────
        lhs, n_h = self._build_kkt_lhs(prob, P_sp, G_sp, active_np, n_active)

        # ── Build dense RHS (skip zero tangent terms) ────────────────
        #
        # The full expression is:
        #   dL = dP @ x + dq  +  dA^T @ mu  +  dG_active^T @ lam_active
        #   rhs_eq = dA @ x - db
        #   rhs_ineq = dG_active @ x - dh_active
        #
        # Each addend is skipped when its tangent (dP, dq, dA, …) is
        # structurally zero, saving a matvec or vector add per term.

        rhs_pieces: list[ndarray] = []

        if batched:
            # -- dL accumulator: start from zero, add only live terms --
            dL_np = np.zeros((batch_size, n_var), dtype=_dtype)

            if _nz("P"):
                dL_np = dL_np + d_np["P"] @ x_np
            if _nz("q"):
                dL_np = dL_np + d_np["q"]

            if n_eq > 0:
                if _nz("A"):
                    dL_np = dL_np + d_np["A"].transpose(0, 2, 1) @ mu_np
                    rhs_pieces.append(d_np["A"] @ x_np)
                else:
                    rhs_pieces.append(
                        np.zeros((batch_size, n_eq), dtype=_dtype)
                    )

                if _nz("b"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["b"]

            if n_ineq > 0 and n_active > 0:
                if _nz("G"):
                    dG_active = d_np["G"][:, active_np, :]
                    dL_np = dL_np + dG_active.transpose(0, 2, 1) @ lam_np[active_np]
                    rhs_pieces.append(dG_active @ x_np)
                else:
                    rhs_pieces.append(
                        np.zeros((batch_size, n_active), dtype=_dtype)
                    )

                if _nz("h"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["h"][:, active_np]

            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in [dL_np] + rhs_pieces
            ], axis=1).T
        else:
            # -- dL accumulator: start from zero, add only live terms --
            dL_np = np.zeros(n_var, dtype=_dtype)

            if _nz("P"):
                dL_np = dL_np + d_np["P"] @ x_np
            if _nz("q"):
                dL_np = dL_np + d_np["q"]

            if n_eq > 0:
                if _nz("A"):
                    dL_np = dL_np + d_np["A"].T @ mu_np
                    rhs_pieces.append(d_np["A"] @ x_np)
                else:
                    rhs_pieces.append(np.zeros(n_eq, dtype=_dtype))

                if _nz("b"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["b"]

            if n_ineq > 0 and n_active > 0:
                if _nz("G"):
                    dG_active = d_np["G"][active_np, :]
                    dL_np = dL_np + dG_active.T @ lam_np[active_np]
                    rhs_pieces.append(dG_active @ x_np)
                else:
                    rhs_pieces.append(np.zeros(n_active, dtype=_dtype))

                if _nz("h"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["h"][active_np]

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

        return cast(SolverDiffOutFwdNP, (dx_np, dlam_np, dmu_np)), t

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
    ) -> tuple[SolverDiffOutRevNP, dict[str, float]]:
        """Reverse-mode (VJP) through KKT conditions.

        Args:
            dyn_primals_np: Dynamic ingredients (primal values).
            x_np: Primal variables at solution.
            lam_np: Inequality multipliers at solution.
            mu_np: Equality multipliers at solution.
            g_x: Cotangent of primal variables.
            g_lam: Cotangent of inequality multipliers.
            g_mu: Cotangent of equality multipliers.
            batch_size: Number of problems (0 for single, >0 for batched).

        Returns:
            Tuple of gradient dict and timing dict.

        Raises:
            KeyError: If required ingredients are missing.
            ValueError: If batch_size doesn't match array shapes.
            RuntimeError: If linear solver fails.
        """

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

        return cast(SolverDiffOutRevNP,grads), t


# ── Register in the backend registry ─────────────────────────────────
register_differentiator_backend("sparse_kkt", SparseKKTDifferentiatorBackend)