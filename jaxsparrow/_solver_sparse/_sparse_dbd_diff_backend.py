"""
solver_sparse/_sparse_dbd_diff_backend.py
==========================================
Sparse "Differentiable-by-Design" (DBD) differentiator backend.

Implements the regularized KKT system from

    Zuliani, Balta & Lygeros,
    "Differentiable-by-design Nonlinear Optimization
     for Model Predictive Control", arXiv:2509.12692v3,
    Equations (11) and (12).

This is the sparse analogue of
:class:`DenseDBDDifferentiatorBackend`.  All matrices are assembled
in SciPy CSC form and solved with a configurable sparse linear
solver (default: ``splu``).

The key idea is to add a symmetric perturbation ``B^rho`` to the
standard KKT matrix ``A_theta`` and an offset ``d^rho`` to the
right-hand side so that the augmented system is non-singular even
when the original problem violates SSOSC, LICQ, or SCS.  The
perturbation is parameterised by a scalar ``rho > 0`` and involves
the *inactive* inequality constraints.

Specialisation to QPs
~~~~~~~~~~~~~~~~~~~~~
For the QP

    min  0.5 x'Px + q'x
    s.t. Ax = b,   Gx <= h

the constraint function is ``g(x) = Gx - h``, giving::

    nabla_x g = G
    z_i^2     = 2 (h - Gx)_i      (slack variables for inactive rows)

The regularized KKT LHS becomes (partitioned by *active*
inequality rows ``I`` and equality rows)::

    [ P + H + rho I    G_I^T    A^T  ]
    [ G_I             -rho I     0   ]
    [ A                 0      -rho I]

where  ``H = G_Ibar^T  diag(rho / (z_i^2 + rho^2))  G_Ibar``
and ``Ibar`` denotes inactive constraints.

The RHS receives an additive offset ``l`` in the stationarity
(first) block::

    l = G_Ibar^T  diag(rho / (z_i^2 + rho^2))  nabla_theta g_Ibar

Registered as ``"sparse_dbd"`` in the differentiator backend
registry.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Sequence, Union, cast

import numpy as np
from numpy import ndarray
from scipy.sparse import (
    csc_matrix,
    issparse,
    eye as sp_eye,
    bmat as sp_bmat,
    diags as sp_diags,
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
    SolverDiffOutRevNP,
)
from jaxsparrow._solver_sparse._options import SparseDBDDiffOptionsFull
from jaxsparrow._utils._linear_solvers import (
    SparseLinearSolver,
    get_sparse_linear_solver,
)
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    register_differentiator_backend,
)


# ── Sparse helpers ───────────────────────────────────────────────────

def _sparse_zeros(m: int, n: int, dtype) -> csc_matrix:
    """Create an empty sparse matrix with no stored entries."""
    return csc_matrix((m, n), dtype=dtype)


# =====================================================================
# Sparse DBD backend
# =====================================================================

class SparseDBDDifferentiatorBackend(DifferentiatorBackend):
    """Differentiable-by-Design differentiator for sparse QPs.

    Assembles the regularized KKT system of Zuliani et al.
    (Eqs. 11-12) in sparse CSC form and solves with a configurable
    sparse linear solver.

    Compared with the standard ``sparse_kkt`` backend, the LHS
    receives an additive symmetric perturbation

        ``B^rho = diag(H + rho I,  -rho I,  -rho I)``

    and the RHS receives an offset whose only non-zero block
    is ``l`` (Eq. 12), placed in the stationarity rows.

    Args:
        n_var:  Number of decision variables.
        n_eq:   Number of equality constraints (0 if none).
        n_ineq: Number of inequality constraints (0 if none).
        options: Differentiator options.  Recognised keys beyond
            those of ``sparse_kkt``:

            - ``rho``  *(float, default 1e-5)*:
              Regularisation strength.  Must be ``> 0``.
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        options: SparseDBDDiffOptionsFull,
    ) -> None:
        if n_var < 0 or n_eq < 0 or n_ineq < 0:
            raise ValueError(
                f"Dimensions must be non-negative: "
                f"n_var={n_var}, n_eq={n_eq}, n_ineq={n_ineq}"
            )

        self._n_var = n_var
        self._n_eq = n_eq
        self._n_ineq = n_ineq

        self._dtype = options["dtype"]
        self._bool_dtype = options["bool_dtype"]
        self._cst_tol = options["cst_tol"]
        self._rho: float = options["rho"]
        if self._rho <= 0:
            raise ValueError(f"rho must be > 0, got {self._rho}")

        self._solve_linear_system: SparseLinearSolver = (
            get_sparse_linear_solver(options["linear_solver"])
        )

        # Populated by setup()
        self._fixed: dict[str, Union[csc_matrix, ndarray]] = {}
        self._d_fixed: dict[str, Union[csc_matrix, ndarray]] = {}
        self._d_fixed_batched: dict[str, Union[list[csc_matrix], ndarray]] = {}
        self._sp_indices: dict[str, tuple[ndarray, ndarray]] = {}
        self._dyn_set: Optional[frozenset[str]] = None
        self._zero_tangent_keys: frozenset[str] = frozenset()

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[SparseIngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """Store fixed elements, zero tangents, sparsity indices,
        and dynamic key set.

        Semantics are identical to
        :meth:`SparseKKTDifferentiatorBackend.setup`.
        """
        start = perf_counter()

        n_var, n_eq, n_ineq = self._n_var, self._n_eq, self._n_ineq
        _dtype = self._dtype

        # ── Store fixed elements and zero tangents ───────────────────
        self._fixed = {}
        self._d_fixed = {}

        if fixed_elements is not None:
            # ── Shape validation ─────────────────────────────────────
            _expected_shapes: dict[str, tuple[int, ...]] = {
                "P": (n_var, n_var),
                "q": (n_var,),
                "A": (n_eq, n_var),
                "b": (n_eq,),
                "G": (n_ineq, n_var),
                "h": (n_ineq,),
            }
            for k, v in fixed_elements.items():
                if k in _expected_shapes:
                    shape = v.shape
                    if issparse(v):
                        expected = _expected_shapes[k]
                    else:
                        expected = _expected_shapes[k]
                        shape = np.asarray(v).squeeze().shape
                    if shape != expected:
                        raise ValueError(
                            f"{k} shape {shape} != {expected}"
                        )

            for k, v in fixed_elements.items():
                if issparse(v):
                    self._fixed[k] = csc_matrix(v, dtype=_dtype)
                    mat = csc_matrix(v, dtype=_dtype)
                    self._d_fixed[k] = csc_matrix(
                        (np.zeros(mat.nnz, dtype=_dtype),
                         mat.indices.copy(), mat.indptr.copy()),
                        shape=mat.shape,
                    )
                else:
                    self._fixed[k] = np.asarray(v, dtype=_dtype).squeeze()
                    self._d_fixed[k] = np.zeros_like(
                        np.asarray(v, dtype=_dtype).squeeze()
                    )

        # Zero placeholders for absent constraints
        if n_eq == 0:
            self._fixed["A"] = _sparse_zeros(0, n_var, _dtype)
            self._fixed["b"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["A"] = _sparse_zeros(0, n_var, _dtype)
            self._d_fixed["b"] = np.zeros((0,), dtype=_dtype)
        if n_ineq == 0:
            self._fixed["G"] = _sparse_zeros(0, n_var, _dtype)
            self._fixed["h"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["G"] = _sparse_zeros(0, n_var, _dtype)
            self._d_fixed["h"] = np.zeros((0,), dtype=_dtype)

        # Batched versions
        self._d_fixed_batched = {}
        for k, v in self._d_fixed.items():
            if issparse(v):
                self._d_fixed_batched[k] = [v]
            else:
                self._d_fixed_batched[k] = np.expand_dims(v, 0)

        # ── Dynamic key set ──────────────────────────────────────────
        if dynamic_keys is not None:
            self._dyn_set = frozenset(dynamic_keys)
            all_fixed_keys = frozenset(self._d_fixed.keys())
            self._zero_tangent_keys = all_fixed_keys - self._dyn_set
        else:
            self._dyn_set = None
            self._zero_tangent_keys = frozenset()

        # ── Sparsity indices ─────────────────────────────────────────
        self._sp_indices = {}

        if sparsity_info is not None:
            for key in ("P", "A", "G"):
                if key in sparsity_info:
                    si = sparsity_info[key]
                    self._sp_indices[key] = (si["rows"], si["cols"])

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
        """Return True if the tangent for *key* is potentially
        nonzero."""
        return key not in self._zero_tangent_keys

    def _compute_active_inactive(
        self,
        residual: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Return ``(active, inactive)`` boolean masks of shape
        ``(n_ineq,)``."""
        if self._n_ineq > 0:
            active = np.asarray(
                residual <= self._cst_tol,
                dtype=self._bool_dtype,
            ).reshape(-1)
            return active, ~active
        empty = np.empty(0, dtype=self._bool_dtype)
        return empty, empty

    def _inactive_weights(
        self,
        slack: ndarray,
        inactive: ndarray,
    ) -> ndarray:
        """Compute  ``w_i = rho / (z_i^2 + rho^2)`` for inactive
        constraints, where ``z_i^2 = 2(h - Gx)_i``.

        Slack values that are negative or below ``cst_tol`` are
        clamped to zero before computing the weights.  This avoids
        numerical issues when a constraint is only marginally
        inactive.
        """
        n_inactive = int(np.sum(inactive))
        if self._n_ineq == 0 or n_inactive == 0:
            return np.empty(0, dtype=self._dtype)
        z_sq = 2.0 * slack[inactive]
        z_sq[z_sq < self._cst_tol] = 0.0
        return self._rho / (z_sq + self._rho ** 2)

    # ── Regularized KKT assembly (Eqs. 11-12, sparse) ───────────────

    def _build_regularized_kkt(
        self,
        prob: SparseIngredientsNP,
        x_np: ndarray,
        active: ndarray,
        inactive: ndarray,
        n_active: int,
        G_sp: csc_matrix,
        slack: ndarray,
    ) -> tuple[csc_matrix, int, ndarray, csc_matrix]:
        """Build the regularized KKT LHS in sparse CSC form.

        Returns:
            ``(lhs, n_h, w_inact, G_inact)``

            *  ``lhs``      - regularized KKT matrix (CSC)
            *  ``n_h``      - total constraint rows
                              (``n_eq + n_active``)
            *  ``w_inact``  - weight vector for inactive rows
            *  ``G_inact``  - rows of ``G`` for inactive
                              constraints (CSC)
        """
        n_var, n_eq = self._n_var, self._n_eq
        rho, _dtype = self._rho, self._dtype
        n_inactive = int(np.sum(inactive))

        P_sp: csc_matrix = prob["P"]

        # ── H from inactive constraints (Eq. 12) ────────────────────
        w_inact = self._inactive_weights(slack, inactive)
        if n_inactive > 0:
            G_inact = G_sp[inactive, :]
            W_diag = sp_diags(w_inact, format="csc")
            H_reg = G_inact.T @ W_diag @ G_inact
        else:
            G_inact = _sparse_zeros(0, n_var, _dtype)
            H_reg = _sparse_zeros(n_var, n_var, _dtype)

        # ── Constraint rows (equality + active inequality) ───────────
        C_parts: list[csc_matrix] = []
        if n_eq > 0:
            if "A" not in prob:
                raise ValueError(
                    "A is required when n_eq > 0. "
                    "Provide it via fixed_elements or as a dynamic argument."
                )
            C_parts.append(prob["A"])
        if self._n_ineq > 0 and n_active > 0:
            C_parts.append(G_sp[active, :])

        C_sp = (
            cast(csc_matrix, sp_vstack(C_parts, format="csc"))
            if C_parts
            else _sparse_zeros(0, n_var, _dtype)
        )
        n_h: int = C_sp.shape[0]

        # ── Assemble LHS ────────────────────────────────────────────
        UL = P_sp + H_reg + rho * sp_eye(n_var, dtype=_dtype, format="csc")
        LR = -rho * sp_eye(n_h, dtype=_dtype, format="csc")

        if n_h > 0:
            lhs = cast(csc_matrix, sp_bmat([
                [UL,   C_sp.T],
                [C_sp, LR    ],
            ], format="csc"))
        else:
            lhs = UL.tocsc()

        return lhs, n_h, w_inact, G_inact

    # ── Forward differentiation ──────────────────────────────────────

    def differentiate_fwd(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: SparseIngredientsNP,
        dyn_tangents_np: SparseIngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]:
        """Forward-mode (JVP) through the regularized KKT system.

        Solves ``[A_theta + B^rho] V = -(b_theta + d^rho)`` for
        the tangent direction encoded in *dyn_tangents_np*.

        Sparse matrix tangents (dP, dA, dG) may arrive as:
        - ``csc_matrix`` in the unbatched case, or
        - ``list[csc_matrix]`` in the batched case.

        The batched path loops over batch elements performing sparse
        matvecs, avoiding dense ``(batch, m, n)`` materialization.
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

        # ── Merge primals & tangents ─────────────────────────────────
        prob = cast(SparseIngredientsNP, {**self._fixed, **dyn_primals_np})

        if "P" not in prob or "q" not in prob:
            raise ValueError(
                "P and q are required. "
                "Provide them via fixed_elements or as dynamic arguments."
            )

        if batched:
            d_np: dict[str, Union[ndarray, list[csc_matrix], csc_matrix]] = {
                k: v for k, v in self._d_fixed_batched.items()
                if _nz(k)
            }
            d_np.update(dyn_tangents_np)  # type: ignore
        else:
            d_np = {k: v for k, v in self._d_fixed.items() if _nz(k)}
            d_np.update(dyn_tangents_np)  # type: ignore

        # ── Active / inactive sets ───────────────────────────────────
        if self._n_ineq > 0:
            G_sp = prob["G"]
            Gx = np.asarray(G_sp @ x_np).ravel()
            h_vec = np.asarray(prob["h"], dtype=self._dtype).ravel()
            residual = np.abs(Gx - h_vec)
            slack = h_vec - Gx
        else:
            G_sp = _sparse_zeros(0, self._n_var, _dtype)
            residual = np.empty(0, dtype=_dtype)
            slack = np.empty(0, dtype=_dtype)
        active, inactive = self._compute_active_inactive(residual)
        n_active = int(np.sum(active))
        n_inactive = int(np.sum(inactive))
        t["active_set"] = perf_counter() - start

        # ── Regularized LHS ──────────────────────────────────────────
        start = perf_counter()
        lhs, n_h, w_inact, G_inact = self._build_regularized_kkt(
            prob, x_np, active, inactive, n_active, G_sp, slack,
        )

        # ── Build RHS (b_theta + d^rho) contracted with tangent ──────
        rhs_pieces: list[ndarray] = []

        if batched:
            dL_np = np.zeros((batch_size, n_var), dtype=_dtype)

            if _nz("P"):
                dP_list = d_np["P"]
                for i, dP_i in enumerate(dP_list):  # type: ignore
                    dL_np[i] += dP_i @ x_np
            if _nz("q"):
                dL_np = dL_np + d_np["q"]

            # d^rho offset (l term) in stationarity block
            if n_inactive > 0:
                dg_inact = np.zeros((batch_size, n_inactive), dtype=_dtype)
                if _nz("G"):
                    dG_list = d_np["G"]
                    for i, dG_i in enumerate(dG_list):  # type: ignore
                        dg_inact[i] += np.asarray(
                            dG_i[inactive, :] @ x_np
                        ).ravel()
                if _nz("h"):
                    dg_inact = dg_inact - d_np["h"][:, inactive]
                # l = G_Ibar^T diag(w) dg_Ibar
                dL_np = dL_np + (
                    dg_inact * w_inact[None, :]
                ) @ G_inact.toarray()

            # Equality constraints
            if n_eq > 0:
                if _nz("A"):
                    dA_list = d_np["A"]
                    rhs_eq = np.zeros((batch_size, n_eq), dtype=_dtype)
                    for i, dA_i in enumerate(dA_list):  # type: ignore
                        dL_np[i] += dA_i.T @ mu_np
                        rhs_eq[i] = np.asarray(dA_i @ x_np).ravel()
                    rhs_pieces.append(rhs_eq)
                else:
                    rhs_pieces.append(
                        np.zeros((batch_size, n_eq), dtype=_dtype)
                    )
                if _nz("b"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["b"]

            # Active inequality constraints
            if n_ineq > 0 and n_active > 0:
                lam_active = lam_np[active]
                if _nz("G"):
                    dG_list = d_np["G"]
                    rhs_ineq = np.zeros(
                        (batch_size, n_active), dtype=_dtype
                    )
                    for i, dG_i in enumerate(dG_list):  # type: ignore
                        dG_i_active = dG_i[active, :]
                        dL_np[i] += dG_i_active.T @ lam_active
                        rhs_ineq[i] = np.asarray(
                            dG_i_active @ x_np
                        ).ravel()
                    rhs_pieces.append(rhs_ineq)
                else:
                    rhs_pieces.append(
                        np.zeros((batch_size, n_active), dtype=_dtype)
                    )
                if _nz("h"):
                    rhs_pieces[-1] = (
                        rhs_pieces[-1] - d_np["h"][:, active]
                    )

            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in [dL_np] + rhs_pieces
            ], axis=1).T

        else:  # unbatched
            dL_np_1d = np.zeros(n_var, dtype=_dtype)

            if _nz("P"):
                dL_np_1d = dL_np_1d + d_np["P"] @ x_np
            if _nz("q"):
                dL_np_1d = dL_np_1d + d_np["q"]

            # d^rho offset (l term)
            if n_inactive > 0:
                dg_inact = np.zeros(n_inactive, dtype=_dtype)
                if _nz("G"):
                    dg_inact = dg_inact + np.asarray(
                        d_np["G"][inactive, :] @ x_np
                    ).ravel()
                if _nz("h"):
                    dg_inact = dg_inact - d_np["h"][inactive]
                dL_np_1d = dL_np_1d + np.asarray(
                    G_inact.T @ (w_inact * dg_inact)
                ).ravel()

            if n_eq > 0:
                if _nz("A"):
                    dL_np_1d = dL_np_1d + d_np["A"].T @ mu_np
                    rhs_pieces.append(
                        np.asarray(d_np["A"] @ x_np).ravel()
                    )
                else:
                    rhs_pieces.append(np.zeros(n_eq, dtype=_dtype))
                if _nz("b"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["b"]

            if n_ineq > 0 and n_active > 0:
                if _nz("G"):
                    dG_active = d_np["G"][active, :]
                    dL_np_1d = dL_np_1d + dG_active.T @ lam_np[active]
                    rhs_pieces.append(
                        np.asarray(dG_active @ x_np).ravel()
                    )
                else:
                    rhs_pieces.append(np.zeros(n_active, dtype=_dtype))
                if _nz("h"):
                    rhs_pieces[-1] = (
                        rhs_pieces[-1] - d_np["h"][active]
                    )

            rhs = np.hstack([dL_np_1d] + rhs_pieces)

        t["build_system"] = perf_counter() - start

        # ── Sparse solve ─────────────────────────────────────────────
        start = perf_counter()
        sol = self._solve_linear_system(lhs, -rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract dx, dlam, dmu ────────────────────────────────────
        if batched:
            dx_np = sol[:n_var, :].T
            dmu_np = (
                sol[n_var:n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=_dtype)
            )
            dlam_np = np.zeros((batch_size, n_ineq), dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[:, active] = sol[n_var + n_eq:, :].T
        else:
            dx_np = sol[:n_var]
            dmu_np = (
                sol[n_var:n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            dlam_np = np.zeros(n_ineq, dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[active] = sol[n_var + n_eq:]

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
        """Reverse-mode (VJP) through the regularized KKT system.

        The regularized LHS ``A_theta + B^rho`` is symmetric in
        its block structure, so the same matrix is used for the
        adjoint solve.  Parameter cotangents additionally receive
        a correction from the ``l`` offset for inactive
        constraints.

        The adjoint of the forward ``l`` contribution::

            dL_x += G_Ibar^T diag(w) (dG_Ibar @ x - dh_Ibar)

        yields the following corrections to the parameter
        cotangents (sparse-indexed for G, dense for h)::

            grad_G[inactive_nnz] -= w[row] * (G_Ibar @ v_x)[row] * x[col]
            grad_h[inactive]     += w * G_Ibar @ v_x
        """
        t: dict[str, float] = {}
        batched = batch_size > 0

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        # ── Merge primals ────────────────────────────────────────────
        prob = cast(SparseIngredientsNP, {**self._fixed, **dyn_primals_np})

        # ── Active / inactive ────────────────────────────────────────
        start = perf_counter()
        if self._n_ineq > 0:
            G_sp = prob["G"]
            Gx = np.asarray(G_sp @ x_np).ravel()
            h_vec = np.asarray(prob["h"], dtype=self._dtype).ravel()
            residual = np.abs(Gx - h_vec)
            slack = h_vec - Gx
        else:
            G_sp = _sparse_zeros(0, self._n_var, _dtype)
            residual = np.empty(0, dtype=_dtype)
            slack = np.empty(0, dtype=_dtype)
        active, inactive = self._compute_active_inactive(residual)
        n_active = int(np.sum(active))
        n_inactive = int(np.sum(inactive))
        t["active_set"] = perf_counter() - start

        # ── Regularized LHS ──────────────────────────────────────────
        start = perf_counter()
        lhs, n_h, w_inact, G_inact = self._build_regularized_kkt(
            prob, x_np, active, inactive, n_active, G_sp, slack,
        )

        # ── RHS from cotangent vectors ───────────────────────────────
        if batched:
            if g_x.shape[0] == 1 and g_x.ndim > 1:
                g_x = np.broadcast_to(
                    g_x, (batch_size, *g_x.shape[1:])
                )
            if g_lam.shape[0] == 1 and g_lam.ndim > 1:
                g_lam = np.broadcast_to(
                    g_lam, (batch_size, *g_lam.shape[1:])
                )
            if g_mu.shape[0] == 1 and g_mu.ndim > 1:
                g_mu = np.broadcast_to(
                    g_mu, (batch_size, *g_mu.shape[1:])
                )

            rhs_parts: list[ndarray] = [g_x.T]
            if n_eq > 0:
                rhs_parts.append(g_mu.T)
            if n_ineq > 0 and n_active > 0:
                rhs_parts.append(g_lam[:, active].T)
            rhs = np.vstack(rhs_parts)
        else:
            rhs_list: list[ndarray] = [g_x]
            if n_eq > 0:
                rhs_list.append(g_mu)
            if n_ineq > 0 and n_active > 0:
                rhs_list.append(g_lam[active])
            rhs = np.hstack(rhs_list)

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

        # Pre-compute inactive-constraint adjoint correction
        if n_inactive > 0:
            if batched:
                Gv = np.asarray(G_inact @ v_x.T)   # (n_inactive, batch)
                wGv = w_inact[:, None] * Gv          # (n_inactive, batch)
            else:
                Gv = np.asarray(G_inact @ v_x).ravel()  # (n_inactive,)
                wGv = w_inact * Gv                        # (n_inactive,)

        grads: dict[str, ndarray] = {}
        _sp = self._sp_indices

        if batched:
            # ── P (sparse-indexed) ───────────────────────────────────
            if self._need("P") and "P" in _sp:
                P_rows, P_cols = _sp["P"]
                grads["P"] = -(v_x[:, P_rows] * x_np[P_cols])

            # ── q (dense) ────────────────────────────────────────────
            if self._need("q"):
                grads["q"] = -v_x

            # ── A, b ─────────────────────────────────────────────────
            if n_eq > 0:
                if self._need("A") and "A" in _sp:
                    A_rows, A_cols = _sp["A"]
                    grads["A"] = -(
                        mu_np[A_rows] * v_x[:, A_cols]
                        + v_mu[:, A_rows] * x_np[A_cols]
                    )
                if self._need("b"):
                    grads["b"] = v_mu

            # ── G, h (with inactive correction) ──────────────────────
            if n_ineq > 0:
                if self._need("G") and "G" in _sp:
                    G_rows, G_cols = _sp["G"]
                    g_G = np.zeros(
                        (batch_size, len(G_rows)), dtype=_dtype
                    )
                    # Active contribution (standard KKT)
                    if n_active > 0:
                        active_indices = np.where(active)[0]
                        nnz_row_is_active = active[G_rows]
                        full_to_active = np.empty(n_ineq, dtype=np.intp)
                        full_to_active[active_indices] = np.arange(
                            n_active
                        )
                        active_nnz = np.where(nnz_row_is_active)[0]
                        compressed_rows = full_to_active[
                            G_rows[active_nnz]
                        ]
                        g_G[:, active_nnz] = -(
                            lam_np[G_rows[active_nnz]]
                            * v_x[:, G_cols[active_nnz]]
                            + v_lam_a[:, compressed_rows]
                            * x_np[G_cols[active_nnz]]
                        )
                    # Inactive correction (DBD)
                    if n_inactive > 0:
                        inactive_indices = np.where(inactive)[0]
                        nnz_row_is_inactive = inactive[G_rows]
                        full_to_inactive = np.empty(
                            n_ineq, dtype=np.intp
                        )
                        full_to_inactive[inactive_indices] = np.arange(
                            n_inactive
                        )
                        inactive_nnz = np.where(nnz_row_is_inactive)[0]
                        compressed_inact_rows = full_to_inactive[
                            G_rows[inactive_nnz]
                        ]
                        # wGv has shape (n_inactive, batch)
                        g_G[:, inactive_nnz] -= (
                            wGv[compressed_inact_rows, :].T
                            * x_np[G_cols[inactive_nnz]]
                        )
                    grads["G"] = g_G

                if self._need("h"):
                    g_h = np.zeros(
                        (batch_size, n_ineq), dtype=_dtype
                    )
                    if n_active > 0:
                        g_h[:, active] = v_lam_a
                    if n_inactive > 0:
                        g_h[:, inactive] += wGv.T
                    grads["h"] = g_h

        else:  # unbatched
            # ── P (sparse-indexed) ───────────────────────────────────
            if self._need("P") and "P" in _sp:
                P_rows, P_cols = _sp["P"]
                grads["P"] = -(v_x[P_rows] * x_np[P_cols])

            # ── q (dense) ────────────────────────────────────────────
            if self._need("q"):
                grads["q"] = -v_x

            # ── A, b ─────────────────────────────────────────────────
            if n_eq > 0:
                if self._need("A") and "A" in _sp:
                    A_rows, A_cols = _sp["A"]
                    grads["A"] = -(
                        mu_np[A_rows] * v_x[A_cols]
                        + v_mu[A_rows] * x_np[A_cols]
                    )
                if self._need("b"):
                    grads["b"] = v_mu

            # ── G, h (with inactive correction) ──────────────────────
            if n_ineq > 0:
                if self._need("G") and "G" in _sp:
                    G_rows, G_cols = _sp["G"]
                    g_G = np.zeros(len(G_rows), dtype=_dtype)
                    # Active contribution (standard KKT)
                    if n_active > 0:
                        active_indices = np.where(active)[0]
                        nnz_row_is_active = active[G_rows]
                        full_to_active = np.empty(n_ineq, dtype=np.intp)
                        full_to_active[active_indices] = np.arange(
                            n_active
                        )
                        active_nnz = np.where(nnz_row_is_active)[0]
                        compressed_rows = full_to_active[
                            G_rows[active_nnz]
                        ]
                        g_G[active_nnz] = -(
                            lam_np[G_rows[active_nnz]]
                            * v_x[G_cols[active_nnz]]
                            + v_lam_a[compressed_rows]
                            * x_np[G_cols[active_nnz]]
                        )
                    # Inactive correction (DBD)
                    if n_inactive > 0:
                        inactive_indices = np.where(inactive)[0]
                        nnz_row_is_inactive = inactive[G_rows]
                        full_to_inactive = np.empty(
                            n_ineq, dtype=np.intp
                        )
                        full_to_inactive[inactive_indices] = np.arange(
                            n_inactive
                        )
                        inactive_nnz = np.where(nnz_row_is_inactive)[0]
                        compressed_inact_rows = full_to_inactive[
                            G_rows[inactive_nnz]
                        ]
                        g_G[inactive_nnz] -= (
                            wGv[compressed_inact_rows]
                            * x_np[G_cols[inactive_nnz]]
                        )
                    grads["G"] = g_G

                if self._need("h"):
                    g_h = np.zeros(n_ineq, dtype=_dtype)
                    if n_active > 0:
                        g_h[active] = v_lam_a
                    if n_inactive > 0:
                        g_h[inactive] += wGv
                    grads["h"] = g_h

        t["compute_grads"] = perf_counter() - start

        return cast(SolverDiffOutRevNP, grads), t


# ── Register in the backend registry ─────────────────────────────────
register_differentiator_backend(
    "sparse_dbd", SparseDBDDifferentiatorBackend
)