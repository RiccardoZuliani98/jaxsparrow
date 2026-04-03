"""
solver_dense/_dense_dbd_backend.py
===================================
Dense "Differentiable-by-Design" (DBD) differentiator backend.

Implements the regularized KKT system from

    Zuliani, Balta & Lygeros,
    "Differentiable-by-design Nonlinear Optimization
     for Model Predictive Control", arXiv:2509.12692v3,
    Equations (11) and (12).

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

Derivation
~~~~~~~~~~
The offset ``l`` lands in the stationarity block because it
originates from eliminating the inactive-constraint block of
the full (unreduced) system.  Specifically, row 3 of the E
matrix in the proof of Lemma 3 gives

    v_{3,Ibar} = diag(w) [G_Ibar v_1 - nabla_theta g_Ibar]

Substituting back into row 1:

    G_Ibar^T v_{3,Ibar} = H v_1 - l

which adds ``H`` to the (1,1) block of the LHS and ``l`` to
the first block of the RHS.

Registered as ``"dense_dbd"`` in the differentiator backend
registry.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Sequence, cast

import numpy as np
from numpy import ndarray

from jaxsparrow._solver_dense._types import (
    DenseIngredientsNP,
    DenseIngredientsTangentsNP,
)
from jaxsparrow._types_common import (
    SolverOutputNP,
    SolverDiffOutFwdNP,
    SolverDiffOutRevNP,
)
from jaxsparrow._solver_dense._options import DenseDBDDiffOptionsFull
from jaxsparrow._utils._linear_solvers import (
    DenseLinearSolver,
    get_dense_linear_solver,
)
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    register_differentiator_backend,
)
from jaxsparrow._solver_sparse._converters import SparsityInfo


class DenseDBDDifferentiatorBackend(DifferentiatorBackend):
    """Differentiable-by-Design differentiator for dense QPs.

    Assembles the regularized KKT system of Zuliani et al.
    (Eqs. 11-12) and solves with a configurable dense linear
    solver.

    Compared with the standard ``dense_kkt`` backend, the LHS
    receives an additive symmetric perturbation

        ``B^rho = diag(H + rho I,  -rho I,  -rho I)``

    and the RHS receives an offset whose only non-zero block
    is ``l`` (Eq. 12), placed in the stationarity rows.

    Args:
        n_var:  Number of decision variables.
        n_eq:   Number of equality constraints (0 if none).
        n_ineq: Number of inequality constraints (0 if none).
        options: Differentiator options.  Recognised keys beyond
            those of ``dense_kkt``:

            - ``rho``  *(float, default 1e-5)*:
              Regularisation strength.  Must be ``> 0``.
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        options: DenseDBDDiffOptionsFull,
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
        self._rho = options["rho"]
        if self._rho <= 0:
            raise ValueError(f"rho must be > 0, got {self._rho}")

        self._solve_linear_system: DenseLinearSolver = (
            get_dense_linear_solver(options["linear_solver"])
        )

        # Populated by setup()
        self._fixed: dict[str, ndarray] = {}
        self._d_fixed: dict[str, ndarray] = {}
        self._d_fixed_batched: dict[str, ndarray] = {}
        self._dyn_set: Optional[frozenset[str]] = None
        self._zero_tangent_keys: frozenset[str] = frozenset()

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[DenseIngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """Store fixed elements, zero tangents, and dynamic key set.

        Semantics are identical to
        :meth:`DenseKKTDifferentiatorBackend.setup`.
        """
        start = perf_counter()

        n_var, n_eq, n_ineq = self._n_var, self._n_eq, self._n_ineq
        _dtype = self._dtype

        self._fixed = {}
        self._d_fixed = {}

        if fixed_elements is not None:
            if "P" in fixed_elements:
                P = np.asarray(fixed_elements["P"])
                if P.shape != (n_var, n_var):
                    raise ValueError(
                        f"P shape {P.shape} != ({n_var}, {n_var})"
                    )
            if "q" in fixed_elements:
                q = np.asarray(fixed_elements["q"])
                if q.shape != (n_var,):
                    raise ValueError(
                        f"q shape {q.shape} != ({n_var},)"
                    )
            if "A" in fixed_elements:
                A = np.asarray(fixed_elements["A"])
                if A.shape != (n_eq, n_var):
                    raise ValueError(
                        f"A shape {A.shape} != ({n_eq}, {n_var})"
                    )
            if "b" in fixed_elements:
                b = np.asarray(fixed_elements["b"]).squeeze()
                if b.shape != (n_eq,):
                    raise ValueError(
                        f"b shape {b.shape} != ({n_eq},)"
                    )
            if "G" in fixed_elements:
                G = np.asarray(fixed_elements["G"])
                if G.shape != (n_ineq, n_var):
                    raise ValueError(
                        f"G shape {G.shape} != ({n_ineq}, {n_var})"
                    )
            if "h" in fixed_elements:
                h = np.asarray(fixed_elements["h"]).squeeze()
                if h.shape != (n_ineq,):
                    raise ValueError(
                        f"h shape {h.shape} != ({n_ineq},)"
                    )

            self._fixed = {
                k: np.asarray(v, dtype=_dtype).squeeze()
                for k, v in fixed_elements.items()
            }
            self._d_fixed = {
                k: np.zeros_like(v, dtype=_dtype)
                for k, v in self._fixed.items()
            }

        # Zero placeholders for absent constraints
        if n_eq == 0:
            for k in ("A", "b"):
                shape = (0, n_var) if k == "A" else (0,)
                self._fixed[k] = np.zeros(shape, dtype=_dtype)
                self._d_fixed[k] = np.zeros(shape, dtype=_dtype)
        if n_ineq == 0:
            for k in ("G", "h"):
                shape = (0, n_var) if k == "G" else (0,)
                self._fixed[k] = np.zeros(shape, dtype=_dtype)
                self._d_fixed[k] = np.zeros(shape, dtype=_dtype)

        self._d_fixed_batched = {
            k: np.expand_dims(v, axis=0)
            for k, v in self._d_fixed.items()
        }

        if dynamic_keys is not None:
            self._dyn_set = frozenset(dynamic_keys)
            all_fixed_keys = frozenset(self._d_fixed.keys())
            self._zero_tangent_keys = all_fixed_keys - self._dyn_set
        else:
            self._dyn_set = None
            self._zero_tangent_keys = frozenset()

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
            active = residual <= self._cst_tol
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

    # ── regularized KKT assembly (Eqs. 11-12) ───────────────────────

    def _build_regularized_kkt(
        self,
        prob_np: DenseIngredientsNP,
        x_np: ndarray,
        active: ndarray,
        inactive: ndarray,
        n_active: int,
        slack: ndarray,
    ) -> tuple[ndarray, int, ndarray, ndarray]:
        """Build the regularized KKT LHS and auxiliary data.

        Returns:
            ``(lhs, n_h, w_inact, G_inact)``

            *  ``lhs``      - regularized KKT matrix
            *  ``n_h``      - total constraint rows
                              (``n_eq + n_active``)
            *  ``w_inact``  - weight vector for inactive rows
            *  ``G_inact``  - rows of ``G`` for inactive
                              constraints
        """
        n_var, n_eq = self._n_var, self._n_eq
        rho, _dtype = self._rho, self._dtype
        n_inactive = int(np.sum(inactive))

        P = prob_np["P"]

        # ── H from inactive constraints (Eq. 12) ────────────────────
        w_inact = self._inactive_weights(slack, inactive)
        if n_inactive > 0:
            G_inact = prob_np["G"][inactive, :] #type: ignore
            H_reg = G_inact.T @ (w_inact[:, None] * G_inact)
        else:
            G_inact = np.empty((0, n_var), dtype=_dtype)
            H_reg = np.zeros((n_var, n_var), dtype=_dtype)

        # ── Constraint rows (equality + active inequality) ───────────
        C_parts: list[ndarray] = []
        if n_eq > 0:
            C_parts.append(prob_np["A"]) #type: ignore
        if self._n_ineq > 0 and n_active > 0:
            C_parts.append(prob_np["G"][active, :]) #type: ignore

        C_np = (
            np.vstack(C_parts)
            if C_parts
            else np.empty((0, n_var), dtype=_dtype)
        )
        n_h = C_np.shape[0]

        # ── Assemble LHS ────────────────────────────────────────────
        UL = P + H_reg + rho * np.eye(n_var, dtype=_dtype) #type: ignore
        LR = -rho * np.eye(n_h, dtype=_dtype)

        lhs = np.block([
            [UL,   C_np.T],
            [C_np, LR    ],
        ])

        return lhs, n_h, w_inact, G_inact

    # ── Forward differentiation ──────────────────────────────────────

    def differentiate_fwd(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: DenseIngredientsNP,
        dyn_tangents_np: DenseIngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]:
        """Forward-mode (JVP) through the regularized KKT system.

        Solves ``[A_theta + B^rho] V = -(b_theta + d^rho)`` for
        the tangent direction encoded in *dyn_tangents_np*.
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
        prob_np = cast(
            DenseIngredientsNP, {**self._fixed, **dyn_primals_np}
        )
        if batched:
            d_np: dict[str, ndarray] = {
                k: v
                for k, v in self._d_fixed_batched.items()
                if _nz(k)
            }
            d_np.update(dyn_tangents_np)  # type: ignore
        else:
            d_np = {
                k: v for k, v in self._d_fixed.items() if _nz(k)
            }
            d_np.update(dyn_tangents_np)  # type: ignore

        # ── Active / inactive sets ───────────────────────────────────
        Gx = prob_np["G"] @ x_np if self._n_ineq > 0 else np.empty(0, dtype=_dtype)
        h_vec = prob_np["h"] if self._n_ineq > 0 else np.empty(0, dtype=_dtype)
        residual = np.abs(Gx - h_vec)
        slack = h_vec - Gx
        active, inactive = self._compute_active_inactive(residual)
        n_active = int(np.sum(active))
        n_inactive = int(np.sum(inactive))
        t["active_set"] = perf_counter() - start

        # ── regularized LHS ──────────────────────────────────────────
        start = perf_counter()
        lhs, n_h, w_inact, G_inact = self._build_regularized_kkt(
            prob_np, x_np, active, inactive, n_active, slack,
        )

        # ── Build RHS  (b_theta + d^rho) contracted with tangent ────
        # Three blocks: stationarity, equality, active-inequality.
        # The ``l`` offset from Eq. 12 is added to the stationarity
        # block.
        rhs_pieces: list[ndarray] = []

        if batched:
            dL_np = np.zeros((batch_size, n_var), dtype=_dtype)

            if _nz("P"):
                dL_np = dL_np + d_np["P"] @ x_np
            if _nz("q"):
                dL_np = dL_np + d_np["q"]

            # d^rho offset (l term) in stationarity block
            if n_inactive > 0:
                # tangent of g_Ibar = dG_Ibar @ x - dh_Ibar
                dg_inact = np.zeros(
                    (batch_size, n_inactive), dtype=_dtype
                )
                if _nz("G"):
                    dg_inact = (
                        dg_inact
                        + d_np["G"][:, inactive, :] @ x_np
                    )
                if _nz("h"):
                    dg_inact = dg_inact - d_np["h"][:, inactive]
                # l · d_theta = G_Ibar^T diag(w) dg_Ibar
                dL_np = dL_np + (
                    dg_inact * w_inact[None, :]
                ) @ G_inact

            # Equality constraints
            if n_eq > 0:
                if _nz("A"):
                    dL_np = (
                        dL_np
                        + d_np["A"].transpose(0, 2, 1) @ mu_np
                    )
                    rhs_pieces.append(d_np["A"] @ x_np)
                else:
                    rhs_pieces.append(
                        np.zeros((batch_size, n_eq), dtype=_dtype)
                    )
                if _nz("b"):
                    rhs_pieces[-1] = rhs_pieces[-1] - d_np["b"]

            # Active inequality constraints
            if n_ineq > 0 and n_active > 0:
                if _nz("G"):
                    dG_act = d_np["G"][:, active, :]
                    dL_np = (
                        dL_np
                        + dG_act.transpose(0, 2, 1) @ lam_np[active]
                    )
                    rhs_pieces.append(dG_act @ x_np)
                else:
                    rhs_pieces.append(
                        np.zeros(
                            (batch_size, n_active), dtype=_dtype
                        )
                    )
                if _nz("h"):
                    rhs_pieces[-1] = (
                        rhs_pieces[-1] - d_np["h"][:, active]
                    )

            rhs = np.concatenate(
                [
                    np.broadcast_to(p, (batch_size, p.shape[1]))
                    if p.shape[0] == 1
                    else p
                    for p in [dL_np] + rhs_pieces
                ],
                axis=1,
            ).T

        else:  # unbatched
            dL_np = np.zeros(n_var, dtype=_dtype)

            if _nz("P"):
                dL_np = dL_np + d_np["P"] @ x_np
            if _nz("q"):
                dL_np = dL_np + d_np["q"]

            # d^rho offset (l term)
            if n_inactive > 0:
                dg_inact = np.zeros(n_inactive, dtype=_dtype)
                if _nz("G"):
                    dg_inact = (
                        dg_inact + d_np["G"][inactive, :] @ x_np
                    )
                if _nz("h"):
                    dg_inact = dg_inact - d_np["h"][inactive]
                dL_np = dL_np + G_inact.T @ (w_inact * dg_inact)

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
                    dG_act = d_np["G"][active, :]
                    dL_np = dL_np + dG_act.T @ lam_np[active]
                    rhs_pieces.append(dG_act @ x_np)
                else:
                    rhs_pieces.append(
                        np.zeros(n_active, dtype=_dtype)
                    )
                if _nz("h"):
                    rhs_pieces[-1] = (
                        rhs_pieces[-1] - d_np["h"][active]
                    )

            rhs = np.hstack([dL_np] + rhs_pieces)

        t["build_system"] = perf_counter() - start

        # ── Solve ────────────────────────────────────────────────────
        start = perf_counter()
        sol = self._solve_linear_system(lhs, -rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract dx, dlam, dmu ────────────────────────────────────
        if batched:
            dx_np = sol[:n_var, :].T
            dmu_np = (
                sol[n_var : n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=_dtype)
            )
            dlam_np = np.zeros((batch_size, n_ineq), dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[:, active] = sol[n_var + n_eq :, :].T
        else:
            dx_np = sol[:n_var]
            dmu_np = (
                sol[n_var : n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            dlam_np = np.zeros(n_ineq, dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[active] = sol[n_var + n_eq :]

        return cast(SolverDiffOutFwdNP, (dx_np, dlam_np, dmu_np)), t

    # ── Reverse differentiation ──────────────────────────────────────

    def differentiate_rev(
        self,
        dyn_primals_np: DenseIngredientsNP,
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
        cotangents::

            grad_G[inactive,:] -= outer(w * G_Ibar @ v_x,  x)
            grad_h[inactive]   += w * G_Ibar @ v_x
        """
        t: dict[str, float] = {}
        batched = batch_size > 0

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        # ── Merge primals ────────────────────────────────────────────
        prob_np = cast(
            DenseIngredientsNP, {**self._fixed, **dyn_primals_np}
        )

        # ── Active / inactive ────────────────────────────────────────
        start = perf_counter()
        Gx = prob_np["G"] @ x_np if self._n_ineq > 0 else np.empty(0, dtype=_dtype)
        h_vec = prob_np["h"] if self._n_ineq > 0 else np.empty(0, dtype=_dtype)
        residual = np.abs(Gx - h_vec)
        slack = h_vec - Gx
        active, inactive = self._compute_active_inactive(residual)
        n_active = int(np.sum(active))
        n_inactive = int(np.sum(inactive))
        t["active_set"] = perf_counter() - start

        # ── regularized LHS ──────────────────────────────────────────
        start = perf_counter()
        lhs, n_h, w_inact, G_inact = self._build_regularized_kkt(
            prob_np, x_np, active, inactive, n_active, slack,
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

        # ── Solve adjoint system ─────────────────────────────────────
        start = perf_counter()
        v = self._solve_linear_system(lhs, rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract adjoint variables ────────────────────────────────
        start = perf_counter()
        if batched:
            v_x = v[:n_var, :].T
            v_mu = (
                v[n_var : n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=_dtype)
            )
            v_lam_a = (
                v[n_var + n_eq :, :].T
                if (n_ineq > 0 and n_active > 0)
                else np.empty((batch_size, 0), dtype=_dtype)
            )
        else:
            v_x = v[:n_var]
            v_mu = (
                v[n_var : n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            v_lam_a = (
                v[n_var + n_eq :]
                if (n_ineq > 0 and n_active > 0)
                else np.empty(0, dtype=_dtype)
            )
        t["extract_adjoint"] = perf_counter() - start

        # ── Parameter cotangents ─────────────────────────────────────
        start = perf_counter()

        # Pre-compute inactive-constraint adjoint correction
        if n_inactive > 0:
            if batched:
                Gv = G_inact @ v_x.T          # (n_inactive, batch)
                wGv = w_inact[:, None] * Gv    # (n_inactive, batch)
            else:
                Gv = G_inact @ v_x             # (n_inactive,)
                wGv = w_inact * Gv             # (n_inactive,)

        grads: dict[str, ndarray] = {}

        if batched:
            if self._need("P"):
                grads["P"] = -(
                    v_x[:, :, None] * x_np[None, None, :]
                )
            if self._need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                if self._need("A"):
                    grads["A"] = -(
                        mu_np[None, :, None] * v_x[:, None, :]
                        + v_mu[:, :, None] * x_np[None, None, :]
                    )
                if self._need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if self._need("G"):
                    g_G = np.zeros(
                        (batch_size, n_ineq, n_var), dtype=_dtype
                    )
                    if n_active > 0:
                        g_G[:, active, :] = -(
                            lam_np[active][None, :, None]
                            * v_x[:, None, :]
                            + v_lam_a[:, :, None]
                            * x_np[None, None, :]
                        )
                    if n_inactive > 0:
                        g_G[:, inactive, :] -= (
                            wGv.T[:, :, None] * x_np[None, None, :]
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
            if self._need("P"):
                grads["P"] = -np.outer(v_x, x_np)
            if self._need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                if self._need("A"):
                    grads["A"] = -(
                        np.outer(mu_np, v_x)
                        + np.outer(v_mu, x_np)
                    )
                if self._need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if self._need("G"):
                    g_G = np.zeros_like(prob_np["G"])  # type: ignore
                    if n_active > 0:
                        g_G[active, :] = -(
                            np.outer(lam_np[active], v_x)
                            + np.outer(v_lam_a, x_np)
                        )
                    if n_inactive > 0:
                        g_G[inactive, :] -= np.outer(wGv, x_np)
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
    "dense_dbd", DenseDBDDifferentiatorBackend
)