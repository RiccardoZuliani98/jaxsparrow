"""
solver_dense/_dense_kkt_backend.py
==================================
Dense KKT differentiator backend.

Implements the :class:`DifferentiatorBackend` protocol for dense
problems, using ``numpy.block`` for KKT assembly and a configurable
dense linear solver (default: ``numpy.linalg.solve``).

Registered as ``"kkt"`` in the differentiator backend registry.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Sequence, cast

import numpy as np
from numpy import ndarray
from jaxtyping import Float, Bool

from jaxsparrow._solver_dense._types import (
    DenseIngredientsNP,
    DenseIngredientsTangentsNP,
)
from jaxsparrow._types_common import SolverOutputNP, SolverDiffOutNP
from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._solver_dense._options import DEFAULT_DIFF_OPTIONS
from jaxsparrow._utils._linear_solvers import DenseLinearSolver, get_dense_linear_solver
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    register_differentiator_backend,
)


class DenseKKTDifferentiatorBackend(DifferentiatorBackend):
    """KKT-based differentiator for dense problems.

    Assembles the KKT system using ``numpy.block`` and solves
    with a configurable dense linear solver. Fixed elements, zero
    tangents, and the linear solver are all set up once at
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

        self._solve_linear_system: DenseLinearSolver = get_dense_linear_solver(
            options_parsed["linear_solver"]
        )

        # Populated by setup()
        self._fixed: DenseIngredientsNP = {}
        self._d_fixed: DenseIngredientsTangentsNP = {}
        self._d_fixed_batched: DenseIngredientsTangentsNP = {}
        self._dyn_set: Optional[frozenset[str]] = None

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[DenseIngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info=None,
    ) -> dict[str, float]:
        """Store fixed elements, zero tangents, and dynamic key set."""
        start = perf_counter()

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        # ── Fixed elements and zero tangents ─────────────────────────
        self._fixed = {}
        self._d_fixed = {}

        if fixed_elements is not None:
            self._fixed = cast(DenseIngredientsNP, {
                k: np.array(v, dtype=_dtype).squeeze()
                for k, v in fixed_elements.items()
            })
            self._d_fixed = cast(DenseIngredientsTangentsNP, {
                k: np.zeros_like(v, dtype=_dtype).squeeze()
                for k, v in fixed_elements.items()
            })

        # Zero placeholders for absent constraints
        if n_eq == 0:
            self._fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
            self._fixed["b"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
            self._d_fixed["b"] = np.zeros((0,), dtype=_dtype)
        if n_ineq == 0:
            self._fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
            self._fixed["h"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
            self._d_fixed["h"] = np.zeros((0,), dtype=_dtype)

        self._d_fixed_batched = cast(DenseIngredientsTangentsNP, {
            k: np.expand_dims(v, 0) for k, v in self._d_fixed.items()
        })

        # ── Dynamic key set ──────────────────────────────────────────
        if dynamic_keys is not None:
            self._dyn_set = frozenset(dynamic_keys)
        else:
            self._dyn_set = None

        return {"setup": perf_counter() - start}

    # ── Helpers ──────────────────────────────────────────────────────

    def _need(self, key: str) -> bool:
        """Return True if gradient is needed for *key*."""
        return self._dyn_set is None or key in self._dyn_set

    def _compute_active_set(
        self, prob_np: dict, x_np: ndarray,
    ) -> Bool[ndarray, "n_ineq"]:
        """Compute the active inequality constraint mask."""
        if self._n_ineq > 0:
            assert "G" in prob_np and "h" in prob_np, (
                "G and h are required when n_ineq > 0. "
                "Provide them via fixed_elements or as dynamic arguments."
            )
            return np.asarray(
                np.abs(prob_np["G"] @ x_np - prob_np["h"])
                <= self._cst_tol,
                dtype=self._bool_dtype,
            ).reshape(-1)
        return np.empty(0, dtype=self._bool_dtype)

    def _build_kkt_lhs(
        self, prob_np: dict, active_np: ndarray, n_active: int,
    ) -> tuple[ndarray, int]:
        """Build the dense KKT LHS matrix. Returns ``(lhs, n_h)``."""
        n_var = self._n_var
        _dtype = self._dtype

        assert "P" in prob_np and "q" in prob_np, (
            "P and q are required. "
            "Provide them via fixed_elements or as dynamic arguments."
        )

        H_parts: list[ndarray] = []
        if self._n_eq > 0:
            assert "A" in prob_np and "b" in prob_np, (
                "A and b are required when n_eq > 0. "
                "Provide them via fixed_elements or as dynamic arguments."
            )
            H_parts.append(prob_np["A"])
        if self._n_ineq > 0 and n_active > 0:
            H_parts.append(prob_np["G"][active_np, :])

        if H_parts:
            H_np = np.vstack(H_parts)
        else:
            H_np = np.empty((0, n_var), dtype=_dtype)

        n_h: int = H_np.shape[0]

        lhs = np.block([
            [prob_np["P"], H_np.T],
            [H_np, np.zeros((n_h, n_h), dtype=_dtype)],
        ])
        return lhs, n_h

    # ── Forward differentiation ──────────────────────────────────────

    def differentiate_fwd(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: DenseIngredientsNP,
        dyn_tangents_np: DenseIngredientsTangentsNP,
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

        # ── Merge primals and tangents ───────────────────────────────
        prob_np = cast(DenseIngredientsNP, {**self._fixed, **dyn_primals_np})

        if batched:
            d_np = cast(dict[str, ndarray], {**self._d_fixed_batched, **dyn_tangents_np})
        else:
            d_np = cast(dict[str, ndarray], {**self._d_fixed, **dyn_tangents_np})

        # ── Active set and KKT LHS ──────────────────────────────────
        active_np = self._compute_active_set(prob_np, x_np)
        n_active = int(np.sum(active_np))
        lhs, n_h = self._build_kkt_lhs(prob_np, active_np, n_active)

        # ── Build dense RHS ──────────────────────────────────────────
        dL_np: ndarray = d_np["P"] @ x_np + d_np["q"]
        rhs_pieces: list[ndarray] = []

        if batched:
            if n_eq > 0:
                dL_np = dL_np + d_np["A"].transpose(0, 2, 1) @ mu_np
                rhs_pieces.append(d_np["A"] @ x_np - d_np["b"])
            if n_ineq > 0 and n_active > 0:
                dG_active: ndarray = d_np["G"][:, active_np, :]
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
        dyn_primals_np: DenseIngredientsNP,
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
        prob_np = cast(DenseIngredientsNP, {**self._fixed, **dyn_primals_np})

        # ── Active set and KKT LHS ──────────────────────────────────
        start = perf_counter()
        active_np = self._compute_active_set(prob_np, x_np)
        n_active = int(np.sum(active_np))
        t["active_set"] = perf_counter() - start

        start = perf_counter()
        lhs, n_h = self._build_kkt_lhs(prob_np, active_np, n_active)

        # ── Build dense RHS from cotangent vectors ───────────────────
        if batched:
            if g_x.shape[0] == 1 and g_x.ndim > 1:
                g_x = np.broadcast_to(g_x, (batch_size, *g_x.shape[1:]))
            if g_lam.shape[0] == 1 and g_lam.ndim > 1:
                g_lam = np.broadcast_to(g_lam, (batch_size, *g_lam.shape[1:]))
            if g_mu.shape[0] == 1 and g_mu.ndim > 1:
                g_mu = np.broadcast_to(g_mu, (batch_size, *g_mu.shape[1:]))

            rhs_parts: list[ndarray] = [g_x.T]
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

        # ── Solve ────────────────────────────────────────────────────
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

        if batched:
            if self._need("P"):
                grads["P"] = -(v_x[:, :, None] * x_np[None, None, :])

            if self._need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                if self._need("A"):
                    term1: ndarray = mu_np[None, :, None] * v_x[:, None, :]
                    term2: ndarray = v_mu[:, :, None] * x_np[None, None, :]
                    grads["A"] = -(term1 + term2)
                if self._need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if self._need("G"):
                    g_G: ndarray = np.zeros((batch_size, n_ineq, n_var), dtype=_dtype)
                    if n_active > 0:
                        lam_active = lam_np[active_np]
                        term1 = lam_active[None, :, None] * v_x[:, None, :]
                        term2 = v_lam_a[:, :, None] * x_np[None, None, :]
                        g_G[:, active_np, :] = -(term1 + term2)
                    grads["G"] = g_G
                if self._need("h"):
                    g_h_full: ndarray = np.zeros((batch_size, n_ineq), dtype=_dtype)
                    if n_active > 0:
                        g_h_full[:, active_np] = v_lam_a
                    grads["h"] = g_h_full

        else:
            if self._need("P"):
                grads["P"] = -np.outer(v_x, x_np)

            if self._need("q"):
                grads["q"] = -v_x

            if n_eq > 0:
                if self._need("A"):
                    grads["A"] = -(np.outer(mu_np, v_x) + np.outer(v_mu, x_np))
                if self._need("b"):
                    grads["b"] = v_mu

            if n_ineq > 0:
                if self._need("G"):
                    g_G = np.zeros_like(prob_np["G"])
                    if n_active > 0:
                        g_G[active_np, :] = -(
                            np.outer(lam_np[active_np], v_x)
                            + np.outer(v_lam_a, x_np)
                        )
                    grads["G"] = g_G
                if self._need("h"):
                    g_h_full = np.zeros(n_ineq, dtype=_dtype)
                    if n_active > 0:
                        g_h_full[active_np] = v_lam_a
                    grads["h"] = g_h_full

        t["compute_grads"] = perf_counter() - start

        return grads, t


# ── Register in the backend registry ─────────────────────────────────
register_differentiator_backend("dense_kkt", DenseKKTDifferentiatorBackend)