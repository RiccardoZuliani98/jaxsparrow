"""
solver_dense/_dense_kkt_backend.py
==================================
Dense KKT differentiator backend.

Implements the :class:`DifferentiatorBackend` protocol for dense
problems, using ``numpy.block`` for KKT assembly and a configurable
dense linear solver (default: ``numpy.linalg.solve``).

Registered as ``"dense_kkt"`` in the differentiator backend registry.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Sequence, cast, Any

import numpy as np
from numpy import ndarray

from jaxsparrow._solver_dense._types import (
    DenseIngredientsNP,
    DenseIngredientsTangentsNP,
)
from jaxsparrow._types_common import (
    SolverOutputNP, 
    SolverDiffOutFwdNP,
    SolverDiffOutRevNP
)
from jaxsparrow._solver_dense._options import DenseKKTDiffOptionsFull
from jaxsparrow._utils._linear_solvers import DenseLinearSolver, get_dense_linear_solver
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    register_differentiator_backend,
)
from jaxsparrow._solver_sparse._converters import SparsityInfo


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
        options: Differentiator options. Supported keys:
            - dtype: Floating point dtype (default: np.float64)
            - bool_dtype: Boolean dtype (default: np.bool_)
            - cst_tol: Constraint tolerance (default: 1e-8)
            - linear_solver: Solver name (default: "numpy_solve")

    Raises:
        ValueError: If dimensions are negative or inconsistent.
        TypeError: If options contain invalid values.
    """

    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        options: DenseKKTDiffOptionsFull,
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

        self._solve_linear_system: DenseLinearSolver = get_dense_linear_solver(
            options["linear_solver"]
        )

        # Populated by setup()
        self._fixed: dict[str, ndarray] = {}
        self._d_fixed: dict[str, ndarray] = {}
        self._d_fixed_batched: dict[str, ndarray] = {}
        self._dyn_set: Optional[frozenset[str]] = None
        # Keys whose tangents are structurally zero (fixed and not
        # in dynamic_keys).  Forward-mode skips all arithmetic
        # involving these tangents instead of multiplying by zero.
        self._zero_tangent_keys: frozenset[str] = frozenset()

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        fixed_elements: Optional[DenseIngredientsNP] = None,
        dynamic_keys: Optional[Sequence[str]] = None,
        sparsity_info: Optional[SparsityInfo] = None,
    ) -> dict[str, float]:
        """Store fixed elements, zero tangents, and dynamic key set.

        Args:
            fixed_elements: Ingredients constant across calls.
                Must have correct dimensions for the problem.
            dynamic_keys: Keys for which gradients are needed.
                None means gradients for all keys.
            sparsity_info: needed for compatibility with sparse
                backend

        Returns:
            Timing dictionary with "setup" key.

        Raises:
            ValueError: If fixed_elements dimensions don't match
                n_var, n_eq, n_ineq.
        """
        start = perf_counter()

        n_var = self._n_var
        n_eq = self._n_eq
        n_ineq = self._n_ineq
        _dtype = self._dtype

        # ── Fixed elements and zero tangents ─────────────────────────
        self._fixed = {}
        self._d_fixed = {}

        if fixed_elements is not None:
            # Validate dimensions
            if "P" in fixed_elements:
                P = np.asarray(fixed_elements["P"])
                if P.shape != (n_var, n_var):
                    raise ValueError(
                        f"P shape {P.shape} doesn't match (n_var, n_var) = ({n_var}, {n_var})"
                    )
            
            if "q" in fixed_elements:
                q = np.asarray(fixed_elements["q"])
                if q.shape != (n_var,):
                    raise ValueError(
                        f"q shape {q.shape} doesn't match (n_var,) = ({n_var},)"
                    )
            
            # Similar validation for A, b, G, h
            
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
            self._fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
            self._fixed["b"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["A"] = np.zeros((0, n_var), dtype=_dtype)
            self._d_fixed["b"] = np.zeros((0,), dtype=_dtype)
        if n_ineq == 0:
            self._fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
            self._fixed["h"] = np.zeros((0,), dtype=_dtype)
            self._d_fixed["G"] = np.zeros((0, n_var), dtype=_dtype)
            self._d_fixed["h"] = np.zeros((0,), dtype=_dtype)

        self._d_fixed_batched = {
            k: np.expand_dims(v, axis=0) for k, v in self._d_fixed.items()
        }

        # ── Dynamic key set ──────────────────────────────────────────
        if dynamic_keys is not None:
            self._dyn_set = frozenset(dynamic_keys)
        else:
            self._dyn_set = None

        # ── Determine which fixed keys have structurally zero tangents ──
        if dynamic_keys is not None:
            all_fixed_keys = frozenset(self._d_fixed.keys())
            dyn = frozenset(dynamic_keys)
            self._zero_tangent_keys = all_fixed_keys - dyn
        else:
            self._zero_tangent_keys = frozenset()

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
        prob_np: DenseIngredientsNP, 
        x_np: ndarray,
    ) -> ndarray:
        """Compute the active inequality constraint mask.
        
        Returns:
            Boolean array of shape (n_ineq,) where True indicates
            an active inequality constraint.
        """
        if self._n_ineq > 0:
            if "G" not in prob_np or "h" not in prob_np:
                raise KeyError(
                    "G and h are required when n_ineq > 0. "
                    "Provide them via fixed_elements or as dynamic arguments."
                )
            return np.abs(prob_np["G"] @ x_np - prob_np["h"]) <= self._cst_tol
        return np.empty(0, dtype=self._bool_dtype)

    def _build_kkt_lhs(
        self, 
        prob_np: DenseIngredientsNP, 
        active_np: ndarray, 
        n_active: int,
    ) -> tuple[ndarray, int]:
        """Build the dense KKT LHS matrix.
        
        Returns:
            Tuple of (lhs_matrix, n_h) where n_h is the number of
            active constraints (equalities + active inequalities).
        """
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
            H_parts.append(prob_np["G"][active_np, :]) #type: ignore

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
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]:
        """Forward-mode (JVP) through KKT conditions.

        Args:
            sol_np: Tuple of (x, lam, mu) solution arrays.
            dyn_primals_np: Dynamic ingredients (primal values).
            dyn_tangents_np: Tangents for dynamic ingredients.
            batch_size: Number of problems (0 for single, >0 for batched).

        Returns:
            Tuple of (dx, dlam, dmu) tangents and timing dict.
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
        prob_np = cast(DenseIngredientsNP, {**self._fixed, **dyn_primals_np})

        # ── Merge tangents (only for keys with nonzero tangents) ─────
        if batched:
            d_np: dict[str, ndarray] = {
                k: v for k, v in self._d_fixed_batched.items()
                if _nz(k)
            }
            d_np.update(dyn_tangents_np) #type: ignore
        else:
            d_np = {k: v for k, v in self._d_fixed.items() if _nz(k)}
            d_np.update(dyn_tangents_np) #type: ignore

        # ── Active set and KKT LHS ──────────────────────────────────
        active_np = self._compute_active_set(prob_np, x_np)
        n_active = int(np.sum(active_np))
        lhs, n_h = self._build_kkt_lhs(prob_np, active_np, n_active)

        # ── Build dense RHS (skip zero tangent terms) ────────────────
        rhs_pieces: list[ndarray] = []

        if batched:
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
                    dG_active: ndarray = d_np["G"][:, active_np, :]
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
        """

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
                    g_G = np.zeros_like(prob_np["G"]) #type: ignore
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

        return cast(SolverDiffOutRevNP,grads), t


# ── Register in the backend registry ─────────────────────────────────
register_differentiator_backend("dense_kkt", DenseKKTDifferentiatorBackend)