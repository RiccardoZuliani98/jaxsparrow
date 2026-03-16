from jax import custom_jvp, pure_callback
import jax.numpy as jnp
from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
import jax
import logging
from typing import Optional, cast
from numpy import ndarray
from jaxtyping import Float, Bool

from .parsing_utils import parse_options
from .solver_dense_types import DenseProblemIngredients, DenseProblemIngredientsNP
from .solver_dense_options import (
    DEFAULT_SOLVER_OPTIONS, 
    SolverOptions, 
)

# Dimension key (used in jaxtyping annotations throughout):
#
#   nv  = n_var   — number of decision variables
#   ne  = n_eq    — number of equality constraints
#   ni  = n_ineq  — number of inequality constraints
#   na  = n_active — number of active inequality constraints (runtime)
#   B   = batch   — batch dimension (JVP vmap path only)


#TODO: add warmstart

# Expected ndim for each QP ingredient (unbatched).
# Used to detect batching in the JVP path.
_EXPECTED_NDIM: dict[str, int] = {
    "P": 2, "q": 1, "A": 2, "b": 1, "G": 2, "h": 1,
}


def setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: Optional[DenseProblemIngredients] = None,
    options: Optional[SolverOptions] = None,
):
    """Set up a differentiable dense QP solver.

    Creates and returns a ``solver(**runtime)`` callable that solves
    quadratic programs of the form::

        min  0.5 x^T P x + q^T x
        s.t. A x  = b
             G x <= h

    The solver is compatible with ``jax.jvp`` and ``jax.grad`` via
    implicit differentiation of the KKT conditions.

    Elements declared in ``fixed_elements`` are stored as numpy arrays
    and are **never** converted to JAX or routed through the traced
    path. This avoids unnecessary round-trips but means that fixed
    elements are treated as constants for differentiation — their
    tangents are always zero. To differentiate through an element,
    it must be supplied at runtime (not fixed).

    Args:
        n_var: Number of decision variables.
        n_ineq: Number of inequality constraints (may be 0).
        n_eq: Number of equality constraints (may be 0).
        fixed_elements: Optional dict of QP ingredients that remain
            constant across calls. Valid keys are ``P``, ``q``, ``A``,
            ``b``, ``G``, ``h``. Fixed elements are treated as
            constants for differentiation.
        options: Optional solver configuration (backend solver name,
            dtype, tolerances, etc.).

    Returns:
        A callable ``solver(**runtime) -> dict[str, jax.Array]``.
    """
    # start logging
    logger = logging.getLogger(__name__)

    # parse user options
    _options_parsed = parse_options(options, DEFAULT_SOLVER_OPTIONS)

    # extract dtype for simplicity
    _dtype = _options_parsed['dtype']

    # parse and process fixed elements if present
    _fixed: DenseProblemIngredientsNP = {}
    if fixed_elements is not None:
        _fixed = cast(
            DenseProblemIngredientsNP,
            {k: np.array(v, dtype=_dtype).squeeze() for k, v in fixed_elements.items()}
        )

    # form shapes of problem solution
    _fwd_shapes = {
        "x":        jax.ShapeDtypeStruct((n_var,),  _dtype),
        "lam":      jax.ShapeDtypeStruct((n_ineq,), _dtype),
        "mu":       jax.ShapeDtypeStruct((n_eq,),   _dtype),
        "active":   jax.ShapeDtypeStruct((n_ineq,), jnp.bool_),
    }

    # Shapes for the tangent-only callback — no primal solution.
    # Must match the return order of _kkt_diff: (dx, dlam, dmu, dactive).
    # The forward pass is handled separately so that vmap can see the
    # primal output does not depend on the batched tangents.
    _tangent_shapes = (
        jax.ShapeDtypeStruct((n_var,),  _dtype),             # dx
        jax.ShapeDtypeStruct((n_ineq,), _dtype),             # dlam
        jax.ShapeDtypeStruct((n_eq,),   _dtype),             # dmu
        jax.ShapeDtypeStruct((n_ineq,), jax.dtypes.float0),  # dactive
    )

    # gather all required keys
    _required_keys = ("P", "q")
    if n_eq > 0:
        _required_keys += ("A", "b")
    if n_ineq > 0:
        _required_keys += ("G", "h")

    # gather keys required at runtime (i.e., not fixed).
    # Only these flow through JAX's traced path.
    _dynamic_keys = tuple(k for k in _required_keys if k not in _fixed)

    # transform to sets for speed
    _required_keys_set = set(_required_keys)
    _dynamic_keys_set = set(_dynamic_keys)
    _fixed_keys_set = set(_fixed.keys())

    logger.info(
        f"Setting up QP with {n_var} variables, "
        f"{n_eq} equalities, {n_ineq} inequalities."
    )
    logger.info(f"Fixed variables: {_fixed_keys_set or 'none'}")
    logger.info(f"Dynamic variables: {_dynamic_keys_set or 'none'}")

    def _fmt_times(t: dict[str, float]) -> str:
        """Format a timing dict into a single-line summary string.

        Args:
            t: Dict mapping stage names to elapsed seconds.

        Returns:
            Formatted string, e.g. ``"solve=1.2e-03  active=4.5e-05"``.
        """
        return "  ".join(f"{k}={v:.3e}s" for k, v in t.items())

    # =================================================================
    # SETUP SOLVER
    # =================================================================

    #region

    def _solve_qp_numpy(**kwargs: ndarray) -> tuple[
        Float[ndarray, " nv"],      # x
        Float[ndarray, " ni"],      # lam
        Float[ndarray, " ne"],      # mu
        Bool[ndarray, " ni"],       # active
        dict[str, float],           # timing
    ]:
        """Solve the QP in pure numpy via ``qpsolvers``.

        All required ingredients must be present in ``kwargs``; no merging
        with ``_fixed`` is performed here (that happens upstream).

        When the problem has no equality constraints (``n_eq == 0``),
        ``A``, ``b``, and ``mu`` are absent / empty. Likewise, when
        there are no inequality constraints (``n_ineq == 0``), ``G``,
        ``h``, ``lam``, and ``active`` are absent / empty.

        Args:
            **kwargs: Numpy arrays for the QP ingredients:

                - ``P`` (nv, nv): Positive semi-definite cost matrix.
                - ``q`` (nv,): Linear cost vector.
                - ``A`` (ne, nv): Equality constraint matrix
                  (required when ``n_eq > 0``).
                - ``b`` (ne,): Equality constraint vector
                  (required when ``n_eq > 0``).
                - ``G`` (ni, nv): Inequality constraint matrix
                  (required when ``n_ineq > 0``).
                - ``h`` (ni,): Inequality constraint vector
                  (required when ``n_ineq > 0``).

        Returns:
            A tuple ``(x, lam, mu, active, t)`` where:

                - ``x`` (nv,): Primal solution.
                - ``lam`` (ni,): Dual variables for inequality
                  constraints. Empty when ``n_ineq == 0``.
                - ``mu`` (ne,): Dual variables for equality
                  constraints. Empty when ``n_eq == 0``.
                - ``active`` (ni,): Boolean mask of active inequality
                  constraints. Empty when ``n_ineq == 0``.
                - ``t``: Timing dict with keys ``problem_setup``,
                  ``solve``, ``retrieve``, ``active_set``.

        Raises:
            ValueError: If any required ingredient is missing.
            AssertionError: If the solver fails or array shapes are
                inconsistent with declared problem dimensions.
        """
        # Safety check — all required keys should already be present
        missing = _required_keys_set - set(kwargs)
        if missing:
            raise ValueError(
                f"Missing QP ingredients: {sorted(missing)}. "
                f"Provide them either via fixed_elements at setup or "
                f"at call time."
            )

        # Validate shapes against declared problem dimensions
        assert kwargs["P"].shape == (n_var, n_var)
        assert kwargs["q"].shape == (n_var,)
        if n_eq > 0:
            assert kwargs["A"].shape == (n_eq, n_var)
            assert kwargs["b"].shape == (n_eq,)
        if n_ineq > 0:
            assert kwargs["G"].shape == (n_ineq, n_var)
            assert kwargs["h"].shape == (n_ineq,)

        t: dict[str, float] = {}

        # Build qpsolvers Problem
        start = perf_counter()
        prob = Problem(**kwargs)
        t["problem_setup"] = perf_counter() - start

        # Solve QP
        start = perf_counter()
        sol = solve_problem(prob, solver=_options_parsed["solver"])
        assert sol.found, "QP solver failed to find a solution."
        t["solve"] = perf_counter() - start

        # Recover primal / dual variables
        start = perf_counter()
        x: Float[ndarray, " nv"] = (
            np.asarray(sol.x, dtype=_dtype).reshape(-1)
        )

        if n_eq > 0:
            mu: Float[ndarray, " ne"] = (
                np.asarray(sol.y, dtype=_dtype).reshape(-1)
            )
        else:
            mu = np.empty(0, dtype=_dtype)

        if n_ineq > 0:
            lam: Float[ndarray, " ni"] = (
                np.asarray(sol.z, dtype=_dtype).reshape(-1)
            )
        else:
            lam = np.empty(0, dtype=_dtype)
        t["retrieve"] = perf_counter() - start

        # Determine active set: |Gx − h| <= tolerance
        start = perf_counter()
        if n_ineq > 0:
            active: Bool[ndarray, " ni"] = np.asarray(
                np.abs(kwargs["G"] @ sol.x - kwargs["h"])
                <= _options_parsed["cst_tol"],
                dtype=np.bool_,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=np.bool_)
        t["active_set"] = perf_counter() - start

        return x, lam, mu, active, t

    def _solve_qp(*dynamic_vals: jax.Array) -> dict[str, jax.Array]:
        """Bridge between JAX ``pure_callback`` and the numpy QP solver.

        Receives only the **dynamic** QP ingredients (those not in
        ``_fixed``) as positional JAX arrays in ``_dynamic_keys`` order,
        converts them to numpy, merges with the pre-stored ``_fixed``
        numpy arrays (no conversion needed for those), and delegates to
        ``_solve_qp_numpy``.

        Args:
            *dynamic_vals: JAX arrays corresponding to ``_dynamic_keys``,
                in order.

        Returns:
            Solution dict with numpy arrays (``pure_callback`` handles
            the numpy → JAX conversion):

                - ``x``      (nv,):  Primal solution.
                - ``lam``    (ni,):  Inequality duals.
                - ``mu``     (ne,):  Equality duals.
                - ``active`` (ni,):  Active-set boolean mask.
        """
        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Convert only dynamic JAX arrays to numpy ─────────────────
        start = perf_counter()
        dynamic_np = {
            k: np.asarray(v, dtype=_dtype).squeeze()
            for k, v in zip(_dynamic_keys, dynamic_vals)
        }
        t["convert_to_numpy"] = perf_counter() - start

        # ── Merge with pre-stored fixed arrays (no conversion) ───────
        prob_np = {**_fixed, **dynamic_np}

        # ── Solve in numpy ───────────────────────────────────────────
        x_np, lam_np, mu_np, active_np, t_solve = _solve_qp_numpy(**prob_np)
        t.update(t_solve)

        # ── Return numpy arrays directly ─────────────────────────────
        # pure_callback handles numpy → JAX conversion using _fwd_shapes.
        result = {
            "x":      x_np,
            "lam":    lam_np,
            "mu":     mu_np,
            "active": active_np,
        }

        t["total"] = perf_counter() - t_start
        logger.info(f"_solve_qp | {_fmt_times(t)}")

        return result

    #endregion

    # =================================================================
    # SETUP DIFFERENTIATOR
    # =================================================================

    #region
    def _kkt_diff(*args: ndarray) -> tuple[
        Float[ndarray, " nv"],      # dx
        Float[ndarray, " ni"],      # dlam
        Float[ndarray, " ne"],      # dmu
        ndarray,                    # dactive (float0)
    ]:
        """Implicit differentiation of the KKT conditions.

        Returns **only the tangents**. The primal solution is computed
        separately in the forward pass (``_solver_dynamic``) and is
        not duplicated here — this avoids expensive broadcasting when
        the JVP is vmapped.

        The QP is still solved internally (needed to build the KKT
        system), but the numpy solution is not converted to JAX or
        broadcast to batch size.

        Receives only the **dynamic** primals and tangents (those not
        in ``_fixed``). Fixed primals are merged from the ``_fixed``
        closure (already numpy, no conversion). Fixed tangents are
        set to zero since fixed elements are not traced by JAX.

        Batching is detected automatically: when the tangent arrays
        carry an extra leading dimension compared to the primals, the
        linear system is solved for every tangent vector simultaneously
        via a matrix RHS.

        Args:
            *args: ``2 * len(_dynamic_keys)`` arrays. The first half
                are dynamic primals, the second half are dynamic
                tangents, both in ``_dynamic_keys`` order.

        Returns:
            A tuple ``(dx, dlam, dmu, dactive)`` where:

                - ``dx``      (nv,) | (B, nv):   Tangent of primal.
                - ``dlam``    (ni,) | (B, ni):   Tangent of ineq duals.
                - ``dmu``     (ne,) | (B, ne):   Tangent of eq duals.
                - ``dactive`` (ni,) | (B, ni):   Zero tangent (float0).
        """
        t_start = perf_counter()
        t: dict[str, float] = {}
        n_dyn = len(_dynamic_keys)

        # ── Split positional args into dynamic primals and tangents ──
        dyn_primal_vals = args[:n_dyn]
        dyn_tangent_vals = args[n_dyn:]

        # ── Convert dynamic primals to numpy, squeeze batch-1 dims ───
        start = perf_counter()
        dyn_primals_np: dict[str, ndarray] = {
            k: np.asarray(v, dtype=_dtype).squeeze()
            for k, v in zip(_dynamic_keys, dyn_primal_vals)
        }
        t["convert_primals"] = perf_counter() - start

        # ── Merge with fixed arrays (already numpy, no conversion) ───
        prob_np: dict[str, ndarray] = {**_fixed, **dyn_primals_np}

        # ── Forward solve (numpy only, result not returned) ──────────
        x_np, lam_np, mu_np, active_np, t_solve = _solve_qp_numpy(**prob_np)
        t.update({f"fwd_{k}": v for k, v in t_solve.items()})

        # ── Convert dynamic tangents to numpy (keep batch dim) ───────
        start = perf_counter()
        dyn_tangents_np: dict[str, ndarray] = {
            k: np.asarray(v, dtype=_dtype)
            for k, v in zip(_dynamic_keys, dyn_tangent_vals)
        }
        t["convert_tangents"] = perf_counter() - start

        # ── Detect batching ──────────────────────────────────────────
        if n_dyn > 0:
            first_key = _dynamic_keys[0]
            batched = (
                dyn_tangents_np[first_key].ndim
                == _EXPECTED_NDIM[first_key] + 1
            )
        else:
            batched = False

        # ── Build full tangent dict: zeros for fixed, actual for dyn ─
        if batched:
            batch_size = max(
                v.shape[0] for v in dyn_tangents_np.values()
            )
            d_np: dict[str, ndarray] = {}
            for k in _required_keys:
                if k in dyn_tangents_np:
                    d_np[k] = dyn_tangents_np[k]
                else:
                    d_np[k] = np.zeros(
                        (1, *_fixed[k].shape), dtype=_dtype
                    )
        else:
            batch_size = 0
            d_np = {}
            for k in _required_keys:
                if k in dyn_tangents_np:
                    d_np[k] = dyn_tangents_np[k]
                else:
                    d_np[k] = np.zeros_like(_fixed[k])

        # ── Retrieve numpy arrays with short aliases ─────────────────
        P_np: Float[ndarray, "nv nv"] = prob_np["P"]
        A_np: Float[ndarray, "ne nv"] = prob_np.get(
            "A", np.empty((0, n_var), dtype=_dtype)
        )
        G_np: Float[ndarray, "ni nv"] = prob_np.get(
            "G", np.empty((0, n_var), dtype=_dtype)
        )

        dP_np = d_np["P"]
        dq_np = d_np["q"]
        dA_np = d_np.get(
            "A",
            np.empty((0, n_var), dtype=_dtype) if not batched
            else np.empty((1, 0, n_var), dtype=_dtype),
        )
        db_np = d_np.get(
            "b",
            np.empty((0,), dtype=_dtype) if not batched
            else np.empty((1, 0), dtype=_dtype),
        )
        dG_np = d_np.get(
            "G",
            np.empty((0, n_var), dtype=_dtype) if not batched
            else np.empty((1, 0, n_var), dtype=_dtype),
        )
        dh_np = d_np.get(
            "h",
            np.empty((0,), dtype=_dtype) if not batched
            else np.empty((1, 0), dtype=_dtype),
        )

        # ── Build KKT system LHS (same for batched and unbatched) ────
        start = perf_counter()

        n_active = int(np.sum(active_np))
        H_parts: list[Float[ndarray, "_ nv"]] = []
        if n_eq > 0:
            H_parts.append(A_np)
        if n_ineq > 0 and n_active > 0:
            H_parts.append(G_np[active_np, :])

        if H_parts:
            H_np: Float[ndarray, "nh nv"] = np.vstack(H_parts)
        else:
            H_np = np.empty((0, n_var), dtype=_dtype)

        n_h = H_np.shape[0]

        lhs: Float[ndarray, "nv_nh nv_nh"] = np.block([
            [P_np,  H_np.T],
            [H_np,  np.zeros((n_h, n_h), dtype=_dtype)],
        ])

        # ── Build RHS (differs between batched and unbatched) ────────
        if batched:
            dL_np: Float[ndarray, "B nv"] = dP_np @ x_np + dq_np
            if n_eq > 0:
                dL_np = dL_np + dA_np.transpose(0, 2, 1) @ mu_np
            if n_ineq > 0 and n_active > 0:
                dL_np = dL_np + (
                    dG_np[:, active_np, :]
                    .transpose(0, 2, 1) @ lam_np[active_np]
                )

            rhs_pieces: list[Float[ndarray, "_ _"]] = [dL_np]
            if n_eq > 0:
                rhs_pieces.append(dA_np @ x_np - db_np)
            if n_ineq > 0 and n_active > 0:
                rhs_pieces.append(
                    dG_np[:, active_np, :] @ x_np - dh_np[:, active_np]
                )

            rhs: Float[ndarray, "nv_nh B"] = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in rhs_pieces
            ], axis=1).T

        else:
            dL_np: Float[ndarray, " nv"] = dP_np @ x_np + dq_np
            if n_eq > 0:
                dL_np = dL_np + dA_np.T @ mu_np
            if n_ineq > 0 and n_active > 0:
                dL_np = dL_np + (
                    dG_np[active_np, :].T @ lam_np[active_np]
                )

            rhs_parts: list[Float[ndarray, " _"]] = [dL_np]
            if n_eq > 0:
                rhs_parts.append(dA_np @ x_np - db_np)
            if n_ineq > 0 and n_active > 0:
                rhs_parts.append(
                    dG_np[active_np, :] @ x_np - dh_np[active_np]
                )
            rhs: Float[ndarray, " nv_nh"] = np.hstack(rhs_parts)

        t["build_system"] = perf_counter() - start

        # ── Solve the linear system ──────────────────────────────────
        start = perf_counter()
        sol = np.linalg.solve(lhs, -rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract dx, dlam, dmu from the solution ──────────────────
        if batched:
            dx_np: Float[ndarray, "B nv"] = sol[:n_var, :].T
            dmu_np: Float[ndarray, "B ne"] = (
                sol[n_var:n_var + n_eq, :].T
                if n_eq > 0
                else np.empty((batch_size, 0), dtype=_dtype)
            )
            dlam_np: Float[ndarray, "B ni"] = np.zeros(
                (batch_size, n_ineq), dtype=_dtype
            )
            if n_ineq > 0 and n_active > 0:
                dlam_np[:, active_np] = sol[n_var + n_eq:, :].T
            dactive = np.zeros(
                (batch_size, n_ineq), dtype=jax.dtypes.float0
            )
        else:
            dx_np: Float[ndarray, " nv"] = sol[:n_var]
            dmu_np: Float[ndarray, " ne"] = (
                sol[n_var:n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            dlam_np: Float[ndarray, " ni"] = np.zeros(n_ineq, dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[active_np] = sol[n_var + n_eq:]
            dactive = np.zeros(n_ineq, dtype=jax.dtypes.float0)

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_diff | {_fmt_times(t)}")

        return dx_np, dlam_np, dmu_np, dactive

    def _kkt_diff_callback(
        primals_dict: dict[str, jax.Array],
        tangents_dict: dict[str, jax.Array],
    ) -> tuple[
        Float[jax.Array, " nv"],    # dx
        Float[jax.Array, " ni"],    # dlam
        Float[jax.Array, " ne"],    # dmu
        jax.Array,                  # dactive (float0)
    ]:
        """Wrap ``_kkt_diff`` in a ``pure_callback`` for use inside JAX traces.

        Extracts only the **dynamic** keys from the dicts and passes
        them positionally (primals first, then tangents) as expected
        by ``_kkt_diff``.

        Returns only tangents — the primal solution is computed
        separately in the forward pass.

        Args:
            primals_dict: Dict mapping ``_dynamic_keys`` to primal
                JAX arrays.
            tangents_dict: Dict mapping ``_dynamic_keys`` to tangent
                JAX arrays (same shapes as primals, or with leading
                batch dim ``B``).

        Returns:
            A tuple ``(dx, dlam, dmu, dactive)``.
        """
        primal_vals = tuple(primals_dict[k] for k in _dynamic_keys)
        tangent_vals = tuple(tangents_dict[k] for k in _dynamic_keys)
        return pure_callback(
            _kkt_diff,
            _tangent_shapes,
            *primal_vals,
            *tangent_vals,
            vmap_method="expand_dims",
        )

    #endregion

    # =================================================================
    # COMBINE INTO A UNIQUE FUNCTION
    # =================================================================

    #region
    @custom_jvp
    def _solver_dynamic(*dynamic_vals: jax.Array) -> dict[str, jax.Array]:
        """Internal solver with positional-only dynamic arguments.

        Wraps ``_solve_qp`` in a ``pure_callback`` so that the numpy QP
        solver can be called from within JAX-traced code. Only the
        **dynamic** (non-fixed) ingredients are passed positionally in
        ``_dynamic_keys`` order; fixed ingredients are merged inside
        ``_solve_qp`` from the ``_fixed`` closure without any
        JAX-to-numpy conversion.

        Args:
            *dynamic_vals: JAX arrays corresponding to
                ``_dynamic_keys``, in order.

        Returns:
            Solution dict with JAX arrays:

                - ``x``      (nv,):  Primal solution.
                - ``lam``    (ni,):  Inequality duals.
                - ``mu``     (ne,):  Equality duals.
                - ``active`` (ni,):  Active-set boolean mask.
        """
        return pure_callback(_solve_qp, _fwd_shapes, *dynamic_vals)

    @_solver_dynamic.defjvp
    def _solver_dynamic_jvp(
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """JVP rule for ``_solver_dynamic`` via implicit differentiation.

        The forward pass and tangent computation are **separate**
        callbacks. This allows JAX's vmap to see that the primal
        output does not depend on the batched tangents, so vmap
        broadcasts the primal result via metadata (zero cost) instead
        of materializing a full batch copy.

        The QP is solved twice (once in the forward callback, once
        inside ``_kkt_diff`` to build the KKT system). This is
        acceptable because the solve cost is small compared to the
        cost of broadcasting large solution vectors to batch size.

        Args:
            primals: Primal inputs in ``_dynamic_keys`` order.
            tangents: Tangent inputs with the same structure as
                ``primals``.

        Returns:
            ``(primal_out, tangent_out)`` where both are solution dicts.
        """
        # ── Forward pass (unbatched, vmap broadcasts automatically) ──
        res = _solver_dynamic(*primals)

        # ── Tangent computation (only tangents returned) ─────────────
        primals_dict: dict[str, jax.Array] = dict(
            zip(_dynamic_keys, primals)
        )
        tangents_dict: dict[str, jax.Array] = dict(
            zip(_dynamic_keys, tangents)
        )

        dx, dlam, dmu, dactive = _kkt_diff_callback(
            primals_dict, tangents_dict
        )

        tangents_out: dict[str, jax.Array] = {
            "x":      dx,
            "lam":    dlam,
            "mu":     dmu,
            "active": dactive,
        }

        return res, tangents_out

    def solver(**runtime: jax.Array) -> dict[str, jax.Array]:
        """Solve a (possibly partially fixed) QP problem.

        The full set of QP ingredients is ``{P, q, A, b, G, h}``. Any
        subset may be fixed at setup time via ``fixed_elements``; the
        remainder must be supplied here at call time.

        Fixed elements are stored as numpy arrays and never enter
        JAX's trace. This avoids unnecessary conversions but means
        their tangents are always zero. To differentiate through an
        element, supply it at runtime instead of fixing it at setup.

        Args:
            **runtime: QP ingredients as JAX arrays. Must cover at
                least every key in ``_dynamic_keys`` (i.e. every
                ingredient not already in ``fixed_elements``). Keys
                that overlap with ``fixed_elements`` are ignored with
                a warning. Valid keys and shapes:

                    - ``P`` (nv, nv): PSD cost matrix.
                    - ``q`` (nv,): Linear cost vector.
                    - ``A`` (ne, nv): Equality constraint matrix.
                    - ``b`` (ne,): Equality RHS.
                    - ``G`` (ni, nv): Inequality constraint matrix.
                    - ``h`` (ni,): Inequality RHS.

        Returns:
            Solution dict with JAX arrays:

                - ``x``      (nv,):  Primal solution.
                - ``lam``    (ni,):  Inequality duals.
                - ``mu``     (ne,):  Equality duals.
                - ``active`` (ni,):  Active-set boolean mask.

        Raises:
            ValueError: If any dynamic ingredient is missing.
        """
        # Warn if user passes keys that are already fixed
        overridden = set(runtime) & _fixed_keys_set
        if overridden:
            logger.warning(
                f"Ignoring runtime values for fixed keys: "
                f"{sorted(overridden)}. These were fixed at setup and "
                f"are not part of the traced path (tangents are zero). "
                f"To differentiate through them, create a new solver "
                f"without fixing them."
            )

        # Check that all dynamic keys are provided
        missing = _dynamic_keys_set - set(runtime)
        if missing:
            raise ValueError(
                f"Missing QP ingredients: {sorted(missing)}. "
                f"Provide them at call time (these are not fixed)."
            )

        # Pass only dynamic values through the traced path
        dynamic_vals: tuple[jax.Array, ...] = tuple(
            runtime[k] for k in _dynamic_keys
        )

        return _solver_dynamic(*dynamic_vals)

    #endregion

    # =================================================================

    return solver