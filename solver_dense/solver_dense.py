from jax import custom_jvp, pure_callback
import jax.numpy as jnp
from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
import jax
import logging
from typing import Optional, cast
from numpy import ndarray

from solver_dense.parsing_utils import parse_options
from solver_dense.solver_dense_types import DenseProblemIngredients, DenseProblemIngredientsNP
from solver_dense.solver_dense_options import (
    DEFAULT_SOLVER_OPTIONS, 
    SolverOptions, 
)

# This function allows overwriting arguments 

#TODO: add dimensions and annotate dimensions in array?
#TODO: we should allow the user not to pass e.g. A,b, or G,h
#TODO: add warmstart
#TODO: setup_dense_solver should allow passing some of the elements explicitly,
# e.g. if the user passes P, q, then we should treat those as constant and possibly
# pre-store them and convert them inside the solver / differentiator
#TODO: don't convert back and forth e.g. dP if it's all zeros, that is, only convert
# nonzero elements! same with QP elements!


def setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: Optional[DenseProblemIngredients] = None,
    options: Optional[SolverOptions] = None,
):
    # start logging
    logger = logging.getLogger(__name__)

    # parse user options
    _options_parsed = parse_options(options,DEFAULT_SOLVER_OPTIONS)

    # extract dtype for simplicity
    _dtype = _options_parsed['dtype']

    # parse and process fixed elements if present
    _fixed : DenseProblemIngredientsNP = {}
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

    # form shapes of jvp
    _jvp_shapes = (
        jax.ShapeDtypeStruct((n_var,),      _dtype),
        jax.ShapeDtypeStruct((n_ineq,),     _dtype),
        jax.ShapeDtypeStruct((n_eq,),       _dtype),
        jax.ShapeDtypeStruct((n_ineq,),     jax.dtypes.float0),
        _fwd_shapes,
    )
    
    # gather requires keys
    _required_keys = ("P", "q")
    if n_eq > 0:
        _required_keys += ("A", "b")
    if n_ineq > 0:
        _required_keys += ("G", "h")

    # gather keys required at runtime (i.e., not fixed)
    _dynamic_keys = [k for k in _required_keys if k not in _fixed]

    # transform to sets for speed
    _required_keys_set = set(_required_keys)
    _dynamic_keys_set = set(_dynamic_keys)

    logger.debug(f"Setting up QP with {n_var} variables, {n_eq} equalities, {n_ineq} inequalities.")
    logger.debug(f"Fixed variables: {set(_fixed.keys()) or 'none'}")


    # =================================================================
    # SETUP SOLVER
    # =================================================================

    #region

    def _solve_qp_numpy(**kwargs: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Solve the QP in pure numpy via ``qpsolvers``.

        All required ingredients must be present in ``kwargs``; no merging
        with ``_fixed`` is performed here (that happens upstream in
        ``solver()``).

        When the problem has no equality constraints (``n_eq == 0``),
        ``A``, ``b``, and ``mu`` are absent / empty. Likewise, when
        there are no inequality constraints (``n_ineq == 0``), ``G``,
        ``h``, ``lam``, and ``active`` are absent / empty.

        Args:
            **kwargs: Numpy arrays for the QP ingredients. Required keys
                are always ``P`` and ``q``. ``A`` and ``b`` are required
                when ``n_eq > 0``; ``G`` and ``h`` when ``n_ineq > 0``.

        Returns:
            A tuple ``(x, lam, mu, active, t_dict)`` where:
                x: Primal solution, shape ``(n_var,)``.
                lam: Dual variables for inequality constraints,
                    shape ``(n_ineq,)``. Empty when ``n_ineq == 0``.
                mu: Dual variables for equality constraints,
                    shape ``(n_eq,)``. Empty when ``n_eq == 0``.
                active: Boolean mask of active inequality constraints,
                    shape ``(n_ineq,)``. Empty when ``n_ineq == 0``.

        Raises:
            ValueError: If any required ingredient is missing.
            AssertionError: If the solver fails to find a solution or
                array shapes are inconsistent with the declared problem
                dimensions.
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

        # Build qpsolvers Problem
        start = perf_counter()
        prob = Problem(**kwargs)
        logger.debug(f"problem_setup: {perf_counter() - start}")

        # Solve QP
        start = perf_counter()
        sol = solve_problem(prob, solver=_options_parsed["solver"])
        assert sol.found, "QP solver failed to find a solution."
        logger.debug(f"solve: {perf_counter() - start}")

        # Recover primal / dual variables
        start = perf_counter()
        x = np.asarray(sol.x, dtype=_dtype).reshape(-1)

        if n_eq > 0:
            mu = np.asarray(sol.y, dtype=_dtype).reshape(-1)
        else:
            mu = np.empty(0, dtype=_dtype)

        if n_ineq > 0:
            lam = np.asarray(sol.z, dtype=_dtype).reshape(-1)
        else:
            lam = np.empty(0, dtype=_dtype)

        logger.debug(f"retrieve: {perf_counter() - start}")

        # Determine active set: |Gx − h| <= tolerance
        start = perf_counter()
        if n_ineq > 0:
            active = np.asarray(
                np.abs(kwargs["G"] @ sol.x - kwargs["h"])
                <= _options_parsed["cst_tol"],
                dtype=np.bool_,
            ).reshape(-1)
        else:
            active = np.empty(0, dtype=np.bool_)
        logger.debug(f"active set determination: {perf_counter() - start}")

        return x, lam, mu, active
    
    def _solve_qp(*vals: jax.Array) -> dict[str, jax.Array]:
        """Bridge between JAX ``pure_callback`` and the numpy QP solver.

        Receives all QP ingredients as positional JAX arrays in
        ``_required_keys`` order, converts them to numpy, delegates to
        ``_solve_qp_numpy`` for the actual solve, and converts the
        solution back to JAX arrays.

        This function is invoked inside ``pure_callback`` and therefore
        executes outside of JAX's tracing — all operations are eager
        numpy.

        When equality or inequality constraints are absent
        (``n_eq == 0`` / ``n_ineq == 0``), the corresponding keys are
        not present in ``_required_keys`` and are simply omitted. The
        returned duals and active-set mask have shape ``(0,)`` in that
        case.

        Args:
            *vals: JAX arrays corresponding to ``_required_keys``, in
                order (i.e. P, q, [A, b,] [G, h]).

        Returns:
            Solution dict with keys ``x``, ``lam``, ``mu``, ``active``
            as JAX arrays. ``lam`` and ``active`` have shape ``(0,)``
            when ``n_ineq == 0``; ``mu`` has shape ``(0,)`` when
            ``n_eq == 0``.
        """
        t_start = perf_counter()

        # ── Map positional args back to named ingredients ────────────
        kwargs = dict(zip(_required_keys, vals))

        # ── Convert JAX arrays to numpy ──────────────────────────────
        start = perf_counter()
        prob_np = {
            k: np.asarray(v, dtype=_dtype).squeeze()
            for k, v in kwargs.items()
        }
        logger.debug(f"convert_to_numpy: {perf_counter() - start:.3e}s")

        # ── Solve in numpy ───────────────────────────────────────────
        x_np, lam_np, mu_np, active_np = _solve_qp_numpy(**prob_np)

        # ── Convert solution back to JAX ─────────────────────────────
        start = perf_counter()
        result = {
            "x":      jnp.array(x_np, dtype=_dtype),
            "lam":    jnp.array(lam_np, dtype=_dtype),
            "mu":     jnp.array(mu_np, dtype=_dtype),
            "active": jnp.array(active_np, dtype=jnp.bool_),
        }
        logger.debug(f"convert_to_jax: {perf_counter() - start:.3e}s")

        logger.debug(f"DenseQP total: {perf_counter() - t_start:.3e}s")

        return result


    #endregion

    # =================================================================
    # SETUP DIFFERENTIATOR
    # =================================================================

    #region
    def _kkt_diff(*args: ndarray) -> tuple[
        ndarray, ndarray, ndarray, ndarray, dict[str, jax.Array]
    ]:
        """Implicit differentiation of the KKT conditions.

        Solves the QP forward, then differentiates through the KKT
        optimality conditions by solving a linear system whose structure
        depends on the active set of inequality constraints.

        The function receives all primals followed by all tangents as
        positional numpy arrays, both ordered according to
        ``_required_keys`` (i.e. P, q, [A, b,] [G, h]).

        Batching is detected automatically: when the tangent arrays
        carry an extra leading dimension compared to the primals, the
        linear system is solved for every tangent vector simultaneously
        via a matrix RHS.

        Args:
            *args: ``2 * len(_required_keys)`` numpy arrays. The first
                half are primals, the second half are tangents, both in
                ``_required_keys`` order.

        Returns:
            A tuple ``(dx, dmu, dlam, dactive, sol)`` where:
                dx: Tangent of the primal solution.
                dmu: Tangent of the inequality duals.
                dlam: Tangent of the equality duals.
                dactive: Zero tangent for the active-set mask
                    (non-differentiable), dtype ``float0``.
                sol: Primal / dual solution dict (``x``, ``lam``,
                    ``mu``, ``active``) as JAX arrays, used by the
                    JVP rule as the primal output.
        """
        t_start = perf_counter()
        n_keys = len(_required_keys)

        # ── Split positional args into primals and tangents ──────────
        primal_vals = args[:n_keys]
        tangent_vals = args[n_keys:]
        primals = dict(zip(_required_keys, primal_vals))
        tangents = dict(zip(_required_keys, tangent_vals))

        # ── Convert primals to numpy and squeeze batch-1 dims ────────
        start = perf_counter()
        prob_np = {
            k: np.asarray(v, dtype=_dtype).squeeze()
            for k, v in primals.items()
        }
        logger.debug(f"kkt_diff convert_primals: {perf_counter() - start:.3e}s")

        # ── Forward solve ────────────────────────────────────────────
        x_np, lam_np, mu_np, active_np = _solve_qp_numpy(**prob_np)

        # ── Convert tangents to numpy (keep batch dimension) ─────────
        start = perf_counter()
        d_np = {
            k: np.asarray(v, dtype=_dtype)
            for k, v in tangents.items()
        }
        logger.debug(f"kkt_diff convert_tangents: {perf_counter() - start:.3e}s")

        # ── Detect batching ──────────────────────────────────────────
        # Tangents have an extra leading dim when vmap expands them.
        batched = d_np["P"].ndim == 3

        # ── Retrieve numpy arrays (with short aliases) ───────────────
        P_np = prob_np["P"]
        A_np = prob_np.get("A", np.empty((0, n_var), dtype=_dtype))
        G_np = prob_np.get("G", np.empty((0, n_var), dtype=_dtype))

        dP_np = d_np["P"]
        dq_np = d_np["q"]
        dA_np = d_np.get("A", np.empty((0, n_var), dtype=_dtype) if not batched
                         else np.empty((1, 0, n_var), dtype=_dtype))
        db_np = d_np.get("b", np.empty((0,), dtype=_dtype) if not batched
                         else np.empty((1, 0), dtype=_dtype))
        dG_np = d_np.get("G", np.empty((0, n_var), dtype=_dtype) if not batched
                         else np.empty((1, 0, n_var), dtype=_dtype))
        dh_np = d_np.get("h", np.empty((0,), dtype=_dtype) if not batched
                         else np.empty((1, 0), dtype=_dtype))

        # ── Build KKT system LHS (same for batched and unbatched) ────
        start = perf_counter()

        # Stack equality and active inequality constraint rows
        n_active = int(np.sum(active_np))
        H_parts = []
        if n_eq > 0:
            H_parts.append(A_np)
        if n_ineq > 0 and n_active > 0:
            H_parts.append(G_np[active_np, :])

        if H_parts:
            H_np = np.vstack(H_parts)                      # (n_eq + n_active, n_var)
        else:
            H_np = np.empty((0, n_var), dtype=_dtype)

        n_h = H_np.shape[0]
        lhs = np.block([
            [P_np,  H_np.T],
            [H_np,  np.zeros((n_h, n_h), dtype=_dtype)],
        ])

        # ── Build RHS (differs between batched and unbatched) ────────
        if batched:
            # Tangents: (batch, ...), primals/duals: unbatched.
            # dL = dP x + dq + dA^T mu + dG_active^T lam_active
            dL_np = dP_np @ x_np + dq_np
            if n_eq > 0:
                dL_np = dL_np + dA_np.transpose(0, 2, 1) @ mu_np
            if n_ineq > 0 and n_active > 0:
                dL_np = dL_np + (dG_np[:, active_np, :]
                                 .transpose(0, 2, 1) @ lam_np[active_np])

            # Collect rhs pieces: each is (batch_i, n_i)
            rhs_pieces = [dL_np]
            if n_eq > 0:
                rhs_pieces.append(dA_np @ x_np - db_np)
            if n_ineq > 0 and n_active > 0:
                rhs_pieces.append(
                    dG_np[:, active_np, :] @ x_np - dh_np[:, active_np]
                )

            # Broadcast to common batch size and concatenate
            batch_size = max(p.shape[0] for p in rhs_pieces)
            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in rhs_pieces
            ], axis=1).T                                    # (n_var + n_h, batch)

        else:
            # dL = dP x + dq + dA^T mu + dG_active^T lam_active
            dL_np = dP_np @ x_np + dq_np
            if n_eq > 0:
                dL_np = dL_np + dA_np.T @ mu_np
            if n_ineq > 0 and n_active > 0:
                dL_np = dL_np + (dG_np[active_np, :].T
                                 @ lam_np[active_np])

            rhs_parts = [dL_np]
            if n_eq > 0:
                rhs_parts.append(dA_np @ x_np - db_np)
            if n_ineq > 0 and n_active > 0:
                rhs_parts.append(
                    dG_np[active_np, :] @ x_np - dh_np[active_np]
                )
            rhs = np.hstack(rhs_parts)                     # (n_var + n_h,)
            batch_size = 0

        logger.debug(f"kkt_diff build_system: {perf_counter() - start:.3e}s")

        # ── Solve the linear system ──────────────────────────────────
        start = perf_counter()
        sol = np.linalg.solve(lhs, -rhs)
        logger.debug(f"kkt_diff lin_solve: {perf_counter() - start:.3e}s")

        # ── Extract dx, dmu, dlam from the solution ──────────────────
        if batched:
            dx_np  = sol[:n_var, :].T                       # (B, n_var)
            dmu_np = (sol[n_var:n_var + n_eq, :].T
                      if n_eq > 0
                      else np.empty((batch_size, 0), dtype=_dtype))
            dlam_np = np.zeros((batch_size, n_ineq), dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[:, active_np] = sol[n_var + n_eq:, :].T
        else:
            dx_np  = sol[:n_var]                            # (n_var,)
            dmu_np = (sol[n_var:n_var + n_eq]
                      if n_eq > 0
                      else np.empty(0, dtype=_dtype))
            dlam_np = np.zeros(n_ineq, dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[active_np] = sol[n_var + n_eq:]

        # ── Convert primal/dual solution to JAX ──────────────────────
        start = perf_counter()
        if batch_size > 0:
            res = {
                "x":      jnp.broadcast_to(jnp.array(x_np, dtype=_dtype),        (batch_size, n_var)),
                "lam":    jnp.broadcast_to(jnp.array(lam_np, dtype=_dtype),       (batch_size, n_ineq)),
                "mu":     jnp.broadcast_to(jnp.array(mu_np, dtype=_dtype),        (batch_size, n_eq)),
                "active": jnp.broadcast_to(jnp.array(active_np, dtype=jnp.bool_), (batch_size, n_ineq)),
            }
        else:
            res = {
                "x":      jnp.array(x_np, dtype=_dtype),
                "lam":    jnp.array(lam_np, dtype=_dtype),
                "mu":     jnp.array(mu_np, dtype=_dtype),
                "active": jnp.array(active_np, dtype=jnp.bool_),
            }
        logger.debug(f"kkt_diff convert_sol: {perf_counter() - start:.3e}s")

        # Active-set mask is not differentiable
        dactive = np.zeros(
            res["active"].shape, dtype=jax.dtypes.float0
        )

        logger.info(f"kkt_diff total: {perf_counter() - t_start:.3e}s")

        return dx_np, dlam_np, dmu_np, dactive, res
    
    def _kkt_diff_callback(primals_dict, tangents_dict):
        """Wrap ``_kkt_diff`` in a ``pure_callback`` for use inside JAX traces.

        Converts the named dicts back to positional args in
        ``_required_keys`` order (primals first, then tangents) as
        expected by ``_kkt_diff``.

        Args:
            primals_dict: Dict mapping ``_required_keys`` to primal
                JAX arrays.
            tangents_dict: Dict mapping ``_required_keys`` to tangent
                JAX arrays.

        Returns:
            The output of ``_kkt_diff``: a tuple
            ``(dx, dmu, dlam, dactive, sol)``.
        """
        primal_vals = tuple(primals_dict[k] for k in _required_keys)
        tangent_vals = tuple(tangents_dict[k] for k in _required_keys)
        return pure_callback(
            _kkt_diff,
            _jvp_shapes,
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
    def _solver_dynamic(*vals: jax.Array) -> dict[str, jax.Array]:
        """Internal solver with positional-only arguments.

        Wraps ``_solve_qp`` in a ``pure_callback`` so that the numpy QP solver
        can be called from within JAX-traced code. All required QP ingredients
        are passed positionally in ``_required_keys`` order, so that every
        argument is visible to JAX's tracer and carries correct tangent
        information.

        Fixed elements supplied at setup time act as defaults; they are merged
        in ``solver()`` before reaching this function, so from this function's
        perspective every required ingredient is always present.

        Args:
            *vals (jax.Array): JAX arrays corresponding to ``_required_keys``,
                in the same order (i.e. P, q, [A, b,] [G, h]).

        Returns:
            dict[str, jax.Array]: Solution dict with keys ``x``, ``lam``,
                ``mu``, ``active``.
        """
        return pure_callback(_solve_qp, _fwd_shapes, *vals)

    @_solver_dynamic.defjvp
    def _solver_dynamic_jvp(
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """JVP rule for ``_solver_dynamic`` via implicit differentiation of the KKT conditions.

        ``primals`` and ``tangents`` are positional tuples whose order matches
        ``_required_keys``. Any element — whether originally fixed or supplied
        at runtime — flows through the traced path, so its tangent is computed
        correctly by JAX regardless of whether it was declared fixed at setup.

        Args:
            primals (tuple[jax.Array, ...]): Primal inputs corresponding to
                ``_required_keys`` order.
            tangents (tuple[jax.Array, ...]): Tangent inputs with the same
                structure as ``primals``. Elements that do not depend on any
                upstream traced computation will naturally be zero; elements
                that do (even if they override a fixed default) will carry
                the correct nonzero tangent.

        Returns:
            tuple[dict[str, jax.Array], dict[str, jax.Array]]: Tuple of
                ``(primal_out, tangent_out)`` where both are solution dicts
                with keys ``x``, ``lam``, ``mu``, ``active``.
        """
        # Recover named access by zipping with the stable key order
        primals_dict  : dict[str, jax.Array] = dict(zip(_required_keys, primals))
        tangents_dict : dict[str, jax.Array] = dict(zip(_required_keys, tangents))

        # Differentiate via KKT implicit differentiation
        dx, dlam, dmu, dactive, res = _kkt_diff_callback(primals_dict, tangents_dict)

        tangents_out: dict[str, jax.Array] = {
            "x":      dx,
            "lam":    dlam,
            "mu":     dmu,
            "active": dactive,
        }

        return res, tangents_out

    def solver(**runtime: jax.Array) -> dict[str, jax.Array]:
        """Solve a (possibly partially fixed) QP problem.

        The full set of QP ingredients is ``{P, q, A, b, G, h}``. Any subset
        may be fixed at setup time via ``fixed_elements``; the remainder must
        be supplied here at call time. If an ingredient is supplied both at
        setup and at call time, the **runtime value takes precedence** — this
        allows overriding defaults without rebuilding the solver or
        re-triggering JIT compilation.

        Every ingredient (fixed or runtime) is routed through JAX's tracer,
        so gradients are correct regardless of whether a value was originally
        declared as fixed.

        Args:
            **runtime (jax.Array): QP ingredients as JAX arrays. Must cover
                at least every key not already present in ``fixed_elements``.
                May also include keys that *are* fixed, in which case the
                runtime value overrides the default. Valid keys are:

                - ``P`` (n_var, n_var): positive semi-definite cost matrix
                - ``q`` (n_var,): linear cost vector
                - ``A`` (n_eq, n_var): equality constraint matrix
                - ``b`` (n_eq,): equality constraint vector
                - ``G`` (n_ineq, n_var): inequality constraint matrix
                - ``h`` (n_ineq,): inequality constraint vector

        Returns:
            dict[str, jax.Array]: Solution dict with keys:

                - ``x`` (n_var,): primal solution
                - ``lam`` (n_ineq,): dual variables for inequality constraints
                - ``mu`` (n_eq,): dual variables for equality constraints
                - ``active`` (n_ineq,): boolean active-set mask

        Raises:
            ValueError: If any required ingredient is missing from both
                ``fixed_elements`` and ``runtime``.
        """
        # Merge: runtime values override fixed defaults
        merged = {**_fixed, **runtime}

        missing = _required_keys_set - set(merged)
        if missing:
            raise ValueError(
                f"Missing QP ingredients: {sorted(missing)}. "
                f"Provide them via fixed_elements at setup or at call time."
            )

        # Build positional args in _required_keys order — every ingredient
        # passes through the traced path so that tangents are always correct.
        vals: tuple[jax.Array, ...] = tuple(merged[k] for k in _required_keys)

        return _solver_dynamic(*vals)
    
    #endregion

    # =================================================================

    return solver