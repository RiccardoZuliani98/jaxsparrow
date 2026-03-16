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
    """Set up a differentiable dense QP solver.

    Creates and returns a ``solver(**runtime)`` callable that solves
    quadratic programs of the form::

        min  0.5 x^T P x + q^T x
        s.t. A x  = b
             G x <= h

    The solver is compatible with ``jax.jvp`` and ``jax.grad`` via
    implicit differentiation of the KKT conditions.

    Args:
        n_var: Number of decision variables.
        n_ineq: Number of inequality constraints (may be 0).
        n_eq: Number of equality constraints (may be 0).
        fixed_elements: Optional dict of QP ingredients that remain
            constant across calls. Valid keys are ``P``, ``q``, ``A``,
            ``b``, ``G``, ``h``. Fixed elements act as defaults and
            can be overridden at call time.
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

    # form shapes of jvp output — must be a tuple matching the
    # return order of _kkt_diff: (dx, dlam, dmu, dactive, sol)
    _jvp_shapes = (
        jax.ShapeDtypeStruct((n_var,),  _dtype),             # dx
        jax.ShapeDtypeStruct((n_ineq,), _dtype),             # dlam (inequality duals)
        jax.ShapeDtypeStruct((n_eq,),   _dtype),             # dmu  (equality duals)
        jax.ShapeDtypeStruct((n_ineq,), jax.dtypes.float0),  # dactive
        _fwd_shapes,                                          # sol
    )

    # gather required keys
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

    logger.info(
        f"Setting up QP with {n_var} variables, "
        f"{n_eq} equalities, {n_ineq} inequalities."
    )
    logger.info(f"Fixed variables: {set(_fixed.keys()) or 'none'}")

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
        with ``_fixed`` is performed here (that happens upstream in
        ``solver()``).

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
            Solution dict with JAX arrays:

                - ``x``      (nv,):  Primal solution.
                - ``lam``    (ni,):  Inequality duals.
                - ``mu``     (ne,):  Equality duals.
                - ``active`` (ni,):  Active-set boolean mask.
        """
        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Map positional args back to named ingredients ────────────
        kwargs = dict(zip(_required_keys, vals))

        # ── Convert JAX arrays to numpy ──────────────────────────────
        start = perf_counter()
        prob_np = {
            k: np.asarray(v, dtype=_dtype).squeeze()
            for k, v in kwargs.items()
        }
        t["convert_to_numpy"] = perf_counter() - start

        # ── Solve in numpy ───────────────────────────────────────────
        x_np, lam_np, mu_np, active_np, t_solve = _solve_qp_numpy(**prob_np)
        t.update(t_solve)

        # ── Convert solution back to JAX ─────────────────────────────
        start = perf_counter()
        result: dict[str, jax.Array] = {
            "x":      jnp.array(x_np, dtype=_dtype),       # (nv,)
            "lam":    jnp.array(lam_np, dtype=_dtype),      # (ni,)
            "mu":     jnp.array(mu_np, dtype=_dtype),       # (ne,)
            "active": jnp.array(active_np, dtype=jnp.bool_),  # (ni,)
        }
        t["convert_to_jax"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_solve_qp | {_fmt_times(t)}")

        return result

    #endregion

    # =================================================================
    # SETUP DIFFERENTIATOR
    # =================================================================

    #region
    def _kkt_diff(*args: ndarray) -> tuple[
        Float[ndarray, " nv"],                          # dx
        Float[ndarray, " ni"],                          # dlam
        Float[ndarray, " ne"],                          # dmu
        ndarray,                                        # dactive (float0)
        dict[str, jax.Array],                           # sol
    ]:
        """Implicit differentiation of the KKT conditions.

        Solves the QP forward, then differentiates through the KKT
        optimality conditions by solving a linear system whose structure
        depends on the active set of inequality constraints.

        The function receives all primals followed by all tangents as
        positional arrays, both ordered according to
        ``_required_keys`` (i.e. P, q, [A, b,] [G, h]).

        Batching is detected automatically: when the tangent arrays
        carry an extra leading dimension compared to the primals, the
        linear system is solved for every tangent vector simultaneously
        via a matrix RHS.

        Args:
            *args: ``2 * len(_required_keys)`` arrays. The first
                half are primals, the second half are tangents, both in
                ``_required_keys`` order.

                Primals (squeezed to remove batch-1 dims):

                    - ``P``  (nv, nv): Cost matrix.
                    - ``q``  (nv,):    Linear cost.
                    - ``A``  (ne, nv): Equality constraint matrix.
                    - ``b``  (ne,):    Equality RHS.
                    - ``G``  (ni, nv): Inequality constraint matrix.
                    - ``h``  (ni,):    Inequality RHS.

                Tangents (unbatched or batched with leading ``B`` dim):

                    - ``dP`` (nv, nv) | (B, nv, nv)
                    - ``dq`` (nv,)    | (B, nv)
                    - ``dA`` (ne, nv) | (B, ne, nv)
                    - ``db`` (ne,)    | (B, ne)
                    - ``dG`` (ni, nv) | (B, ni, nv)
                    - ``dh`` (ni,)    | (B, ni)

        Returns:
            A tuple ``(dx, dlam, dmu, dactive, sol)`` where:

                - ``dx``      (nv,) | (B, nv):   Tangent of primal.
                - ``dlam``    (ni,) | (B, ni):   Tangent of ineq duals.
                - ``dmu``     (ne,) | (B, ne):   Tangent of eq duals.
                - ``dactive`` (ni,) | (B, ni):   Zero tangent (float0).
                - ``sol``: Primal / dual solution dict as JAX arrays.
        """
        t_start = perf_counter()
        t: dict[str, float] = {}
        n_keys = len(_required_keys)

        # ── Split positional args into primals and tangents ──────────
        primal_vals = args[:n_keys]
        tangent_vals = args[n_keys:]
        primals = dict(zip(_required_keys, primal_vals))
        tangents = dict(zip(_required_keys, tangent_vals))

        # ── Convert primals to numpy and squeeze batch-1 dims ────────
        start = perf_counter()
        prob_np: dict[str, ndarray] = {
            k: np.asarray(v, dtype=_dtype).squeeze()
            for k, v in primals.items()
        }
        t["convert_primals"] = perf_counter() - start

        # ── Forward solve ────────────────────────────────────────────
        # x_np:      (nv,)
        # lam_np:    (ni,)
        # mu_np:     (ne,)
        # active_np: (ni,)  bool
        x_np, lam_np, mu_np, active_np, t_solve = _solve_qp_numpy(**prob_np)
        t.update({f"fwd_{k}": v for k, v in t_solve.items()})

        # ── Convert tangents to numpy (keep batch dimension) ─────────
        start = perf_counter()
        d_np: dict[str, ndarray] = {
            k: np.asarray(v, dtype=_dtype)
            for k, v in tangents.items()
        }
        t["convert_tangents"] = perf_counter() - start

        # ── Detect batching ──────────────────────────────────────────
        # Tangents have an extra leading dim when vmap expands them.
        batched = d_np["P"].ndim == 3

        # ── Retrieve numpy arrays with short aliases ─────────────────
        P_np: Float[ndarray, "nv nv"] = prob_np["P"]
        A_np: Float[ndarray, "ne nv"] = prob_np.get(
            "A", np.empty((0, n_var), dtype=_dtype)
        )
        G_np: Float[ndarray, "ni nv"] = prob_np.get(
            "G", np.empty((0, n_var), dtype=_dtype)
        )

        # Tangent aliases — shape depends on batching
        # Unbatched: same shape as primals
        # Batched:   (B, ...) with leading batch dim
        dP_np = d_np["P"]              # (nv,nv) | (B,nv,nv)
        dq_np = d_np["q"]              # (nv,)   | (B,nv)
        dA_np = d_np.get(              # (ne,nv) | (B,ne,nv)
            "A",
            np.empty((0, n_var), dtype=_dtype) if not batched
            else np.empty((1, 0, n_var), dtype=_dtype),
        )
        db_np = d_np.get(              # (ne,)   | (B,ne)
            "b",
            np.empty((0,), dtype=_dtype) if not batched
            else np.empty((1, 0), dtype=_dtype),
        )
        dG_np = d_np.get(              # (ni,nv) | (B,ni,nv)
            "G",
            np.empty((0, n_var), dtype=_dtype) if not batched
            else np.empty((1, 0, n_var), dtype=_dtype),
        )
        dh_np = d_np.get(              # (ni,)   | (B,ni)
            "h",
            np.empty((0,), dtype=_dtype) if not batched
            else np.empty((1, 0), dtype=_dtype),
        )

        # ── Build KKT system LHS (same for batched and unbatched) ────
        start = perf_counter()

        # H stacks equality rows and active inequality rows
        n_active = int(np.sum(active_np))
        H_parts: list[Float[ndarray, "_ nv"]] = []
        if n_eq > 0:
            H_parts.append(A_np)                            # (ne, nv)
        if n_ineq > 0 and n_active > 0:
            H_parts.append(G_np[active_np, :])              # (na, nv)

        if H_parts:
            H_np: Float[ndarray, "nh nv"] = np.vstack(H_parts)
        else:
            H_np = np.empty((0, n_var), dtype=_dtype)

        n_h = H_np.shape[0]                                # ne + na

        # LHS: [ P    H^T ]   shape (nv+nh, nv+nh)
        #      [ H    0   ]
        lhs: Float[ndarray, "nv_nh nv_nh"] = np.block([
            [P_np,  H_np.T],
            [H_np,  np.zeros((n_h, n_h), dtype=_dtype)],
        ])

        # ── Build RHS (differs between batched and unbatched) ────────
        if batched:
            # dL: (B, nv)
            # dL = dP @ x + dq + dA^T @ mu + dG_active^T @ lam_active
            dL_np: Float[ndarray, "B nv"] = dP_np @ x_np + dq_np
            if n_eq > 0:
                # dA_np:           (B, ne, nv)
                # dA^T @ mu:       (B, nv, ne) @ (ne,) -> (B, nv)
                dL_np = dL_np + dA_np.transpose(0, 2, 1) @ mu_np
            if n_ineq > 0 and n_active > 0:
                # dG[:, active, :]:       (B, na, nv)
                # transposed @ lam_active: (B, nv, na) @ (na,) -> (B, nv)
                dL_np = dL_np + (
                    dG_np[:, active_np, :]
                    .transpose(0, 2, 1) @ lam_np[active_np]
                )

            # Collect RHS pieces, each (batch_i, n_i)
            rhs_pieces: list[Float[ndarray, "_ _"]] = [dL_np]  # (B, nv)
            if n_eq > 0:
                # dA @ x - db: (B, ne)
                rhs_pieces.append(dA_np @ x_np - db_np)
            if n_ineq > 0 and n_active > 0:
                # dG[:, active, :] @ x - dh[:, active]: (B, na)
                rhs_pieces.append(
                    dG_np[:, active_np, :] @ x_np - dh_np[:, active_np]
                )

            # Broadcast to common batch size and concatenate
            batch_size = max(p.shape[0] for p in rhs_pieces)
            rhs: Float[ndarray, "nv_nh B"] = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1]))
                if p.shape[0] == 1 else p
                for p in rhs_pieces
            ], axis=1).T                                    # (nv+nh, B)

        else:
            # dL: (nv,)
            # dL = dP @ x + dq + dA^T @ mu + dG_active^T @ lam_active
            dL_np: Float[ndarray, " nv"] = dP_np @ x_np + dq_np
            if n_eq > 0:
                # dA^T @ mu: (nv, ne) @ (ne,) -> (nv,)
                dL_np = dL_np + dA_np.T @ mu_np
            if n_ineq > 0 and n_active > 0:
                # dG[active]^T @ lam[active]: (nv, na) @ (na,) -> (nv,)
                dL_np = dL_np + (
                    dG_np[active_np, :].T @ lam_np[active_np]
                )

            # Stack: [dL; dA @ x - db; dG_active @ x - dh_active]
            rhs_parts: list[Float[ndarray, " _"]] = [dL_np]  # (nv,)
            if n_eq > 0:
                # dA @ x - db: (ne,)
                rhs_parts.append(dA_np @ x_np - db_np)
            if n_ineq > 0 and n_active > 0:
                # dG[active] @ x - dh[active]: (na,)
                rhs_parts.append(
                    dG_np[active_np, :] @ x_np - dh_np[active_np]
                )
            rhs: Float[ndarray, " nv_nh"] = np.hstack(rhs_parts)
            batch_size = 0

        t["build_system"] = perf_counter() - start

        # ── Solve the linear system ──────────────────────────────────
        # lhs @ sol = -rhs
        # Unbatched: sol is (nv+nh,)
        # Batched:   sol is (nv+nh, B)
        start = perf_counter()
        sol = np.linalg.solve(lhs, -rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract dx, dmu, dlam from the solution ──────────────────
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
                dlam_np[:, active_np] = sol[n_var + n_eq:, :].T  # (B, na)
        else:
            dx_np: Float[ndarray, " nv"] = sol[:n_var]
            dmu_np: Float[ndarray, " ne"] = (
                sol[n_var:n_var + n_eq]
                if n_eq > 0
                else np.empty(0, dtype=_dtype)
            )
            dlam_np: Float[ndarray, " ni"] = np.zeros(n_ineq, dtype=_dtype)
            if n_ineq > 0 and n_active > 0:
                dlam_np[active_np] = sol[n_var + n_eq:]      # (na,)

        # ── Convert primal/dual solution to JAX ──────────────────────
        start = perf_counter()
        if batch_size > 0:
            res: dict[str, jax.Array] = {
                "x":      jnp.broadcast_to(                 # (B, nv)
                    jnp.array(x_np, dtype=_dtype),
                    (batch_size, n_var),
                ),
                "lam":    jnp.broadcast_to(                  # (B, ni)
                    jnp.array(lam_np, dtype=_dtype),
                    (batch_size, n_ineq),
                ),
                "mu":     jnp.broadcast_to(                  # (B, ne)
                    jnp.array(mu_np, dtype=_dtype),
                    (batch_size, n_eq),
                ),
                "active": jnp.broadcast_to(                  # (B, ni)
                    jnp.array(active_np, dtype=jnp.bool_),
                    (batch_size, n_ineq),
                ),
            }
        else:
            res = {
                "x":      jnp.array(x_np, dtype=_dtype),       # (nv,)
                "lam":    jnp.array(lam_np, dtype=_dtype),      # (ni,)
                "mu":     jnp.array(mu_np, dtype=_dtype),       # (ne,)
                "active": jnp.array(active_np, dtype=jnp.bool_),  # (ni,)
            }
        t["convert_sol"] = perf_counter() - start

        # Active-set mask is not differentiable — zero tangent
        dactive = np.zeros(                                  # (ni,) | (B, ni)
            res["active"].shape, dtype=jax.dtypes.float0
        )

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_diff | {_fmt_times(t)}")

        return dx_np, dlam_np, dmu_np, dactive, res

    def _kkt_diff_callback(
        primals_dict: dict[str, jax.Array],
        tangents_dict: dict[str, jax.Array],
    ) -> tuple[
        Float[jax.Array, " nv"],    # dx
        Float[jax.Array, " ni"],    # dlam
        Float[jax.Array, " ne"],    # dmu
        jax.Array,                  # dactive (float0)
        dict[str, jax.Array],       # sol
    ]:
        """Wrap ``_kkt_diff`` in a ``pure_callback`` for use inside JAX traces.

        Converts the named dicts back to positional args in
        ``_required_keys`` order (primals first, then tangents) as
        expected by ``_kkt_diff``.

        Args:
            primals_dict: Dict mapping ``_required_keys`` to primal
                JAX arrays:

                    - ``P`` (nv, nv), ``q`` (nv,),
                    - ``A`` (ne, nv), ``b`` (ne,),
                    - ``G`` (ni, nv), ``h`` (ni,).

            tangents_dict: Dict mapping ``_required_keys`` to tangent
                JAX arrays (same shapes as primals, or with leading
                batch dim ``B``).

        Returns:
            The output of ``_kkt_diff``: a tuple
            ``(dx, dlam, dmu, dactive, sol)``.
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

        Wraps ``_solve_qp`` in a ``pure_callback`` so that the numpy QP
        solver can be called from within JAX-traced code. All required QP
        ingredients are passed positionally in ``_required_keys`` order,
        so that every argument is visible to JAX's tracer and carries
        correct tangent information.

        Fixed elements supplied at setup time act as defaults; they are
        merged in ``solver()`` before reaching this function, so from
        this function's perspective every required ingredient is always
        present.

        Args:
            *vals: JAX arrays corresponding to ``_required_keys``, in
                the same order:

                    - ``P`` (nv, nv): Cost matrix.
                    - ``q`` (nv,): Linear cost.
                    - ``A`` (ne, nv): Equality constraints (if present).
                    - ``b`` (ne,): Equality RHS (if present).
                    - ``G`` (ni, nv): Inequality constraints (if present).
                    - ``h`` (ni,): Inequality RHS (if present).

        Returns:
            Solution dict with JAX arrays:

                - ``x``      (nv,):  Primal solution.
                - ``lam``    (ni,):  Inequality duals.
                - ``mu``     (ne,):  Equality duals.
                - ``active`` (ni,):  Active-set boolean mask.
        """
        return pure_callback(_solve_qp, _fwd_shapes, *vals)

    @_solver_dynamic.defjvp
    def _solver_dynamic_jvp(
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """JVP rule for ``_solver_dynamic`` via implicit differentiation.

        ``primals`` and ``tangents`` are positional tuples whose order
        matches ``_required_keys``. Any element — whether originally
        fixed or supplied at runtime — flows through the traced path,
        so its tangent is computed correctly by JAX regardless of
        whether it was declared fixed at setup.

        Args:
            primals: Primal inputs in ``_required_keys`` order.

                - ``P`` (nv, nv), ``q`` (nv,),
                - ``A`` (ne, nv), ``b`` (ne,),
                - ``G`` (ni, nv), ``h`` (ni,).

            tangents: Tangent inputs with the same structure as
                ``primals``. Elements that do not depend on any
                upstream traced computation will naturally be zero;
                elements that do (even if they override a fixed
                default) will carry the correct nonzero tangent.

        Returns:
            ``(primal_out, tangent_out)`` where both are solution dicts:

                - ``x``      (nv,):  Primal / tangent of primal.
                - ``lam``    (ni,):  Inequality duals / tangent.
                - ``mu``     (ne,):  Equality duals / tangent.
                - ``active`` (ni,):  Active mask / zero tangent.
        """
        # Recover named access by zipping with the stable key order
        primals_dict: dict[str, jax.Array] = dict(
            zip(_required_keys, primals)
        )
        tangents_dict: dict[str, jax.Array] = dict(
            zip(_required_keys, tangents)
        )

        # Differentiate via KKT implicit differentiation
        # Returns: (dx, dlam, dmu, dactive, res)
        dx, dlam, dmu, dactive, res = _kkt_diff_callback(
            primals_dict, tangents_dict
        )

        tangents_out: dict[str, jax.Array] = {
            "x":      dx,       # (nv,)
            "lam":    dlam,     # (ni,)
            "mu":     dmu,      # (ne,)
            "active": dactive,  # (ni,) float0
        }

        return res, tangents_out

    def solver(**runtime: jax.Array) -> dict[str, jax.Array]:
        """Solve a (possibly partially fixed) QP problem.

        The full set of QP ingredients is ``{P, q, A, b, G, h}``. Any
        subset may be fixed at setup time via ``fixed_elements``; the
        remainder must be supplied here at call time. If an ingredient
        is supplied both at setup and at call time, the **runtime value
        takes precedence** — this allows overriding defaults without
        rebuilding the solver or re-triggering JIT compilation.

        Every ingredient (fixed or runtime) is routed through JAX's
        tracer, so gradients are correct regardless of whether a value
        was originally declared as fixed.

        Args:
            **runtime: QP ingredients as JAX arrays. Must cover at
                least every key not already present in
                ``fixed_elements``. May also include keys that *are*
                fixed, in which case the runtime value overrides the
                default. Valid keys and shapes:

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
            ValueError: If any required ingredient is missing from
                both ``fixed_elements`` and ``runtime``.
        """
        # Merge: runtime values override fixed defaults
        merged = {**_fixed, **runtime}

        missing = _required_keys_set - set(merged)
        if missing:
            raise ValueError(
                f"Missing QP ingredients: {sorted(missing)}. "
                f"Provide them via fixed_elements at setup or at "
                f"call time."
            )

        # Build positional args in _required_keys order — every
        # ingredient passes through the traced path so that tangents
        # are always correct.
        vals: tuple[jax.Array, ...] = tuple(
            merged[k] for k in _required_keys
        )

        return _solver_dynamic(*vals)

    #endregion

    # =================================================================

    return solver