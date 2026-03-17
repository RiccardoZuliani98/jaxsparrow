from jax import custom_jvp, custom_vjp, pure_callback, Array
import jax.numpy as jnp
from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
import jax
import logging
from typing import Optional, cast
from numpy import ndarray
from jaxtyping import Float, Bool

from src.utils.parsing_utils import parse_options
from src.utils.printing_utils import fmt_times
from src.utils.timing_utils import TimingRecorder
from src.solver_dense.solver_dense_options import (
    DEFAULT_SOLVER_OPTIONS, 
    SolverOptions, 
)
from src.solver_dense.solvers import create_dense_qp_solver
from src.solver_dense.differentiators import create_dense_kkt_differentiator_fwd

# Dimension key (used in jaxtyping annotations throughout):
#
#   nv  = n_var   — number of decision variables
#   ne  = n_eq    — number of equality constraints
#   ni  = n_ineq  — number of inequality constraints
#   na  = n_active — number of active inequality constraints (runtime)
#   B   = batch   — batch dimension (JVP vmap path only)

#TODO: add better type annotations using the types in solver_dense_types
#TODO: add a finite difference utility similar in principle to TimingRecorder.
# this should only use the numpy sub-solver and therefore not contribute to 
# the overall timings
#TODO: add reverse mode tests
#TODO: add value function and envelope theorem?

# Expected ndim for each QP ingredient (unbatched).
# Used to detect batching in the JVP path.
_EXPECTED_NDIM: dict[str, int] = {
    "P": 2, "q": 1, "A": 2, "b": 1, "G": 2, "h": 1,
}


def setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: Optional[dict[str,ndarray]] = None,
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

    # timing recorder — collects structured timing dicts across calls.
    # Accessible after setup as ``solver.timings``.
    _timings = TimingRecorder()

    # parse user options
    _options_parsed = parse_options(options, DEFAULT_SOLVER_OPTIONS)

    # extract dtype for simplicity
    _dtype = _options_parsed['dtype']

    # form shapes of problem solution
    _fwd_shapes = {
        "x":        jax.ShapeDtypeStruct((n_var,),  _dtype),
        "lam":      jax.ShapeDtypeStruct((n_ineq,), _dtype),
        "mu":       jax.ShapeDtypeStruct((n_eq,),   _dtype),
    }

    # Shapes for the combined JVP callback: tangents + solution.
    # Must match the return order of _kkt_diff:
    # (dx, dlam, dmu, sol_dict).
    # pure_callback uses these to convert numpy → JAX internally.
    _jvp_shapes = (
        jax.ShapeDtypeStruct((n_var,),  _dtype),             # dx
        jax.ShapeDtypeStruct((n_ineq,), _dtype),             # dlam
        jax.ShapeDtypeStruct((n_eq,),   _dtype),             # dmu
        _fwd_shapes,                                          # sol
    )

    # gather all required keys
    _required_keys = ("P", "q")
    if n_eq > 0:
        _required_keys += ("A", "b")
    if n_ineq > 0:
        _required_keys += ("G", "h")

    # gather fixed keys
    _fixed_keys_set = set(fixed_elements.keys()) if fixed_elements is not None else set()

    # gather keys required at runtime (i.e., not fixed).
    # Only these flow through JAX's traced path.
    _dynamic_keys = tuple(k for k in _required_keys if k not in _fixed_keys_set)

    # transform to sets for speed
    _required_keys_set = set(_required_keys)
    _dynamic_keys_set = set(_dynamic_keys)

    _n_dyn = len(_dynamic_keys)

    # Expected shapes for each QP ingredient (unbatched).
    # Used for VJP backward output shapes and batch detection.
    _expected_shapes: dict[str, tuple[int, ...]] = {
        "P": (n_var, n_var),
        "q": (n_var,),
        "A": (n_eq, n_var),
        "b": (n_eq,),
        "G": (n_ineq, n_var),
        "h": (n_ineq,),
    }

    # Number of required keys (used for arg layout in VJP callbacks).
    _n_req = len(_required_keys)

    # Shapes for the VJP forward callback: solution + active set +
    # all required problem matrices (already merged with fixed).
    # Returned as a flat tuple so residuals carry everything the
    # backward needs — no re-merging with _fixed required.
    _vjp_fwd_shapes = (
        jax.ShapeDtypeStruct((n_var,),  _dtype),                          # x
        jax.ShapeDtypeStruct((n_ineq,), _dtype),                          # lam
        jax.ShapeDtypeStruct((n_eq,),   _dtype),                          # mu
        jax.ShapeDtypeStruct((n_ineq,), _options_parsed["bool_dtype"]),   # active
        *(jax.ShapeDtypeStruct(_expected_shapes[k], _dtype)               # prob
          for k in _required_keys),
    )

    # Shapes for the VJP backward callback: one cotangent per dynamic key,
    # matching the shape of the corresponding primal input.
    _vjp_bwd_shapes = tuple(
        jax.ShapeDtypeStruct(_expected_shapes[k], _dtype)
        for k in _dynamic_keys
    )

    # Mutable warmstart slot. Written by solver() before each call,
    # read by _solve_qp_numpy() inside the callback, cleared after use.
    # Stored as a one-element list so nested functions can mutate it.
    _warmstart: list[Optional[ndarray]] = [None]

    # check shape of fixed_elements if present
    if fixed_elements is not None:
        for key,val in fixed_elements.items():
            assert val.shape == _expected_shapes[key]

    # choose numpy solver and store it in _solve_qp_numpy. Note that
    # the fixed elements are passed to the solver and will never be 
    # seen again, since they will get hard-coded inside the solver.
    if _options_parsed["solver_type"] == "qp_solvers":
        _solve_qp_numpy = create_dense_qp_solver(
            n_eq=n_eq, 
            n_ineq=n_ineq, 
            options=_options_parsed["solver"],
            fixed_elements=fixed_elements
        )
    else:
        raise ValueError("Only qp_solvers is available as 'solver_type'")
    
    # choose differentiator, once again some elements will be fixed
    # insider for speed
    if _options_parsed["differentiator_type"] == "kkt_fwd":
        _diff_forward_numpy = create_dense_kkt_differentiator_fwd(
            n_var=n_var,
            n_eq=n_eq,
            n_ineq=n_ineq,
            options=_options_parsed["differentiator"],
            fixed_elements=fixed_elements
        )
    else:
        raise ValueError("Only differentiator available is 'kkt_fwd'")

    logger.info(
        f"Setting up QP with {n_var} variables, "
        f"{n_eq} equalities, {n_ineq} inequalities."
    )
    logger.info(f"Fixed variables: {_fixed_keys_set or 'none'}")
    logger.info(f"Dynamic variables: {_dynamic_keys_set or 'none'}")


    # =================================================================
    # BASE SOLVER
    # =================================================================

    #region
    def _solve_qp(*dynamic_vals: jax.Array) -> dict[str, Array]:
        """Bridge between JAX ``pure_callback`` and the numpy QP solver.

        Receives only the **dynamic** QP ingredients (those not in
        ``_fixed``) as positional JAX arrays in ``_dynamic_keys`` order,
        converts them to numpy, merges with the pre-stored ``_fixed``
        numpy arrays (no conversion needed for those), and delegates to
        ``_solve_qp_numpy``.

        Returns numpy arrays directly — ``pure_callback`` handles
        the numpy → JAX conversion using ``_fwd_shapes``.

        Args:
            *dynamic_vals: JAX arrays corresponding to ``_dynamic_keys``,
                in order.

        Returns:
            Solution dict with numpy arrays:

                - ``x``      (nv,):  Primal solution.
                - ``lam``    (ni,):  Inequality duals.
                - ``mu``     (ne,):  Equality duals.
                - ``active`` (ni,):  Active-set boolean mask.
        """
        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Convert only dynamic JAX arrays to numpy ─────────────────
        start = perf_counter()
        dynamic_np: dict[str, ndarray] = {}
        for k, v in zip(_dynamic_keys, dynamic_vals):
            #TODO: this is the only line that has to change for sparse mode,
            # can we keep a unique solver skeleton?
            arr = np.asarray(v, dtype=_dtype)
            if arr.ndim > _EXPECTED_NDIM[k]:
                arr = arr[0]  # all batch entries are identical for primals
            dynamic_np[k] = arr
        t["convert_to_numpy"] = perf_counter() - start

        # ── Run checks ───────────────────────────────────────────────
        if _options_parsed["debug"]:
            missing = _dynamic_keys_set - set(dynamic_np)
            if missing:
                raise ValueError(
                    f"Missing QP ingredients: {sorted(missing)}. "
                    f"Provide them either via fixed_elements at setup or "
                    f"at call time."
                )

            # Validate shapes against declared problem dimensions
            for key in _dynamic_keys_set:
                assert dynamic_np[key].shape == _expected_shapes[key]

        # ── Add warmstarting ─────────────────────────────────────────
        prob_np = dynamic_np
        if _warmstart[0] is not None: 
            dynamic_np["warmstart"] = _warmstart[0]

        # ── Solve in numpy ───────────────────────────────────────────
        x_np, lam_np, mu_np, _, t_solve = _solve_qp_numpy(**prob_np)

        # Clear warmstart after use so it doesn't persist
        _warmstart[0] = None

        # store time
        t.update(t_solve)

        # ── Return jax arrays directly ────────────────────────────────
        # pure_callback handles numpy → JAX conversion using _fwd_shapes.
        start = perf_counter()
        result = {
            "x":      jnp.array(x_np,dtype=_dtype),
            "lam":    jnp.array(lam_np,dtype=_dtype),
            "mu":     jnp.array(mu_np,dtype=_dtype),
        }
        t["convert_to_jax"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_solve_qp | {fmt_times(t)}")
        _timings.record("_solve_qp", t)

        return result
    #endregion

    # =================================================================
    # FORWARD SOLVE FOR VJP PATH
    # =================================================================

    #region
    def _diff_forward(*args: ndarray) -> tuple[
        Float[Array, " nv"],      # dx
        Float[Array, " ni"],      # dlam
        Float[Array, " ne"],      # dmu
        dict[str, Array],         # sol (numpy, not JAX)
    ]:
        """Implicit differentiation of the KKT conditions.

        Solves the QP forward, then differentiates through the KKT
        optimality conditions by solving a linear system. Returns
        both the tangents and the primal solution as **numpy arrays**.
        ``pure_callback`` handles all numpy → JAX conversion using
        ``_jvp_shapes``, avoiding redundant Python-level ``jnp.array``
        calls.

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
            A tuple ``(dx, dlam, dmu, sol)`` where:

                - ``dx``      (nv,) | (B, nv):   Tangent of primal.
                - ``dlam``    (ni,) | (B, ni):   Tangent of ineq duals.
                - ``dmu``     (ne,) | (B, ne):   Tangent of eq duals.
                - ``sol``: Primal / dual solution dict as numpy arrays.
                  Broadcast to ``(B, ...)`` when batched.
        """

        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Split positional args into dynamic primals and tangents ──
        dyn_primal_vals = args[:_n_dyn]
        dyn_tangent_vals = args[_n_dyn:]

        # ── Convert dynamic primals to numpy, squeeze batch-1 dims ───
        start = perf_counter()
        dyn_primals_np: dict[str, ndarray] = {}
        for k, v in zip(_dynamic_keys, dyn_primal_vals):
            #TODO: adapt this to the sparse case
            arr = np.asarray(v, dtype=_dtype)
            if arr.ndim > _EXPECTED_NDIM[k]:
                arr = arr[0]  # all batch entries are identical for primals
            dyn_primals_np[k] = arr
        t["convert_to_numpy"] = perf_counter() - start

        # ── Run checks ───────────────────────────────────────────────
        if _options_parsed["debug"]:
            missing = _dynamic_keys_set - set(dyn_primals_np)
            if missing:
                raise ValueError(
                    f"Missing QP ingredients: {sorted(missing)}. "
                    f"Provide them either via fixed_elements at setup or "
                    f"at call time."
                )

            # Validate shapes against declared problem dimensions
            for key in _dynamic_keys_set:
                assert dyn_primals_np[key].shape == _expected_shapes[key]

        # ── Add warmstarting ─────────────────────────────────────────
        prob_np = dyn_primals_np
        if _warmstart[0] is not None: 
            dyn_primals_np["warmstart"] = _warmstart[0]

        # ── Solve in numpy ───────────────────────────────────────────
        x_np, lam_np, mu_np, active_np, t_solve = _solve_qp_numpy(**prob_np)

        # Clear warmstart after use so it doesn't persist
        _warmstart[0] = None

        # store time
        t.update(t_solve)

        # ── Convert dynamic tangents to numpy (keep batch dim) ───────
        start = perf_counter()
        dyn_tangents_np: dict[str, ndarray] = {
            #TODO: here for sparse too
            k: np.asarray(v, dtype=_dtype)
            for k, v in zip(_dynamic_keys, dyn_tangent_vals)
        }
        t["convert_tangents"] = perf_counter() - start

        # ── Detect batching ──────────────────────────────────────────
        if _n_dyn > 0:

            # get a dynamic key
            first_key = _dynamic_keys[0]

            # check if dimension was augmented by one, if so, batching
            # was applied
            batched = (
                dyn_tangents_np[first_key].ndim
                == _EXPECTED_NDIM[first_key] + 1
            )
        else:

            # no batching if there are no dynamic arguments
            batched = False

        # get batch size
        batch_size = max(v.shape[0] for v in dyn_tangents_np.values()) if batched else 0


        # ── Call differentiator ──────────────────────────────────────
        dx_np, dlam_np, dmu_np, t_diff = _diff_forward_numpy(
            x_np=x_np, 
            lam_np=lam_np,
            mu_np=mu_np,
            active_np=active_np,
            dyn_primals_np=dyn_primals_np,
            dyn_tangents_np=dyn_tangents_np,
            batch_size=batch_size
        )
        t.update(t_diff)

        # ── Build solution dict (numpy only) ─────────────────────────
        # np.broadcast_to returns a read-only view (zero cost).
        # .copy() materializes a contiguous array for pure_callback.
        start = perf_counter()
        if batch_size > 0:
            res: dict[str, Array] = {
                "x":      jnp.array(np.broadcast_to(x_np, (batch_size, n_var)).copy(),dtype=_dtype),
                "lam":    jnp.array(np.broadcast_to(lam_np, (batch_size, n_ineq)).copy(),dtype=_dtype),
                "mu":     jnp.array(np.broadcast_to(mu_np, (batch_size, n_eq)).copy(),dtype=_dtype),
            }
        else:
            res = {
                "x":      jnp.array(x_np,dtype=_dtype),
                "lam":    jnp.array(lam_np,dtype=_dtype),
                "mu":     jnp.array(mu_np,dtype=_dtype),
            }
        dx = jnp.array(dx_np, dtype=_dtype)
        dlam = jnp.array(dlam_np, dtype=_dtype)
        dmu = jnp.array(dmu_np, dtype=_dtype)
        t["build_sol"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_diff | {fmt_times(t)}")
        _timings.record("_kkt_diff", t)

        return dx, dlam, dmu, res
    #endregion

    # =================================================================
    # SETUP VJP DIFFERENTIATOR
    # =================================================================

    #region

    #TODO: this should be moved outside
    def _kkt_vjp(*args: ndarray) -> tuple[ndarray, ...]:
        """Adjoint (VJP) differentiation of the KKT conditions.

        Given the pre-merged problem matrices, forward solution,
        active set, and output cotangents, solves the adjoint KKT
        system and computes parameter cotangents via implicit
        differentiation.

        The problem matrices are received **already merged** (dynamic
        + fixed) from the VJP forward pass residuals — no second
        conversion of ``dynamic_vals`` or re-merging with ``_fixed``
        is performed here.

        The adjoint system exploits the symmetry of the KKT matrix:
        because ``M = M^T``, the adjoint linear system is identical
        to the forward one, with the cotangent vector as the RHS.

        After solving for the adjoint variables ``v = M^{-1} g``, the
        parameter cotangents are obtained analytically::

            g_P = -outer(v_x, x)
            g_q = -v_x
            g_A = -(outer(mu, v_x) + outer(v_mu, x))
            g_b = v_mu
            g_G[active] = -(outer(lam[active], v_x)
                            + outer(v_lam_a, x))
            g_h[active] = v_lam_a

        Args:
            *args: ``_n_req + 7`` arrays laid out as:
                - ``args[:_n_req]``:       Merged problem matrices
                  (``_required_keys`` order).
                - ``args[_n_req]``:        ``x``      (nv,)
                - ``args[_n_req + 1]``:    ``lam``    (ni,)
                - ``args[_n_req + 2]``:    ``mu``     (ne,)
                - ``args[_n_req + 3]``:    ``active`` (ni,) bool
                - ``args[_n_req + 4]``:    ``g_x``    (nv,)
                - ``args[_n_req + 5]``:    ``g_lam``  (ni,)
                - ``args[_n_req + 6]``:    ``g_mu``   (ne,)

        Returns:
            A tuple of numpy arrays — one cotangent per **dynamic**
            key, in ``_dynamic_keys`` order.
        """
        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Unpack arguments ─────────────────────────────────────────
        # Problem matrices (already merged, no conversion needed)
        prob_np: dict[str, ndarray] = {
            k: np.asarray(args[i], dtype=_dtype)
            for i, k in enumerate(_required_keys)
        }
        off = _n_req
        x_np      = np.asarray(args[off],     dtype=_dtype)
        lam_np    = np.asarray(args[off + 1], dtype=_dtype)
        mu_np     = np.asarray(args[off + 2], dtype=_dtype)
        active_np = np.asarray(args[off + 3], dtype=np.bool_)
        g_x       = np.asarray(args[off + 4], dtype=_dtype)
        g_lam     = np.asarray(args[off + 5], dtype=_dtype)
        g_mu      = np.asarray(args[off + 6], dtype=_dtype)

        # ── Retrieve problem matrices ────────────────────────────────
        P_np: Float[ndarray, "nv nv"] = prob_np["P"]
        A_np: Float[ndarray, "ne nv"] = prob_np.get(
            "A", np.empty((0, n_var), dtype=_dtype)
        )
        G_np: Float[ndarray, "ni nv"] = prob_np.get(
            "G", np.empty((0, n_var), dtype=_dtype)
        )

        # ── Build KKT matrix LHS (symmetric → adjoint = forward) ────
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

        # ── Build RHS from cotangent vectors ─────────────────────────
        rhs_parts: list[Float[ndarray, " _"]] = [g_x]
        if n_eq > 0:
            rhs_parts.append(g_mu)
        if n_ineq > 0 and n_active > 0:
            rhs_parts.append(g_lam[active_np])

        rhs: Float[ndarray, " nv_nh"] = np.hstack(rhs_parts)

        t["build_system"] = perf_counter() - start

        # ── Solve the adjoint system: lhs @ v = rhs ──────────────────
        start = perf_counter()
        v = np.linalg.solve(lhs, rhs)
        t["lin_solve"] = perf_counter() - start

        # ── Extract adjoint variables ────────────────────────────────
        v_x: Float[ndarray, " nv"] = v[:n_var]
        v_mu: Float[ndarray, " ne"] = (
            v[n_var:n_var + n_eq]
            if n_eq > 0
            else np.empty(0, dtype=_dtype)
        )
        v_lam_a: Float[ndarray, " na"] = (
            v[n_var + n_eq:]
            if (n_ineq > 0 and n_active > 0)
            else np.empty(0, dtype=_dtype)
        )

        # ── Compute parameter cotangents ─────────────────────────────
        start = perf_counter()

        grads: dict[str, ndarray] = {}

        grads["P"] = -np.outer(v_x, x_np)
        grads["q"] = -v_x

        if n_eq > 0:
            grads["A"] = -(
                np.outer(mu_np, v_x) + np.outer(v_mu, x_np)
            )
            grads["b"] = v_mu

        if n_ineq > 0:
            g_G = np.zeros_like(G_np)
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

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_vjp | {fmt_times(t)}")
        _timings.record("_kkt_vjp", t)

        # ── Return in _dynamic_keys order ────────────────────────────
        return tuple(grads[k] for k in _dynamic_keys)

    #endregion

    # =================================================================
    # COMBINE INTO A UNIQUE FUNCTION
    # =================================================================

    #region

    # -----------------------------------------------------------------
    # JVP path (differentiator = "kkt_dense")
    # -----------------------------------------------------------------

    @custom_jvp
    def _solver_dynamic_jvp_mode(
        *dynamic_vals: jax.Array,
    ) -> dict[str, jax.Array]:
        """Internal solver (JVP path) with positional-only dynamic args.

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
        """
        return pure_callback(_solve_qp, _fwd_shapes, *dynamic_vals)

    @_solver_dynamic_jvp_mode.defjvp
    def _solver_dynamic_jvp_rule(
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """JVP rule for ``_solver_dynamic_jvp_mode`` via implicit diff.

        Both the forward solve and KKT differentiation happen inside a
        **single** ``pure_callback`` call (``_kkt_diff``). This avoids
        paying the callback dispatch overhead twice. All numpy → JAX
        conversion is handled internally by ``pure_callback`` via
        ``_jvp_shapes``, so no Python-level ``jnp.array`` calls are
        needed.

        Args:
            primals: Primal inputs in ``_dynamic_keys`` order.
            tangents: Tangent inputs with the same structure as
                ``primals``.

        Returns:
            ``(primal_out, tangent_out)`` where both are solution dicts.
        """

        # call forward differentiator
        dx, dlam, dmu, res = pure_callback(
            _diff_forward,
            _jvp_shapes,
            *primals,
            *tangents,
            vmap_method="expand_dims",
        )

        tangents_out: dict[str, jax.Array] = {
            "x":      dx,
            "lam":    dlam,
            "mu":     dmu,
        }

        return res, tangents_out

    # -----------------------------------------------------------------
    # VJP path (differentiator = "kkt_dense_vjp")
    # -----------------------------------------------------------------

    # @custom_vjp
    # def _solver_dynamic_vjp_mode(
    #     *dynamic_vals: jax.Array,
    # ) -> dict[str, jax.Array]:
    #     """Internal solver (VJP path) with positional-only dynamic args.

    #     Identical forward semantics to the JVP path. The difference
    #     is in how derivatives are computed: the VJP path stores the
    #     forward solution as residuals and solves the *adjoint* KKT
    #     system in the backward pass.

    #     Args:
    #         *dynamic_vals: JAX arrays corresponding to
    #             ``_dynamic_keys``, in order.

    #     Returns:
    #         Solution dict with JAX arrays:

    #             - ``x``      (nv,):  Primal solution.
    #             - ``lam``    (ni,):  Inequality duals.
    #             - ``mu``     (ne,):  Equality duals.
    #     """
    #     return pure_callback(_solve_qp, _fwd_shapes, *dynamic_vals)

    # def _solver_dynamic_vjp_fwd(
    #     *dynamic_vals: jax.Array,
    # ) -> tuple[
    #     dict[str, jax.Array],
    #     tuple[jax.Array, ...],
    # ]:
    #     """Forward pass for ``custom_vjp``.

    #     Runs the QP solve via a single ``pure_callback`` that converts
    #     dynamic inputs to numpy **once**, merges with ``_fixed``,
    #     solves the QP, and returns the solution, active set, **and**
    #     all merged problem matrices as a flat tuple.

    #     The merged problem matrices are stored as residuals so that
    #     the backward pass can feed them directly to the adjoint
    #     solve — no second numpy conversion or re-merging is needed.

    #     All residuals are JAX arrays that ``custom_vjp`` treats as
    #     non-differentiable constants.

    #     Args:
    #         *dynamic_vals: Dynamic primal inputs in
    #             ``_dynamic_keys`` order.

    #     Returns:
    #         ``(primal_out, residuals)`` where:

    #             - ``primal_out``: Solution dict (``x``, ``lam``,
    #               ``mu``).
    #             - ``residuals``: Flat tuple
    #               ``(x, lam, mu, active, P, q, [A, b,] [G, h])``.
    #               All JAX arrays, non-differentiable.
    #     """
    #     result = pure_callback(
    #         _solve_qp_vjp_fwd, _vjp_fwd_shapes, *dynamic_vals,
    #     )

    #     # result is a flat tuple: (x, lam, mu, active, prob[0], ...)
    #     x, lam, mu, active = result[:4]
    #     prob_arrays = result[4:]

    #     primal_out: dict[str, jax.Array] = {
    #         "x":   x,
    #         "lam": lam,
    #         "mu":  mu,
    #     }

    #     # Residuals: solution + active + merged problem matrices.
    #     residuals = (x, lam, mu, active, *prob_arrays)

    #     return primal_out, residuals

    # def _solver_dynamic_vjp_bwd(
    #     residuals: tuple[jax.Array, ...],
    #     g: dict[str, jax.Array],
    # ) -> tuple[jax.Array, ...]:
    #     """Backward pass for ``custom_vjp``.

    #     Receives the stored residuals (forward solution + active set
    #     + pre-merged problem matrices) and the output cotangent dict.
    #     Passes the problem matrices, solution, and cotangents through
    #     ``_kkt_vjp`` via ``pure_callback`` to solve the adjoint KKT
    #     system in numpy and return parameter cotangents.

    #     Because the problem matrices were already merged with
    #     ``_fixed`` during the forward pass, no re-merging or second
    #     numpy conversion of dynamic inputs is needed.

    #     Args:
    #         residuals: ``(x, lam, mu, active, P, q, [A, b,] [G, h])``
    #             from the forward pass. All JAX arrays.
    #         g: Cotangent dict with keys ``x``, ``lam``, ``mu``.

    #     Returns:
    #         A tuple of cotangents, one per dynamic input, in
    #         ``_dynamic_keys`` order.
    #     """
    #     x, lam, mu, active = residuals[:4]
    #     prob_arrays = residuals[4:]

    #     g_x   = g["x"]
    #     g_lam = g["lam"]
    #     g_mu  = g["mu"]

    #     # Layout: *prob_arrays, x, lam, mu, active, g_x, g_lam, g_mu
    #     grad_vals: tuple[jax.Array, ...] = pure_callback(
    #         _kkt_vjp,
    #         _vjp_bwd_shapes,
    #         *prob_arrays, x, lam, mu, active, g_x, g_lam, g_mu,
    #     )

    #     return grad_vals

    # _solver_dynamic_vjp_mode.defvjp(
    #     _solver_dynamic_vjp_fwd,
    #     _solver_dynamic_vjp_bwd,
    # )

    # -----------------------------------------------------------------
    # Select differentiator
    # -----------------------------------------------------------------

    _diff_name = _options_parsed["differentiator_type"]

    if _diff_name == "kkt_fwd":
        _solver_dynamic = _solver_dynamic_jvp_mode
    elif _diff_name == "kkt_bwd":
        _solver_dynamic = _solver_dynamic_vjp_mode
    else:
        raise ValueError(
            f"Unknown differentiator: {_diff_name!r}. "
            f"Supported: 'kkt_fwd' (JVP), 'kkt_bwd' (VJP)."
        )

    logger.info(f"Differentiator: {_diff_name}")

    # -----------------------------------------------------------------
    # Public solver callable
    # -----------------------------------------------------------------

    def solver(
        *,
        warmstart: Optional[jax.Array] = None,
        **runtime: jax.Array,
    ) -> dict[str, jax.Array]:
        """Solve a (possibly partially fixed) QP problem.

        The full set of QP ingredients is ``{P, q, A, b, G, h}``. Any
        subset may be fixed at setup time via ``fixed_elements``; the
        remainder must be supplied here at call time.

        Fixed elements are stored as numpy arrays and never enter
        JAX's trace. This avoids unnecessary conversions but means
        their tangents are always zero. To differentiate through an
        element, supply it at runtime instead of fixing it at setup.

        The differentiation mode (JVP or VJP) is selected at setup
        time via the ``differentiator`` option:

            - ``"kkt_dense"``     — forward-mode (JVP), compatible
              with ``jax.jvp`` and ``jax.vmap(jax.jvp(...))``.
            - ``"kkt_dense_vjp"`` — reverse-mode (VJP), compatible
              with ``jax.grad`` and ``jax.value_and_grad``.

        Args:
            warmstart: Optional initial guess for the primal solution,
                shape ``(n_var,)``. Passed to the underlying QP solver
                to speed up convergence. Not differentiated through.
                Cleared after each solve.
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

        Raises:
            ValueError: If any dynamic ingredient is missing.
        """
        # Store warmstart in the mutable closure slot (converted to
        # numpy). _solve_qp_numpy reads it and clears it after use.
        if warmstart is not None:
            _warmstart[0] = np.asarray(warmstart, dtype=_dtype).reshape(-1)
        else:
            _warmstart[0] = None

        # Warn if user passes keys that are already fixed
        if _options_parsed["debug"]:
            overridden = set(runtime.keys()) & _fixed_keys_set
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

    solver.timings = _timings  # type: ignore[attr-defined]

    return solver