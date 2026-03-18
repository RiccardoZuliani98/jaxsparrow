from jax import custom_jvp, custom_vjp, pure_callback, Array
import jax.numpy as jnp
from time import perf_counter
import numpy as np
import jax
import logging
from typing import Optional, cast
from numpy import ndarray

from src.utils.parsing_utils import parse_options
from src.utils.printing_utils import fmt_times
from src.utils.timing_utils import TimingRecorder
from src.solver_dense.solver_dense_options import (
    DEFAULT_CONSTRUCTOR_OPTIONS, 
    ConstructorOptions,
    ConstructorOptions
)
from src.solver_dense.solvers import create_dense_qp_solver
from src.solver_dense.differentiators import (
    create_dense_kkt_differentiator_fwd, 
    create_dense_kkt_differentiator_rev
)
from src.solver_dense.solver_dense_types import (
    DenseQPIngredientsNP, 
    QPDiffOut, 
    QPOutput, 
    DenseQPIngredientsTangentsNP
)

#TODO: create sparse solver, I left some TODOs in this file pointing at
# where changes should be made
#TODO: add a finite difference utility similar in principle to TimingRecorder.
# this should only use the numpy sub-solver and therefore not contribute to 
# the overall timings
#TODO: add value function and envelope theorem?
#TODO: fix test for vmapped vjp unconstrained => what is going wrong?

# Expected ndim for each QP ingredient (unbatched).
# Used to detect batching in the JVP path.
_EXPECTED_NDIM: dict[str, int] = {
    "P": 2, "q": 1, "A": 2, "b": 1, "G": 2, "h": 1,
}


def setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: Optional[DenseQPIngredientsNP] = None,
    options: Optional[ConstructorOptions] = None,
):

    # start logging
    logger = logging.getLogger(__name__)

    # timing recorder — collects structured timing dicts across calls.
    # Accessible after setup as ``solver.timings``.
    _timings = TimingRecorder()

    # parse user options
    _options_parsed = parse_options(options, DEFAULT_CONSTRUCTOR_OPTIONS)

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
    _jvp_shapes = ({
            "x":    jax.ShapeDtypeStruct((n_var,),  _dtype),    # dx
            "lam":  jax.ShapeDtypeStruct((n_ineq,), _dtype),    # dlam
            "mu":   jax.ShapeDtypeStruct((n_eq,),   _dtype),    # dmu
        },
        _fwd_shapes,                                            # sol
    )

    # gather all required keys
    _required_keys: tuple[str,...] = ("P", "q")
    if n_eq > 0:
        _required_keys += ("A", "b")
    if n_ineq > 0:
        _required_keys += ("G", "h")

    # gather fixed keys
    _fixed_keys_set : set[str] = set(fixed_elements.keys()) if fixed_elements is not None else set()

    # gather keys required at runtime (i.e., not fixed).
    # Only these flow through JAX's traced path.
    _dynamic_keys = tuple(k for k in _required_keys if k not in _fixed_keys_set)

    # transform to sets for speed
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

    # Shapes for the VJP backward callback: one cotangent per dynamic key,
    # matching the shape of the corresponding primal input.
    _vjp_bwd_shapes = {
        k: jax.ShapeDtypeStruct(_expected_shapes[k], _dtype)
        for k in _dynamic_keys
    }

    # Mutable warmstart slot. Written by solver() before each call,
    # read by _solve_qp_numpy() inside the callback, cleared after use.
    # Stored as a one-element list so nested functions can mutate it.
    _warmstart: list[Optional[ndarray]] = [None]

    # check shape of fixed_elements if present
    if fixed_elements is not None:
        for key,val in fixed_elements.items():
            assert val.shape == _expected_shapes[key] #type:ignore

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
    if _options_parsed["differentiator_type"] in ("kkt_fwd", "kkt_rev"):
        _diff_forward_numpy = create_dense_kkt_differentiator_fwd(
            n_var=n_var,
            n_eq=n_eq,
            n_ineq=n_ineq,
            options=_options_parsed["differentiator"],
            fixed_elements=fixed_elements
        )
        _diff_reverse_numpy = create_dense_kkt_differentiator_rev(
            n_var=n_var,
            n_eq=n_eq,
            n_ineq=n_ineq,
            options=_options_parsed["differentiator"],
            fixed_elements=fixed_elements
        )
    else:
        raise ValueError("Only differentiator available is 'kkt'")

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
    def _solve_qp(*dynamic_vals: jax.Array) -> QPOutput:

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
            prob_np["warmstart"] = _warmstart[0]

        # ── Solve in numpy ───────────────────────────────────────────
        sol_np, t_solve = _solve_qp_numpy(**prob_np)
        x_np, lam_np, mu_np, _ = sol_np

        # Clear warmstart after use so it doesn't persist
        _warmstart[0] = None

        # store time
        t.update(t_solve)

        # ── Return jax arrays directly ────────────────────────────────
        # pure_callback handles numpy → JAX conversion using _fwd_shapes.
        start = perf_counter()
        result = cast(QPOutput, {
            "x":      jnp.array(x_np,dtype=_dtype),
            "lam":    jnp.array(lam_np,dtype=_dtype),
            "mu":     jnp.array(mu_np,dtype=_dtype),
        })
        t["convert_to_jax"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_solve_qp | {fmt_times(t)}")
        _timings.record("_solve_qp", t)

        return result
    #endregion

    # =================================================================
    # FORWARD MODE DIFFERENTIATION
    # =================================================================

    #region
    def _diff_forward(*args: ndarray) -> tuple[QPDiffOut, QPOutput]:

        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Split positional args into dynamic primals and tangents ──
        dyn_primal_vals = args[:_n_dyn]
        dyn_tangent_vals = args[_n_dyn:]

        # ── Convert dynamic primals to numpy, squeeze batch-1 dims ───
        start = perf_counter()
        dyn_primals_np: DenseQPIngredientsNP = {}
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
        prob_np = cast(dict[str,ndarray], dyn_primals_np.copy())
        if _warmstart[0] is not None: 
            prob_np["warmstart"] = _warmstart[0]

        # ── Solve in numpy ───────────────────────────────────────────
        sol_np, t_solve = _solve_qp_numpy(**prob_np)
        x_np, lam_np, mu_np, _ = sol_np

        # Clear warmstart after use so it doesn't persist
        _warmstart[0] = None

        # store time
        t.update(t_solve)

        # ── Convert dynamic tangents to numpy (keep batch dim) ───────
        start = perf_counter()
        dyn_tangents_np = cast(DenseQPIngredientsTangentsNP, {
            #TODO: here for sparse too
            k: np.asarray(v, dtype=_dtype)
            for k, v in zip(_dynamic_keys, dyn_tangent_vals)
        })
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
        batch_size : int = max(v.shape[0] for v in dyn_tangents_np.values()) if batched else 0 # type: ignore[union-attr]


        # ── Call differentiator ──────────────────────────────────────
        diff_out_np, t_diff = _diff_forward_numpy(
            sol_np=sol_np,
            dyn_primals_np=dyn_primals_np,
            dyn_tangents_np=dyn_tangents_np,
            batch_size=batch_size
        )
        dx_np, dlam_np, dmu_np = diff_out_np
        t.update(t_diff)

        # ── Build solution dict (numpy only) ─────────────────────────
        # np.broadcast_to returns a read-only view (zero cost).
        # .copy() materializes a contiguous array for pure_callback.
        start = perf_counter()
        if batch_size > 0:
            res : QPOutput ={
                "x":      jnp.array(np.broadcast_to(x_np, (batch_size, n_var)).copy(),dtype=_dtype),
                "lam":    jnp.array(np.broadcast_to(lam_np, (batch_size, n_ineq)).copy(),dtype=_dtype),
                "mu":     jnp.array(np.broadcast_to(mu_np, (batch_size, n_eq)).copy(),dtype=_dtype),
            }
        else:
            res : QPOutput = {
                "x":      jnp.array(x_np,dtype=_dtype),
                "lam":    jnp.array(lam_np,dtype=_dtype),
                "mu":     jnp.array(mu_np,dtype=_dtype),
            }
        
        diff_out : QPDiffOut = {
            "x": jnp.array(dx_np, dtype=_dtype),
            "lam": jnp.array(dlam_np, dtype=_dtype),
            "mu": jnp.array(dmu_np, dtype=_dtype)
        }
        
        t["build_sol"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_diff | {fmt_times(t)}")
        _timings.record("_kkt_diff", t)

        return diff_out, res
    #endregion

    # =================================================================
    # REVERSE MODE DIFFERENTIATION
    # =================================================================

    #region
    def _diff_reverse(*args: ndarray) -> dict[str, Array]:

        t_start = perf_counter()
        t: dict[str, float] = {}


        # ── Unpack arguments ─────────────────────────────────────────
        start = perf_counter()

        # Problem matrices (only dynamic entries)
        prob_np = cast(DenseQPIngredientsNP,{
            #TODO: update this for sparse mode
            k: np.asarray(args[i], dtype=_dtype)
            for i, k in enumerate(_dynamic_keys)
        })

        for k in prob_np:
            while prob_np[k].ndim > _EXPECTED_NDIM[k]:
                prob_np[k] = prob_np[k][0]

        off = _n_dyn
        x_np      = np.asarray(args[off],     dtype=_dtype)
        lam_np    = np.asarray(args[off + 1], dtype=_dtype)
        mu_np     = np.asarray(args[off + 2], dtype=_dtype)

        # Squeeze solution vectors too (expected: 1-D)
        while x_np.ndim > 1:   x_np   = x_np[0]
        while lam_np.ndim > 1:  lam_np = lam_np[0]
        while mu_np.ndim > 1:   mu_np  = mu_np[0]

        g_x       = np.asarray(args[off + 3], dtype=_dtype)
        g_lam     = np.asarray(args[off + 4], dtype=_dtype)
        g_mu      = np.asarray(args[off + 5], dtype=_dtype)

        t["convert_to_numpy"] = perf_counter() - start


        # ── Detect batching ──────────────────────────────────────────
        # Batching is detected by checking if the cotangents have an extra dimension
        batched = (g_x.ndim == 2) if _n_dyn > 0 else False

        # get batch size
        batch_size = max([g_x.shape[0],g_lam.shape[0],g_mu.shape[0]]) if batched else 0 


        # ── Call differentiator ──────────────────────────────────────
        grads_np, t_diff = _diff_reverse_numpy(
            dyn_primals_np=prob_np,
            x_np=x_np,
            lam_np=lam_np,
            mu_np=mu_np,
            g_x=g_x,
            g_lam=g_lam,
            g_mu=g_mu,
            batch_size=batch_size
        )
        t.update(t_diff)


        # ── Build solution dict ──────────────────────────────────────

        # convert solution back to jax
        #TODO: this should change in sparse mode
        grads = {k: jnp.asarray(grads_np[k], dtype=_dtype) for k in _dynamic_keys}

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_vjp | {fmt_times(t)}")
        _timings.record("_kkt_vjp", t)

        return grads

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
    ) -> QPOutput:
        return pure_callback(_solve_qp, _fwd_shapes, *dynamic_vals)

    @_solver_dynamic_jvp_mode.defjvp
    def _solver_dynamic_jvp_rule(
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
    ) -> tuple[QPOutput, QPDiffOut]:

        # call forward differentiator
        tangents_out, res = pure_callback(
            _diff_forward,
            _jvp_shapes,
            *primals,
            *tangents,
            vmap_method="expand_dims",
        )

        return res, tangents_out

    # -----------------------------------------------------------------
    # VJP path (differentiator = "kkt_dense_vjp")
    # -----------------------------------------------------------------

    @custom_vjp
    def _solver_dynamic_vjp_mode(
        *dynamic_vals: jax.Array,
    ) -> QPOutput:

        return pure_callback(
            _solve_qp, 
            _fwd_shapes, 
            *dynamic_vals, 
            # vmap_method="broadcast_all"
        )

    def _solver_dynamic_vjp_fwd(
        *dynamic_vals: jax.Array,
    ) -> tuple[QPOutput,tuple[jax.Array, ...],
    ]:

        result = pure_callback(
            _solve_qp, 
            _fwd_shapes, 
            *dynamic_vals,
            # vmap_method="broadcast_all"
        )

        # result is a flat tuple: (x, lam, mu, active, prob[0], ...)
        x = result["x"]
        lam = result["lam"]
        mu = result["mu"]

        # Residuals: solution + active + merged problem matrices.
        residuals = (*dynamic_vals, x, lam, mu)

        return result, residuals

    def _solver_dynamic_vjp_bwd(
        residuals: tuple[jax.Array, ...],
        g: dict[str, jax.Array],
    ) -> tuple[jax.Array, ...]:

        # unpack cotangents
        g_x   = g["x"]
        g_lam = g["lam"]
        g_mu  = g["mu"]

        print(f"Shapes of cotangent: {g_x.shape}, {g_lam.shape}, {g_mu.shape}")

        # Layout: *prob_arrays, x, lam, mu, active, g_x, g_lam, g_mu
        grad_vals = pure_callback(
            _diff_reverse,
            _vjp_bwd_shapes,
            *residuals, g_x, g_lam, g_mu,
            vmap_method="broadcast_all"
        )

        return tuple(grad_vals[k] for k in _dynamic_keys)

    _solver_dynamic_vjp_mode.defvjp(
        _solver_dynamic_vjp_fwd,
        _solver_dynamic_vjp_bwd,
    )

    # =================================================================
    # SOLVER FUNCTION TO BE RETURNED
    # =================================================================

    _diff_name = _options_parsed["differentiator_type"]

    if _diff_name == "kkt_fwd":
        _solver_dynamic = _solver_dynamic_jvp_mode
    elif _diff_name == "kkt_rev":
        _solver_dynamic = _solver_dynamic_vjp_mode
    else:
        raise ValueError(
            f"Unknown differentiator: {_diff_name!r}. "
            f"Supported: 'kkt_fwd' (JVP), 'kkt_rev' (VJP)."
        )

    logger.info(f"Differentiator: {_diff_name}")

    def solver(
        *,
        warmstart: Optional[jax.Array] = None,
        **runtime: jax.Array,
    ) -> QPOutput:

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

    solver.timings = _timings

    return solver