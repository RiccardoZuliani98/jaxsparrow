"""
solver_common.py
================
Shared scaffolding for dense and sparse differentiable solvers.

This module contains all format-agnostic logic:
  - JAX custom_jvp / custom_vjp registration and pure_callback wiring
  - Batch detection
  - Timing and finite-difference bookkeeping
  - Warmstart management
  - The public ``solver()`` closure

Dense and sparse entry-points call :func:`build_solver` and supply:
  1. A *converter* that turns JAX arrays into the numpy representation
     expected by the underlying solver (dense ndarray or scipy CSC).
  2. Pre-built solver / differentiator callables.
  3. The ``_vjp_bwd_shapes`` dict (because the sparse path returns
     gradients w.r.t. nonzero values, not full matrices).
"""

from __future__ import annotations

from jax import custom_jvp, custom_vjp, pure_callback, Array
import jax
import jax.numpy as jnp
import numpy as np
import logging
from time import perf_counter
from typing import Optional, cast, Callable, Any, Union
from numpy import ndarray

from jaxsparrow._utils._printing_utils import fmt_times
from jaxsparrow._utils._timing_utils import TimingRecorder
from jaxsparrow._types_common import (
    SolverDiffOutFwd,
    SolverDiffOutRev,
    SolverOutput
)
from jaxsparrow._options_common import ConstructorOptionsFull
from jaxsparrow._utils._fd_recorder import FiniteDifferenceRecorder
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix


SolverInputNP = Union[ndarray, csc_matrix]
SolverInput = Union[jax.Array, BCOO]


# ── Constants ────────────────────────────────────────────────────────
 
# Expected ndim for each ingredient (unbatched).
# Used to detect batching in the JVP / VJP paths.
EXPECTED_NDIM: dict[str, int] = {
    "P": 2, "q": 1, "A": 2, "b": 1, "G": 2, "h": 1,
}
 
# Expected shapes for each ingredient (unbatched), given problem dims.
def make_expected_shapes(
    n_var: int, n_eq: int, n_ineq: int
) -> dict[str, tuple[int, ...]]:
    return {
        "P": (n_var, n_var),
        "q": (n_var,),
        "A": (n_eq, n_var),
        "b": (n_eq,),
        "G": (n_ineq, n_var),
        "h": (n_ineq,),
    }
 
 
# ── Key partitioning helpers ─────────────────────────────────────────
 
def compute_required_keys(
    n_eq: int, n_ineq: int
) -> tuple[str, ...]:
    """Return the tuple of ingredient names required by the problem."""
    keys: tuple[str, ...] = ("P", "q")
    if n_eq > 0:
        keys += ("A", "b")
    if n_ineq > 0:
        keys += ("G", "h")
    return keys
 
 
def compute_dynamic_keys(
    required_keys: tuple[str, ...],
    fixed_keys: set[str],
) -> tuple[str, ...]:
    """Keys that are *not* fixed and must flow through JAX's traced path."""
    return tuple(k for k in required_keys if k not in fixed_keys)

# ── Converter protocol ───────────────────────────────────────────────
#
# Dense and sparse callers each supply a *converter* with the following
# signature.  It receives a JAX array (or BCOO) and a key name, and
# returns the numpy representation expected by the solver.
#
#   converter(key: str, jax_val: jax.Array, dtype) -> ndarray | csc_matrix
#
# The common code never inspects the returned object — it just passes
# it through to the solver / differentiator callables.

PrimalConverter = Callable[[str, Any, Any], Any]
"""(key, jax_value, dtype) -> numpy representation."""

TangentConverter = Callable[[str, Any, Any], Any]
"""(key, jax_value, dtype) -> numpy tangent representation."""

GradToJaxConverter = Callable[[str, Any, Any], Array]
"""(key, numpy_grad, dtype) -> JAX array for backward pass."""


# ── Builder ──────────────────────────────────────────────────────────

def build_solver(
    *,
    # Problem dimensions
    n_var: int,
    n_ineq: int,
    n_eq: int,
    # Parsed options (already merged with defaults)
    options_parsed: ConstructorOptionsFull,
    # Fixed elements (dense ndarray or sparse — opaque to us)
    fixed_keys_set: set[str],
    # Pre-built numpy solver & differentiators
    solver_numpy: Callable[..., Any],
    diff_forward_numpy: Callable[..., Any],
    diff_reverse_numpy: Callable[..., Any],
    # Converters
    primal_converter: PrimalConverter,
    tangent_converter: TangentConverter,
    grad_to_jax: GradToJaxConverter,
    # VJP backward shapes — caller builds these because sparse differs
    vjp_bwd_shapes: dict[str, jax.ShapeDtypeStruct] | None = None,
    # Finite-difference recorder (caller creates it)
    fd_check: FiniteDifferenceRecorder | None = None,
):
    """
    Wire up JAX custom differentiation rules around a numpy solver.

    Returns the public ``solver`` callable (with ``.timings`` and
    ``.fd_check`` attributes).
    """

    logger = logging.getLogger(__name__)
    _dtype = options_parsed["dtype"]

    # ── Timing recorder ──────────────────────────────────────────────
    _timings = TimingRecorder()

    # ── FD recorder (use caller's or create a default) ───────────────
    _fd_check = fd_check or FiniteDifferenceRecorder(
        enabled=options_parsed["fd_check"],
        eps=options_parsed["fd_eps"]
    )

    # ── Key partitioning ─────────────────────────────────────────────
    _required_keys = compute_required_keys(n_eq, n_ineq)
    _dynamic_keys = compute_dynamic_keys(_required_keys, fixed_keys_set)
    _dynamic_keys_set = set(_dynamic_keys)
    _n_dyn = len(_dynamic_keys)
    _expected_shapes = make_expected_shapes(n_var, n_eq, n_ineq)

    # ── JAX shape structs for pure_callback ──────────────────────────
    _fwd_shapes = {
        "x":   jax.ShapeDtypeStruct((n_var,),  _dtype),
        "lam": jax.ShapeDtypeStruct((n_ineq,), _dtype),
        "mu":  jax.ShapeDtypeStruct((n_eq,),   _dtype),
    }

    _jvp_shapes = (
        {
            "x":   jax.ShapeDtypeStruct((n_var,),  _dtype),
            "lam": jax.ShapeDtypeStruct((n_ineq,), _dtype),
            "mu":  jax.ShapeDtypeStruct((n_eq,),   _dtype),
        },
        _fwd_shapes,
    )

    # VJP backward shapes — if not supplied, default to dense.
    _vjp_bwd_shapes = vjp_bwd_shapes or {
        k: jax.ShapeDtypeStruct(_expected_shapes[k], _dtype)
        for k in _dynamic_keys
    }

    # ── Warmstart slot ───────────────────────────────────────────────
    _warmstart: list[Optional[ndarray]] = [None]

    # ── Aliases for readability ──────────────────────────────────────
    _solver_numpy = solver_numpy
    _diff_forward_numpy = diff_forward_numpy
    _diff_reverse_numpy = diff_reverse_numpy
    _primal_conv = primal_converter
    _tangent_conv = tangent_converter
    _grad_to_jax = grad_to_jax

    # =================================================================
    # BASE SOLVER CALLBACK
    # =================================================================

    def _solver(*dynamic_vals: jax.Array) -> SolverOutput:

        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Convert dynamic JAX arrays to numpy ──────────────────────
        start = perf_counter()
        dynamic_np: dict[str, Any] = {
            k: _primal_conv(k, v, _dtype)
            for k, v in zip(_dynamic_keys, dynamic_vals)
        }
        t["convert_to_numpy"] = perf_counter() - start

        # ── Debug checks ─────────────────────────────────────────────
        if options_parsed["debug"]:
            missing = _dynamic_keys_set - set(dynamic_np)
            if missing:
                raise ValueError(
                    f"Missing ingredients: {sorted(missing)}. "
                    f"Provide them either via fixed_elements at setup "
                    f"or at call time."
                )

        # ── Warmstart ────────────────────────────────────────────────
        prob_np = dynamic_np
        if _warmstart[0] is not None:
            prob_np["warmstart"] = _warmstart[0]

        # ── Solve ────────────────────────────────────────────────────
        sol_np, t_solve = _solver_numpy(**prob_np)
        x_np, lam_np, mu_np = sol_np
        t.update(t_solve)

        # ── Return JAX arrays ────────────────────────────────────────
        start = perf_counter()
        result = cast(SolverOutput, {
            "x":   jnp.array(x_np, dtype=_dtype),
            "lam": jnp.array(lam_np, dtype=_dtype),
            "mu":  jnp.array(mu_np, dtype=_dtype),
        })
        t["convert_to_jax"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_solver | {fmt_times(t)}")
        _timings.record("_solver", t)

        return result

    # =================================================================
    # FORWARD DIFFERENTIATION CALLBACK
    # =================================================================

    def _diff_forward(*args) -> tuple[SolverDiffOutFwd, SolverOutput]:

        t_start = perf_counter()
        t: dict[str, float] = {}

        dyn_primal_vals = args[:_n_dyn]
        dyn_tangent_vals = args[_n_dyn:]

        # ── Convert primals ──────────────────────────────────────────
        start = perf_counter()
        dyn_primals_np: dict[str, Any] = {
            k: _primal_conv(k, v, _dtype)
            for k, v in zip(_dynamic_keys, dyn_primal_vals)
        }
        t["convert_to_numpy"] = perf_counter() - start

        # ── Debug checks ─────────────────────────────────────────────
        if options_parsed["debug"]:
            missing = _dynamic_keys_set - set(dyn_primals_np)
            if missing:
                raise ValueError(
                    f"Missing ingredients: {sorted(missing)}."
                )

        # ── Warmstart ────────────────────────────────────────────────
        prob_np = dict(dyn_primals_np)
        if _warmstart[0] is not None:
            prob_np["warmstart"] = _warmstart[0]

        # ── Solve ────────────────────────────────────────────────────
        sol_np, t_solve = _solver_numpy(**prob_np)
        x_np, lam_np, mu_np = sol_np
        _warmstart[0] = None
        t.update(t_solve)

        # ── Convert tangents ─────────────────────────────────────────
        
        start = perf_counter()
        dyn_tangents_np: dict[str, Any] = {
            k: _tangent_conv(k, v, _dtype)
            for k, v in zip(_dynamic_keys, dyn_tangent_vals)
        }
        t["convert_tangents"] = perf_counter() - start

        # ── Detect batching ──────────────────────────────────────────
        if _n_dyn > 0:
            batch_sizes = []
            for k, v in zip(_dynamic_keys, dyn_tangent_vals):
                expected = EXPECTED_NDIM[k]
                if v.ndim > expected:
                    batch_sizes.append(v.shape[0])
            batched = len(batch_sizes) > 0
            batch_size = max(batch_sizes) if batched else 0
        else:
            batched = False
            batch_size = 0

        # ── Differentiate ────────────────────────────────────────────
        diff_out_np, t_diff = _diff_forward_numpy(
            sol_np=sol_np,
            dyn_primals_np=dyn_primals_np,
            dyn_tangents_np=dyn_tangents_np,
            batch_size=batch_size,
        )
        dx_np, dlam_np, dmu_np = diff_out_np
        t.update(t_diff)

        # ── Build JAX results ────────────────────────────────────────
        start = perf_counter()
        if batch_size > 0:
            res: SolverOutput = {
                "x":   jnp.array(np.broadcast_to(x_np, (batch_size, n_var)).copy(), dtype=_dtype),
                "lam": jnp.array(np.broadcast_to(lam_np, (batch_size, n_ineq)).copy(), dtype=_dtype),
                "mu":  jnp.array(np.broadcast_to(mu_np, (batch_size, n_eq)).copy(), dtype=_dtype),
            }
        else:
            res: SolverOutput = {
                "x":   jnp.array(x_np, dtype=_dtype),
                "lam": jnp.array(lam_np, dtype=_dtype),
                "mu":  jnp.array(mu_np, dtype=_dtype),
            }

        diff_out: SolverDiffOutFwd = {
            "x":   jnp.array(dx_np, dtype=_dtype),
            "lam": jnp.array(dlam_np, dtype=_dtype),
            "mu":  jnp.array(dmu_np, dtype=_dtype),
        }
        t["build_sol"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_diff | {fmt_times(t)}")
        _timings.record("_kkt_diff", t)

        # ── FD check ─────────────────────────────────────────────────
        if options_parsed["fd_check"]:
            _run_fd_check_jvp(
                dyn_primals_np, dyn_tangents_np,
                dx_np, dlam_np, dmu_np, batched,
            )

        return diff_out, res

    # =================================================================
    # REVERSE DIFFERENTIATION CALLBACK
    # =================================================================

    def _diff_reverse(*args) -> SolverDiffOutRev:

        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Unpack ───────────────────────────────────────────────────
        start = perf_counter()
        prob_np: dict[str, Any] = {
            k: _primal_conv(k, args[i], _dtype)
            for i, k in enumerate(_dynamic_keys)
        }

        off = _n_dyn
        x_np   = np.asarray(args[off],     dtype=_dtype)
        lam_np = np.asarray(args[off + 1], dtype=_dtype)
        mu_np  = np.asarray(args[off + 2], dtype=_dtype)

        # Squeeze solution vectors (expected: 1-D)
        while x_np.ndim > 1:   x_np   = x_np[0]
        while lam_np.ndim > 1: lam_np = lam_np[0]
        while mu_np.ndim > 1:  mu_np  = mu_np[0]

        g_x   = np.asarray(args[off + 3], dtype=_dtype)
        g_lam = np.asarray(args[off + 4], dtype=_dtype)
        g_mu  = np.asarray(args[off + 5], dtype=_dtype)
        t["convert_to_numpy"] = perf_counter() - start

        # ── Detect batching ──────────────────────────────────────────
        batched = (g_x.ndim == 2) if _n_dyn > 0 else False
        batch_size = (
            max(g_x.shape[0], g_lam.shape[0], g_mu.shape[0])
            if batched else 0
        )

        # ── Differentiate ────────────────────────────────────────────
        start = perf_counter()
        grads_np, t_diff = _diff_reverse_numpy(
            dyn_primals_np=prob_np,
            x_np=x_np, lam_np=lam_np, mu_np=mu_np,
            g_x=g_x, g_lam=g_lam, g_mu=g_mu,
            batch_size=batch_size,
        )
        t.update(t_diff)
        t["total_numpy_operations"] = perf_counter()-start

        # ── Convert back to JAX ──────────────────────────────────────
        start = perf_counter()
        grads = {
            k: _grad_to_jax(k, grads_np[k], _dtype)
            for k in _dynamic_keys
        }
        t["convert_to_jax"] = perf_counter()-start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_vjp | {fmt_times(t)}")
        _timings.record("_kkt_vjp", t)

        # ── FD check ─────────────────────────────────────────────────
        if options_parsed["fd_check"]:
            _run_fd_check_vjp(
                prob_np, grads_np,
                g_x, g_lam, g_mu, batched,
            )

        return cast(SolverDiffOutRev, grads)

    # =================================================================
    # FD CHECK HELPERS
    # =================================================================

    def _run_fd_check_jvp(
        dyn_primals_np, dyn_tangents_np,
        dx_np, dlam_np, dmu_np, batched,
    ):
        if not batched:
            _fd_check.check_jvp(
                solve_fn=_solver_numpy,
                dyn_primals_np=dyn_primals_np,
                dyn_tangents_np=dyn_tangents_np,
                dx_analytic=dx_np,
                dlam_analytic=dlam_np,
                dmu_analytic=dmu_np,
                dynamic_keys=_dynamic_keys,
            )
        else:
            first_tangents = {k: v[0] for k, v in dyn_tangents_np.items()}
            _fd_check.check_jvp(
                solve_fn=_solver_numpy,
                dyn_primals_np=dyn_primals_np,
                dyn_tangents_np=first_tangents,
                dx_analytic=dx_np[0] if dx_np.ndim > 1 else dx_np,
                dlam_analytic=dlam_np[0] if dlam_np.ndim > 1 else dlam_np,
                dmu_analytic=dmu_np[0] if dmu_np.ndim > 1 else dmu_np,
                dynamic_keys=_dynamic_keys,
            )

    def _run_fd_check_vjp(
        prob_np, grads_np,
        g_x, g_lam, g_mu, batched,
    ):
        if not batched:
            _fd_check.check_vjp(
                solve_fn=_solver_numpy,
                dyn_primals_np=prob_np,
                grads_analytic=grads_np,
                g_x=g_x, g_lam=g_lam, g_mu=g_mu,
                dynamic_keys=_dynamic_keys,
            )
        else:
            _fd_check.check_vjp(
                solve_fn=_solver_numpy,
                dyn_primals_np=prob_np,
                grads_analytic={k: v[0] for k, v in grads_np.items()},
                g_x=g_x[0], g_lam=g_lam[0], g_mu=g_mu[0],
                dynamic_keys=_dynamic_keys,
            )

    # =================================================================
    # JAX CUSTOM JVP / VJP REGISTRATION
    # =================================================================

    # ── JVP path ─────────────────────────────────────────────────────

    @custom_jvp
    def _solver_dynamic_jvp_mode(
        *dynamic_vals: SolverInput,
    ) -> SolverOutput:
        return pure_callback(_solver, _fwd_shapes, *dynamic_vals)

    @_solver_dynamic_jvp_mode.defjvp
    def _solver_dynamic_jvp_rule(
        primals: tuple[SolverInput, ...],
        tangents: tuple[SolverInput, ...],
    ) -> tuple[SolverOutput, SolverDiffOutFwd]:
        tangents_out, res = pure_callback(
            _diff_forward,
            _jvp_shapes,
            *primals,
            *tangents,
            vmap_method="expand_dims",
        )
        return res, tangents_out

    # ── VJP path ─────────────────────────────────────────────────────
    @custom_vjp
    def _solver_dynamic_vjp_mode(
        *dynamic_vals: SolverInput,
    ) -> SolverOutput:
        return pure_callback(
            _solver, _fwd_shapes, *dynamic_vals,
        )

    def _solver_dynamic_vjp_fwd(
        *dynamic_vals: SolverInput,
    ) -> tuple[SolverOutput, tuple[SolverInput, ...]]:
        result = pure_callback(
            _solver, _fwd_shapes, *dynamic_vals,
        )
        x = result["x"]
        lam = result["lam"]
        mu = result["mu"]
        residuals = (*dynamic_vals, x, lam, mu)
        return result, residuals
    
    def _solver_dynamic_vjp_bwd(
        residuals: tuple[SolverInput, ...],
        g: dict[str, jax.Array],
    ) -> tuple[SolverInput, ...]:
        g_x   = g["x"]
        g_lam = g["lam"]
        g_mu  = g["mu"]
        grad_vals = pure_callback(
            _diff_reverse,
            _vjp_bwd_shapes,
            *residuals, g_x, g_lam, g_mu,
            vmap_method="expand_dims",
        )
        return tuple(grad_vals[k] for k in _dynamic_keys)

    _solver_dynamic_vjp_mode.defvjp(
        _solver_dynamic_vjp_fwd,
        _solver_dynamic_vjp_bwd,
    )

    # =================================================================
    # SELECT DIFFERENTIATOR MODE
    # =================================================================

    _diff_name = options_parsed["diff_mode"]
    if _diff_name == "fwd":
        _solver_dynamic = _solver_dynamic_jvp_mode
    elif _diff_name == "rev":
        _solver_dynamic = _solver_dynamic_vjp_mode
    else:
        raise ValueError(
            f"Unknown differentiator: {_diff_name!r}. "
            f"Supported: 'fwd' (JVP), 'rev' (VJP)."
        )

    logger.info(f"Differentiator: {_diff_name}")

    # =================================================================
    # PUBLIC SOLVER CLOSURE
    # =================================================================

    def solver(
        warmstart: Optional[jax.Array] = None,
        **runtime: SolverInput,
    ) -> SolverOutput:
        """
        Solve a convex problem and return (x, lam, mu).

        Parameters
        ----------
        warmstart : optional JAX array
            Initial guess for the primal variable *x*.
        **runtime : SolverInput
            Dynamic ingredients (those not fixed at setup).
        """

        # Store warmstart in mutable closure slot
        if warmstart is not None:
            _warmstart[0] = np.asarray(warmstart, dtype=_dtype).reshape(-1)
        else:
            _warmstart[0] = None

        # Debug warnings
        if options_parsed["debug"]:
            overridden = set(runtime.keys()) & fixed_keys_set
            if overridden:
                logger.warning(
                    f"Ignoring runtime values for fixed keys: "
                    f"{sorted(overridden)}. These were fixed at setup and "
                    f"are not part of the traced path."
                )
            missing = _dynamic_keys_set - set(runtime)
            if missing:
                raise ValueError(
                    f"Missing ingredients: {sorted(missing)}. "
                    f"Provide them at call time (these are not fixed)."
                )

        # Pass only dynamic values through the traced path
        dynamic_vals: tuple[SolverInput, ...] = tuple(
            runtime[k] for k in _dynamic_keys
        )

        return _solver_dynamic(*dynamic_vals)

    # ── Attach diagnostics ───────────────────────────────────────────
    solver.timings = _timings
    solver.fd_check = _fd_check

    return solver