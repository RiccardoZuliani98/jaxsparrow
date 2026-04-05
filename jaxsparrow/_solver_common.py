"""
_solver_common.py
=================
Shared scaffolding for dense and sparse differentiable solvers.

Every ``pure_callback`` receives exactly ONE flat 1-D numpy vector
as input and returns either a dict of numpy arrays (solve, fwd) or
a single flat numpy vector (rev).  This reduces ``device_put``
overhead to a single transfer per callback invocation.

All callbacks use ``vmap_method="legacy_vectorized"`` so that only
vmapped arguments acquire a leading batch dimension.  Non-vmapped
arguments keep their original shape.
"""

from __future__ import annotations

from jax import custom_jvp, custom_vjp, pure_callback
import jax
import jax.numpy as jnp
import numpy as np
import logging
from time import perf_counter
from typing import Optional, Callable, Any, Union, Sequence
from numpy import ndarray

from jaxsparrow._utils._printing_utils import fmt_times
from jaxsparrow._utils._timing_utils import TimingRecorder
from jaxsparrow._types_common import SolverOutput
from jaxsparrow._options_common import ConstructorOptionsFull
from jaxsparrow._utils._fd_recorder import FiniteDifferenceRecorder
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix


SolverInputNP = Union[ndarray, csc_matrix]
SolverInput = Union[jax.Array, BCOO]


# ── Constants ────────────────────────────────────────────────────────

EXPECTED_NDIM: dict[str, int] = {
    "P": 2, "q": 1, "A": 2, "b": 1, "G": 2, "h": 1,
}

def make_expected_shapes(
    n_var: int, n_eq: int, n_ineq: int,
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

def compute_required_keys(n_eq: int, n_ineq: int) -> tuple[str, ...]:
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
    return tuple(k for k in required_keys if k not in fixed_keys)


# ── Callable protocols ───────────────────────────────────────────────

StackFn = Callable[[Sequence[str], Sequence[Any]], jax.Array]
UnstackPrimalsFn = Callable[[ndarray, Any], dict[str, Any]]
UnstackTangentsFn = Callable[[ndarray, Any], dict[str, Any]]


# ── Builder ──────────────────────────────────────────────────────────

def build_solver(
    *,
    n_var: int,
    n_ineq: int,
    n_eq: int,
    options_parsed: ConstructorOptionsFull,
    fixed_keys_set: set[str],
    solver_numpy: Callable[..., Any],
    diff_forward_numpy: Callable[..., Any],
    diff_reverse_numpy: Callable[..., Any],
    # Ingredient stacker / unstacker
    stack_fn: StackFn,
    unstack_primals_fn: UnstackPrimalsFn,
    unstack_tangents_fn: UnstackTangentsFn,
    packed_length: int,
    # Forward-diff packer (primals+tangents → one vector)
    fwd_stack_fn: Callable[..., jax.Array],
    fwd_unstack_fn: Callable[..., tuple[ndarray, ndarray]],
    # Reverse-diff packer (residuals+sol+cotangents → one vector)
    rev_stack_fn: Callable[..., jax.Array],
    rev_unstack_fn: Callable[..., tuple],
    rev_total_length: int,
    # VJP backward shapes
    vjp_bwd_shapes: dict[str, jax.ShapeDtypeStruct] | None = None,
    fd_check: FiniteDifferenceRecorder | None = None,
):
    """
    Wire up JAX custom differentiation rules around a numpy solver.

    Every ``pure_callback`` receives exactly one flat vector.
    All use ``vmap_method="legacy_vectorized"``.
    """

    logger = logging.getLogger(__name__)
    _dtype: jnp.dtype = options_parsed["dtype"]

    _timings = TimingRecorder()
    _fd_check = fd_check or FiniteDifferenceRecorder(
        enabled=options_parsed["fd_check"],
        eps=options_parsed["fd_eps"],
    )

    _required_keys = compute_required_keys(n_eq, n_ineq)
    _dynamic_keys = compute_dynamic_keys(_required_keys, fixed_keys_set)
    _dynamic_keys_set = set(_dynamic_keys)
    _n_dyn = len(_dynamic_keys)
    _expected_shapes = make_expected_shapes(n_var, n_eq, n_ineq)

    _stack = stack_fn
    _unstack_primals = unstack_primals_fn
    _unstack_tangents = unstack_tangents_fn
    _packed_len = packed_length

    _fwd_stack = fwd_stack_fn
    _fwd_unstack = fwd_unstack_fn
    _rev_stack = rev_stack_fn
    _rev_unstack = rev_unstack_fn
    _rev_total_len = rev_total_length

    # ── JAX shape structs ────────────────────────────────────────────
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

    _vjp_bwd_flat_shape = jax.ShapeDtypeStruct((_packed_len,), _dtype)

    _warmstart: list[Optional[jax.Array]] = [None]

    _solver_numpy_fn = solver_numpy
    _diff_forward_numpy_fn = diff_forward_numpy
    _diff_reverse_numpy_fn = diff_reverse_numpy

    # =================================================================
    # BASE SOLVER CALLBACK  (1 flat vector in → dict out)
    # =================================================================

    def _solver(packed: ndarray) -> dict[str, ndarray]:

        t_start = perf_counter()
        t: dict[str, float] = {}

        start = perf_counter()
        packed_np = np.asarray(packed, dtype=_dtype)
        while packed_np.ndim > 1:
            packed_np = packed_np[0]
        dynamic_np = _unstack_primals(packed_np, _dtype)
        t["convert_to_numpy"] = perf_counter() - start

        prob_np = dynamic_np
        if _warmstart[0] is not None:
            prob_np["warmstart"] = np.asarray(_warmstart[0])

        sol_np, t_solve = _solver_numpy_fn(**prob_np)
        x_np, lam_np, mu_np = sol_np
        t.update(t_solve)

        result = {"x": x_np, "lam": lam_np, "mu": mu_np}

        t["total"] = perf_counter() - t_start
        logger.info(f"_solver | {fmt_times(t)}")
        _timings.record("_solver", t)
        return result

    # =================================================================
    # FORWARD DIFF CALLBACK  (1 flat vector in → (dict, dict) out)
    # =================================================================

    def _diff_forward(
        packed_all: ndarray,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray]]:

        t_start = perf_counter()
        t: dict[str, float] = {}

        packed_all_np = np.asarray(packed_all, dtype=_dtype)
        primals_flat, tangents_flat = _fwd_unstack(packed_all_np)

        # ── Unpack primals (1-D) ─────────────────────────────────────
        start = perf_counter()
        dyn_primals_np = _unstack_primals(primals_flat, _dtype)
        t["convert_to_numpy"] = perf_counter() - start

        prob_np = dict(dyn_primals_np)
        if _warmstart[0] is not None:
            prob_np["warmstart"] = np.asarray(_warmstart[0])

        # ── Solve ────────────────────────────────────────────────────
        sol_np, t_solve = _solver_numpy_fn(**prob_np)
        x_np, lam_np, mu_np = sol_np
        _warmstart[0] = None
        t.update(t_solve)

        # ── Unpack tangents (1-D or 2-D) ─────────────────────────────
        start = perf_counter()
        dyn_tangents_np = _unstack_tangents(tangents_flat, _dtype)
        t["convert_tangents"] = perf_counter() - start

        batched = tangents_flat.ndim == 2
        batch_size = tangents_flat.shape[0] if batched else 0

        # ── Differentiate ────────────────────────────────────────────
        diff_out_np, t_diff = _diff_forward_numpy_fn(
            sol_np=sol_np,
            dyn_primals_np=dyn_primals_np,
            dyn_tangents_np=dyn_tangents_np,
            batch_size=batch_size,
        )
        dx_np, dlam_np, dmu_np = diff_out_np
        t.update(t_diff)

        # ── Build results ────────────────────────────────────────────
        start = perf_counter()
        if batch_size > 0:
            res = {
                "x":   np.broadcast_to(x_np, (batch_size, n_var)).copy(),
                "lam": np.broadcast_to(lam_np, (batch_size, n_ineq)).copy(),
                "mu":  np.broadcast_to(mu_np, (batch_size, n_eq)).copy(),
            }
        else:
            res = {"x": x_np, "lam": lam_np, "mu": mu_np}

        diff_out = {"x": dx_np, "lam": dlam_np, "mu": dmu_np}
        t["build_sol"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_diff | {fmt_times(t)}")
        _timings.record("_kkt_diff", t)

        if options_parsed["fd_check"]:
            _run_fd_check_jvp(
                dyn_primals_np, dyn_tangents_np,
                dx_np, dlam_np, dmu_np, batched,
            )

        return diff_out, res

    # =================================================================
    # REVERSE DIFF CALLBACK  (1 flat vector in → 1 flat vector out)
    # =================================================================

    def _diff_reverse(packed_all: ndarray) -> ndarray:

        t_start = perf_counter()
        t: dict[str, float] = {}

        start = perf_counter()
        packed_all_np = np.asarray(packed_all, dtype=_dtype)
        (packed_res, x_np, lam_np, mu_np,
         g_x_np, g_lam_np, g_mu_np) = _rev_unstack(packed_all_np, _dtype)

        # Unpack primals from residuals
        prob_np = _unstack_primals(packed_res, _dtype)
        t["convert_to_numpy"] = perf_counter() - start

        # ── Detect batching (from cotangent shape) ───────────────────
        batched = g_x_np.ndim == 2
        batch_size = g_x_np.shape[0] if batched else 0

        # ── Differentiate ────────────────────────────────────────────
        start = perf_counter()
        grads_np, t_diff = _diff_reverse_numpy_fn(
            dyn_primals_np=prob_np,
            x_np=x_np, lam_np=lam_np, mu_np=mu_np,
            g_x=g_x_np, g_lam=g_lam_np, g_mu=g_mu_np,
            batch_size=batch_size,
        )
        t.update(t_diff)
        t["total_numpy_operations"] = perf_counter() - start

        if options_parsed["fd_check"]:
            _run_fd_check_vjp(
                prob_np, grads_np,
                g_x_np, g_lam_np, g_mu_np, batched,
            )

        # ── Pack gradients ───────────────────────────────────────────
        start = perf_counter()
        grad_parts = []
        for k in _dynamic_keys:
            g = grads_np[k]
            if batched and g.ndim > 1:
                grad_parts.append(g)
            else:
                grad_parts.append(g.ravel())
        if batched:
            packed_grads = np.concatenate(grad_parts, axis=1)
        else:
            packed_grads = np.concatenate(grad_parts)
        t["pack_grads"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_vjp | {fmt_times(t)}")
        _timings.record("_kkt_vjp", t)

        return packed_grads

    # =================================================================
    # FD CHECK HELPERS
    # =================================================================

    def _run_fd_check_jvp(
        dyn_primals_np, dyn_tangents_np,
        dx_np, dlam_np, dmu_np, batched,
    ):
        if not batched:
            _fd_check.check_jvp(
                solve_fn=_solver_numpy_fn,
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
                solve_fn=_solver_numpy_fn,
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
                solve_fn=_solver_numpy_fn,
                dyn_primals_np=prob_np,
                grads_analytic=grads_np,
                g_x=g_x, g_lam=g_lam, g_mu=g_mu,
                dynamic_keys=_dynamic_keys,
            )
        else:
            _fd_check.check_vjp(
                solve_fn=_solver_numpy_fn,
                dyn_primals_np=prob_np,
                grads_analytic={k: v[0] for k, v in grads_np.items()},
                g_x=g_x[0], g_lam=g_lam[0], g_mu=g_mu[0],
                dynamic_keys=_dynamic_keys,
            )

    # =================================================================
    # GRADIENT UNPACKING HELPER (JAX side)
    # =================================================================

    _grad_offsets: list[tuple[str, int, int]] = []
    _offset = 0
    for k in _dynamic_keys:
        if vjp_bwd_shapes is not None and k in vjp_bwd_shapes:
            n = 1
            for s in vjp_bwd_shapes[k].shape:
                n *= s
        else:
            n = 1
            for s in _expected_shapes[k]:
                n *= s
        _grad_offsets.append((k, _offset, _offset + n))
        _offset += n

    def _unpack_flat_grads(flat_grads: jax.Array) -> tuple[jax.Array, ...]:
        return tuple(
            flat_grads[start:end]
            for _, start, end in _grad_offsets
        )

    # =================================================================
    # JAX CUSTOM JVP / VJP REGISTRATION
    # =================================================================

    # ── JVP path ─────────────────────────────────────────────────────

    @custom_jvp
    def _solver_dynamic_jvp_mode(
        *dynamic_vals: SolverInput,
    ) -> SolverOutput:
        packed = _stack(_dynamic_keys, dynamic_vals)
        return pure_callback(_solver, _fwd_shapes, packed)

    # Shape for the single packed fwd-diff input
    _fwd_packed_shape = jax.ShapeDtypeStruct((2 * _packed_len,), _dtype)

    @_solver_dynamic_jvp_mode.defjvp
    def _solver_dynamic_jvp_rule(
        primals: tuple[SolverInput, ...],
        tangents: tuple[SolverInput, ...],
    ) -> tuple[SolverOutput, dict[str, jax.Array]]:
        packed_primals = _stack(_dynamic_keys, primals)
        packed_tangents = _stack(_dynamic_keys, tangents)
        packed_all = _fwd_stack(packed_primals, packed_tangents)

        tangents_out, res = pure_callback(
            _diff_forward,
            _jvp_shapes,
            packed_all,
            vmap_method="legacy_vectorized",
        )
        return res, tangents_out

    # ── VJP path ─────────────────────────────────────────────────────

    @custom_vjp
    def _solver_dynamic_vjp_mode(
        *dynamic_vals: SolverInput,
    ) -> SolverOutput:
        packed = _stack(_dynamic_keys, dynamic_vals)
        return pure_callback(_solver, _fwd_shapes, packed)

    def _solver_dynamic_vjp_fwd(
        *dynamic_vals: SolverInput,
    ) -> tuple[SolverOutput, tuple[jax.Array, ...]]:
        packed = _stack(_dynamic_keys, dynamic_vals)
        result = pure_callback(_solver, _fwd_shapes, packed)
        # 4 residual arrays: packed ingredients + x + lam + mu
        residuals = (packed, result["x"], result["lam"], result["mu"])
        return result, residuals

    def _solver_dynamic_vjp_bwd(
        residuals: tuple[jax.Array, ...],
        g: dict[str, jax.Array],
    ) -> tuple[jax.Array, ...]:
        packed_residuals, x_sol, lam_sol, mu_sol = residuals
        g_x   = g["x"]
        g_lam = g["lam"]
        g_mu  = g["mu"]

        # Pack everything into ONE vector
        packed_all = _rev_stack(
            packed_residuals, x_sol, lam_sol, mu_sol,
            g_x, g_lam, g_mu,
        )

        flat_grads = pure_callback(
            _diff_reverse,
            _vjp_bwd_flat_shape,
            packed_all,
            vmap_method="legacy_vectorized",
        )
        return _unpack_flat_grads(flat_grads)

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
        """Solve a convex problem and return (x, lam, mu)."""

        if warmstart is not None:
            _warmstart[0] = jnp.asarray(warmstart, dtype=_dtype).reshape(-1)
        else:
            _warmstart[0] = None

        if options_parsed["debug"]:
            overridden = set(runtime.keys()) & fixed_keys_set
            if overridden:
                logger.warning(
                    f"Ignoring runtime values for fixed keys: "
                    f"{sorted(overridden)}."
                )
            missing = _dynamic_keys_set - set(runtime)
            if missing:
                raise ValueError(
                    f"Missing ingredients: {sorted(missing)}."
                )

        dynamic_vals: tuple[SolverInput, ...] = tuple(
            runtime[k] for k in _dynamic_keys
        )
        return _solver_dynamic(*dynamic_vals)

    solver.timings = _timings
    solver.fd_check = _fd_check
    return solver