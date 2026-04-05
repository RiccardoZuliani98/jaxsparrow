"""
_solver_common.py
=================
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

All ``pure_callback`` functions return raw numpy arrays.  The
callback infrastructure applies ``tree_map(np.asarray, ...)`` on
return values, so explicit ``jnp.array()`` wrapping inside
callbacks is unnecessary and would double the conversion cost.

VJP residual compression
-------------------------
By default, the VJP forward pass saves full dynamic primals as
residuals.  For sparse problems this is wasteful: BCOO matrices
carry indices that are constant (fixed at setup via sparsity
patterns).  The optional ``residual_extractor``,
``residual_reconstructor``, and ``vjp_residual_shapes`` parameters
allow the sparse path to save only the nonzero values (``.data``)
as residuals and reconstruct the full CSC matrices on the backward
pass from cached sparsity patterns.  The dense path omits these
parameters and gets the default (full-array) behavior.
"""

from __future__ import annotations

from jax import custom_jvp, custom_vjp, pure_callback
import jax
import jax.numpy as jnp
import numpy as np
import logging
from time import perf_counter
from typing import Optional, Callable, Any, Union
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

# Expected ndim for each ingredient (unbatched).
# Used to detect batching in the JVP / VJP paths.
EXPECTED_NDIM: dict[str, int] = {
    "P": 2, "q": 1, "A": 2, "b": 1, "G": 2, "h": 1,
}

# Expected shapes for each ingredient (unbatched), given problem dims.
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

def compute_required_keys(
    n_eq: int, n_ineq: int,
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


# ── Converter protocols ──────────────────────────────────────────────
#
# Dense and sparse callers each supply converters with the following
# signatures.  The common code never inspects the returned objects —
# it just passes them through to the solver / differentiator.

PrimalConverter = Callable[[str, Any, Any], Any]
"""(key, jax_value, dtype) -> numpy representation."""

TangentConverter = Callable[[str, Any, Any], Any]
"""(key, jax_value, dtype) -> numpy tangent representation."""

ResidualExtractor = Callable[[str, Any], jax.Array]
"""(key, jax_value) -> minimal JAX array to save as VJP residual.

For dense keys this is the identity.  For sparse BCOO keys this
extracts ``.data`` (the nonzero values), discarding the constant
indices.
"""

ResidualReconstructor = Callable[[str, Any, Any], Any]
"""(key, residual_array, dtype) -> numpy primal.

Reconstructs a full numpy primal (ndarray or csc_matrix) from the
minimal residual saved by ``ResidualExtractor``.  For dense keys
this is equivalent to the primal converter.  For sparse keys it
rebuilds a CSC matrix from the residual data vector and cached
sparsity indices.
"""


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
    # Converters (JAX → numpy)
    primal_converter: PrimalConverter,
    tangent_converter: TangentConverter,
    # VJP backward shapes (gradient shapes per dynamic key)
    vjp_bwd_shapes: dict[str, jax.ShapeDtypeStruct] | None = None,
    # Optional: minimal VJP residuals (sparse path only)
    residual_extractor: ResidualExtractor | None = None,
    residual_reconstructor: ResidualReconstructor | None = None,
    vjp_residual_shapes: dict[str, jax.ShapeDtypeStruct] | None = None,
    # Finite-difference recorder (caller creates it)
    fd_check: FiniteDifferenceRecorder | None = None,
):
    """
    Wire up JAX custom differentiation rules around a numpy solver.

    All callbacks return raw numpy arrays.  The ``pure_callback``
    infrastructure handles numpy-to-JAX conversion automatically
    via ``tree_map(np.asarray, ...)``.

    Parameters
    ----------
    residual_extractor : optional
        If provided, used in the VJP forward pass to extract minimal
        residuals from dynamic primals (e.g. ``.data`` from BCOO).
        When ``None``, full primals are saved as residuals.
    residual_reconstructor : optional
        If provided, used in the VJP backward callback to reconstruct
        numpy primals from minimal residuals.  When ``None``, the
        standard ``primal_converter`` is used.
    vjp_residual_shapes : optional
        Per-key ``ShapeDtypeStruct`` for the minimal residuals.
        Required when ``residual_extractor`` is provided.  When
        ``None``, residual shapes are the full expected shapes.

    Returns the public ``solver`` callable (with ``.timings`` and
    ``.fd_check`` attributes).
    """

    logger = logging.getLogger(__name__)
    _dtype: jnp.dtype = options_parsed["dtype"]

    # ── Timing recorder ──────────────────────────────────────────────
    _timings = TimingRecorder()

    # ── FD recorder (use caller's or create a default) ───────────────
    _fd_check = fd_check or FiniteDifferenceRecorder(
        enabled=options_parsed["fd_check"],
        eps=options_parsed["fd_eps"],
    )

    # ── Key partitioning ─────────────────────────────────────────────
    _required_keys = compute_required_keys(n_eq, n_ineq)
    _dynamic_keys = compute_dynamic_keys(_required_keys, fixed_keys_set)
    _dynamic_keys_set = set(_dynamic_keys)
    _n_dyn = len(_dynamic_keys)
    _expected_shapes = make_expected_shapes(n_var, n_eq, n_ineq)

    # ── Residual mode ────────────────────────────────────────────────
    #
    # When all three residual params are provided, the VJP path saves
    # minimal residuals (e.g. only .data for sparse BCOO keys).
    # Otherwise it falls back to saving full primals.
    _use_minimal_residuals = (
        residual_extractor is not None
        and residual_reconstructor is not None
        and vjp_residual_shapes is not None
    )
    _extract_residual = residual_extractor
    _reconstruct_from_residual = residual_reconstructor

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

    # VJP backward shapes — gradient shapes per dynamic key.
    # If not supplied, default to the full expected shapes.
    _vjp_bwd_shapes = vjp_bwd_shapes or {
        k: jax.ShapeDtypeStruct(_expected_shapes[k], _dtype)
        for k in _dynamic_keys
    }

    # ── Warmstart slot ───────────────────────────────────────────────
    _warmstart: list[Optional[jax.Array]] = [None]

    # ── Aliases for readability ──────────────────────────────────────
    _solver_numpy = solver_numpy
    _diff_forward_numpy = diff_forward_numpy
    _diff_reverse_numpy = diff_reverse_numpy
    _primal_conv = primal_converter
    _tangent_conv = tangent_converter

    # =================================================================
    # BASE SOLVER CALLBACK
    # =================================================================

    def _solver(
        *dynamic_vals: jax.Array | BCOO,
    ) -> dict[str, ndarray]:
        """Solve the QP and return raw numpy arrays.

        Returns a dict with keys ``x``, ``lam``, ``mu`` whose values
        are numpy ndarrays.  ``pure_callback`` converts them to JAX
        arrays automatically.
        """

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
            prob_np["warmstart"] = np.asarray(_warmstart[0])

        # ── Solve ────────────────────────────────────────────────────
        sol_np, t_solve = _solver_numpy(**prob_np)
        x_np, lam_np, mu_np = sol_np
        t.update(t_solve)

        # ── Return numpy arrays ──────────────────────────────────────
        result = {"x": x_np, "lam": lam_np, "mu": mu_np}

        t["total"] = perf_counter() - t_start
        logger.info(f"_solver | {fmt_times(t)}")
        _timings.record("_solver", t)

        return result

    # =================================================================
    # FORWARD DIFFERENTIATION CALLBACK
    # =================================================================

    def _diff_forward(
        *args: jax.Array | BCOO,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray]]:
        """Solve + forward-mode differentiate, returning raw numpy.

        Returns ``(diff_out, res)`` where both are dicts with keys
        ``x``, ``lam``, ``mu`` containing numpy ndarrays.
        """

        t_start = perf_counter()
        t: dict[str, float] = {}

        dyn_primal_vals = args[:_n_dyn]
        dyn_tangent_vals = args[_n_dyn:]

        # ── Convert primals ──────────────────────────────────────────
        start = perf_counter()
        #TODO: this needs to change after I stack all inputs together,
        # use a function from the "converter"
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
            prob_np["warmstart"] = np.asarray(_warmstart[0])

        # ── Solve ────────────────────────────────────────────────────
        sol_np, t_solve = _solver_numpy(**prob_np)
        x_np, lam_np, mu_np = sol_np
        _warmstart[0] = None
        t.update(t_solve)

        # ── Convert tangents ─────────────────────────────────────────
        start = perf_counter()
        #TODO: this needs to change after I stack all inputs together,
        # use a function from the "converter"
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

    def _diff_reverse(
        *args: jax.Array | BCOO,
    ) -> dict[str, ndarray]:
        """Reverse-mode differentiate, returning raw numpy gradients.

        The first ``_n_dyn`` args are VJP residuals.  When minimal
        residuals are enabled, these are compressed representations
        (e.g. only the nonzero values for sparse keys); the
        reconstructor rebuilds full numpy primals from them.  When
        disabled, these are full primals converted via the standard
        primal converter.

        Returns a dict keyed by dynamic parameter names whose values
        are numpy ndarrays.
        """

        t_start = perf_counter()
        t: dict[str, float] = {}

        # ── Unpack primals from residuals ────────────────────────────
        start = perf_counter()
        #TODO: this is the same as above, we need to somehow unstack the
        # QP ingredients
        if _use_minimal_residuals:
            prob_np: dict[str, Any] = {
                k: _reconstruct_from_residual(k, args[i], _dtype)  # type: ignore
                for i, k in enumerate(_dynamic_keys)
            }
        else:
            prob_np = {
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
        t["total_numpy_operations"] = perf_counter() - start

        t["total"] = perf_counter() - t_start
        logger.info(f"_kkt_vjp | {fmt_times(t)}")
        _timings.record("_kkt_vjp", t)

        # ── FD check ─────────────────────────────────────────────────
        if options_parsed["fd_check"]:
            _run_fd_check_vjp(
                prob_np, grads_np,
                g_x, g_lam, g_mu, batched,
            )

        return grads_np

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
        # Convert BCOO → .data, leave dense arrays as-is
        converted = []
        for k, v in zip(_dynamic_keys, dynamic_vals):
            if isinstance(v, BCOO):
                converted.append(v.data)          # pass only the nonzero values
            else:
                converted.append(v)               # dense JAX array
        return pure_callback(_solver, _fwd_shapes, *converted)

    @_solver_dynamic_jvp_mode.defjvp
    def _solver_dynamic_jvp_rule(
        primals: tuple[SolverInput, ...],
        tangents: tuple[SolverInput, ...],
    ) -> tuple[SolverOutput, dict[str, jax.Array]]:
        #TODO: this needs to change, there should be a converter that comes in
        # and does the "stacking"
        primals_converted = []
        for k, v in zip(_dynamic_keys, primals):
            if isinstance(v, BCOO):
                primals_converted.append(v.data)
            else:
                primals_converted.append(v)

        #TODO: same exact thing here
        tangents_converted = []
        for k, v in zip(_dynamic_keys, tangents):
            if isinstance(v, BCOO):
                tangents_converted.append(v.data)
            else:
                tangents_converted.append(v)

        #TODO: we might change to "legacy_vectorized"
        tangents_out, res = pure_callback(
            _diff_forward,
            _jvp_shapes,
            *primals_converted,
            *tangents_converted,
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

    if _use_minimal_residuals:

        # Minimal residuals: save only .data for sparse BCOO keys,
        # full arrays for dense keys.  Shapes come from
        # vjp_residual_shapes.

        #TODO: same here, we need to do some stacking, and we must not
        # branch based on "_use_minimal_residuals"
        def _solver_dynamic_vjp_fwd(
            *dynamic_vals: SolverInput,
        ) -> tuple[SolverOutput, tuple[jax.Array, ...]]:
            result = pure_callback(
                _solver, _fwd_shapes, *dynamic_vals,
            )
            saved = tuple(
                _extract_residual(k, v)  # type: ignore
                for k, v in zip(_dynamic_keys, dynamic_vals)
            )
            residuals = (*saved, result["x"], result["lam"], result["mu"])
            return result, residuals

    else:
        # Default: save full primals as residuals.

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

    #TODO: here we need to also put some stacking converter
    def _solver_dynamic_vjp_bwd(
        residuals: tuple[jax.Array, ...],
        g: dict[str, jax.Array],
    ) -> tuple[jax.Array, ...]:
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
            _warmstart[0] = jnp.asarray(warmstart, dtype=_dtype).reshape(-1)
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