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
#TODO: add reverse mode tests
#TODO: add value function and envelope theorem?
#TODO: vmap for vjp

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
    """Set up a differentiable dense QP solver with configurable differentiation mode.
    
    Creates and returns a callable ``solver(**runtime)`` that solves quadratic
    programs of the form::
    
        minimize    (1/2) xᵀ P x + qᵀ x
        subject to  A x = b
                    G x ≤ h
    
    The solver supports both forward-mode (JVP) and reverse-mode (VJP)
    automatic differentiation via implicit differentiation of the KKT
    optimality conditions. The differentiation mode is selected at setup
    time via the ``differentiator_type`` option.
    
    **Differentiation Modes:**
    
        - ``"kkt_fwd"`` (JVP): Forward-mode autodiff. Efficient for functions
          with few inputs relative to outputs. Compatible with ``jax.jvp``
          and ``jax.vmap(jax.jvp(...))``.
          
        - ``"kkt_rev"`` (VJP): Reverse-mode autodiff. Efficient for functions
          with many inputs and few outputs. Compatible with ``jax.grad``,
          ``jax.value_and_grad``, and ``jax.vjp``.
    
    **Fixed Elements for Performance:**
    
    Elements declared in ``fixed_elements`` are stored as numpy arrays and
    are **never** converted to JAX or routed through the traced path. This
    avoids unnecessary Python-JAX round-trips and numpy conversions, but
    means that fixed elements are treated as constants for differentiation —
    their tangents/cotangents are always zero.
    
    To differentiate through an element, it must be supplied at runtime
    (not fixed). This gives you fine-grained control over the trade-off
    between performance and differentiability.
    
    **Warm-starting:**
    
    The solver supports warm-starting via an optional ``warmstart`` argument
    at call time. The warmstart initial guess is passed to the underlying
    numpy solver and cleared after each solve. It is **not** differentiated
    through and does not affect the computed derivatives.
    
    **Timing and Profiling:**
    
    The returned solver object has a ``.timings`` attribute that collects
    detailed timing information across calls. This is useful for profiling
    and debugging performance bottlenecks.
    
    Args:
        n_var: Number of decision variables. Must be positive.
        
        n_ineq: Number of inequality constraints. May be 0 if no inequalities.
            Defaults to 0.
            
        n_eq: Number of equality constraints. May be 0 if no equalities.
            Defaults to 0.
            
        fixed_elements: Optional dictionary of QP ingredients that remain
            constant across all solver calls. Valid keys are:
            
            - ``"P"``: Cost matrix, shape ``(n_var, n_var)``
            - ``"q"``: Linear cost vector, shape ``(n_var,)``
            - ``"A"``: Equality constraint matrix, shape ``(n_eq, n_var)``
            - ``"b"``: Equality RHS vector, shape ``(n_eq,)``
            - ``"G"``: Inequality constraint matrix, shape ``(n_ineq, n_var)``
            - ``"h"``: Inequality RHS vector, shape ``(n_ineq,)``
            
            Fixed elements are treated as constants for differentiation and
            are never converted to JAX arrays.
            
        options: Optional configuration dictionary controlling the solver and
            differentiator behavior. Supports the following keys:
            
            - ``"dtype"``: Data type for arrays (e.g., ``np.float64``).
              Defaults to ``np.float64``.
              
            - ``"solver_type"``: Backend QP solver. Currently only
              ``"qp_solvers"`` is supported.
              
            - ``"solver"``: Solver-specific options passed to the underlying
              QP solver (e.g., tolerances, max iterations).
              
            - ``"differentiator_type"``: Differentiation mode. One of:
              ``"kkt_fwd"`` (JVP) or ``"kkt_rev"`` (VJP). Defaults to
              ``"kkt_fwd"``.
              
            - ``"differentiator"``: Differentiator-specific options (e.g.,
              regularization, linear solver type).
              
            - ``"debug"``: If True, enables additional shape and consistency
              checks. Defaults to False.
            
            See ``solver_dense_options.py`` for defaults and available options.
    
    Returns:
        Callable[[**runtime], QPOutput]: A solver function with signature:
        
            ``solver(*, warmstart: Optional[jax.Array] = None, **runtime: jax.Array) -> QPOutput``
        
        The returned solver also has a ``.timings`` attribute (a ``TimingRecorder``
        instance) that accumulates timing information across calls.
    
    Raises:
        ValueError: If an unsupported solver type or differentiator is requested,
            or if fixed elements have incorrect shapes.
        AssertionError: If debug mode is enabled and shape validation fails.
    
    Example:
        >>> # Set up a solver with fixed cost matrices
        >>> solver = setup_dense_solver(
        ...     n_var=2, n_eq=1,
        ...     fixed_elements={
        ...         "P": np.array([[4., 1.], [1., 2.]]),
        ...         "A": np.array([[1., 1.]]),
        ...     }
        ... )
        >>> 
        >>> # Solve with varying linear cost and RHS
        >>> q = jnp.array([1., 2.])
        >>> b = jnp.array([1.])
        >>> sol = solver(q=q, b=b)
        >>> 
        >>> # Compute gradients via reverse-mode
        >>> grad_q = jax.grad(lambda q, b: solver(q=q, b=b)["x"].sum())(q, b)
    
    Notes:
        - The solver is **not** jit-compilable due to the use of ``pure_callback``.
          It is designed for use inside JAX-transformed functions where the
          callback overhead is acceptable.
        - Batching is supported only in differentiation (via JVP/VJP), not in
          the forward solve itself. When batched tangents/cotangents are provided,
          the linear systems are solved with matrix RHS for efficiency.
        - The active set from the QP solution is used internally but never
          exposed to the user.
    """

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
    if _options_parsed["differentiator_type"] == "kkt_fwd" or "kkt_rev":
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
        """Bridge between JAX ``pure_callback`` and the numpy QP solver.
        
        This function is designed to be called via ``pure_callback``. It receives
        only the **dynamic** QP ingredients (those not in ``fixed_elements``) as
        positional JAX arrays in ``_dynamic_keys`` order, converts them to numpy,
        merges them with the pre-stored fixed numpy arrays, and delegates to the
        underlying numpy solver ``_solve_qp_numpy``.
        
        The function handles warm-starting by reading from the mutable
        ``_warmstart`` slot and clears it after use. Timing information is
        collected and recorded via ``_timings``.
        
        Args:
            *dynamic_vals: Variable-length argument list of JAX arrays.
                Must contain exactly ``len(_dynamic_keys)`` arrays, in the order
                specified by ``_dynamic_keys``. Each array corresponds to a
                dynamic QP ingredient (P, q, A, b, G, h) that was not fixed at
                setup time.
        
        Returns:
            QPOutput: A dictionary containing the solution as JAX arrays:

                - ``x``      (n_var,):  Primal solution.
                - ``lam``    (n_ineq,):  Inequality duals.
                - ``mu``     (n_eq,):  Equality duals.

        Raises:
            ValueError: If debug mode is enabled and any dynamic ingredient is
                missing or has incorrect shape.
    
        Notes:
            - Batching is **not** supported in the primal solve path. If batched
            inputs are detected (via extra leading dimension), only the first
            batch element is used. This is because the primal solution should
            be identical across the batch when differentiating.
            - The warmstart array, if provided, is converted to numpy and passed
            to the underlying solver. It is cleared after each solve to prevent
            accidental reuse.
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
        """Forward-mode implicit differentiation of the KKT conditions.
        
        This function is designed to be called via ``pure_callback`` from the JVP
        rule. It performs two tasks in a single numpy-side computation:
        
        1. Solves the forward QP problem to obtain the primal/dual solution
        2. Differentiates through the KKT optimality conditions to compute
        tangents of the solution with respect to the input parameters
        
        The function receives both primal inputs and their tangents for all
        dynamic parameters, solves the linearized KKT system, and returns both
        the primal solution and the tangents of the solution variables.
        
        Batching is handled efficiently: when tangents carry an extra batch
        dimension (compared to the primals), the linear system is solved once
        with a matrix right-hand side, yielding tangents for all batch elements
        simultaneously.
        
        Args:
            *args: Variable-length argument list of numpy arrays.
                The first ``len(_dynamic_keys)`` arrays are the primal values
                of the dynamic QP ingredients (P, q, A, b, G, h) in order.
                The next ``len(_dynamic_keys)`` arrays are the corresponding
                tangents for each dynamic ingredient.
                
                Example: If ``_dynamic_keys = ('P', 'q')``, then:
                    - ``args[0]``: P (primal)
                    - ``args[1]``: q (primal)
                    - ``args[2]``: tangent of P
                    - ``args[3]``: tangent of q
        
        Returns:
            
            - ``tangents_out`` (QPDiffOut): Dictionary containing the tangents
            of the solution variables:
            
                * ``x``: Tangent of primal solution.
                Shape ``(n_var,)`` if unbatched, or ``(batch_size, n_var)``
                if batched.
                * ``lam``: Tangent of inequality duals.
                Shape ``(n_ineq,)`` if unbatched, or ``(batch_size, n_ineq)``
                if batched.
                * ``mu``: Tangent of equality duals.
                Shape ``(n_eq,)`` if unbatched, or ``(batch_size, n_eq)``
                if batched.
                
            - ``solution`` (QPOutput): Dictionary containing the primal
            solution (the result of the forward QP solve):
            
                * ``x``: Primal solution vector.
                Shape ``(n_var,)`` if unbatched, or broadcast to
                ``(batch_size, n_var)`` if batched.
                * ``lam``: Inequality dual variables.
                Shape ``(n_ineq,)`` if unbatched, or broadcast to
                ``(batch_size, n_ineq)`` if batched.
                * ``mu``: Equality dual variables.
                Shape ``(n_eq,)`` if unbatched, or broadcast to
                ``(batch_size, n_eq)`` if batched.
        
        Notes:
            - Fixed elements (provided at setup) are **not** passed as arguments.
            Their tangents are implicitly zero and are handled inside the
            differentiator implementation.
            - The primal solution is broadcast (not tiled) to match the batch
            dimension when batching is detected, as the primal solution is
            identical for all batch elements.
            - This function returns numpy arrays; ``pure_callback`` handles the
            conversion to JAX arrays using ``_jvp_shapes``.
            - The active set from the QP solution is used internally for
            inequality handling but is not returned.
        """

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
        """Reverse-mode (VJP) implicit differentiation of the KKT conditions.
        
        This function is designed to be called via ``pure_callback`` from the VJP
        backward pass. It solves the adjoint KKT system to compute cotangents
        (gradients) of the solution with respect to all dynamic input parameters.
        
        The function leverages the symmetry of the KKT matrix: because the KKT
        matrix M is symmetric (M = M^T), the adjoint linear system is identical
        to the forward system used in JVP differentiation. The only difference
        is the right-hand side: instead of tangents, we use the output cotangents
        (g_x, g_lam, g_mu) from the reverse pass.
        
        After solving M·v = [g_x; g_lam; g_mu] for the adjoint variables v,
        the parameter cotangents are computed analytically using matrix calculus:
        
            g_P = -outer(v_x, x)
            g_q = -v_x
            g_A = -(outer(mu, v_x) + outer(v_mu, x))
            g_b = v_mu
            g_G[active] = -(outer(lam[active], v_x) + outer(v_lam_a, x))
            g_h[active] = v_lam_a
        
        where v = [v_x; v_lam; v_mu] is the solution of the adjoint system, and
        `active` is the set of active inequality constraints from the forward solve.
        
        Batching is supported: when the input cotangents carry an extra batch
        dimension (indicated by g_x.ndim == 2), the adjoint system is solved once
        with a matrix right-hand side, and parameter cotangents are computed for
        all batch elements simultaneously.
        
        Args:
            *args: Variable-length argument list of numpy arrays with the following
                layout (where N = len(_dynamic_keys)):
                
                - ``args[0:N]``: Dynamic problem matrices in ``_dynamic_keys`` order.
                    These are the **merged** matrices (dynamic + fixed) from the
                    forward pass, already converted to numpy.
                
                - ``args[N]``:     ``x``      - Primal solution, shape ``(n_var,)``
                - ``args[N+1]``:   ``lam``    - Inequality duals, shape ``(n_ineq,)``
                - ``args[N+2]``:   ``mu``     - Equality duals, shape ``(n_eq,)``
                - ``args[N+3]``:   ``g_x``    - Cotangent of x, shape ``(n_var,)`` or ``(batch_size, n_var)``
                - ``args[N+4]``:   ``g_lam``  - Cotangent of lam, shape ``(n_ineq,)`` or ``(batch_size, n_ineq)``
                - ``args[N+5]``:   ``g_mu``   - Cotangent of mu, shape ``(n_eq,)`` or ``(batch_size, n_eq)``
                
                Note: The active set is not passed explicitly; it is reconstructed
                inside the differentiator from the solution and problem data.
        
        Returns:
            Dict[str, Array]: A dictionary mapping each dynamic key to its cotangent
            (gradient) as a JAX array. The returned dictionary has exactly the keys
            in ``_dynamic_keys``, with shapes:
            
                - For matrix inputs (P, A, G): same shape as the primal input
                - For vector inputs (q, b, h): same shape as the primal input
            
            When batched, the cotangents have shape ``(batch_size, ...)`` matching
            the batch dimension of the input cotangents.
        
        Raises:
            ValueError: If the input arguments cannot be unpacked correctly or if
                the differentiator fails (propagated from ``_diff_reverse_numpy``).
        
        Notes:
            - This function receives **merged** problem matrices (dynamic + fixed)
            as residuals from the forward pass. This avoids re-converting dynamic
            inputs to numpy and re-merging with fixed elements in the backward pass.
            - Fixed elements (provided at setup) are **not** present in the input
            arguments and do **not** receive cotangents (their gradients are zero).
            - The active set from the forward solve is used to handle inequality
            constraints correctly but is not exposed in the public API.
            - This function returns JAX arrays directly; ``pure_callback`` handles
            the conversion from numpy using ``_vjp_bwd_shapes``.
            - Warmstarting is **not** used in the reverse pass, as it only affects
            the forward solve.
        """

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
        off = _n_dyn
        x_np      = np.asarray(args[off],     dtype=_dtype)
        lam_np    = np.asarray(args[off + 1], dtype=_dtype)
        mu_np     = np.asarray(args[off + 2], dtype=_dtype)
        g_x       = np.asarray(args[off + 3], dtype=_dtype)
        g_lam     = np.asarray(args[off + 4], dtype=_dtype)
        g_mu      = np.asarray(args[off + 5], dtype=_dtype)

        t["convert_to_numpy"] = perf_counter() - start


        # ── Detect batching ──────────────────────────────────────────
        # Batching is detected by checking if the cotangents have an extra dimension
        batched = (g_x.ndim == 2) if _n_dyn > 0 else False

        # get batch size
        batch_size : int = max(v.shape[0] for v in dyn_tangents_np.values()) if batched else 0 # type: ignore[union-attr]


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

                - ``x``      (n_var,):  Primal solution.
                - ``lam``    (n_ineq,):  Inequality duals.
                - ``mu``     (n_eq,):  Equality duals.
        """
        return pure_callback(_solve_qp, _fwd_shapes, *dynamic_vals)

    @_solver_dynamic_jvp_mode.defjvp
    def _solver_dynamic_jvp_rule(
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
    ) -> tuple[QPOutput, QPDiffOut]:
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
        """Internal solver for reverse-mode differentiation (VJP path).
        
        This function provides the forward pass for reverse-mode automatic
        differentiation via `custom_vjp`. It has identical forward semantics to
        the JVP path (`_solver_dynamic_jvp_mode`), but stores additional residuals
        for use in the backward pass.
        
        The function is designed to be called with only the **dynamic** QP
        ingredients (those not fixed at setup). Fixed ingredients are merged
        inside the numpy callback without ever entering JAX's tracing mechanism.
        
        The differentiation behavior is:
            - Compatible with `jax.grad`, `jax.value_and_grad`, and `jax.vjp`
            - Uses implicit differentiation of the KKT conditions
            - More memory-efficient than JVP for functions with many inputs
            - May be slower than JVP for functions with many outputs
        
        Args:
            *dynamic_vals: Variable-length argument list of JAX arrays.
                Must contain exactly ``len(_dynamic_keys)`` arrays, in the order
                specified by ``_dynamic_keys``. Each array corresponds to a
                dynamic QP ingredient (P, q, A, b, G, h) that was not fixed at
                setup time.
                
                Example: If ``_dynamic_keys = ('P', 'q', 'G', 'h')``, then:
                    - ``dynamic_vals[0]``: P matrix, shape ``(n_var, n_var)``
                    - ``dynamic_vals[1]``: q vector, shape ``(n_var,)``
                    - ``dynamic_vals[2]``: G matrix, shape ``(n_ineq, n_var)``
                    - ``dynamic_vals[3]``: h vector, shape ``(n_ineq,)``

        Returns:
            QPOutput: A dictionary containing the solution as JAX arrays:
            
                - ``x``:   Primal solution vector, shape ``(n_var,)``
                - ``lam``: Inequality dual variables, shape ``(n_ineq,)``
                - ``mu``:  Equality dual variables, shape ``(n_eq,)``
        
        Notes:
            - This function **does not** support batching in the forward pass.
            If batched inputs are provided, only the first batch element is used
            for the solve. This is because the primal solution must be identical
            across the batch for differentiation to be meaningful.
            - The function stores the dynamic inputs, solution, and (implicitly)
            the active set as residuals for the backward pass. This avoids
            re-computing or re-converting these values.
            - Warmstarting is supported via the `warmstart` argument to the
            public `solver()` function, but is not part of the traced path.
        """
        return pure_callback(
            _solve_qp, 
            _fwd_shapes, 
            *dynamic_vals, 
            vmap_method="sequential"
        )

    def _solver_dynamic_vjp_fwd(
        *dynamic_vals: jax.Array,
    ) -> tuple[QPOutput,tuple[jax.Array, ...],
    ]:
        """Forward pass for custom VJP, saving residuals for backward differentiation.
        
        This function is called by JAX during the forward pass of reverse-mode
        differentiation. It solves the QP problem and packages the results along
        with all necessary residuals for the backward pass.
        
        The residuals include:
            1. The dynamic input arrays (to avoid re-converting them in backward)
            2. The primal solution (x, lam, mu) (to compute analytical gradients)
        
        Note: The active set is **not** explicitly stored as a residual; it is
        reconstructed inside the backward differentiator from the problem data
        and solution. This trades a small amount of recomputation for reduced
        memory usage.
        
        Args:
            *dynamic_vals: Variable-length argument list of JAX arrays.
                Same as in ``_solver_dynamic_vjp_mode``: the dynamic QP ingredients
                in ``_dynamic_keys`` order.
        
        Returns:
            A tuple ``(primal_out, residuals)`` where:
            
                - ``primal_out`` (QPOutput): The solution dictionary containing
                ``x``, ``lam``, and ``mu`` as JAX arrays. This is the value
                returned to the user and used in the forward pass.
                
                - ``residuals`` (Tuple[jax.Array, ...]): A flat tuple of JAX arrays
                saved for the backward pass. The layout is:
                
                    * First ``len(_dynamic_keys)`` arrays: The original dynamic
                    inputs (P, q, A, b, G, h) in order.
                    * Next 3 arrays: The solution (x, lam, mu) in that order.
                    
                This layout must match the unpacking in ``_solver_dynamic_vjp_bwd``.
        
        Notes:
            - The residuals are treated as non-differentiable constants by JAX.
            - All arrays in residuals are JAX arrays, but they will be converted
            to numpy inside the backward callback.
            - The total number of residuals is ``len(_dynamic_keys) + 3``.
            - This function is **not** meant to be called directly by users; it's
            part of the ``custom_vjp`` machinery.
        """
        result = pure_callback(
            _solve_qp, 
            _fwd_shapes, 
            *dynamic_vals,
            vmap_method="sequential"
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
        """Backward pass for custom VJP, computing parameter cotangents.
        
        This function is called by JAX during the backward pass of reverse-mode
        differentiation. It receives the residuals saved in the forward pass and
        the cotangents of the outputs, and computes the cotangents (gradients)
        of all dynamic input parameters.
        
        The backward computation proceeds in three steps:
            1. Unpack residuals into dynamic inputs and forward solution
            2. Pass these along with output cotangents to the numpy-side
            adjoint differentiator ``_diff_reverse`` via ``pure_callback``
            3. Return the computed parameter cotangents in the correct order
        
        The adjoint differentiation exploits the symmetry of the KKT matrix:
        solving M·v = g (where g are the output cotangents) yields adjoint
        variables v, from which parameter cotangents are derived analytically.
        
        Args:
            residuals: A flat tuple of JAX arrays saved from the forward pass.
                Layout matches the output of ``_solver_dynamic_vjp_fwd``:
                
                - First ``len(_dynamic_keys)`` arrays: Dynamic inputs in order
                - Next 3 arrays: Solution (x, lam, mu) in that order
                
            g: A dictionary mapping output names to their cotangents.
                Contains exactly three keys:
                
                - ``"x"``:   Cotangent of primal solution.
                Shape ``(n_var,)`` if unbatched, or ``(batch_size, n_var)`` if batched.
                - ``"lam"``: Cotangent of inequality duals.
                Shape ``(n_ineq,)`` if unbatched, or ``(batch_size, n_ineq)`` if batched.
                - ``"mu"``:  Cotangent of equality duals.
                Shape ``(n_eq,)`` if unbatched, or ``(batch_size, n_eq)`` if batched.
        
        Returns:
            Tuple[jax.Array, ...]: A tuple of cotangents, one per dynamic input,
            in the same order as ``_dynamic_keys``. The shape of each cotangent
            matches the shape of the corresponding input:
            
                - For matrix inputs (P, A, G): same shape as the primal input
                - For vector inputs (q, b, h): same shape as the primal input
            
            When batched, the cotangents have shape ``(batch_size, ...)`` matching
            the batch dimension of the input cotangents.
        
        Raises:
            ValueError: If the residuals tuple has an unexpected length or if the
                cotangent dictionary is missing required keys.
        
        Notes:
            - The backward pass handles batching automatically: if the input
            cotangents have a batch dimension, the adjoint system is solved
            once with a matrix right-hand side.
            - Fixed elements (provided at setup) **do not** receive cotangents
            and are not present in the output tuple.
            - The active set from the forward solve is reconstructed inside the
            numpy differentiator; it is not stored as a residual to save memory.
            - This function is **not** meant to be called directly by users; it's
            part of the ``custom_vjp`` machinery.
        """

        # unpack cotangents
        g_x   = g["x"]
        g_lam = g["lam"]
        g_mu  = g["mu"]

        # Layout: *prob_arrays, x, lam, mu, active, g_x, g_lam, g_mu
        grad_vals = pure_callback(
            _diff_reverse,
            _vjp_bwd_shapes,
            *residuals, g_x, g_lam, g_mu,
            vmap_method="expand_dims"
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
        """Solve a (possibly partially fixed) quadratic program.
        
        This function is the main entry point for solving QPs after setup.
        It accepts the runtime QP ingredients (those not fixed at setup) as
        keyword arguments, solves the problem using the configured numpy
        backend, and returns the solution as JAX arrays.
        
        The full set of possible QP ingredients is ``{P, q, A, b, G, h}``.
        Any subset may be fixed at setup time via ``fixed_elements``; the
        remainder must be supplied here at call time. Fixed elements are
        stored as numpy arrays and never enter JAX's trace, which avoids
        unnecessary conversions but means their derivatives are always zero.
        
        **Warm-starting:**
        
        If a warmstart initial guess is provided, it is converted to numpy and
        passed to the underlying solver. The warmstart is **cleared after each
        solve** to prevent accidental reuse across calls. Warmstart is **not**
        differentiated through and does not affect computed derivatives.
        
        **Differentiation Behavior:**
        
        The differentiation mode (JVP or VJP) is selected at setup time:
        
            - ``"kkt_fwd"``: Forward-mode. The solver is compatible with
            ``jax.jvp`` and ``jax.vmap(jax.jvp(...))``. Efficient for
            problems with few inputs relative to outputs.
            
            - ``"kkt_rev"``: Reverse-mode. The solver is compatible with
            ``jax.grad``, ``jax.value_and_grad``, and ``jax.vjp``.
            Efficient for problems with many inputs and few outputs.
        
        **Batching:**
        
        The forward solve itself does **not** support batching — if batched
        inputs are provided (with an extra leading dimension), only the first
        batch element is used. This is because the primal solution must be
        identical across the batch for differentiation to be meaningful.
        
        However, differentiation **does** support batching:
        
            - In JVP mode, if tangents have a batch dimension, the linearized
            KKT system is solved once with a matrix right-hand side.
            - In VJP mode, if output cotangents have a batch dimension, the
            adjoint system is solved once with a matrix right-hand side.
        
        **Debug Mode:**
        
        If ``debug=True`` was set in the constructor options, additional
        checks are performed:
        
            - Verification that all dynamic ingredients are provided
            - Shape validation against declared problem dimensions
            - Warnings when runtime keys overlap with fixed elements
        
        Args:
            warmstart: Optional initial guess for the primal solution.
                Shape ``(n_var,)``. Passed to the underlying QP solver to
                speed up convergence. **Not differentiated through.**
                Cleared after each solve.
                
            **runtime: QP ingredients as JAX arrays. Must cover at least every
                key in ``_dynamic_keys`` (i.e., every ingredient not already
                fixed at setup). Keys that overlap with ``fixed_elements`` are
                ignored with a warning (in debug mode).
                
                Valid keys and their required shapes:
                
                - ``P``: Cost matrix, shape ``(n_var, n_var)``. Must be
                positive semidefinite.
                - ``q``: Linear cost vector, shape ``(n_var,)``.
                - ``A``: Equality constraint matrix, shape ``(n_eq, n_var)``.
                Required if ``n_eq > 0``.
                - ``b``: Equality RHS vector, shape ``(n_eq,)``.
                Required if ``n_eq > 0``.
                - ``G``: Inequality constraint matrix, shape ``(n_ineq, n_var)``.
                Required if ``n_ineq > 0``.
                - ``h``: Inequality RHS vector, shape ``(n_ineq,)``.
                Required if ``n_ineq > 0``.
        
        Returns:
            QPOutput: A dictionary containing the solution as JAX arrays:
            
                - ``x``:   Primal solution vector.
                Shape ``(n_var,)`` (unbatched) or broadcast to ``(batch_size, n_var)``
                when differentiating with batched tangents/cotangents.
                
                - ``lam``: Inequality dual variables (Lagrange multipliers for
                inequality constraints). Shape ``(n_ineq,)`` or broadcast to
                ``(batch_size, n_ineq)`` when batched.
                
                - ``mu``:  Equality dual variables (Lagrange multipliers for
                equality constraints). Shape ``(n_eq,)`` or broadcast to
                ``(batch_size, n_eq)`` when batched.
        
        Raises:
            ValueError: If any dynamic ingredient is missing (in debug mode),
                or if the warmstart array has incorrect shape.
            Warning: In debug mode, warns if runtime keys overlap with fixed
                elements (those inputs are ignored).
        
        Example:
            >>> # Basic usage
            >>> sol = solver(P=P_mat, q=q_vec, A=A_mat, b=b_vec)
            >>> x_opt, lam_opt, mu_opt = sol["x"], sol["lam"], sol["mu"]
            
            >>> # With warm-starting
            >>> sol = solver(warmstart=x0, P=P_mat, q=q_vec)
            
            >>> # Gradient computation (reverse-mode)
            >>> def loss(q):
            ...     sol = solver(q=q, P=P_fixed)  # P is fixed at setup
            ...     return sol["x"].sum()
            >>> grad_q = jax.grad(loss)(q)
            
            >>> # Forward-mode differentiation
            >>> primals = (q,)
            >>> tangents = (dq,)
            >>> sol, sol_dot = jax.jvp(lambda q: solver(q=q, P=P_fixed), primals, tangents)
        
        Notes:
            - The solver is **not** jit-compilable. Use it inside JAX-transformed
            functions where the callback overhead is acceptable.
            - Fixed elements are **never** traced by JAX, so they cannot be
            differentiated through. To differentiate through an element,
            provide it at runtime instead of fixing it at setup.
            - The active set from the QP solution is used internally but never
            returned to the user.
            - When batching is detected during differentiation, the primal
            solution is broadcast (not tiled) to match the batch dimension,
            as the primal solution is identical for all batch elements.
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

    solver.timings = _timings

    return solver