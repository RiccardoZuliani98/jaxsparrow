from jax import custom_jvp, ShapeDtypeStruct, pure_callback, Array
import jax.numpy as jnp
from time import perf_counter
import numpy as np
from qpsolvers import Problem, solve_problem
import jax
import logging
from typing import (
    Final, 
    TypedDict, 
    Dict, 
    Tuple, 
    TypeAlias, 
    Optional, 
    cast
)
from jax.experimental.sparse import BCOO
from numpy import ndarray

from parsing_utils import parse_options
from solver_dense_types import DenseProblemIngredients, DenseProblemIngredientsNP
from solver_dense_options import (

    DEFAULT_SOLVER_OPTIONS, 
    SolverOptions, 
    SolverOptionsFull
)


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
    _jvp_shapes = {
        "dx":       jax.ShapeDtypeStruct((n_var,),   _dtype),
        "dmu":      jax.ShapeDtypeStruct((n_ineq,),  _dtype),
        "dlam":     jax.ShapeDtypeStruct((n_eq,),    _dtype),
        "dactive":  jax.ShapeDtypeStruct((n_ineq,),  jax.dtypes.float0),
        "sol":      _fwd_shapes,
    }
    
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

    logger.info(f"Setting up QP with {n_var} variables, {n_eq} equalities, {n_ineq} inequalities.")
    logger.info(f"Fixed variables: {set(_fixed.keys()) or 'none'}")


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
        logger.info(f"problem_setup: {perf_counter() - start}")

        # Solve QP
        start = perf_counter()
        sol = solve_problem(prob, solver=_options_parsed["solver"])
        assert sol.found, "QP solver failed to find a solution."
        logger.info(f"solve: {perf_counter() - start}")

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

        logger.info(f"retrieve: {perf_counter() - start}")

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
        logger.info(f"active set determination: {perf_counter() - start}")

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

        logger.info(f"DenseQP total: {perf_counter() - t_start:.3e}s")

        return result


    #endregion

    # =================================================================
    # SETUP DIFFERENTIATOR
    # =================================================================

    #region

    def _kkt_diff(P, q, A, b, G, h, dP, dq, dA, db, dG, dh):

        # detect batching
        batched = P[0].ndim == 2

        # this function takes in jax.numpy arrays and matrices,
        # the tangent arguments dQ, dq, dF, df, dG, dg can be vectorized,
        # in which case they are passed

        logger.info("Hi, I am kkt differentiator and I am running.")

        if batched:

            assert set(elem.shape[0] for elem in (P, q, A, b, G, h)) == {1}
            
            assert P.shape[1:] == (n_var,n_var)
            assert q.shape[1:] == (n_var,)
            assert A.shape[1:] == (n_eq,n_var)
            assert b.shape[1:] == (n_eq,)
            assert G.shape[1:] == (n_ineq,n_var)
            assert h.shape[1:] == (n_ineq,)

            assert set(elem.ndim for elem in (dP, dA, dG)) == {3}
            assert set(elem.ndim for elem in (dq, db, dh)) == {2}
            
            assert dP.shape[1:] == (n_var,n_var)
            assert dq.shape[1:] == (n_var,)
            assert dA.shape[1:] == (n_eq,n_var)
            assert db.shape[1:] == (n_eq,)
            assert dG.shape[1:] == (n_ineq,n_var)
            assert dh.shape[1:] == (n_ineq,)
        
        else:

            assert P.shape == (n_var,n_var)
            assert q.shape == (n_var,)
            assert A.shape == (n_eq,n_var)
            assert b.shape == (n_eq,)
            assert G.shape == (n_ineq,n_var)
            assert h.shape == (n_ineq,)

            assert dP.shape == (n_var,n_var)
            assert dq.shape == (n_var,)
            assert dA.shape == (n_eq,n_var)
            assert db.shape == (n_eq,)
            assert dG.shape == (n_ineq,n_var)
            assert dh.shape == (n_ineq,)

        start = perf_counter()

        # convert to numpy and squeeze any additional dimension
        P_np, q_np, A_np, b_np, G_np, h_np, t_dict = _convert_qp_ingredients_to_numpy(P, q, A, b, G, h)

        # extra shape must be removed here
        assert P_np.shape == (n_var,n_var)
        assert q_np.shape == (n_var,)
        assert A_np.shape == (n_eq,n_var)
        assert b_np.shape == (n_eq,)
        assert G_np.shape == (n_ineq,n_var)
        assert h_np.shape == (n_ineq,)
        
        # Get primal/dual solution (forward eval)
        x_np, lam_np, mu_np, active_np, t_dict_solve = _solve_qp_numpy(P_np, q_np, A_np, b_np, G_np, h_np)

        assert x_np.shape == (n_var,)
        assert mu_np.shape == (n_eq,)
        assert lam_np.shape == (n_ineq,)
        assert active_np.shape == (n_ineq,)

        # Convert vectors
        start = perf_counter()
        dq_np = np.asarray(dq, dtype=dtype)
        db_np = np.asarray(db, dtype=dtype)
        dh_np = np.asarray(dh, dtype=dtype)
        t_dict["convert_vector_derivatives"] = perf_counter() - start

        # Convert matrices
        start = perf_counter()
        dP_np = np.asarray(dP, dtype=dtype)
        dA_np = np.asarray(dA, dtype=dtype)
        dG_np = np.asarray(dG, dtype=dtype)
        t_dict["convert_matrix_derivatives"] = perf_counter() - start
        
        if batched:

            assert set(elem.ndim for elem in (dP, dA, dG)) == {3}
            assert set(elem.ndim for elem in (dq, db, dh)) == {2}
            
            assert dP_np.shape[1:] == (n_var,n_var)
            assert dq_np.shape[1:] == (n_var,)
            assert dA_np.shape[1:] == (n_eq,n_var)
            assert db_np.shape[1:] == (n_eq,)
            assert dG_np.shape[1:] == (n_ineq,n_var)
            assert dh_np.shape[1:] == (n_ineq,)
        
        else:

            assert dP_np.shape == (n_var,n_var)
            assert dq_np.shape == (n_var,)
            assert dA_np.shape == (n_eq,n_var)
            assert db_np.shape == (n_eq,)
            assert dG_np.shape == (n_ineq,n_var)
            assert dh_np.shape == (n_ineq,)

        # differentiate
        start = perf_counter()

        # H_np = np.vstack((A_np, G_np[active_np,:]))  # (n_eq + n_ineq, n_var)
        # dL_np = dP_np @ x_np + dq_np + dA_np.T @ mu_np + dG_np[active_np,:].T @ lam_np[active_np]
        # rhs = np.concatenate([dL_np, dA_np @ x_np - db_np, dG_np[active_np,:] @ x_np - dh_np[active_np]])
        # n_h = H_np.shape[0]
        # lhs = np.block([[P_np, H_np.T],[H_np, np.zeros((n_h,n_h))]])
        
        H_np = np.vstack((A_np, G_np[active_np, :]))   # (n_eq + n_active, n_var)
        n_h  = H_np.shape[0]
        lhs  = np.block([[P_np, H_np.T], [H_np, np.zeros((n_h, n_h))]])

        if batched:

            # expected dimensions
            # dP_np.shape = (batch,n_var,n_var)
            # dq_np.shape == (batch,n_var,) 
            # dA_np.shape == (batch,n_eq,n_var)
            # db_np.shape == (batch,n_eq,)
            # dG_np.shape == (batch,n_ineq,n_var)
            # dh_np.shape == (batch,n_ineq,)
            # x_np.shape == (n_var,)
            # mu_np.shape == (n_eq,)
            # lam_np.shape == (n_ineq,)
            # active_np.shape == (n_ineq,)
            
            # note that batch dimension of each element may differ,
            # but that's not a problem, numpy will correctly broadcast
            # to the correct output dimension

            # all elements here are (batch, n_var)
            dL_np = (dP_np @ x_np
                     + dq_np
                     + dA_np.transpose(0, 2, 1) @ mu_np
                     + dG_np[:, active_np, :].transpose(0, 2, 1) @ lam_np[active_np])

            rhs_pieces = [
                dL_np,
                dA_np @ x_np - db_np,
                dG_np[:, active_np, :] @ x_np - dh_np[:, active_np],
            ]

            # determine batch size
            batch_size = max(p.shape[0] for p in rhs_pieces)

            # form rhs
            rhs = np.concatenate([
                np.broadcast_to(p, (batch_size, p.shape[1])) if p.shape[0] == 1 else p
                for p in rhs_pieces
            ], axis=1).T

            assert rhs.shape == (n_var + n_eq + int(np.sum(active_np)), batch_size)
        
        else:
            
            # expected dimensions
            # dP_np.shape = (n_var,n_var)
            # dq_np.shape == (n_var,) 
            # dA_np.shape == (n_eq,n_var)
            # db_np.shape == (n_eq,)
            # dG_np.shape == (n_ineq,n_var)
            # dh_np.shape == (n_ineq,)
            # x_np.shape == (n_var,)
            # mu_np.shape == (n_eq,)
            # lam_np.shape == (n_ineq,)
            # active_np.shape == (n_ineq,)

            dL_np = dP_np @ x_np + dq_np + dA_np.T @ mu_np + dG_np[active_np, :].T @ lam_np[active_np]

            assert dL_np.shape == (n_var,)

            rhs = np.hstack((dL_np, dA_np @ x_np - db_np, dG_np[active_np, :] @ x_np - dh_np[active_np]))

            assert rhs.shape == (n_var + n_eq + int(np.sum(active_np)), )

            batch_size = 0

        t_dict["diff_setup"] = perf_counter() - start
        start = perf_counter()
        sol = np.linalg.solve(lhs, -rhs)
        
        if batched:
            dx_np              = sol[:n_var, :].T             # (B, n_var)
            dmu_np             = sol[n_var : n_var + n_eq, :].T  # (B, n_eq)
            dlam_np            = np.zeros((batch_size, n_ineq))
            dlam_np[:, active_np] = sol[n_var + n_eq :, :].T  # (B, n_active)

            assert dx_np.shape == (batch_size,n_var)
            assert dmu_np.shape == (batch_size,n_eq)
            assert dlam_np.shape == (batch_size,n_ineq)

        else:
            dx_np               = sol[:n_var]
            dmu_np              = sol[n_var : n_var + n_eq]
            dlam_np             = np.zeros(n_ineq)
            dlam_np[active_np]  = sol[n_var + n_eq :]

            assert dx_np.shape == (n_var,)
            assert dmu_np.shape == (n_eq,)
            assert dlam_np.shape == (n_ineq,)

        t_dict["diff_lin_solve"] = perf_counter() - start

        # convert to jax and add extra first dimension if in batched mode
        res, t_sol_convert = _convert_solution_to_jax(x_np, lam_np, mu_np, active_np, batch_size, n_var, n_eq, n_ineq)

        t_full = perf_counter() - start

        if _options_parsed["verbose"]:
            logger.info(
                f"DenseQP jvp time {t_full:.3e}s -- "
                f"solve={t_dict_solve["solve"]:.3e}s  "
                f"active set={t_dict_solve["active"]:.3e}s  "
                f"retrieve={t_dict_solve["retrieve"]:.3e}s  "
                f"solve={t_dict_solve["solve"]:.3e}s  "
                f"problem setup={t_dict_solve["problem_setup"]:.3e}s  "
                f"conversion vectors={t_dict["convert_vectors"]:.3e}s  "
                f"conversion vector derivatives={t_dict["convert_vector_derivatives"]:.3e}s  "
                f"conversion matrices={t_dict["convert_matrices"]:.3e}s  "
                f"conversion matrix derivatives={t_dict["convert_matrix_derivatives"]:.3e}s  "
                f"conversion solution={t_sol_convert:.3e}"
                f"derivative setup={t_dict["diff_setup"]:.3e}"
                f"derivative linear solve={t_dict["diff_lin_solve"]:.3e}"
            )

        return dx_np, dmu_np, dlam_np, np.zeros(res["active"].shape, dtype=jax.dtypes.float0), res
    
    def _kkt_diff_callback(primals, tangents):
        return pure_callback(
            _kkt_diff,
            _jvp_shapes,
            *primals, 
            *tangents,
            vmap_method="expand_dims"
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
        dx, dmu, dlam, dactive, res = _kkt_diff_callback(primals_dict, tangents_dict)

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


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)

    epsilon = 1.0

    horizon = 30

    A = jnp.array([[1,1],[0,1]])
    B = jnp.array([[0],[1]])

    xmax = 5
    xmin = -5
    umax = 0.5
    umin = -0.5

    cost_state = 1
    cost_input = 0.1

    nx,nu = B.shape
    N = horizon

    nz = (N+1)*nx + N*nu

    # cost
    P = jnp.diag(
        jnp.hstack((
            jnp.ones((N+1)*nx) * cost_state,
            jnp.ones(N*nu) * cost_input
        ))
    )

    q = jnp.zeros(nz)

    # inequality constraints
    G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))

    h = jnp.hstack((
        jnp.ones((N+1)*nx)*xmax,
        jnp.ones(N*nu)*umax,
        -jnp.ones((N+1)*nx)*xmin,
        -jnp.ones(N*nu)*umin
    ))

    # subdiagonal shift matrix
    S = jnp.diag(jnp.ones(N), -1)

    # state part
    Ax = jnp.kron(jnp.eye(N+1), jnp.eye(nx)) + jnp.kron(S, -A)

    # input part
    Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
    Au = jnp.kron(Su, -B)

    Aeq = jnp.hstack((Ax, Au))

    # parameterized RHS
    def beq(x_init):
        return jnp.hstack((
            x_init,
            jnp.zeros(N*nx)
        ))

    neq = Aeq.shape[0]
    nineq = G.shape[0]

    x0 = jnp.array([-3.0,-1.0])
    dx0 = jnp.array([epsilon,0])


    ## FIRST SOLVER: everything passed at runtime
    solver = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq)
    solve_mpc = lambda x_init: solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)
    sol1 = solve_mpc(x0)
    sol2 = solve_mpc(x0+dx0)

    ## SECOND SOLVER: some stuff fixed at setup
    solver_fixed = setup_dense_solver(n_var=nz,n_ineq=nineq,n_eq=neq,fixed_elements={"P":P,"q":q})
    sol1_fixed = solver_fixed(P=P, q=q, A=Aeq, b=beq(x0), G=G, h=h)
    sol2_fixed = solver_fixed(A=Aeq, b=beq(x0), G=G, h=h)