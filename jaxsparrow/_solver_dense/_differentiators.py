"""
solver_dense/_differentiators.py
================================
Factory functions for dense KKT differentiators.

These are thin wrappers that instantiate a
:class:`DifferentiatorBackend` via the backend registry, call its
:meth:`setup`, and return the bound :meth:`differentiate_fwd` or
:meth:`differentiate_rev` method as the callable closure.

The actual algorithm lives in
:mod:`jaxsparrow._solver_dense._dense_kkt_backend`.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence
from numpy import ndarray

from jaxsparrow._solver_dense._types import (
    DenseIngredientsNP,
    DenseIngredientsTangentsNP,
)
from jaxsparrow._types_common import (
    SolverOutputNP, 
    SolverDiffOutFwdNP, 
    SolverDiffOutRevNP
)
from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._solver_dense._options import DEFAULT_DIFF_OPTIONS
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    get_differentiator_backend,
)

# Ensure the dense backend is registered on import
import jaxsparrow._solver_dense._dense_diff_backend  # noqa: F401


# ── Callable protocols for the returned closures ─────────────────────

class DenseKKTDifferentiatorFwd(Protocol):
    """Signature of the closure returned by
    :func:`create_dense_kkt_differentiator_fwd`."""

    def __call__(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: DenseIngredientsNP,
        dyn_tangents_np: DenseIngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]: ...


class DenseKKTDifferentiatorRev(Protocol):
    """Signature of the closure returned by
    :func:`create_dense_kkt_differentiator_rev`."""

    def __call__(
        self,
        dyn_primals_np: DenseIngredientsNP,
        x_np: ndarray,
        lam_np: ndarray,
        mu_np: ndarray,
        g_x: ndarray,
        g_lam: ndarray,
        g_mu: ndarray,
        batch_size: int,
    ) -> tuple[SolverDiffOutRevNP, dict[str, float]]: ...


# ── Factory functions ────────────────────────────────────────────────

def create_dense_kkt_differentiator_fwd(
    n_var: int,
    n_eq: int,
    n_ineq: int,
    options: Optional[DifferentiatorOptions] = None,
    fixed_elements: Optional[DenseIngredientsNP] = None,
) -> DenseKKTDifferentiatorFwd:
    """Create a forward-mode (JVP) differentiator for dense problems.

    Instantiates the differentiator backend specified by the
    ``"backend"`` key in *options* (default: ``"dense_kkt"``),
    calls :meth:`setup`, and returns the bound
    :meth:`differentiate_fwd` method.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options. The ``"backend"`` key
            selects which :class:`DifferentiatorBackend` to use.
            Remaining keys are passed to the backend constructor.
        fixed_elements: Ingredients constant across calls.

    Returns:
        A callable matching :class:`DenseKKTDifferentiatorFwd`.
    """
    options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
    backend_name: str = options_parsed.get("backend")

    backend: DifferentiatorBackend = get_differentiator_backend(
        backend_name,
        n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
        options=options,
    )
    backend.setup(fixed_elements=fixed_elements)
    return backend.differentiate_fwd


def create_dense_kkt_differentiator_rev(
    n_var: int,
    n_eq: int,
    n_ineq: int,
    options: Optional[DifferentiatorOptions] = None,
    fixed_elements: Optional[DenseIngredientsNP] = None,
    dynamic_keys: Optional[Sequence[str]] = None,
) -> DenseKKTDifferentiatorRev:
    """Create a reverse-mode (VJP) differentiator for dense problems.

    Instantiates the differentiator backend specified by the
    ``"backend"`` key in *options* (default: ``"dense_kkt"``),
    calls :meth:`setup` with dynamic keys, and returns the bound
    :meth:`differentiate_rev` method.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options. The ``"backend"`` key
            selects which :class:`DifferentiatorBackend` to use.
            Remaining keys are passed to the backend constructor.
        fixed_elements: Ingredients constant across calls.
        dynamic_keys: If provided, gradients are computed only for
            these keys. ``None`` means all keys.

    Returns:
        A callable matching :class:`DenseKKTDifferentiatorRev`.
    """
    options_parsed = parse_options(options, DEFAULT_DIFF_OPTIONS)
    backend_name: str = options_parsed.get("backend", "dense_kkt")

    backend: DifferentiatorBackend = get_differentiator_backend(
        backend_name,
        n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
        options=options,
    )
    backend.setup(
        fixed_elements=fixed_elements,
        dynamic_keys=dynamic_keys,
    )
    return backend.differentiate_rev