"""
solver_dense/_differentiators.py
================================
Factory functions for dense KKT differentiators.

These are thin wrappers that instantiate a
:class:`DifferentiatorBackend` via the backend registry, call its
:meth:`setup`, and return the bound :meth:`differentiate_fwd` or
:meth:`differentiate_rev` method as the callable closure.

The ``"backend"`` key in the *options* dict selects the backend
implementation.  Each backend has its own default options defined
in :data:`DIFF_OPTIONS_DEFAULTS`:

- ``"dense_kkt"`` — standard KKT differentiator
  (see :mod:`jaxsparrow._solver_dense._dense_diff_backend`)
- ``"dense_dbd"`` — regularised Differentiable-by-Design
  differentiator
  (see :mod:`jaxsparrow._solver_dense._dense_dbd_diff_backend`)
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
    SolverDiffOutRevNP,
)
from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._solver_dense._options import (
    DIFF_OPTIONS_DEFAULTS,
    DEFAULT_DIFF_BACKEND
)
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    get_differentiator_backend,
)

# Ensure both dense backends are registered on import
import jaxsparrow._solver_dense._dense_diff_backend  # noqa: F401


# ----------------------------------------------------------------------
# Callable protocols for the returned closures
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _resolve_backend_defaults(
    options: Optional[DifferentiatorOptions],
) -> tuple[str, DifferentiatorOptions]:
    """Determine the backend name and its matching defaults.

    The backend is read from ``options["backend"]`` if present,
    otherwise ``_DEFAULT_BACKEND`` is used.  The returned defaults
    dict is the one registered in :data:`DIFF_OPTIONS_DEFAULTS` for
    that backend.

    Returns:
        ``(backend_name, default_options)``

    Raises:
        KeyError: If the resolved backend name has no entry in
            :data:`DIFF_OPTIONS_DEFAULTS`.
    """
    if options is not None and "backend" in options:
        backend_name: str = options["backend"]
    else:
        backend_name = DEFAULT_DIFF_BACKEND

    if backend_name not in DIFF_OPTIONS_DEFAULTS:
        raise KeyError(
            f"Unknown differentiator backend {backend_name!r}.  "
            f"Available backends: {sorted(DIFF_OPTIONS_DEFAULTS)}"
        )

    return backend_name, DIFF_OPTIONS_DEFAULTS[backend_name]


# ----------------------------------------------------------------------
# Factory functions
# ----------------------------------------------------------------------

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

    Each backend has its own default options; user-supplied keys
    override the backend-specific defaults.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options.  The ``"backend"`` key
            selects which :class:`DifferentiatorBackend` to use.
            Remaining keys are merged with that backend's defaults.
        fixed_elements: Ingredients constant across calls.

    Returns:
        A callable matching :class:`DenseKKTDifferentiatorFwd`.
    """
    backend_name, defaults = _resolve_backend_defaults(options)
    options_parsed = parse_options(options, defaults)

    backend: DifferentiatorBackend = get_differentiator_backend(
        backend_name,
        n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
        options=options_parsed,
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

    Each backend has its own default options; user-supplied keys
    override the backend-specific defaults.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options.  The ``"backend"`` key
            selects which :class:`DifferentiatorBackend` to use.
            Remaining keys are merged with that backend's defaults.
        fixed_elements: Ingredients constant across calls.
        dynamic_keys: If provided, gradients are computed only for
            these keys.  ``None`` means all keys.

    Returns:
        A callable matching :class:`DenseKKTDifferentiatorRev`.
    """
    backend_name, defaults = _resolve_backend_defaults(options)
    options_parsed = parse_options(options, defaults)

    backend: DifferentiatorBackend = get_differentiator_backend(
        backend_name,
        n_var=n_var, n_eq=n_eq, n_ineq=n_ineq,
        options=options_parsed,
    )
    backend.setup(
        fixed_elements=fixed_elements,
        dynamic_keys=dynamic_keys,
    )
    return backend.differentiate_rev