"""
solver_sparse/_differentiators.py
=================================
Factory functions for sparse KKT differentiators.

These are thin wrappers that instantiate a
:class:`DifferentiatorBackend` via the backend registry, call its
:meth:`setup`, and return the bound :meth:`differentiate_fwd` or
:meth:`differentiate_rev` method as the callable closure.

The actual algorithms live in
:mod:`jaxsparrow._utils._differentiator_backends`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Protocol

from numpy import ndarray

from jaxsparrow._types_common import (
    SolverOutputNP, 
    SolverDiffOutFwdNP, 
    SolverDiffOutRevNP
)

from jaxsparrow._solver_sparse._types import (
    SparseIngredientsNP,
    SparseIngredientsTangentsNP
)
from jaxsparrow._solver_sparse._converters import SparsityInfo
from jaxsparrow._options_common import DifferentiatorOptions
from jaxsparrow._utils._parsing_utils import parse_options
from jaxsparrow._solver_sparse._options import (
    DIFF_OPTIONS_DEFAULTS,
    DEFAULT_DIFF_BACKEND,
)
from jaxsparrow._utils._diff_backends import (
    DifferentiatorBackend,
    get_differentiator_backend,
)

# Ensure the sparse backends are registered on import
import jaxsparrow._solver_sparse._sparse_diff_backend  # noqa: F401 #import: ignore

# ── Callable protocols for the returned closures ─────────────────────

class SparseKKTDifferentiatorFwd(Protocol):
    """Signature of the closure returned by
    :func:`create_sparse_kkt_differentiator_fwd`."""

    def __call__(
        self,
        sol_np: SolverOutputNP,
        dyn_primals_np: SparseIngredientsNP,
        dyn_tangents_np: SparseIngredientsTangentsNP,
        batch_size: int,
    ) -> tuple[SolverDiffOutFwdNP, dict[str, float]]: ...


class SparseKKTDifferentiatorRev(Protocol):
    """Signature of the closure returned by
    :func:`create_sparse_kkt_differentiator_rev`."""

    def __call__(
        self,
        dyn_primals_np: SparseIngredientsNP,
        x_np: ndarray,
        lam_np: ndarray,
        mu_np: ndarray,
        g_x: ndarray,
        g_lam: ndarray,
        g_mu: ndarray,
        batch_size: int,
    ) -> tuple[SolverDiffOutRevNP, dict[str, float]]: ...

def _resolve_backend_defaults(
    options: Optional[DifferentiatorOptions],
) -> tuple[str, DifferentiatorOptions]:
    """Determine the backend name and its matching defaults.

    The backend is read from ``options["backend"]`` if present,
    otherwise ``DEFAULT_DIFF_BACKEND`` is used.  The returned defaults
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


def create_sparse_kkt_differentiator_fwd(
    n_var: int,
    n_eq: int,
    n_ineq: int,
    options: Optional[DifferentiatorOptions] = None,
    fixed_elements: Optional[SparseIngredientsNP] = None,
) -> SparseKKTDifferentiatorFwd:
    """Create a forward-mode (JVP) differentiator for sparse problems.

    Instantiates the differentiator backend specified by the
    ``"backend"`` key in *options* (default: ``"sparse_kkt"``),
    calls :meth:`setup` with the fixed elements, and returns the bound
    :meth:`differentiate_fwd` method.

    Each backend has its own default options; user-supplied keys
    override the backend-specific defaults.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options. The ``"backend"`` key
            selects which :class:`DifferentiatorBackend` to use.
            Remaining keys are merged with that backend's defaults.
        fixed_elements: Ingredients constant across calls.

    Returns:
        A callable matching :class:`SparseKKTDifferentiatorFwd`.
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


def create_sparse_kkt_differentiator_rev(
    n_var: int,
    n_eq: int,
    n_ineq: int,
    options: Optional[DifferentiatorOptions] = None,
    fixed_elements: Optional[SparseIngredientsNP] = None,
    dynamic_keys: Optional[Sequence[str]] = None,
    sparsity_info: Optional[SparsityInfo] = None,
) -> SparseKKTDifferentiatorRev:
    """Create a reverse-mode (VJP) differentiator for sparse problems.

    Instantiates the differentiator backend specified by the
    ``"backend"`` key in *options* (default: ``"sparse_kkt"``),
    calls :meth:`setup` with the fixed elements, dynamic keys, and
    sparsity info, and returns the bound :meth:`differentiate_rev`
    method.

    Each backend has its own default options; user-supplied keys
    override the backend-specific defaults.

    Args:
        n_var: Number of decision variables.
        n_eq: Number of equality constraints (zero if none).
        n_ineq: Number of inequality constraints (zero if none).
        options: Differentiator options. The ``"backend"`` key
            selects which :class:`DifferentiatorBackend` to use.
            Remaining keys are merged with that backend's defaults.
        fixed_elements: Ingredients constant across calls.
        dynamic_keys: If provided, gradients are computed only for
            these keys.  ``None`` means all keys.
        sparsity_info: Per-key sparsity info from BCOO patterns.

    Returns:
        A callable matching :class:`SparseKKTDifferentiatorRev`.
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
        sparsity_info=sparsity_info,
    )
    return backend.differentiate_rev