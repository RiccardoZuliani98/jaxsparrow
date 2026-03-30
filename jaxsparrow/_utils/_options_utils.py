from jaxsparrow._solver_dense._options import ALL_DENSE_DIFF_OPTIONS, ALL_DENSE_SOLVER_OPTIONS
from jaxsparrow._solver_sparse._options import ALL_SPARSE_DIFF_OPTIONS, ALL_SPARSE_SOLVER_OPTIONS

from typing import Literal, get_origin, get_args, get_type_hints, Dict, Any


def get_options_info(cls, defaults: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract structured info about a TypedDict options class."""
    hints = get_type_hints(cls)
    info = {}
    for field, typ in hints.items():
        origin = get_origin(typ)
        allowed = None
        if origin is Literal:
            allowed = list(get_args(typ))

        if hasattr(typ, '__name__'):
            type_str = typ.__name__
        elif origin is Literal:
            type_str = f"Literal{tuple(allowed)}"
        else:
            type_str = str(typ).replace('typing.', '')

        default = None
        if defaults and field in defaults:
            default = defaults[field]

        info[field] = {
            'type': type_str,
            'allowed': allowed,
            'default': default,
        }
    return info

def print_options(cls, defaults: Dict[str, Any] = None, width: int = 80):
    """Print a human-readable description of the options."""
    info = get_options_info(cls, defaults)
    print(f"Options for {cls.__name__}:")
    print("-" * width)

    for field, data in info.items():
        line = f"{field}: {data['type']}"
        if data['allowed']:
            line += f" ∈ {data['allowed']}"
        if data['default'] is not None:
            default = data['default']
            if isinstance(default, type) and hasattr(default, '__name__'):
                default_str = default.__name__
            elif isinstance(default, str):
                default_str = f"'{default}'"
            else:
                default_str = str(default)
            line += f"  (default: {default_str})"
        print(line)
    print()

def show_options(type_: str, mode: str, backend: str):
    """
    Display available options and defaults for a given solver/differentiator backend.

    Args:
        type_ (str): "solver" or "diff".
        mode (str): "dense" or "sparse".
        backend (str): Backend name (e.g., "dense_kkt", "dense_dbd", "qpsolvers").
    """
    # Select the appropriate registry based on type and mode
    if type_ == "diff":
        if mode == "dense":
            registry = ALL_DENSE_DIFF_OPTIONS
        elif mode == "sparse":
            registry = ALL_SPARSE_DIFF_OPTIONS
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'dense' or 'sparse'.")
    elif type_ == "solver":
        if mode == "dense":
            registry = ALL_DENSE_SOLVER_OPTIONS
        elif mode == "sparse":
            registry = ALL_SPARSE_SOLVER_OPTIONS
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'dense' or 'sparse'.")
    else:
        raise ValueError(f"Unknown type: {type_}. Must be 'solver' or 'diff'.")

    # Check if backend exists in the registry
    if backend not in registry:
        available = list(registry.keys())
        raise ValueError(f"Unknown {type_} backend '{backend}' for mode '{mode}'. Available: {available}")

    # Retrieve the option class and default instance
    entry = registry[backend]
    option_cls = entry["option"]
    defaults = entry["default"]

    # Print the options
    print_options(option_cls, defaults)