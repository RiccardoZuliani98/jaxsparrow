from typing import get_type_hints, get_origin, get_args, Literal, Dict, Any, Optional
import numpy as np

# Import the registries (adjust paths as needed)
from jaxsparrow._solver_dense._options import (
    ALL_DENSE_DIFF_OPTIONS, ALL_DENSE_SOLVER_OPTIONS
)
from jaxsparrow._solver_sparse._options import (
    ALL_SPARSE_DIFF_OPTIONS, ALL_SPARSE_SOLVER_OPTIONS
)

def get_options_info(
    cls: type,
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract structured information about a TypedDict options class.

    For each field, the returned dictionary contains:
        - 'type': string representation of the type annotation
        - 'allowed': list of allowed values if the type is Literal, else None
        - 'default': the default value if provided in `defaults`, else None

    Args:
        cls: The TypedDict class (e.g., DenseKKTDiffOptionsFull).
        defaults: Optional dict mapping field names to default values.
                  Usually the corresponding `DEFAULT_*` dict.

    Returns:
        A dictionary mapping each field name to its info dict.
    """
    hints = get_type_hints(cls)
    info = {}
    for field, typ in hints.items():
        origin = get_origin(typ)
        
        # Handle Literal specially to get allowed values
        if origin is Literal:
            allowed = list(get_args(typ))
            type_str = f"Literal{tuple(allowed)}"
        else:
            allowed = None
            # Convert type to a readable string
            if hasattr(typ, '__name__'):
                type_str = typ.__name__
            else:
                type_str = str(typ).replace('typing.', '')

        # Get default if provided
        default = None
        if defaults and field in defaults:
            default = defaults[field]

        info[field] = {
            'type': type_str,
            'allowed': allowed,
            'default': default,
        }
    return info

def print_options(
    cls: type,
    defaults: Optional[Dict[str, Any]] = None,
    descriptions: Optional[Dict[str, str]] = None,
    width: int = 80
) -> None:
    """
    Print a human-readable description of the options.

    Displays each field with its type, allowed values (if Literal), default
    (if provided), and a short description (if provided).

    Args:
        cls: The TypedDict class (e.g., DenseKKTDiffOptionsFull).
        defaults: Optional dict of default values.
        descriptions: Optional dict mapping field names to descriptions.
        width: Maximum line width for formatting (currently unused but kept
               for potential future use).
    """
    info = get_options_info(cls, defaults)
    print(f"Options for {cls.__name__}:")
    print("-" * 80)

    for field, data in info.items():
        # Field name and type
        line = f"{field}: {data['type']}"
        if data['allowed']:
            line += f" ∈ {data['allowed']}"
        if data['default'] is not None:
            default = data['default']
            # Format default nicely
            if isinstance(default, type) and hasattr(default, '__name__'):
                default_str = default.__name__
            elif isinstance(default, str):
                default_str = f"'{default}'"
            else:
                default_str = str(default)
            line += f"  (default: {default_str})"
        print(line)

        # Print description if available
        if descriptions and field in descriptions:
            print(f"    {descriptions[field]}")

    print()

def show_options(
    type: Literal["solver","diff"], 
    mode: Literal["dense","sparse"], 
    backend: Literal[
        "dense_kkt",
        "dense_dbd",
        "sparse_dbd",
        "dense_kkt",
        "qpsolvers"
    ]
) -> None:
    """
    Display available options, defaults, and descriptions for a given backend.

    Args:
        type (str): "solver" or "diff".
        mode (str): "dense" or "sparse".
        backend (str): Backend name as used in the registries
                       (e.g., "dense_kkt", "dense_dbd", "qpsolvers").

    Raises:
        ValueError: If unknown type, mode, or backend is provided.
    """
    # Select the appropriate registry based on type and mode
    if type == "diff":
        if mode == "dense":
            registry = ALL_DENSE_DIFF_OPTIONS
        elif mode == "sparse":
            registry = ALL_SPARSE_DIFF_OPTIONS
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'dense' or 'sparse'.")
    elif type == "solver":
        if mode == "dense":
            registry = ALL_DENSE_SOLVER_OPTIONS
        elif mode == "sparse":
            registry = ALL_SPARSE_SOLVER_OPTIONS
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'dense' or 'sparse'.")
    else:
        raise ValueError(f"Unknown type: {type}. Must be 'solver' or 'diff'.")

    # Check if backend exists in the registry
    if backend not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown {type} backend '{backend}' for mode '{mode}'. "
            f"Available: {available}"
        )

    # Retrieve the entry
    entry = registry[backend]
    option_cls = entry["option"]
    defaults = entry.get("default")
    descriptions = entry.get("description")  # may be None

    # Print the options with descriptions
    print_options(option_cls, defaults, descriptions)