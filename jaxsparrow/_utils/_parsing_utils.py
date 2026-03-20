"""
utils/_parsing_utils.py
=======================
Option-parsing helpers used throughout jaxsparrow.
"""

from typing import TypeVar, Optional, cast, Any
from collections.abc import Mapping

Full = TypeVar("Full", bound=Mapping[str, Any])
Partial = TypeVar("Partial", bound=Mapping[str, Any])


def parse_options(
    options: Optional[Partial],
    default: Full,
) -> Full:
    """Merge user-supplied options into a complete defaults dict.

    Returns *default* unchanged when *options* is ``None``. Otherwise
    overlays *options* on top of *default*, raising on any key not
    present in the defaults.

    Args:
        options: User-supplied partial options, or ``None`` to accept
            all defaults.
        default: Complete defaults dictionary. Its keys define the
            set of allowed option names.

    Returns:
        A new dictionary with the same type as *default*, containing
        all default values with any user overrides applied.

    Raises:
        TypeError: If *options* contains keys not present in
            *default*.
    """
    if options is None:
        return default

    allowed: set[str] = set(default.keys())
    unknown: set[str] = set(options) - allowed
    if unknown:
        raise TypeError(
            f"Unknown option key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )
    return cast(Full, {**default, **options})