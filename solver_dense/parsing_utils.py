from typing import TypeVar, Optional, cast, Any
from collections.abc import Mapping

Full = TypeVar("Full", bound=Mapping[str, Any])
Partial = TypeVar("Partial", bound=Mapping[str, Any])

def parse_options(
    options: Optional[Partial],
    default: Full,
) -> Full:
    if options is None:
        return default

    allowed = set(default.keys())
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(
            f"Unknown option key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )
    return cast(Full, {**default, **options})