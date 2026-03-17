def fmt_times(t: dict[str, float]) -> str:
    """Format a timing dict into a single-line summary string.

    Args:
        t: Dict mapping stage names to elapsed seconds.

    Returns:
        Formatted string, e.g. ``"solve=1.2e-03  active=4.5e-05"``.
    """
    return "  ".join(f"{k}={v:.3e}s" for k, v in t.items())