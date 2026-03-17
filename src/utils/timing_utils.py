"""
Structured timing recorder for the dense QP solver.

Collects ``dict[str, float]`` timing records emitted by the internal
callbacks (``_solve_qp``, ``_kkt_diff``, ``_kkt_vjp``, etc.) and
provides formatted summary tables with per-key statistics across
multiple calls.

Usage
-----
The recorder is automatically attached to the solver callable as
``solver.timings`` when ``options["debug"]`` is ``True``::

    solver = setup_dense_solver(n_var=10, n_ineq=5, n_eq=3)

    for x0 in batch:
        solver(P=P, q=q, A=A, b=beq(x0), G=G, h=h)

    # Print summary table
    print(solver.timings.summary())

    # Reset for next batch
    solver.timings.reset()

The ``summary()`` method returns a string with one section per
function, showing count / mean / std / min / max for every timing
key in that function.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


class TimingRecorder:
    """Collects and summarises structured timing records.

    Each record is a ``dict[str, float]`` tagged with a function name
    (e.g. ``"_solve_qp"``, ``"_kkt_vjp"``).  The recorder stores
    every record and can produce aggregate statistics on demand.

    Attributes:
        records: Mapping from function name to a list of timing dicts.
            Each dict maps timing-key names (e.g. ``"solve"``,
            ``"lin_solve"``) to elapsed seconds.
        enabled: When ``False``, ``record()`` is a no-op.  This lets
            the solver keep the call without an ``if`` guard.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.records: defaultdict[str, list[dict[str, float]]] = (
            defaultdict(list)
        )
        self.enabled = enabled

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    def record(self, func_name: str, t: dict[str, float]) -> None:
        """Append a timing dict for *func_name*.

        Args:
            func_name: Identifier for the internal function
                (e.g. ``"_solve_qp"``, ``"_kkt_vjp"``).
            t: Timing dict mapping key names to elapsed seconds.
        """
        if self.enabled:
            self.records[func_name].append(t)

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    @property
    def call_count(self) -> dict[str, int]:
        """Number of recorded calls per function."""
        return {k: len(v) for k, v in self.records.items()}

    def get_raw(
        self, func_name: Optional[str] = None,
    ) -> dict[str, list[dict[str, float]]]:
        """Return the raw records, optionally filtered by function.

        Args:
            func_name: If given, return only records for this function
                (wrapped in a single-key dict).  If ``None``, return
                all records.

        Returns:
            ``{func_name: [t_dict, ...], ...}``
        """
        if func_name is not None:
            return {func_name: list(self.records.get(func_name, []))}
        return dict(self.records)

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def stats(
        self,
        func_name: Optional[str] = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Compute per-key statistics for each function.

        Args:
            func_name: If given, compute stats only for this function.

        Returns:
            Nested dict::

                {func_name: {key: {"count": …, "mean": …,
                 "std": …, "min": …, "max": …, "sum": …}}}
        """
        targets = (
            {func_name: self.records[func_name]}
            if func_name is not None
            else self.records
        )

        result: dict[str, dict[str, dict[str, float]]] = {}

        for fname, entries in targets.items():
            if not entries:
                continue

            # Collect all keys that appear in any entry
            all_keys: list[str] = []
            seen: set[str] = set()
            for entry in entries:
                for k in entry:
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)

            key_stats: dict[str, dict[str, float]] = {}
            for k in all_keys:
                vals = np.array(
                    [e[k] for e in entries if k in e], dtype=np.float64,
                )
                if vals.size == 0:
                    continue
                key_stats[k] = {
                    "count": float(vals.size),
                    "mean":  float(np.mean(vals)),
                    "std":   float(np.std(vals)),
                    "min":   float(np.min(vals)),
                    "max":   float(np.max(vals)),
                    "sum":   float(np.sum(vals)),
                }
            result[fname] = key_stats

        return result

    # -----------------------------------------------------------------
    # Formatting
    # -----------------------------------------------------------------

    @staticmethod
    def _fmt_seconds(s: float) -> str:
        """Format seconds into a human-readable string.

        Uses µs for values < 1 ms, ms for values < 1 s, and s
        otherwise.
        """
        if s < 1e-3:
            return f"{s * 1e6:8.1f} µs"
        elif s < 1.0:
            return f"{s * 1e3:8.3f} ms"
        else:
            return f"{s:8.4f}  s"

    def summary(
        self,
        func_name: Optional[str] = None,
    ) -> str:
        """Return a formatted multi-section summary table.

        Each section corresponds to one function and lists every
        timing key with count, mean, std, min, max columns.

        The ``total`` key (if present) is always printed last and
        separated by a thin rule, so the per-step breakdown is
        visually distinct from the overall wall time.

        Args:
            func_name: If given, summarise only this function.

        Returns:
            A ready-to-print string.
        """
        all_stats = self.stats(func_name)

        if not all_stats:
            return "(no timing records)"

        sections: list[str] = []

        for fname, key_stats in all_stats.items():
            n_calls = len(self.records[fname])
            header = f"  {fname}  ({n_calls} calls)"
            sep = "  " + "─" * 90

            lines: list[str] = [sep, header, sep]

            # Column header
            lines.append(
                f"  {'key':<24s}"
                f"{'count':>6s}"
                f"{'mean':>12s}"
                f"{'std':>12s}"
                f"{'min':>12s}"
                f"{'max':>12s}"
                f"{'sum':>12s}"
            )
            lines.append("  " + "─" * 86)

            # Sort keys: "total" last, rest in insertion order
            ordered_keys = [k for k in key_stats if k != "total"]
            has_total = "total" in key_stats

            for k in ordered_keys:
                s = key_stats[k]
                lines.append(
                    f"  {k:<24s}"
                    f"{int(s['count']):>6d}"
                    f"{self._fmt_seconds(s['mean']):>12s}"
                    f"{self._fmt_seconds(s['std']):>12s}"
                    f"{self._fmt_seconds(s['min']):>12s}"
                    f"{self._fmt_seconds(s['max']):>12s}"
                    f"{self._fmt_seconds(s['sum']):>12s}"
                )

            if has_total:
                lines.append("  " + "─" * 86)
                s = key_stats["total"]
                lines.append(
                    f"  {'total':<24s}"
                    f"{int(s['count']):>6d}"
                    f"{self._fmt_seconds(s['mean']):>12s}"
                    f"{self._fmt_seconds(s['std']):>12s}"
                    f"{self._fmt_seconds(s['min']):>12s}"
                    f"{self._fmt_seconds(s['max']):>12s}"
                    f"{self._fmt_seconds(s['sum']):>12s}"
                )

            lines.append(sep)
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{k}: {len(v)}" for k, v in self.records.items()
        )
        return f"TimingRecorder({counts or 'empty'})"

    # -----------------------------------------------------------------
    # Housekeeping
    # -----------------------------------------------------------------

    def reset(self, func_name: Optional[str] = None) -> None:
        """Clear recorded data.

        Args:
            func_name: If given, clear only records for this function.
                If ``None``, clear everything.
        """
        if func_name is not None:
            self.records.pop(func_name, None)
        else:
            self.records.clear()