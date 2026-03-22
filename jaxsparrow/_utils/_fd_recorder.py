"""
Finite-difference accuracy recorder for the dense QP solver.

Runs a central finite-difference check in pure NumPy every time a JVP
or VJP callback fires, then stores the per-output relative and absolute
errors.  The recorder is attached to the solver as ``solver.fd_check``
and produces formatted summary tables analogous to
:class:`TimingRecorder`.

Usage
-----
::

    solver = setup_dense_solver(
        n_var=10, n_ineq=5, n_eq=3,
        options={"fd_check": True},   # enable FD checking
    )

    for x0 in batch:
        _, tangents = jax.jvp(solve, (x0,), (dx0,))

    print(solver.fd_check.summary())
    solver.fd_check.reset()

Design notes
------------
* All computation happens in NumPy so it does **not** contribute to
  JAX tracing / compilation time and does not appear in
  ``TimingRecorder`` entries.
* The recorder re-uses the *same* NumPy sub-solver that the main
  solver already created — no extra solver setup cost.
* Central FD:  df/dp ≈ (f(p + ε·d) − f(p − ε·d)) / (2ε).
* For JVP the "direction" is the tangent vector; for VJP the check
  verifies  gᵀ·(df/dp)·d  for a random probe direction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Callable

import numpy as np
from numpy import ndarray

#TODO: check finite difference checker
class FiniteDifferenceRecorder:
    """Collect and summarise finite-difference accuracy records.

    Each record is a ``dict[str, float]`` tagged with a mode string
    (``"jvp"`` or ``"vjp"``).  Keys include absolute and relative
    errors for each output component (``"x"``, ``"lam"``, ``"mu"``).

    Attributes:
        records: Mapping from mode to a list of accuracy dicts.
        enabled: When ``False``, ``check_*`` methods are no-ops.
        eps: Finite-difference step size.
    """

    def __init__(
        self,
        enabled: bool = False,
        eps: float = 1e-6,
    ) -> None:
        self.records: defaultdict[str, list[dict[str, float]]] = (
            defaultdict(list)
        )
        self.enabled = enabled
        self.eps = eps

    # -----------------------------------------------------------------
    # JVP check
    # -----------------------------------------------------------------

    def check_jvp(
        self,
        solve_fn: Callable[..., tuple],
        dyn_primals_np: dict[str, ndarray],
        dyn_tangents_np: dict[str, ndarray],
        dx_analytic: ndarray,
        dlam_analytic: ndarray,
        dmu_analytic: ndarray,
        dynamic_keys: tuple[str, ...],
        warmstart: Optional[ndarray] = None,
    ) -> None:
        """Run central FD and compare against analytic JVP tangents.

        Parameters
        ----------
        solve_fn
            The NumPy sub-solver.  Called as
            ``solve_fn(**prob_np) -> (sol_tuple, t_dict)``.
        dyn_primals_np
            Dynamic primal arrays (NumPy, unbatched).
        dyn_tangents_np
            Dynamic tangent arrays (NumPy, unbatched — if batched the
            caller should loop over directions externally).
        dx_analytic, dlam_analytic, dmu_analytic
            Analytic tangent outputs to compare against.
        dynamic_keys
            Ordered tuple of dynamic key names.
        warmstart
            Optional warmstart vector forwarded to ``solve_fn``.
        """
        if not self.enabled:
            return

        eps = self.eps

        # Build the full prob dict once (will be mutated temporarily)
        prob_base = dict(dyn_primals_np)
        if warmstart is not None:
            prob_base["warmstart"] = warmstart

        # ── Central FD per dynamic key, accumulate into dx/dlam/dmu ──
        dx_fd = np.zeros_like(dx_analytic)
        dlam_fd = np.zeros_like(dlam_analytic)
        dmu_fd = np.zeros_like(dmu_analytic)

        for key in dynamic_keys:
            tangent = dyn_tangents_np[key]
            original = dyn_primals_np[key]

            prob_plus = dict(prob_base)
            prob_plus[key] = original + eps * tangent

            prob_minus = dict(prob_base)
            prob_minus[key] = original - eps * tangent

            sol_p, _ = solve_fn(**prob_plus)
            sol_m, _ = solve_fn(**prob_minus)

            xp, lamp, mup, _ = sol_p
            xm, lamm, mum, _ = sol_m

            dx_fd += (xp - xm) / (2.0 * eps)
            dlam_fd += (lamp - lamm) / (2.0 * eps)
            dmu_fd += (mup - mum) / (2.0 * eps)

        # ── Compute errors ───────────────────────────────────────────
        record = self._error_record(
            dx_analytic, dlam_analytic, dmu_analytic,
            dx_fd, dlam_fd, dmu_fd,
        )
        self.records["jvp"].append(record)

    # -----------------------------------------------------------------
    # VJP check
    # -----------------------------------------------------------------

    def check_vjp(
        self,
        solve_fn: Callable[..., tuple],
        dyn_primals_np: dict[str, ndarray],
        grads_analytic: dict[str, ndarray],
        g_x: ndarray,
        g_lam: ndarray,
        g_mu: ndarray,
        dynamic_keys: tuple[str, ...],
        warmstart: Optional[ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Run central FD and compare against analytic VJP cotangents.

        For each dynamic key *k* with analytic gradient ``grads[k]``
        the check verifies::

            grads[k]  ≈  Σ_i  g_i · ∂f_i/∂k

        where the sum runs over outputs (x, lam, mu) and the partial
        derivatives are estimated via central FD over each element of
        key *k*.

        Parameters
        ----------
        solve_fn
            NumPy sub-solver, same signature as for ``check_jvp``.
        dyn_primals_np
            Dynamic primal arrays (NumPy, unbatched).
        grads_analytic
            ``{key: gradient_array}`` for each dynamic key.
        g_x, g_lam, g_mu
            Cotangent vectors used in the VJP.
        dynamic_keys
            Ordered tuple of dynamic key names.
        warmstart
            Optional warmstart vector.
        rng
            NumPy random generator (unused currently, reserved for
            randomised probe directions).
        """
        if not self.enabled:
            return

        eps = self.eps

        prob_base = dict(dyn_primals_np)
        if warmstart is not None:
            prob_base["warmstart"] = warmstart

        # For each dynamic key, compute the FD gradient
        grads_fd: dict[str, ndarray] = {}

        for key in dynamic_keys:
            original = dyn_primals_np[key]
            grad_fd = np.zeros_like(original)

            # Iterate over every element of this parameter
            it = np.nditer(original, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index

                # Perturb one element
                e = np.zeros_like(original)
                e[idx] = 1.0

                prob_plus = dict(prob_base)
                prob_plus[key] = original + eps * e

                prob_minus = dict(prob_base)
                prob_minus[key] = original - eps * e

                sol_p, _ = solve_fn(**prob_plus)
                sol_m, _ = solve_fn(**prob_minus)

                xp, lamp, mup, _ = sol_p
                xm, lamm, mum, _ = sol_m

                # df/dp_j for all outputs
                dx = (xp - xm) / (2.0 * eps)
                dlam = (lamp - lamm) / (2.0 * eps)
                dmu = (mup - mum) / (2.0 * eps)

                # VJP: g^T @ df/dp_j
                grad_fd[idx] = (
                    np.dot(g_x.ravel(), dx.ravel())
                    + np.dot(g_lam.ravel(), dlam.ravel())
                    + np.dot(g_mu.ravel(), dmu.ravel())
                )

                it.iternext()

            grads_fd[key] = grad_fd

        # ── Compute per-key errors ───────────────────────────────────
        record: dict[str, float] = {}
        for key in dynamic_keys:
            a = grads_analytic[key]
            f = grads_fd[key]
            abs_err = float(np.max(np.abs(a - f)))
            norm_f = float(np.linalg.norm(f))
            rel_err = abs_err / max(norm_f, 1e-30)
            cos_sim = self._cosine_similarity(a, f)
            record[f"{key}_abs"] = abs_err
            record[f"{key}_rel"] = rel_err
            record[f"{key}_cos"] = cos_sim

        self.records["vjp"].append(record)

    # -----------------------------------------------------------------
    # Internal: error computation
    # -----------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: ndarray, b: ndarray) -> float:
        """Cosine similarity between two vectors, 0.0 if both are zero."""
        a_flat, b_flat = a.ravel(), b.ravel()
        norm_a = float(np.linalg.norm(a_flat))
        norm_b = float(np.linalg.norm(b_flat))
        if norm_a < 1e-30 and norm_b < 1e-30:
            return 1.0  # both zero → perfect agreement
        if norm_a < 1e-30 or norm_b < 1e-30:
            return 0.0  # one zero, one not → no agreement
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    @staticmethod
    def _error_record(
        dx_a: ndarray, dlam_a: ndarray, dmu_a: ndarray,
        dx_f: ndarray, dlam_f: ndarray, dmu_f: ndarray,
    ) -> dict[str, float]:
        """Build an error dict comparing analytic vs FD results."""
        record: dict[str, float] = {}
        for name, a, f in [
            ("x", dx_a, dx_f),
            ("lam", dlam_a, dlam_f),
            ("mu", dmu_a, dmu_f),
        ]:
            if a.size == 0:
                continue
            abs_err = float(np.max(np.abs(a - f)))
            norm_f = float(np.linalg.norm(f))
            rel_err = abs_err / max(norm_f, 1e-30)
            cos_sim = FiniteDifferenceRecorder._cosine_similarity(a, f)
            record[f"{name}_abs"] = abs_err
            record[f"{name}_rel"] = rel_err
            record[f"{name}_cos"] = cos_sim
        return record

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    @property
    def call_count(self) -> dict[str, int]:
        """Number of recorded checks per mode."""
        return {k: len(v) for k, v in self.records.items()}

    def stats(
        self,
        mode: Optional[str] = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Compute per-key statistics for each mode.

        Returns
        -------
        Nested dict::

            {mode: {key: {"count": …, "mean": …, "std": …,
                          "min": …, "max": …}}}
        """
        targets = (
            {mode: self.records[mode]}
            if mode is not None
            else self.records
        )

        result: dict[str, dict[str, dict[str, float]]] = {}

        for mname, entries in targets.items():
            if not entries:
                continue

            # Collect all keys in insertion order
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
                }
            result[mname] = key_stats

        return result

    # -----------------------------------------------------------------
    # Formatting
    # -----------------------------------------------------------------

    @staticmethod
    def _fmt_error(v: float) -> str:
        """Format an error value in scientific notation."""
        return f"{v:12.2e}"

    @staticmethod
    def _fmt_cosine(v: float) -> str:
        """Format a cosine similarity value (range [-1, 1])."""
        return f"{v:12.8f}"

    @staticmethod
    def _is_cosine_key(k: str) -> bool:
        """Check if a key is a cosine similarity metric."""
        return k.endswith("_cos")

    def summary(
        self,
        mode: Optional[str] = None,
    ) -> str:
        """Return a formatted multi-section summary table.

        Each section corresponds to one mode (``"jvp"`` or ``"vjp"``)
        and lists every error key with count, mean, std, min, max.
        Cosine similarity keys are formatted as decimals rather than
        scientific notation.
        """
        all_stats = self.stats(mode)

        if not all_stats:
            return "(no finite-difference records)"

        sections: list[str] = []

        for mname, key_stats in all_stats.items():
            n_checks = len(self.records[mname])
            header = f"  FD check: {mname}  ({n_checks} checks)"
            sep = "  " + "─" * 80

            lines: list[str] = [sep, header, sep]

            lines.append(
                f"  {'key':<24s}"
                f"{'count':>6s}"
                f"{'mean':>12s}"
                f"{'std':>12s}"
                f"{'min':>12s}"
                f"{'max':>12s}"
            )
            lines.append("  " + "─" * 76)

            for k in key_stats:
                s = key_stats[k]
                fmt = self._fmt_cosine if self._is_cosine_key(k) else self._fmt_error
                lines.append(
                    f"  {k:<24s}"
                    f"{int(s['count']):>6d}"
                    f"{fmt(s['mean']):>12s}"
                    f"{fmt(s['std']):>12s}"
                    f"{fmt(s['min']):>12s}"
                    f"{fmt(s['max']):>12s}"
                )

            lines.append(sep)
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{k}: {len(v)}" for k, v in self.records.items()
        )
        return f"FiniteDifferenceRecorder({counts or 'empty'})"

    # -----------------------------------------------------------------
    # Housekeeping
    # -----------------------------------------------------------------

    def reset(self, mode: Optional[str] = None) -> None:
        """Clear recorded data.

        Args:
            mode: If given, clear only records for this mode.
                If ``None``, clear everything.
        """
        if mode is not None:
            self.records.pop(mode, None)
        else:
            self.records.clear()