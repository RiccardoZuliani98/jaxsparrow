"""
tests/test_examples.py
======================
Discover and run every Python script under ``examples/dense`` and
``examples/sparse`` as a subprocess, treating each one as a test case.

The test file expects to live in ``tests/`` alongside the ``examples/``
directory (i.e. both share the same parent).  Each ``.py`` file found
is parametrised as its own pytest case so failures are reported
individually.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ── Discover example scripts ────────────────────────────────────────

# tests/ and examples/ share the same parent directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _PROJECT_ROOT / "examples"

_DENSE_DIR = _EXAMPLES_DIR / "dense"
_SPARSE_DIR = _EXAMPLES_DIR / "sparse"


def _collect_scripts(*dirs: Path) -> list[Path]:
    """Recursively collect all .py files under the given directories."""
    scripts: list[Path] = []
    for d in dirs:
        if d.is_dir():
            scripts.extend(sorted(d.rglob("*.py")))
    return scripts


_EXAMPLE_SCRIPTS = _collect_scripts(_DENSE_DIR, _SPARSE_DIR)

if not _EXAMPLE_SCRIPTS:
    pytest.skip(
        f"No example scripts found under {_EXAMPLES_DIR}",
        allow_module_level=True,
    )


# ── Parametrised test ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "script",
    _EXAMPLE_SCRIPTS,
    ids=[str(s.relative_to(_EXAMPLES_DIR)) for s in _EXAMPLE_SCRIPTS],
)
def test_example_runs(script: Path) -> None:
    """Run a single example script and assert it exits cleanly."""
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"Example {script.relative_to(_EXAMPLES_DIR)} failed "
        f"(exit code {result.returncode}).\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )