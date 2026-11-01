"""Tests for the module and console-script entry points.

The version and help paths are the only ones that can run without a terminal,
so they are exercised in a subprocess exactly the way a user starts the CLI.
The source-level checks keep ``src/mitocode/__main__.py`` a thin wrapper and
guard against the deprecated TUI creeping back into the entry point.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli import app as app_module

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "src"
ENTRYPOINT = SOURCE_ROOT / "mitocode" / "__main__.py"


def _run_module(*argv):
    """Run ``python -m mitocode`` with *argv* and captured output.

    Args:
        *argv: Arguments passed after the module name.

    Returns:
        The completed process.
    """
    env = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "mitocode", *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=30,
        check=False,
    )


def test_python_m_mitocode_version():
    result = _run_module("--version")
    assert result.returncode == 0
    assert "mitocode" in result.stdout.lower()


def test_python_m_mitocode_help_needs_no_tty():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--workspace" in result.stdout
    assert "--permission-mode" in result.stdout


def test_entrypoint_is_a_thin_wrapper():
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert "from mitocode.cli.app import main" in source
    assert "raise SystemExit(main())" in source
    assert "tui" not in source.lower()
    assert "VibeFrontend" not in source


def test_run_wraps_main_in_system_exit(monkeypatch):
    monkeypatch.setattr(app_module, "main", lambda argv=None: 3)
    with pytest.raises(SystemExit) as caught:
        app_module.run()
    assert caught.value.code == 3
