"""Git bootstrap: detect git, warn when missing, auto-init the workspace.

Runs once at CLI startup. Every failure path is a warning, never a blocker:
a workspace without version control is degraded, not broken, and the CLI
must still start.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from mitocode.i18n import t


def ensure_git_repo(root: Path, console: Console) -> bool:
    """Make sure *root* is inside a git repository, initializing one if needed.

    Detection goes through ``git rev-parse`` rather than a ``.git`` lookup,
    so a workspace nested inside a parent repository counts as covered and
    does not get a second, shadowing ``git init``.

    Args:
        root: The workspace root directory.
        console: Console that user-facing messages are printed to.

    Returns:
        True when the workspace is inside a git repository (pre-existing or
        newly initialized). False when git is missing, cannot run, or
        ``git init`` failed — each case prints a localized warning and
        never raises.
    """
    if shutil.which("git") is None:
        console.print(t("cli.git.missing"), style="yellow")
        return False

    probe = _run(["git", "rev-parse", "--is-inside-work-tree"], root)
    if probe is not None and probe.returncode == 0:
        return True

    init = _run(["git", "init"], root)
    if init is None or init.returncode != 0:
        error = ""
        if init is not None and init.stderr:
            error = init.stderr.strip()
        console.print(t("cli.git.init_failed", path=root, error=error), style="yellow")
        return False

    console.print(t("cli.git.initialized", path=root), style="green")
    return True


def _run(cmd: List[str], root: Path) -> Optional[subprocess.CompletedProcess]:
    """Run *cmd* with *root* as working directory, returning None on OSError.

    Args:
        cmd: Command and arguments.
        root: Working directory for the command.

    Returns:
        The completed process result, or ``None`` when an ``OSError``
        (binary vanished between the ``which`` check and the run,
        permissions, ...) prevented the process from starting.
    """
    try:
        return subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
