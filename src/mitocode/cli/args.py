"""Argument parsing and Rich help output for the MitoCode CLI.

The parser deliberately avoids argparse's own ``-h``/``--version`` actions:
both are stored as booleans on :class:`CliArgs` so the startup path (and tests)
can decide when to print help or the version without catching ``SystemExit``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from rich.console import Console
from rich.table import Table

PERMISSION_MODES: tuple[str, ...] = ("ask", "allow", "deny")
"""Accepted values for ``--permission-mode``."""

DEFAULT_VERSION = "0.1.0"
"""Version reported when the installed distribution metadata is unavailable."""


@dataclass(frozen=True)
class CliArgs:  # pylint: disable=too-many-instance-attributes
    """Parsed command line arguments for ``mitocode``.

    Attributes:
        workspace: Root directory the agent operates in.
        permission_mode: Approval policy for tool calls, one of ``ask``,
            ``allow`` or ``deny``.
        mcp_config: Explicit MCP server configuration file.
        no_mcp: Skip reading any MCP configuration.
        verbose: Increase console verbosity.
        debug: Increase file log verbosity.
        initial_prompt: Optional prompt to submit right after startup.
        show_help: ``--help``/``-h`` was requested.
        show_version: ``--version`` was requested.
    """

    workspace: Path = field(default_factory=Path.cwd)
    permission_mode: str = "ask"
    mcp_config: Optional[Path] = None
    no_mcp: bool = False
    verbose: bool = False
    debug: bool = False
    initial_prompt: Optional[str] = None
    show_help: bool = False
    show_version: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mitocode`` argument parser.

    ``add_help`` is disabled so that ``--help`` and ``--version`` become plain
    boolean fields instead of immediate ``SystemExit``.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="mitocode",
        description="Interactive coding agent for your terminal.",
        add_help=False,
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=Path.cwd(),
        metavar="PATH",
        help="Workspace root directory (default: current directory).",
    )
    permissions = parser.add_mutually_exclusive_group()
    permissions.add_argument(
        "--permission-mode",
        choices=PERMISSION_MODES,
        metavar="MODE",
        help="Approval policy for tool calls: ask (default), allow or deny.",
    )
    permissions.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Approve every tool call without asking "
        "(equivalent to --permission-mode allow).",
    )

    parser.add_argument(
        "--mcp-config",
        type=Path,
        metavar="PATH",
        help="Explicit MCP server configuration file "
        "(default: <workspace>/.mitocode/mcp.json).",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Do not read any MCP configuration.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more detail in the terminal.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write more detailed file logs and show the log path on exit.",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        dest="show_version",
        help="Show the version and exit.",
    )
    parser.add_argument(
        "initial_prompt",
        nargs="?",
        default=None,
        metavar="PROMPT",
        help="Optional prompt submitted immediately after startup.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    """Parse command line arguments into :class:`CliArgs`.

    Args:
        argv: Argument list without the program name. ``None`` reads
            :data:`sys.argv`.

    Returns:
        The parsed arguments.

    Raises:
        SystemExit: argparse rejects the input, for example an unknown flag, an
            invalid ``--permission-mode`` or the mutually exclusive permission
            flags used together.
    """
    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)

    permission_mode: Optional[str] = namespace.permission_mode
    if permission_mode is None:
        permission_mode = "allow" if namespace.dangerously_skip_permissions else "ask"

    return CliArgs(
        workspace=namespace.workspace,
        permission_mode=permission_mode,
        mcp_config=namespace.mcp_config,
        no_mcp=namespace.no_mcp,
        verbose=namespace.verbose,
        debug=namespace.debug,
        initial_prompt=namespace.initial_prompt,
        show_help=namespace.show_help,
        show_version=namespace.show_version,
    )


def version_string() -> str:
    """Return the human-readable version banner.

    Returns:
        ``"mitocode <version>"``, where the version comes from the installed
        distribution metadata and falls back to :data:`DEFAULT_VERSION`.
    """
    try:
        version = importlib.metadata.version("mitocode")
    except importlib.metadata.PackageNotFoundError:
        version = DEFAULT_VERSION
    return f"mitocode {version}"


def render_help(
    console: Console, parser: argparse.ArgumentParser | None = None
) -> None:
    """Render the usage line and an option table to ``console``.

    Args:
        console: Rich console to print to.
        parser: Parser to describe. Defaults to :func:`build_parser`.
    """
    active_parser = parser if parser is not None else build_parser()
    console.print(active_parser.format_usage().rstrip(), style="dim")

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Option", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    # argparse exposes its registered actions through ``_actions``; the
    # attribute is stable and only read here (no mutation, no monkeypatching).
    for action in active_parser._actions:  # pylint: disable=protected-access
        if action.help is argparse.SUPPRESS:
            continue
        if action.option_strings:
            name = ", ".join(action.option_strings)
        else:
            metavar = action.metavar
            name = metavar if isinstance(metavar, str) else action.dest
        table.add_row(name, action.help or "")
    console.print(table)
