"""Startup helpers for the MitoCode command line interface.

This module currently holds the two pieces of startup logic that have no
dependencies on the terminal UI: resolving the workspace, and reading the MCP
server configuration.  Both are pure functions over ``pathlib``/JSON input, so
they can be exercised without a console, a provider or a live MCP connection.

Two rules shape every error raised here:

* a message may name a file or directory, but never a configuration value
  (credentials, headers, environment entries, workspace contents);
* the caller decides whether an error is fatal.  A malformed *default* MCP file
  only disables MCP, while a malformed *explicit* one is the user's own
  pointer and therefore a startup failure.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, List, Optional, TypeVar, Union, cast

from loguru import logger as loguru_logger

from mitocode import APP_VERSION, MITOCODE_CONFIG_PATH
from mitocode.cli.args import CliArgs
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.core.exception.mitocode_exception import (
    MitoCodeAppTooOldError,
    MitoCodeMcpConfigError,
    MitoCodeMigrateError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceTooOldError,
)
from mitocode.core.mcp.mcp_config import McpServerConfig
from mitocode.core.workspace.migration import BasicMigration
from mitocode.core.workspace.workspace import Workspace

CONFIG_FILENAME = "config.json"
"""Workspace configuration file name inside the ``.mitocode`` directory."""

DEFAULT_MCP_CONFIG_RELATIVE_PATH = Path(MITOCODE_CONFIG_PATH) / "mcp.json"
"""Default MCP configuration file, relative to the workspace root."""

_ResultT = TypeVar("_ResultT")


async def _awaited(value: Union[_ResultT, Awaitable[_ResultT]]) -> _ResultT:
    """Await a sync-or-async return value.

    ``Workspace`` and ``BasicMigration`` return a coroutine when an event loop is
    already running and a plain value otherwise. The CLI always calls them from
    inside its event loop (an ``async`` function cannot be reached without one),
    yet spelling that out as a helper keeps the union honest for type checkers
    that cannot narrow it from the call site.

    Args:
        value: A plain result, or an awaitable resolving to one.

    Returns:
        The resolved value.
    """
    if asyncio.iscoroutine(value):
        return await cast(Awaitable[_ResultT], value)
    return cast(_ResultT, value)


@dataclass(frozen=True)
class WorkspaceResolution:
    """Outcome of resolving the workspace at startup.

    Attributes:
        workspace: The loaded workspace. Always present, so every ``ToolContext``
            gets a root directory boundary even when nothing was on disk.
        initialized: Whether a ``.mitocode/config.json`` existed. ``False`` means
            the workspace is in memory only and ``/init`` still has to write it.
        migrated_from: Configuration version before an automatic migration, or
            ``None`` when no migration happened.
    """

    workspace: Workspace
    initialized: bool
    migrated_from: Optional[int] = None


async def resolve_workspace(path: Path) -> WorkspaceResolution:
    """Load the workspace at *path*, migrating an outdated configuration.

    A workspace without ``.mitocode/config.json`` is returned as an in-memory
    :class:`Workspace` at the current application version; nothing is written to
    disk, which keeps a read-only checkout usable and leaves initialization to
    the explicit ``/init`` command.

    Args:
        path: Workspace root directory.

    Returns:
        The resolution describing the workspace, whether it came from disk and
        which version it was migrated from.

    Raises:
        MitoCodeCliConfigError: The configuration is unreadable, invalid, written
            by a newer MitoCode, or could not be migrated. The message names the
            configuration file but never its contents.
    """
    workspace_path = Path(path)
    config_file = workspace_path / MITOCODE_CONFIG_PATH / CONFIG_FILENAME

    if not config_file.is_file():
        return WorkspaceResolution(
            workspace=Workspace(workspace_path, workspace_path.name, APP_VERSION),
            initialized=False,
            migrated_from=None,
        )

    try:
        workspace = await _awaited(Workspace.init_from_path(workspace_path))
    except MitoCodeWorkspaceTooOldError:
        return await _migrate_workspace(workspace_path, config_file)
    except MitoCodeAppTooOldError as exc:
        raise _config_error(
            config_file,
            "it was created by a newer version of MitoCode; upgrade MitoCode "
            "to open it",
        ) from exc
    except (MitoCodeWorkspaceInitError, UnicodeDecodeError) as exc:
        raise _config_error(
            config_file,
            "it is unreadable, not valid JSON, or missing required fields",
        ) from exc

    return WorkspaceResolution(
        workspace=workspace, initialized=True, migrated_from=None
    )


async def _migrate_workspace(
    workspace_path: Path, config_file: Path
) -> WorkspaceResolution:
    """Migrate an outdated workspace and reload it.

    Args:
        workspace_path: Workspace root directory.
        config_file: Configuration file that triggered the migration, used in
            error messages.

    Returns:
        The resolution for the migrated workspace, with ``migrated_from`` set to
        the version found before the migration.

    Raises:
        MitoCodeCliConfigError: The migration failed or the migrated
            configuration could not be reloaded.
    """
    previous_version = _config_version(config_file)
    try:
        await _awaited(BasicMigration.migrate(workspace_path))
        workspace = await _awaited(Workspace.init_from_path(workspace_path))
    except (
        MitoCodeMigrateError,
        MitoCodeWorkspaceInitError,
        MitoCodeAppTooOldError,
        UnicodeDecodeError,
    ) as exc:
        raise _config_error(config_file, "automatic version migration failed") from exc

    loguru_logger.debug(
        "Migrated workspace at {} from version {} to {}",
        workspace_path,
        previous_version,
        APP_VERSION,
    )
    return WorkspaceResolution(
        workspace=workspace, initialized=True, migrated_from=previous_version
    )


def _config_version(config_file: Path) -> Optional[int]:
    """Read the ``version`` field of a workspace configuration file.

    Only used after :class:`MitoCodeWorkspaceTooOldError` reported a valid but
    outdated version, so a failure here means the file changed underneath the
    caller; the version is then simply reported as unknown.

    Args:
        config_file: Configuration file to read.

    Returns:
        The configuration version, or ``None`` when it cannot be determined.
    """
    try:
        document = json.loads(config_file.read_text(encoding="utf-8"))
        return int(document["version"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        loguru_logger.debug(
            "Could not determine the previous workspace version: {}", type(exc).__name__
        )
        return None


def _config_error(config_file: Path, reason: str) -> MitoCodeCliConfigError:
    """Build a value-free workspace configuration error.

    Args:
        config_file: Configuration file the error is about.
        reason: Short explanation of what is wrong with it.

    Returns:
        The error to raise, naming the file and what the user can do about it.
    """
    return MitoCodeCliConfigError(
        f"Cannot open the workspace configuration at '{config_file}': {reason}.\n"
        "Fix or remove that file and run mitocode again."
    )


def load_mcp_servers(
    args: CliArgs, workspace: Workspace
) -> Optional[List[McpServerConfig]]:
    """Read the MCP server configuration for one CLI run.

    ``--no-mcp`` skips reading entirely. Otherwise an explicit ``--mcp-config``
    path wins over the default ``<workspace>/.mitocode/mcp.json``. A missing
    default file means "no MCP servers" and returns ``None``; a missing explicit
    file is the user's own pointer and raises.

    Args:
        args: Parsed command line arguments.
        workspace: Resolved workspace, used to locate the default file.

    Returns:
        The validated server configurations in document order, or ``None`` when
        MCP is disabled or no default file exists.

    Raises:
        MitoCodeMcpConfigError: The explicit file is missing or unreadable, or
            either file is not valid JSON or not a valid ``mcpServers`` document.
            The message names the file and whether it is the explicit or default
            one, but never a header, environment or credential value.
    """
    if args.no_mcp:
        return None

    explicit_path = args.mcp_config
    if explicit_path is not None:
        path = Path(explicit_path)
        label = "explicit MCP configuration file (--mcp-config)"
        is_required = True
    else:
        path = workspace.path / DEFAULT_MCP_CONFIG_RELATIVE_PATH
        label = "default MCP configuration file"
        is_required = False

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if not is_required:
            return None
        raise MitoCodeMcpConfigError(f"The {label} '{path}' does not exist.") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise MitoCodeMcpConfigError(
            f"The {label} '{path}' could not be read as UTF-8 text."
        ) from exc

    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise MitoCodeMcpConfigError(
            f"The {label} '{path}' is not valid JSON."
        ) from exc

    try:
        return McpServerConfig.parse_config(document)
    except MitoCodeMcpConfigError as exc:
        raise MitoCodeMcpConfigError(f"The {label} '{path}' is invalid: {exc}") from exc


__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_MCP_CONFIG_RELATIVE_PATH",
    "WorkspaceResolution",
    "load_mcp_servers",
    "resolve_workspace",
]
