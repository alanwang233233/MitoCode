"""MCP wiring for the interactive CLI.

Loads ``.mitocode/mcp.json``, starts a :class:`McpManager` **before** any
agent session is created (sessions snapshot the tool registry in their
constructor — the call-ordering constraint), answers ``/mcp list|add|
remove``, and stops the manager on shutdown. ``/mcp add`` / ``remove``
write the file and rebuild the manager from it, so servers added mid-run
are connected and visible to every *future* session; the running Main
session keeps its constructor-time snapshot, which is the documented
semantics of the registry.
"""

import json
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from mitocode.core.exception.mitocode_exception import MitoCodeBaseError
from mitocode.core.mcp.mcp_manager import McpManager
from mitocode.i18n import t

#: Location of the MCP server configuration, workspace-relative.
MCP_FILE_RELATIVE = Path(".mitocode") / "mcp.json"


def load_mcp_file(root: Path, console: Optional[Console] = None) -> Optional[dict]:
    """Read ``mcp.json``, returning None when absent or unreadable.

    Args:
        root: Workspace root directory.
        console: Console for the corrupt-file warning; None stays silent.

    Returns:
        The raw ``{"mcpServers": {...}}`` mapping, or None when the file
        is missing, unreadable or not a JSON object.
    """
    path = root / MCP_FILE_RELATIVE
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if console is not None:
            console.print(t("cli.cmd.mcp.load_error", error=exc), style="yellow")
        return None
    return data if isinstance(data, dict) else None


def save_mcp_file(root: Path, config: dict) -> Path:
    """Persist *config* as the workspace ``mcp.json``.

    Args:
        root: Workspace root directory.
        config: The full ``{"mcpServers": {...}}`` mapping.

    Returns:
        The written path.
    """
    path = root / MCP_FILE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


class McpController:
    """Owns the live McpManager for one CLI run.

    Attributes:
        root: Workspace root directory.
        tool_manager: Registry MCP tools are published into.
        console: Console for status messages.
        manager: The running manager, or None when nothing is started.
    """

    def __init__(self, root: Path, tool_manager: Any, console: Console) -> None:
        """Wire the controller to its workspace and registry.

        Args:
            root: Workspace root directory.
            tool_manager: Tool registry the servers publish into.
            console: Console for status messages.
        """
        self.root = root
        self.tool_manager = tool_manager
        self.console = console
        self.manager: Optional[McpManager] = None

    async def start(self) -> None:
        """Start the manager from ``mcp.json`` when the file exists.

        Configuration errors are reported and swallowed: a broken
        ``mcp.json`` must not keep the CLI from starting.
        """
        config = load_mcp_file(self.root, self.console)
        if config is None:
            return
        try:
            manager = McpManager(config, tool_manager=self.tool_manager)
            await manager.start()
        except MitoCodeBaseError as exc:
            self.console.print(t("cli.cmd.mcp.load_error", error=exc), style="yellow")
            return
        self.manager = manager
        self._print_summary()

    async def restart(self) -> None:
        """Stop the current manager and start a fresh one from the file."""
        await self.stop()
        await self.start()

    async def stop(self) -> None:
        """Disconnect every server; safe to call repeatedly."""
        if self.manager is not None:
            await self.manager.stop()
            self.manager = None

    def _print_summary(self) -> None:
        """Print the connected/failed/disabled counts after a start."""
        if self.manager is None:
            return
        statuses = self.manager.status()
        connected = sum(1 for s in statuses.values() if s.state == "connected")
        failed = sum(1 for s in statuses.values() if s.state == "failed")
        disabled = sum(1 for s in statuses.values() if s.state == "disabled")
        self.console.print(
            t(
                "cli.cmd.mcp.started",
                connected=connected,
                failed=failed,
                disabled=disabled,
            )
        )


async def _mcp_list(repl: Any) -> None:
    """``/mcp list``: show every configured server's live status."""
    controller: McpController = repl.mcp
    if controller.manager is None:
        repl.console.print(
            t("cli.cmd.mcp.none", path=controller.root / MCP_FILE_RELATIVE)
        )
        return
    repl.console.print(t("cli.cmd.mcp.list_header"))
    for name, status in controller.manager.status().items():
        repl.console.print(t("cli.cmd.mcp.status", name=name, status=status.state))
        if status.error:
            repl.console.print(f"    {status.error}", style="dim")


async def _mcp_add(repl: Any) -> None:
    """``/mcp add``: ask for a server, append it, rebuild the manager."""
    controller: McpController = repl.mcp
    name = (await repl.input_reader(t("cli.cmd.mcp.ask_name"))).strip()
    kind = (await repl.input_reader(t("cli.cmd.mcp.ask_kind"))).strip().lower()
    if kind not in ("stdio", "http"):
        repl.console.print(t("cli.cmd.mcp.invalid_kind"), style="yellow")
        return
    if kind == "stdio":
        command = (await repl.input_reader(t("cli.cmd.mcp.ask_command"))).strip()
        if not name or not command:
            return
        entry = {"command": command}
    else:
        url = (await repl.input_reader(t("cli.cmd.mcp.ask_url"))).strip()
        if not name or not url:
            return
        entry = {"url": url}

    config = load_mcp_file(controller.root, repl.console) or {"mcpServers": {}}
    servers = config.setdefault("mcpServers", {})
    servers[name] = entry
    path = save_mcp_file(controller.root, config)
    await controller.restart()
    repl.console.print(t("cli.cmd.mcp.added", name=name, path=path), style="green")


async def _mcp_remove(repl: Any) -> None:
    """``/mcp remove``: drop a server from the file, rebuild the manager."""
    controller: McpController = repl.mcp
    name = (await repl.input_reader(t("cli.cmd.mcp.ask_remove"))).strip()
    config = load_mcp_file(controller.root, repl.console)
    if config is None or name not in config.get("mcpServers", {}):
        repl.console.print(t("cli.cmd.mcp.not_found", name=name), style="yellow")
        return
    del config["mcpServers"][name]
    save_mcp_file(controller.root, config)
    await controller.restart()
    repl.console.print(t("cli.cmd.mcp.removed", name=name), style="green")


async def cmd_mcp(repl: Any, args: str) -> None:
    """``/mcp``: manage MCP servers (default subcommand: list)."""
    sub = args.split(maxsplit=1)[0].lower() if args.strip() else "list"
    if sub == "list":
        await _mcp_list(repl)
    elif sub == "add":
        await _mcp_add(repl)
    elif sub == "remove":
        await _mcp_remove(repl)
    else:
        repl.console.print(t("cli.cmd.unknown", name=f"mcp {sub}"), style="yellow")


def register_mcp_commands(registry: Any, controller: McpController) -> None:
    """Register the ``/mcp`` command onto *registry*.

    Args:
        registry: The :class:`~mitocode.cli.commands.CommandRegistry` the
            command is added to.
        controller: The controller the command operates on; the REPL must
            expose it as its ``mcp`` attribute.
    """
    from .commands import SlashCommand  # pylint: disable=import-outside-toplevel

    registry.register(SlashCommand("mcp", cmd_mcp, "cli.cmd.mcp.help"))
