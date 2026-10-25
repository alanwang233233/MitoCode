"""Slash-command registry and handlers for the interactive CLI.

Every command is an ``async`` handler registered under a bare name (``help``,
not ``/help``) with a usage string and a one-line description.  ``dispatch`` is
the only entry point: a line that does not start with ``/`` is a prompt, not a
command, and returns ``None``; an unknown command prints one red line and
returns an ordinary :class:`CommandResult` so the REPL keeps running.

Two rules keep commands from taking down the REPL.  Handlers that address an
agent by id verify the id and warn instead of indexing a missing session, and
every handler that can fail for a recoverable reason (a bad model number, an
applied prompt answer, a workspace that cannot be written) reports through
:class:`~mitocode.cli.ui.CliUI` rather than raising.  A command's failures are
never a reason to exit: only ``/exit`` and ``/quit`` return
``exit_requested=True``.

The handlers are deliberately shallow over the objects in
:class:`~mitocode.cli.state.CliState`; the orchestrator, provider, MCP manager
and approver are duck-typed, so the tests drive them with fakes and no command
needs a terminal or a network connection.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from rich.table import Table
from rich.text import Text

from mitocode.cli.state import CommandContext

if TYPE_CHECKING:
    from mitocode.cli.state import CliState

CommandHandler = Callable[
    [CommandContext, List[str]], Awaitable[Optional["CommandResult"]]
]
"""Signature every command handler implements."""

MAX_AGENT_MESSAGE_CHARS = 2000
"""Longest last-assistant message ``/agents <id>`` prints before truncating."""

UNKNOWN_COMMAND_TEMPLATE = "Unknown command /{name} · /help"
"""Line printed for a slash command that is not registered."""


@dataclass(frozen=True)
class Command:
    """One registered slash command.

    Attributes:
        name: Bare command name without the leading slash.
        usage: Usage string starting with ``/``, shown by ``/help``.
        description: One-line human description.
        handler: Coroutine called as ``await handler(ctx, args)``.
    """

    name: str
    usage: str
    description: str
    handler: CommandHandler


@dataclass
class CommandResult:
    """What ``dispatch`` reports back to the REPL.

    Attributes:
        exit_requested: ``True`` only for ``/exit`` and ``/quit``.
    """

    exit_requested: bool = False


COMMANDS: Dict[str, Command] = {}
"""Every registered command, keyed by bare name."""


def command(name: str, usage: str, description: str):  # type: ignore[no-untyped-def]
    """Register the decorated coroutine as a slash command.

    Args:
        name: Bare command name, e.g. ``"help"``.
        usage: Usage string, e.g. ``"/help"``.
        description: One-line description rendered in the help table.

    Returns:
        A decorator that stores the handler and returns it unchanged.
    """

    def decorator(handler: CommandHandler) -> CommandHandler:
        """Store *handler* in :data:`COMMANDS`.

        Args:
            handler: Coroutine implementing the command.

        Returns:
            The same coroutine, so the module keeps a named function.
        """
        COMMANDS[name] = Command(name, usage, description, handler)
        return handler

    return decorator


async def dispatch(line: str, ctx: CommandContext) -> Optional[CommandResult]:
    """Run the slash command in *line*, or report that it is not one.

    Args:
        line: Raw line submitted at the prompt.
        ctx: State, UI and approver available to the handler.

    Returns:
        ``None`` when the line is ordinary prompt text; otherwise a
        :class:`CommandResult`.  Unknown commands return a default result
        (no exit) after printing one error line.
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped[1:].split()
    if not parts:
        ctx.ui.error(UNKNOWN_COMMAND_TEMPLATE.format(name=""))
        return CommandResult()

    name = parts[0].lower()
    entry = COMMANDS.get(name)
    if entry is None:
        ctx.ui.error(UNKNOWN_COMMAND_TEMPLATE.format(name=name))
        return CommandResult()

    result = await entry.handler(ctx, parts[1:])
    return result if result is not None else CommandResult()


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _model_name(state: "CliState") -> str:
    """Return the provider's current default model.

    Args:
        state: Current CLI state.

    Returns:
        The model name, or ``"unknown"`` when the provider reports none.
    """
    return str(getattr(state.provider, "default_model", "") or "unknown")


def _workspace_path(state: "CliState") -> str:
    """Return the workspace root for display.

    Args:
        state: Current CLI state.

    Returns:
        The root path, or ``"."`` when the workspace object has none.
    """
    return str(getattr(state.workspace, "path", "") or ".")


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    """Read one field from a tree-snapshot entry.

    Args:
        entry: Dictionary or attribute-bearing snapshot entry.
        key: Field name.
        default: Value returned when the field is absent.

    Returns:
        The field value, or *default*.
    """
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _mcp_summary(state: "CliState") -> str:
    """Build the one-line MCP summary shown by ``/status``.

    Args:
        state: Current CLI state.

    Returns:
        ``disabled``, ``no servers``, or a comma-separated list of server
        names and states.
    """
    manager = state.mcp_manager
    if manager is None:
        return "disabled" + (f" · {state.mcp_error}" if state.mcp_error else "")
    try:
        statuses = manager.status()
    except Exception:  # pylint: disable=broad-exception-caught
        return "unavailable"
    if not statuses:
        return "no servers"
    rendered = []
    for name, status in statuses.items():
        state_name = (
            status.get("state", "")
            if isinstance(status, dict)
            else getattr(status, "state", "")
        )
        rendered.append(f"{name} {state_name}".rstrip())
    return ", ".join(rendered)


def _snapshot_ids(state: "CliState") -> List[Any]:
    """Return the agent ids from the orchestrator's tree snapshot.

    Args:
        state: Current CLI state.

    Returns:
        The ids in snapshot order; an empty list when the orchestrator raises
        or the snapshot has no ids.
    """
    try:
        snapshot = state.orchestrator.get_tree_snapshot()
    except Exception:  # pylint: disable=broad-exception-caught
        return []
    return [
        entry_id
        for entry in snapshot
        if (entry_id := _entry_value(entry, "id")) is not None
    ]


def _tree_lines(snapshot: List[Any]) -> List[str]:
    """Render a flat tree snapshot with indentation.

    Args:
        snapshot: Entries carrying ``id``, ``name``, ``state``,
            ``parent_id`` and optionally ``sub_agent_ids``.

    Returns:
        One display line per agent, indented by its depth in the tree.
    """
    by_id = {
        entry_id: entry
        for entry in snapshot
        if (entry_id := _entry_value(entry, "id")) is not None
    }

    def depth(entry: Any) -> int:
        """Return how many ancestors an entry has.

        Args:
            entry: Snapshot entry to measure.

        Returns:
            Non-negative nesting depth, capped by a visited set so a corrupted
            snapshot cannot loop.
        """
        seen = {_entry_value(entry, "id")}
        level = 0
        parent_id = _entry_value(entry, "parent_id")
        while parent_id is not None and parent_id in by_id and parent_id not in seen:
            seen.add(parent_id)
            level += 1
            parent_id = _entry_value(by_id[parent_id], "parent_id")
        return level

    lines = []
    for entry in sorted(by_id.values(), key=lambda item: _entry_value(item, "id", 0)):
        entry_id = _entry_value(entry, "id")
        name = _entry_value(entry, "name", "") or "(unnamed)"
        entry_state = _entry_value(entry, "state", "") or "unknown"
        parent_id = _entry_value(entry, "parent_id")
        children = _entry_value(entry, "sub_agent_ids", []) or []
        line = f"{'  ' * depth(entry)}#{entry_id} {name} [{entry_state}]"
        if parent_id is not None:
            line += f" parent=#{parent_id}"
        if children:
            line += " children=" + ",".join(f"#{child}" for child in children)
        lines.append(line)
    return lines


def _last_assistant_text(session: Any) -> Optional[str]:
    """Return the last assistant text in a session's history.

    Args:
        session: Session whose ``messages`` are inspected.

    Returns:
        The most recent non-empty assistant message, truncated to
        :data:`MAX_AGENT_MESSAGE_CHARS`, or ``None`` when there is none.
    """
    for block in reversed(getattr(session, "messages", []) or []):
        role = (
            block.get("role")
            if isinstance(block, dict)
            else getattr(block, "role", None)
        )
        message = (
            block.get("message")
            if isinstance(block, dict)
            else getattr(block, "message", None)
        )
        if role == "assistant" and isinstance(message, str) and message:
            if len(message) > MAX_AGENT_MESSAGE_CHARS:
                return message[:MAX_AGENT_MESSAGE_CHARS] + "…"
            return message
    return None


async def _resolve(value: Any) -> Any:
    """Await *value* when it is awaitable, otherwise return it.

    ``Workspace.reload``/``dump`` return a coroutine inside a running event
    loop and a plain value outside one; handlers always run inside an event
    loop but spelling the union out keeps both shapes working.

    Args:
        value: Plain result or awaitable.

    Returns:
        The resolved value.
    """
    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@command("help", "/help", "Show commands and keyboard shortcuts.")
async def _cmd_help(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render the command table and the prompt's shortcuts.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Command", style="mito.accent", no_wrap=True)
    table.add_column("Description")
    for entry in COMMANDS.values():
        table.add_row(entry.usage, entry.description)
    ctx.ui.console.print(table)
    ctx.ui.console.print(Text("Shortcuts", style="bold mito.accent"))
    ctx.ui.console.print(
        Text(
            "  Enter submit · \\ + Enter newline · Ctrl+L clear · "
            "Ctrl+C interrupt · Ctrl+D exit",
            style="mito.muted",
        )
    )
    return CommandResult()


@command("status", "/status", "Show the session, model, workspace and MCP state.")
async def _cmd_status(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Print one fact per row for the current session.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    session = state.orchestrator.get_session(state.main_id)
    name = ""
    for entry in state.orchestrator.get_tree_snapshot():
        if _entry_value(entry, "id") == state.main_id:
            name = str(_entry_value(entry, "name", "") or "")
            break
    if not name and session is not None:
        name = str(getattr(session, "name", "") or "")
    if not name:
        name = "Main"

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Field", style="mito.muted", no_wrap=True)
    table.add_column("Value")
    table.add_row("session", f"#{state.main_id} {name}")
    table.add_row("model", _model_name(state))
    table.add_row("workspace", _workspace_path(state))
    table.add_row("messages", str(len(getattr(session, "messages", []) or [])))
    table.add_row("tokens", f"~{state.estimate_tokens()}")
    table.add_row("tools", str(len(state.tool_manager.human_docs())))
    table.add_row("agents", str(len(state.orchestrator.get_tree_snapshot())))
    table.add_row("mcp", _mcp_summary(state))
    ctx.ui.console.print(table)
    return CommandResult()


@command("model", "/model [name]", "List models or switch the default model.")
async def _cmd_model(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Switch ``provider.default_model`` from an argument or a prompt answer.

    A numeric answer is an index into ``provider.model_list``; any other
    non-empty answer is a model name.  A name that the endpoint did not
    advertise is used anyway, with a warning, so a configured model cannot
    leave the user stuck on one the list omitted.

    Args:
        ctx: Command context.
        args: Optional model name.

    Returns:
        A default result.
    """
    state = ctx.state
    provider = state.provider
    models = [str(model) for model in (getattr(provider, "model_list", None) or [])]

    if args:
        answer = args[0]
    else:
        if not models:
            ctx.ui.warn(
                "No models reported by the provider; pass a model name instead."
            )
            return CommandResult()
        for index, model in enumerate(models, start=1):
            marker = " (current)" if model == _model_name(state) else ""
            ctx.ui.console.print(Text(f"  {index}. {model}{marker}"))
        try:
            answer = await ctx.ui.ask("  Model number or name: ")
        except (EOFError, KeyboardInterrupt):
            ctx.ui.note("Model unchanged.")
            return CommandResult()
        answer = answer.strip()
        if not answer:
            ctx.ui.note("Model unchanged.")
            return CommandResult()

    if answer.isdigit():
        index = int(answer) - 1
        if not 0 <= index < len(models):
            ctx.ui.warn(f"Model number {answer} is out of range; model unchanged.")
            return CommandResult()
        chosen = models[index]
    else:
        chosen = answer

    if models and chosen not in models:
        ctx.ui.warn(f"'{chosen}' is not in the provider's model list; using it anyway.")
    provider.default_model = chosen
    ctx.ui.note(f"Model set to {chosen}.")
    return CommandResult()


@command("clear", "/clear", "Destroy the main agent and start a new session.")
async def _cmd_clear(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Replace the main session and reset this process's counters.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    await state.orchestrator.destroy_agent(state.main_id)
    state.main_id = await state.orchestrator.create_agent(
        name="Main", description="Interactive session"
    )
    state.turn_count = 0
    state.tool_call_count = 0
    state.total_chars = 0
    state.agents = state.orchestrator.get_tree_snapshot()
    ctx.ui.note(f"New session #{state.main_id}.")
    return CommandResult()


@command("agents", "/agents [id]", "Show the agent tree or one agent's last answer.")
async def _cmd_agents(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render the tree, or the last assistant message for one id.

    Args:
        ctx: Command context.
        args: Optional agent id.

    Returns:
        A default result.
    """
    state = ctx.state
    snapshot = state.orchestrator.get_tree_snapshot()

    if not args:
        lines = _tree_lines(snapshot)
        if not lines:
            ctx.ui.note("No agents.")
            return CommandResult()
        for line in lines:
            ctx.ui.console.print(Text(f"  {line}"))
        return CommandResult()

    raw_id = args[0]
    try:
        agent_id = int(raw_id)
    except ValueError:
        ctx.ui.warn(f"Invalid agent id '{raw_id}'.")
        return CommandResult()

    session = state.orchestrator.get_session(agent_id)
    if session is None:
        ctx.ui.warn(f"No agent #{agent_id}.")
        return CommandResult()

    message = _last_assistant_text(session)
    if message is None:
        ctx.ui.note(f"Agent #{agent_id} has no assistant message yet.")
        return CommandResult()
    ctx.ui.console.print(Text(f"  #{agent_id} {message}"))
    return CommandResult()


@command("tools", "/tools", "List the registered tools.")
async def _cmd_tools(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render tool names and their human descriptions.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    docs = ctx.state.tool_manager.human_docs()
    if not docs:
        ctx.ui.note("No tools registered.")
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Tool", style="mito.accent", no_wrap=True)
    table.add_column("Description")
    for name, description in docs.items():
        table.add_row(name, description or "—")
    ctx.ui.console.print(table)
    return CommandResult()


@command("mcp", "/mcp [refresh [name]]", "Show MCP servers or refresh their tools.")
async def _cmd_mcp(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render MCP status, or reconnect servers and refresh session tools.

    Args:
        ctx: Command context.
        args: Empty for status, otherwise ``refresh`` and an optional server.

    Returns:
        A default result.
    """
    if args and args[0].lower() == "refresh":
        return await _refresh_mcp(ctx, args[1] if len(args) > 1 else None)
    if args:
        ctx.ui.warn("Usage: /mcp [refresh [name]]")
        return CommandResult()
    return _render_mcp_status(ctx)


async def _refresh_mcp(ctx: CommandContext, server: Optional[str]) -> CommandResult:
    """Reconnect MCP servers and re-snapshot every session's tools.

    Args:
        ctx: Command context.
        server: Server to refresh, or ``None`` for all configured servers.

    Returns:
        A default result.
    """
    state = ctx.state
    manager = state.mcp_manager
    if manager is None:
        ctx.ui.warn("No MCP servers configured.")
        return CommandResult()

    try:
        await manager.refresh(server)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        ctx.ui.error(f"MCP refresh failed: {exc}")
        return CommandResult()

    for agent_id in _snapshot_ids(state):
        session = state.orchestrator.get_session(agent_id)
        refresh = getattr(session, "refresh_tools", None)
        if callable(refresh):
            refresh()
    ctx.ui.note("MCP tools refreshed.")
    return CommandResult()


def _render_mcp_status(ctx: CommandContext) -> CommandResult:
    """Print the server status table, or explain why MCP is unavailable.

    Args:
        ctx: Command context.

    Returns:
        A default result.
    """
    state = ctx.state
    manager = state.mcp_manager
    if manager is None:
        if state.mcp_error:
            ctx.ui.warn(f"MCP unavailable: {state.mcp_error}")
        else:
            ctx.ui.note("No MCP servers configured.")
        return CommandResult()

    statuses = manager.status()
    if not statuses:
        ctx.ui.note("No MCP servers configured.")
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Server", style="mito.accent", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Tools", justify="right", no_wrap=True)
    table.add_column("Error")
    for name, status in statuses.items():
        entry_state = (
            status.get("state", "")
            if isinstance(status, dict)
            else getattr(status, "state", "")
        )
        error = (
            status.get("error", "")
            if isinstance(status, dict)
            else getattr(status, "error", "")
        )
        tool_count = (
            status.get("tool_count", 0)
            if isinstance(status, dict)
            else getattr(status, "tool_count", 0)
        )
        table.add_row(name, str(entry_state), str(tool_count), str(error or ""))
    ctx.ui.console.print(table)
    return CommandResult()


@command(
    "permissions",
    "/permissions [reset]",
    "Show the approval policy and cached always-allow requests.",
)
async def _cmd_permissions(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Show the policy/cache, or clear the cache on ``reset``.

    Args:
        ctx: Command context.
        args: Optional ``reset``.

    Returns:
        A default result.
    """
    approver = ctx.approver or ctx.state.approver
    if approver is None:
        ctx.ui.note("No interactive approver configured.")
        return CommandResult()

    if args and args[0].lower() == "reset":
        approver.reset_cache()
        ctx.ui.note("Permission cache cleared.")
        return CommandResult()
    if args:
        ctx.ui.warn("Usage: /permissions [reset]")
        return CommandResult()

    policy = getattr(approver, "policy", "ask")
    policy_text = getattr(policy, "value", str(policy))
    ctx.ui.console.print(Text(f"  Policy: {policy_text}"))
    cached = getattr(approver, "cached_requests", None)
    raw_entries: Any = cached() if callable(cached) else []
    entries = sorted(raw_entries)
    if not entries:
        ctx.ui.note("No always-allow requests cached.")
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Action", style="mito.accent", no_wrap=True)
    table.add_column("Detail")
    for entry in entries:
        if isinstance(entry, tuple) and len(entry) == 2:
            table.add_row(str(entry[0]), str(entry[1]))
        else:
            table.add_row(str(entry), "")
    ctx.ui.console.print(table)
    return CommandResult()


@command("verbose", "/verbose", "Toggle full tool output.")
async def _cmd_verbose(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Toggle verbose output on the state and the UI together.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    enabled = not ctx.state.verbose
    ctx.state.verbose = enabled
    ctx.ui.verbose = enabled
    ctx.ui.note(f"Verbose output {'on' if enabled else 'off'}.")
    return CommandResult()


@command("cost", "/cost", "Show approximate tokens used by this session.")
async def _cmd_cost(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Print the character/4 token estimate for the main session.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    ctx.ui.note(f"Approximate tokens: ~{ctx.state.estimate_tokens()}")
    return CommandResult()


@command("init", "/init", "Persist the in-memory workspace configuration.")
async def _cmd_init(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Write the workspace to disk, or reload an already initialized one.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    workspace = state.workspace
    path = Path(_workspace_path(state))

    if state.workspace_initialized:
        reload = getattr(workspace, "reload", None)
        if callable(reload):
            try:
                await _resolve(reload())
            except Exception as exc:  # pylint: disable=broad-exception-caught
                ctx.ui.error(f"Could not reload the workspace: {exc}")
                return CommandResult()
        ctx.ui.note(f"Workspace already initialized at {path}.")
        return CommandResult()

    config_path = getattr(workspace, "config_path", None)
    try:
        if isinstance(config_path, Path):
            config_path.mkdir(parents=True, exist_ok=True)
        result = await _resolve(workspace.dump())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        ctx.ui.error(f"Could not initialize the workspace: {exc}")
        return CommandResult()

    if result:
        state.workspace_initialized = True
        ctx.ui.note(f"Workspace initialized at {path}.")
    else:
        ctx.ui.warn("Workspace was not initialized.")
    return CommandResult()


@command("log", "/log", "Show the log file path.")
async def _cmd_log(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Print the file the CLI's logs are written to.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    ctx.ui.note(str(ctx.state.log_path))
    return CommandResult()


@command("exit", "/exit", "Exit MitoCode.")
async def _cmd_exit(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Ask the REPL to shut down.

    Args:
        ctx: Unused command context.
        args: Unused extra arguments.

    Returns:
        A result with ``exit_requested=True``.
    """
    del ctx, args
    return CommandResult(exit_requested=True)


@command("quit", "/quit", "Exit MitoCode (same as /exit).")
async def _cmd_quit(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Ask the REPL to shut down, aliasing ``/exit``.

    Args:
        ctx: Unused command context.
        args: Unused extra arguments.

    Returns:
        A result with ``exit_requested=True``.
    """
    del ctx, args
    return CommandResult(exit_requested=True)


__all__ = [
    "COMMANDS",
    "Command",
    "CommandHandler",
    "CommandResult",
    "MAX_AGENT_MESSAGE_CHARS",
    "UNKNOWN_COMMAND_TEMPLATE",
    "command",
    "dispatch",
]
