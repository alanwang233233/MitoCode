"""Startup assembly for the interactive CLI.

One long-lived event loop builds everything in order — git bootstrap,
workspace, provider resolution, tool registry, orchestrator with the
default system prompt, Main session, REPL. Two orderings are
load-bearing: the workspace's sync-or-async wrappers must be *awaited*
inside the loop (unlike ``__main__``'s TUI path, which never runs in
one), and — once phase two lands — the MCP manager must finish starting
before any session is created, because sessions snapshot the tool
registry in their constructor.
"""

import asyncio
import inspect
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError
from mitocode.core.tool.builtin.context import ToolContext
from mitocode.core.tool.builtin.register import register_builtin_tools
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace
from mitocode.i18n import t

from .approval import InteractiveApprover
from .commands import default_provider_factory
from .git_bootstrap import ensure_git_repo
from .mcp_setup import McpController, register_mcp_commands
from .provider_config import resolve_provider_config
from .repl import CliRepl
from .system_prompt import load_system_prompt


async def _await_or_value(value: Any) -> Any:
    """Resolve a sync-or-async wrapper result inside a running loop.

    Args:
        value: The wrapper's return value — either the plain result or an
            awaitable producing it.

    Returns:
        The underlying value.
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def _open_workspace(root: Path) -> Workspace:
    """Open (or create) the workspace rooted at *root*.

    Args:
        root: The workspace root directory.

    Returns:
        An opened workspace; a directory without configuration gets a
        fresh one.

    Raises:
        MitoCodeBaseError: When an existing configuration cannot be opened
            (corrupt file, version needing migration, ...) and a fresh
            workspace cannot be created either.
    """
    try:
        return await _await_or_value(Workspace.init_from_path(root))
    except MitoCodeBaseError:
        return await _await_or_value(
            Workspace.new_workspace(root, name=root.name, override=False)
        )


async def _never_reader(prompt_text: str) -> str:
    """Placeholder input port, replaced by the production port at run."""
    raise EOFError


async def cli_main(
    root: Optional[Path] = None,
    console: Optional[Console] = None,
    input_reader: Any = None,
    provider_factory: Any = None,
) -> Optional[CliRepl]:
    """Assemble the interactive CLI and run it to completion.

    Args:
        root: Workspace root; defaults to the current directory.
        console: Console for startup output; defaults to a fresh one.
        input_reader: Injected input port (tests). None binds the
            prompt_toolkit production port.
        provider_factory: Provider builder (tests); None uses the
            production ChatProviderOpenAI factory.

    Returns:
        The REPL that ran, for inspection in tests; None when a workspace
        error prevented the launch.
    """
    console = console if console is not None else Console()
    root = Path(root) if root is not None else Path.cwd()
    ensure_git_repo(root, console)

    try:
        workspace = await _open_workspace(root)
    except MitoCodeBaseError as exc:
        console.print(t("cli.cli.launch_error", error=exc), style="bold red")
        return None

    tool_manager = ToolManager()
    context = ToolContext(workspace=workspace, tool_manager=tool_manager)
    register_builtin_tools(tool_manager, context)

    # MCP must finish starting before any session is created: sessions
    # snapshot the tool registry in their constructor, so a later start
    # would leave the Main agent unable to call mcp__ tools.
    mcp_controller = McpController(
        root=root, tool_manager=tool_manager, console=console
    )
    await mcp_controller.start()

    config = resolve_provider_config(root, console)
    provider = None
    if config is not None:
        try:
            provider = await (provider_factory or default_provider_factory)(
                config.model,
                {
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "timeout": config.timeout,
                },
            )
        except MitoCodeBaseError as exc:
            console.print(t("cli.cli.launch_error", error=exc), style="yellow")
            provider = None

    orchestrator = AgentOrchestrator(
        tool_manager=tool_manager,
        default_provider=provider,
        default_system_prompt=load_system_prompt(workspace),
    )
    context.attach_orchestrator(orchestrator)

    main_agent_id = -1
    if provider is not None:
        main_agent_id = await orchestrator.create_agent(
            name="Main", description="Chief orchestrator"
        )

    console.print(t("cli.cli.welcome"))
    if provider is None:
        console.print(t("cli.cli.no_provider_hint"), style="yellow")

    repl = CliRepl(
        orchestrator,
        main_agent_id,
        input_reader if input_reader is not None else _never_reader,
        console,
        root=root,
        provider_factory=provider_factory,
        mcp=mcp_controller,
    )
    register_mcp_commands(repl.commands, mcp_controller)
    # Phase two: the interactive approver replaces the permissive default
    # for this CLI run only; the TUI path and bare ToolContext defaults
    # keep AlwaysAllow.
    context.approver = InteractiveApprover(
        root=root, ask=repl.ask_approval, console=console
    )
    try:
        if input_reader is not None:
            await repl.run()
        else:
            from .prompt_io import run_with_prompt_toolkit

            await run_with_prompt_toolkit(repl)
    finally:
        await mcp_controller.stop()
    return repl


def run() -> None:
    """Typer-facing entry point: one asyncio loop for the whole session."""
    asyncio.run(cli_main())
