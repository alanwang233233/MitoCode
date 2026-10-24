"""Typer-based command line entry point for MitoCode.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import typer
from dotenv import load_dotenv
from rich.console import Console

from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.chat_provider.chat_provider_openai import ChatProviderOpenAI
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError
from mitocode.core.tool.builtin.context import ToolContext
from mitocode.core.tool.builtin.register import register_builtin_tools
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace
from mitocode.i18n import t as translate
from mitocode.tui.ui_main import VibeFrontend

app = typer.Typer(
    name="mitocode",
    help=translate("cli.app_help"),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

#: Seconds the ``tui`` command waits after printing the deprecation notice.
TUI_DEPRECATED_GRACE_SECONDS = 3

#: Fallback provider configuration when the environment says nothing.
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 60

#: Name of the dotenv file loaded from the current directory.
ENV_FILE_NAME = ".env"

#: Environment variables the provider configuration is read from, in
#: priority order.  ``OPENAI_*`` names are accepted as the de-facto standard
#: spellings; the ``MITOCODE_*`` names win so MitoCode can be pointed at a
#: different backend than other tools sharing the shell.
API_KEY_ENV_VARS = ("MITOCODE_API_KEY", "OPENAI_API_KEY")
BASE_URL_ENV_VARS = ("MITOCODE_BASE_URL", "OPENAI_BASE_URL")
MODEL_ENV_VARS = ("MITOCODE_MODEL",)
TIMEOUT_ENV_VARS = ("MITOCODE_TIMEOUT",)


def _first_env(names: Tuple[str, ...]) -> Optional[str]:
    """Return the first non-blank value among the named environment variables.

    Args:
        names: Variable names in priority order.

    Returns:
        The trimmed value of the first variable that is set and non-blank,
        or ``None`` when none is.
    """
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _load_env_file() -> None:
    """Load ``.env`` from the current directory when it exists.

    Values already present in the environment always win over the file
    (``load_dotenv``'s default), so the file is a convenience rather than an
    override.  The current directory is used verbatim -- the same directory
    the command treats as its workspace.
    """
    env_path = Path.cwd() / ENV_FILE_NAME
    if env_path.is_file():
        load_dotenv(env_path)
        console.print(translate("cli.tui.loaded_env_file", path=env_path), style="dim")


def _resolve_timeout() -> int:
    """Resolve the provider request timeout from the environment.

    Args:
        None.

    Returns:
        ``MITOCODE_TIMEOUT`` parsed as seconds, or
        :data:`DEFAULT_TIMEOUT_SECONDS` when unset.  A non-integer value is
        reported and the default is used rather than aborting the launch.
    """
    raw = _first_env(TIMEOUT_ENV_VARS)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return int(raw)
    except ValueError:
        console.print(
            translate(
                "cli.tui.invalid_timeout",
                value=raw,
                default=DEFAULT_TIMEOUT_SECONDS,
            ),
            style="yellow",
        )
        return DEFAULT_TIMEOUT_SECONDS


def _resolve_provider_config() -> Tuple[str, Dict[str, Any]]:
    """Build the chat provider configuration from the environment.

    Args:
        None.

    Returns:
        A ``(default_model, settings)`` pair ready for
        :meth:`ChatProviderOpenAI.factory`.

    Raises:
        typer.Exit: With exit code 1 when no API key is configured.
    """
    api_key = _first_env(API_KEY_ENV_VARS)
    if api_key is None:
        console.print(translate("cli.api_key_missing"), style="bold red")
        raise typer.Exit(code=1)
    settings: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": _first_env(BASE_URL_ENV_VARS) or DEFAULT_BASE_URL,
        "timeout": _resolve_timeout(),
    }
    model = _first_env(MODEL_ENV_VARS) or DEFAULT_MODEL
    return model, settings


def _prepare_workspace() -> Workspace:
    """Open the workspace rooted at the current directory.

    An existing workspace configuration is loaded as-is; a directory without
    one gets a fresh configuration.  ``override=False`` matters for the
    corrupt-config case: re-reading raises again and the error surfaces
    instead of the configuration being deleted behind the user's back.

    Args:
        None.

    Returns:
        The workspace rooted at the current directory.

    Raises:
        MitoCodeBaseError: If the directory has a configuration that cannot
            be opened (corrupt file, version needing migration, ...).
    """
    cwd = Path.cwd()
    # The Workspace constructors are sync-or-async wrappers declared to
    # return ``Workspace | Awaitable[Workspace]``; the CLI never runs inside
    # an event loop, so the Awaitable half of the union is unreachable here.
    try:
        return cast(Workspace, Workspace.init_from_path(cwd))
    except MitoCodeBaseError:
        return cast(
            Workspace, Workspace.new_workspace(cwd, name=cwd.name, override=False)
        )


def _build_orchestrator() -> AgentOrchestrator:
    """Assemble the tool registry, provider and orchestrator for the TUI.

    The ``Main`` agent session must be created after the built-in tools are
    registered: sessions snapshot the tool registry in their constructor.

    Args:
        None.

    Returns:
        An orchestrator with the eleven built-in tools registered, a default
        provider built from the environment configuration, and a ``Main``
        agent session ready for the TUI.

    Raises:
        typer.Exit: With exit code 1 when no API key is configured.
        MitoCodeBaseError: If the workspace or the provider cannot be set up.
    """
    model, settings = _resolve_provider_config()
    tool_manager = ToolManager()
    context = ToolContext(workspace=_prepare_workspace(), tool_manager=tool_manager)
    register_builtin_tools(tool_manager, context)
    provider = asyncio.run(
        ChatProviderOpenAI.factory(
            default_model=model,
            default_model_list=None,
            default_messages=None,
            default_tools=None,
            stream=True,
            settings=settings,
        )
    )
    orchestrator = AgentOrchestrator(
        tool_manager=tool_manager, default_provider=provider
    )
    context.attach_orchestrator(orchestrator)
    asyncio.run(
        orchestrator.create_agent(
            name="Main",
            description="Chief orchestrator",
        )
    )
    return orchestrator


@app.command()
def tui() -> None:
    """Launch the terminal UI (deprecated, no longer maintained)."""
    console.print(translate("cli.tui.deprecated_warning"), style="bold yellow")
    console.print(
        translate("cli.tui.launching_in", seconds=TUI_DEPRECATED_GRACE_SECONDS),
        style="bold yellow",
    )
    time.sleep(TUI_DEPRECATED_GRACE_SECONDS)
    _load_env_file()
    try:
        orchestrator = _build_orchestrator()
    except MitoCodeBaseError as exc:
        console.print(translate("cli.tui.launch_error", error=exc), style="bold red")
        raise typer.Exit(code=1) from exc
    VibeFrontend(orchestrator=orchestrator).run()


@app.command()
def cli() -> None:
    """Start the interactive CLI chat mode."""
    # Imported at call time so `mitocode tui` never pays for prompt_toolkit.
    from mitocode.cli.bootstrap import run as run_cli  # pylint: disable=import-outside-toplevel

    try:
        run_cli()
    except MitoCodeBaseError as exc:
        console.print(translate("cli.cli.launch_error", error=exc), style="bold red")
        raise typer.Exit(code=1) from exc


@app.command()
def webui() -> None:
    """Placeholder for the web UI."""
    console.print(translate("cli.not_implemented", name="webui"))
    raise typer.Exit(code=1)


def main() -> None:
    """Entry point shared by the ``mitocode`` console script and ``python -m mitocode``."""
    app()


if __name__ == "__main__":
    main()
