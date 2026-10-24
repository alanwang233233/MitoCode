"""Slash-command registry and dispatch for the interactive CLI.

Commands are registered once into a :class:`CommandRegistry`; the REPL
feeds every input line through :meth:`CommandRegistry.dispatch`, which
claims lines starting with ``/`` and forwards everything else as chat.
Handlers receive the REPL instance, so they can reach the workspace root,
the console, the injected input port, the main session and the
orchestrator without the registry owning any of that state.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from mitocode.core.chat_provider.chat_provider_openai import ChatProviderOpenAI
from mitocode.i18n import t

from .provider_config import connect_wizard, switch_model


@dataclass(frozen=True)
class SlashCommand:
    """One registered slash command.

    Attributes:
        name: Primary name, without the leading slash.
        handler: Async callable ``(repl, args) -> None`` executing the
            command; *args* is the raw text after the command name.
        help_key: i18n key of the one-line help text.
        aliases: Alternative names that answer to the same handler.
    """

    name: str
    handler: Callable[[Any, str], Awaitable[None]]
    help_key: str
    aliases: Tuple[str, ...] = ()


class CommandRegistry:
    """Name-indexed registry of slash commands, aliases included."""

    def __init__(self) -> None:
        self._by_name: Dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """Register *command* under its name and every alias.

        Args:
            command: The command to register. A later registration under
                the same name replaces the earlier one.
        """
        self._by_name[command.name] = command
        for alias in command.aliases:
            self._by_name[alias] = command

    def lookup(self, name: str) -> Optional[SlashCommand]:
        """Return the command registered under *name*, or None.

        Args:
            name: Name without the slash, case-insensitive at dispatch.

        Returns:
            The matching command, or None when unknown.
        """
        return self._by_name.get(name)

    def commands(self) -> List[SlashCommand]:
        """Return primary commands in registration order, aliases skipped.

        Returns:
            Unique command objects; a command reachable through an alias
            appears exactly once.
        """
        seen: List[SlashCommand] = []
        for command in self._by_name.values():
            if all(existing is not command for existing in seen):
                seen.append(command)
        return seen

    async def dispatch(self, line: str, repl: Any) -> bool:
        """Handle *line* when it is a slash command.

        Args:
            line: The raw input line.
            repl: The REPL instance handlers operate on.

        Returns:
            True when the line was a command (known or unknown — an
            unknown command is reported, not forwarded as chat); False
            when the line is ordinary chat text.
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return False
        parts = stripped[1:].split(maxsplit=1)
        if not parts or not parts[0]:
            return False
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        command = self.lookup(name)
        if command is None:
            repl.console.print(t("cli.cmd.unknown", name=name), style="yellow")
            return True
        await command.handler(repl, args)
        return True


async def default_provider_factory(model: str, settings: dict) -> Any:
    """Build the production provider from a model id and settings.

    Args:
        model: Default model id for the provider.
        settings: Provider settings (api_key / base_url / timeout).

    Returns:
        A freshly initialised :class:`ChatProviderOpenAI`.
    """
    return await ChatProviderOpenAI.factory(
        default_model=model,
        default_model_list=None,
        default_messages=None,
        default_tools=None,
        stream=True,
        settings=settings,
    )


async def cmd_help(repl: Any, args: str) -> None:
    """``/help``: list every registered command with its help line."""
    for command in repl.commands.commands():
        repl.console.print(f"  /{command.name} — {t(command.help_key)}")


async def cmd_clear(repl: Any, args: str) -> None:
    """``/clear``: reset the main session's context, system prompt kept."""
    repl.session.clear_context()
    repl.console.print(t("cli.cmd.cleared"), style="dim")


async def cmd_exit(repl: Any, args: str) -> None:
    """``/exit``: request a clean shutdown of the REPL."""
    repl.request_exit()


async def cmd_connect(repl: Any, args: str) -> None:
    """``/connect`` (alias ``/login``): run the provider wizard and apply it."""
    config = await connect_wizard(repl.root, repl.input_reader, repl.console)
    if config is not None:
        await repl.apply_provider_config(config)


async def cmd_model(repl: Any, args: str) -> None:
    """``/model``: show the live model list and switch the default model."""
    await switch_model(
        repl.session.provider, repl.root, repl.input_reader, repl.console
    )


def build_default_registry() -> CommandRegistry:
    """Build the registry with every first-phase command registered.

    Returns:
        The populated registry; phase-two commands (``/mcp``) are added to
        the same registry by their own setup module.
    """
    registry = CommandRegistry()
    registry.register(SlashCommand("help", cmd_help, "cli.cmd.help.help"))
    registry.register(SlashCommand("clear", cmd_clear, "cli.cmd.clear.help"))
    registry.register(SlashCommand("exit", cmd_exit, "cli.cmd.exit.help"))
    registry.register(
        SlashCommand("connect", cmd_connect, "cli.cmd.connect.help", aliases=("login",))
    )
    registry.register(SlashCommand("model", cmd_model, "cli.cmd.model.help"))
    return registry
