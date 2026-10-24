"""The interactive REPL loop driving ``mitocode cli``.

:class:`CliRepl` owns the read-dispatch-render cycle: it reads a line
through the injected input port, claims slash commands, sends chat text
to the main agent (queueing it while a turn is running), and renders
every bridged event through :class:`~mitocode.cli.rendering.EventRenderer`.
All decisions — interrupt, exit, queueing — live here as small methods,
so the prompt_toolkit layer stays a thin shell and every behaviour is
testable without a terminal.
"""

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, List

from rich.console import Console

from mitocode.core.agent.agent_state import AgentState
from mitocode.i18n import t

from .commands import (
    CommandRegistry,
    build_default_registry,
    default_provider_factory,
)
from .provider_config import ProviderConfig
from .rendering import EventRenderer

#: Type of the injected input port: renders the prompt, returns the line.
InputReader = Callable[[str], Awaitable[str]]

#: Prompt shown while waiting for input.
PROMPT_TEXT = "❯ "


class CliRepl:
    """Interactive REPL over an orchestrator's main agent session.

    Attributes:
        orchestrator: Backend owning sessions and the event queue.
        main_agent_id: The agent chat messages are sent to.
        input_reader: Injected async input port.
        console: Console for chat output and feedback.
        root: Workspace root directory, used by commands.
        commands: The slash-command registry.
        provider_factory: Builder for ``/connect`` provider swaps; None
            means the production default.
    """

    def __init__(
        self,
        orchestrator: Any,
        main_agent_id: int,
        input_reader: InputReader,
        console: Console,
        root: Path | None = None,
        commands: CommandRegistry | None = None,
        provider_factory: Any | None = None,
        mcp: Any | None = None,
    ) -> None:
        """Wire the REPL to its backend and I/O ports.

        Args:
            orchestrator: Object providing ``get_session``,
                ``send_message``, ``cancel_agent``, ``get_tree_snapshot``
                and ``event_queue``.
            main_agent_id: Id of the agent chat input is routed to.
            input_reader: Async callable rendering a prompt and returning
                the entered line; the production port raises ``EOFError``
                on end of input.
            console: Rich console for all output.
            root: Workspace root; defaults to the current directory.
            commands: Registry to use; defaults to the built-in set.
            provider_factory: Provider builder for ``/connect`` swaps;
                None falls back to the production factory.
            mcp: The MCP controller exposed to ``/mcp`` commands; None
                leaves the command unregistered.
        """
        self.orchestrator = orchestrator
        self.main_agent_id = main_agent_id
        self.input_reader = input_reader
        self.console = console
        self.root = Path(root) if root is not None else Path.cwd()
        self.commands = commands if commands is not None else build_default_registry()
        self.provider_factory = provider_factory
        self.mcp = mcp
        self._queued: List[str] = []
        self._exit_requested = False
        self._approval_waiter: Any | None = None
        self._renderer = EventRenderer(console, orchestrator, main_agent_id)

    @property
    def session(self) -> Any:
        """The main agent session, or None when it no longer exists."""
        return self.orchestrator.get_session(self.main_agent_id)

    def request_exit(self) -> None:
        """Ask the main loop to finish after the current input."""
        self._exit_requested = True

    def request_interrupt(self) -> None:
        """Interrupt the main agent's running turn, if any.

        Cooperative: the session's cancel event is set, so the runtime
        stops at the next chunk boundary and the text generated so far
        stays in the history.
        """
        session = self.session
        if session is None or session.state != AgentState.RUNNING:
            return
        session.cancel()
        self.console.print(t("cli.cli.interrupted"), style="yellow")

    async def run(self) -> None:
        """Run the read-dispatch loop until exit is requested or EOF."""
        consumer = asyncio.create_task(self._consume_events())
        try:
            while not self._exit_requested:
                try:
                    line = await self.input_reader(PROMPT_TEXT)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    self.console.print(t("cli.cli.exit_hint"), style="dim")
                    continue
                if (
                    self._approval_waiter is not None
                    and not self._approval_waiter.done()
                ):
                    waiter, self._approval_waiter = self._approval_waiter, None
                    waiter.set_result(line)
                    continue
                line = line.strip()
                if not line:
                    continue
                if await self.commands.dispatch(line, self):
                    if self._exit_requested:
                        break
                    continue
                await self._send(line)
        finally:
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
            await self._shutdown()

    async def ask_approval(self, prompt_text: str) -> str:
        """Ask the user one question and return their next input line.

        The prompt is printed above the standing input line and the
        REPL's main loop routes the user's next typed line here instead
        of treating it as chat or a command — this is how the
        interactive approver prompts without a second concurrent
        prompt session.

        Args:
            prompt_text: The question line to show.

        Returns:
            The line the user typed in answer.
        """
        self.console.print(prompt_text, style="yellow", end="")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._approval_waiter = future
        return await future

    async def _send(self, text: str) -> None:
        """Send chat text, or queue it while a turn is running.

        Args:
            text: The user's chat message.
        """
        session = self.session
        if session is None:
            self.console.print(t("cli.cli.no_provider_hint"), style="yellow")
            return
        if session.state == AgentState.RUNNING:
            self._queued.append(text)
            self.console.print(t("cli.cli.queued", text=text), style="dim")
            return
        await self.orchestrator.send_message(self.main_agent_id, text)

    async def apply_provider_config(self, config: ProviderConfig) -> bool:
        """Make *config* live: swap the session's provider or create Main.

        Args:
            config: The provider configuration to apply.

        Returns:
            True when the provider is live. False only when the main
            session exists and is mid-run — retry after the turn ends.
        """
        provider = await (self.provider_factory or default_provider_factory)(
            config.model,
            {
                "api_key": config.api_key,
                "base_url": config.base_url,
                "timeout": config.timeout,
            },
        )
        session = self.session
        if session is None:
            self.orchestrator.set_default_provider(provider)
            self.main_agent_id = await self.orchestrator.create_agent(
                name="Main", description="Chief orchestrator"
            )
        elif session.state == AgentState.RUNNING:
            self.console.print(t("cli.cli.provider_busy"), style="yellow")
            return False
        else:
            session.provider = provider
            self.orchestrator.set_default_provider(provider)
        self.console.print(
            t("cli.cli.provider_swapped", model=config.model, base_url=config.base_url),
            style="green",
        )
        return True

    async def _consume_events(self) -> None:
        """Render queued backend events forever; drain on tree sync.

        Draining keys off the ``sync`` event rather than ``turn_end``:
        the orchestrator pushes sync from ``_run_session``'s finally
        block, *after* the session state has left RUNNING, so a queued
        message can never be sent into a turn that is still winding down.
        """
        while True:
            agent_id, event = await self.orchestrator.event_queue.get()
            if event.get("type") == "sync":
                await self._maybe_drain()
                continue
            self._renderer.render(event)

    async def _maybe_drain(self) -> None:
        """Send queued messages once the main agent is idle again."""
        session = self.session
        if session is None or session.state == AgentState.RUNNING:
            return
        while self._queued:
            text = self._queued.pop(0)
            await self.orchestrator.send_message(self.main_agent_id, text)

    async def _shutdown(self) -> None:
        """Cancel every live agent so no task outlives the REPL."""
        for node in self.orchestrator.get_tree_snapshot():
            agent_id = node.get("id")
            if agent_id is not None:
                await self.orchestrator.cancel_agent(agent_id)
