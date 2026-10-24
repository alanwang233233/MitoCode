"""Rich rendering of bridged orchestrator events for the CLI.

The renderer consumes the *bridged* event dicts (``token`` / ``tool_call``
/ ``tool_execution`` / ``tool_result`` / ``turn_end`` / ``cancelled``)
that come off ``orchestrator.event_queue``. It prints plain lines rather
than a Rich ``Live`` layout, which keeps streaming output compatible with
``patch_stdout`` and makes the whole thing assertable through a recording
``Console``.
"""

import json
from typing import Any

from rich.console import Console

from mitocode.i18n import t


class EventRenderer:
    """Render bridged agent events onto a Rich console.

    Attributes:
        console: The console everything is printed to.
        orchestrator: Source of session names for sub-agent labels.
        main_agent_id: The agent whose output carries no prefix.
    """

    def __init__(self, console: Console, orchestrator: Any, main_agent_id: int) -> None:
        """Initialise the renderer.

        Args:
            console: Console that receives every printed line.
            orchestrator: Object providing ``get_session(agent_id)`` for
                sub-agent name lookup.
            main_agent_id: Agent id rendered without a label prefix.
        """
        self.console = console
        self.orchestrator = orchestrator
        self.main_agent_id = main_agent_id
        self._in_text = False

    def render(self, event: dict) -> None:
        """Render one bridged event.

        Args:
            event: A bridged event dict with at least ``type`` and
                ``agent_id`` keys.
        """
        etype = event.get("type")
        if etype == "token":
            self._token(event)
        elif etype == "tool_call":
            self._tool_call(event)
        elif etype == "tool_execution":
            self._tool_execution(event)
        elif etype == "tool_result":
            self._tool_result(event)
        elif etype in ("turn_end", "cancelled"):
            self._close_text()

    def _agent_prefix(self, agent_id: Any) -> str:
        """Return the label prefix for *agent_id*, empty for the main agent.

        Args:
            agent_id: The producing agent's id.

        Returns:
            A ``#id name`` label followed by a space for sub-agents, an
            empty string for the main agent.
        """
        if agent_id == self.main_agent_id:
            return ""
        name = ""
        session = self.orchestrator.get_session(agent_id)
        if session is not None:
            name = getattr(session, "name", "")
        return t("cli.cli.agent_label", id=agent_id, name=name) + " "

    def _token(self, event: dict) -> None:
        """Stream one assistant text delta."""
        if not self._in_text:
            self.console.print()
            self.console.print(
                self._agent_prefix(event.get("agent_id")), end="", markup=False
            )
            self._in_text = True
        self.console.print(event.get("content", ""), end="", markup=False)

    def _close_text(self) -> None:
        """End an open streaming text run."""
        if self._in_text:
            self.console.print()
            self._in_text = False

    def _tool_call(self, event: dict) -> None:
        """Render a tool-call header (name plus human description)."""
        self._close_text()
        action = event.get("action") or "tool"
        description = event.get("description") or ""
        line = self._agent_prefix(event.get("agent_id")) + f"▶ {action}"
        self.console.print(line, style="cyan", markup=False)
        if description:
            self.console.print(f"  {description}", style="dim", markup=False)

    def _tool_execution(self, event: dict) -> None:
        """Render the parsed arguments a tool is being invoked with."""
        args = event.get("args") or {}
        rendered = json.dumps(args, ensure_ascii=False, default=str)
        self.console.print(f"  args: {rendered}", style="dim", markup=False)

    def _tool_result(self, event: dict) -> None:
        """Render a tool result block, styled by success."""
        result = str(event.get("result", ""))
        marker = "✔" if event.get("success") else "✘"
        style = "green" if event.get("success") else "red"
        self.console.print(f"  {marker} {result}", style=style, markup=False)
