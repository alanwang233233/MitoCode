"""Single consumer of the orchestrator's event queue that drives the CLI UI.

The CLI has exactly one :class:`EventPump` task for the whole process. It reads
``(agent_id, event)`` tuples from ``orchestrator.event_queue`` and renders them
through :class:`~mitocode.cli.ui.CliUI`. Keeping a single consumer is what makes
the transcript orderly: the UI never needs a lock, because nothing else writes
to it from an event.

Two rules shape the rendering. The main agent owns the transcript, so its
tokens, tool lines, cancellations and failures are all shown. A background
agent only gets a dim ``↳ <agent>`` line per tool call and turn boundary: its
prose never reaches the main reading area, which the user can inspect with
``/agents`` instead.

:meth:`EventPump.wait_idle` is the second half of the contract. A turn is only
allowed to redraw the prompt once every event it produced has been rendered, so
the last token cannot be lost. The pump therefore counts events that have been
taken off the queue but not handled yet (:attr:`EventPump._pending`) and
``wait_idle`` waits on a condition until that count is zero *and* the queue is
empty. Two details make that exact:

* An item is removed from the queue and counted as pending without an
  intervening ``await``, so no other coroutine can ever observe "queue empty,
  nothing pending" between the two steps.
* Waiters re-test both values while holding the condition lock, and the pump
  signals the condition under the same lock after handling an event.

Handlers are deliberately synchronous. Everything they call -- the streaming
renderer and the UI's print methods -- is synchronous too, so an event cannot
stop half-rendered; the only suspension point in the loop is the queue itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger as loguru_logger
from rich.text import Text

from mitocode.cli.format import ERROR_PREFIX, format_result, format_tool_call
from mitocode.cli.state import CliState
from mitocode.cli.ui import CliUI

AGENT_PREFIX = "↳ "
"""Gutter prefixed to every line contributed by a background agent."""

ToolCall = Tuple[str, Mapping[str, Any]]
"""The call name and its arguments, remembered until the result arrives."""


def _describe(payload: Any) -> str:
    """Return a log-safe description of a queue item or event.

    Logging must never raise, and it must never print a payload's contents: an
    event can carry model output. A mapping is described by its ``type`` key; any
    other object only by its class name, so ``None``, a string or a list cannot
    break the log call that is supposed to contain a failure.

    Args:
        payload: Anything taken off the queue or out of an event.

    Returns:
        The event's ``type`` value as text, ``"<missing>"`` for a mapping
        without one, or ``"<ClassName>"`` for a non-mapping payload.
    """
    if isinstance(payload, Mapping):
        return str(payload.get("type", "<missing>"))
    return f"<{type(payload).__name__}>"


class EventPump:
    """Render orchestrator events into the CLI until cancelled.

    One instance lives for the whole process. The REPL starts :meth:`run` as a
    task and awaits :meth:`wait_idle` at the end of every turn; commands may
    replace :attr:`CliState.main_id` at any time (``/clear`` does), so every
    event handler reads it afresh instead of caching the main agent's id.

    Attributes:
        _queue: Source of ``(agent_id, event)`` tuples; the orchestrator owns
            the writing end.
        _ui: Terminal surface every line is rendered through.
        _state: Mutable process state the pump reads (main id, agents, verbose,
            workspace) and updates (agent table, counters).
        _cond: Condition signalled whenever the pump becomes idle.
        _pending: Events taken off the queue whose handler has not finished yet.
        _calls: ``(agent_id, call id)`` to ``(tool name, arguments)``, so a
            result can be formatted with the arguments of its own call even
            when two agents reuse a call id.
    """

    def __init__(
        self,
        queue: "asyncio.Queue[Tuple[int, Dict[str, Any]]]",
        ui: CliUI,
        state: CliState,
    ) -> None:
        """Wire the pump to its queue, surface and state.

        Args:
            queue: The orchestrator's event queue. The pump is its only
                consumer.
            ui: UI every event is rendered through.
            state: Process state holding the main agent's id and the counters
                the pump maintains.
        """
        self._queue = queue
        self._ui = ui
        self._state = state
        self._cond = asyncio.Condition()
        self._pending = 0
        self._calls: Dict[Tuple[int, str], ToolCall] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume events until the task is cancelled.

        The loop never returns on its own: the queue is a live stream of turns.
        It is total: no payload -- ``None``, a string, a list, an object without
        ``.get``, a bare non-pair item or a handler that raises -- can escape it.
        Anything unusable is logged with a type-safe description and skipped, so
        a malformed event cannot kill the pump and park every later
        :meth:`wait_idle` caller forever.

        Raises:
            asyncio.CancelledError: When the owning application cancels the
                task at shutdown.
        """
        while True:
            item = await self._queue.get()
            # Count the item before any other suspension point: a concurrent
            # ``wait_idle`` must never see an empty queue and a zero count while
            # this item is still in flight.
            self._pending += 1
            try:
                self._consume(item)
            except Exception:  # pylint: disable=broad-except
                # Belt and braces: ``_consume`` already contains its own
                # failures, and ``CancelledError`` is a ``BaseException`` so
                # shutdown still propagates. Never touch the payload here.
                loguru_logger.exception(
                    "CLI event pump failed while consuming a {!r} queue item",
                    type(item).__name__,
                )
            finally:
                await self._settle()

    def _consume(self, item: Any) -> None:
        """Unpack one queue item and render its event, containing failures.

        Args:
            item: A ``(agent_id, event)`` pair as written by the orchestrator.
                A queue that is written to by foreign code may also contain
                something else, which is logged and dropped.

        Raises:
            asyncio.CancelledError: When the pump is cancelled mid-event.
        """
        try:
            agent_id, event = item
        except (TypeError, ValueError):
            loguru_logger.warning(
                "CLI event pump dropped a malformed queue item {}",
                _describe(item),
            )
            return

        try:
            self._handle(agent_id, event)
        except Exception:  # pylint: disable=broad-except
            loguru_logger.exception(
                "CLI event pump could not render event type {}",
                _describe(event),
            )

    async def wait_idle(self) -> None:
        """Wait until every queued event has been rendered.

        Returns only when the queue is empty and no event is being handled. A
        caller uses it after a turn finishes, which is what keeps the last
        token from being dropped when the prompt comes back.

        Raises:
            asyncio.CancelledError: When the waiter itself is cancelled.
        """
        async with self._cond:
            while self._pending or not self._queue.empty():
                await self._cond.wait()

    async def _settle(self) -> None:
        """Mark one event as handled and wake ``wait_idle`` when idle.

        Raises:
            asyncio.CancelledError: When the pump is cancelled while settling,
                which only happens at shutdown.
        """
        async with self._cond:
            self._pending -= 1
            if not self._pending and self._queue.empty():
                self._cond.notify_all()

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    def _handle(self, agent_id: int, event: Any) -> None:
        """Route one event to its handler, ignoring anything unusable.

        A non-mapping payload is logged and dropped rather than handed to a
        handler: the queue is filled by the orchestrator, but the pump is the
        component that must stay alive when the contract is broken.

        Args:
            agent_id: Id of the agent that produced the event; ``-1`` for tree
                snapshots.
            event: Adapted event dict from the orchestrator's event bridge.
                Any other object is refused.

        Raises:
            asyncio.CancelledError: When the pump is cancelled mid-event.
        """
        if not isinstance(event, Mapping):
            loguru_logger.warning(
                "CLI event pump ignored a non-mapping event {}",
                _describe(event),
            )
            return

        event_type = event.get("type")
        if event_type == "sync":
            self._handle_sync(event)
        elif event_type == "token":
            self._handle_token(agent_id, event)
        elif event_type == "tool_start":
            self._handle_tool_start(agent_id, event)
        elif event_type == "tool_result":
            self._handle_tool_result(agent_id, event)
        elif event_type == "turn_end":
            self._handle_turn_end(agent_id)
        elif event_type == "turn_cancelled":
            self._handle_turn_cancelled(agent_id)
        elif event_type == "error":
            self._handle_error(agent_id, event)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_sync(self, event: Mapping[str, Any]) -> None:
        """Replace the agent table from a tree snapshot without printing.

        ``/agents`` reads :attr:`CliState.agents` and the prompt toolbar shows
        its length, so the snapshot is stored silently rather than rendered.

        Args:
            event: ``sync`` event carrying an ``agents`` list of node dicts.
        """
        agents = event.get("agents")
        self._state.agents = list(agents) if isinstance(agents, (list, tuple)) else []
        self._ui.agent_count = len(self._state.agents)

    def _handle_token(self, agent_id: int, event: Mapping[str, Any]) -> None:
        """Stream main-agent prose; drop a background agent's prose.

        Args:
            agent_id: Id of the emitting agent.
            event: ``token`` event carrying a ``content`` delta.
        """
        if not self._is_main(agent_id):
            return
        content = event.get("content")
        if not isinstance(content, str) or not content:
            return
        self._state.total_chars += len(content)
        self._ui.stream.feed(content)

    def _handle_tool_start(self, agent_id: int, event: Mapping[str, Any]) -> None:
        """Render a tool header and remember the call's arguments.

        The arguments are recorded before rendering because the result handler
        needs them to build the preview (``Edit`` prints a diff from its
        arguments, not from its output).

        Args:
            agent_id: Id of the agent making the call.
            event: ``tool_start`` event with ``id``, ``name``, ``args`` and
                ``display_param``.
        """
        call_id = str(event.get("id", ""))
        name = str(event.get("name", ""))
        args = event.get("args")
        if not isinstance(args, Mapping):
            args = {}
        display_param = str(event.get("display_param") or "")

        self._calls[(agent_id, call_id)] = (name, args)
        call = format_tool_call(name, args, display_param, self._root())
        if self._is_main(agent_id):
            self._state.tool_call_count += 1
            self._ui.begin_tool(call)
        else:
            self._ui.begin_tool(call, self._prefix(agent_id))

    def _handle_tool_result(self, agent_id: int, event: Mapping[str, Any]) -> None:
        """Render a tool result, matched to its call by agent and call id.

        A failure -- a falsy ``success`` or an ``Error:``-prefixed body -- is
        rendered with the error style so the line stands out even when the tool
        only returned a message.

        Args:
            agent_id: Id of the agent that produced the result.
            event: ``tool_result`` event with ``id``, ``result`` and
                ``success``.
        """
        call_id = str(event.get("id", ""))
        name, args = self._calls.pop((agent_id, call_id), ("", {}))
        raw = event.get("result")
        result = "" if raw is None else str(raw)
        preview = format_result(
            name,
            args,
            result,
            verbose=bool(self._state.verbose),
            root=self._root(),
        )
        failed = event.get(
            "success", True
        ) is False or preview.plain.strip().lower().startswith(ERROR_PREFIX)
        if failed:
            preview = Text(preview.plain, style="mito.error")

        if self._is_main(agent_id):
            self._ui.finish_tool(preview)
        else:
            self._ui.finish_tool(preview, self._prefix(agent_id))

    def _handle_turn_end(self, agent_id: int) -> None:
        """Flush the main transcript, or note that a background turn finished.

        Args:
            agent_id: Id of the agent whose turn ended.
        """
        self._forget_calls(agent_id)
        if self._is_main(agent_id):
            self._ui.stream.flush()
            return
        self._ui.note(f"{self._prefix(agent_id)}finished")

    def _handle_turn_cancelled(self, agent_id: int) -> None:
        """Flush the main transcript and report the interruption.

        Args:
            agent_id: Id of the agent whose turn was cancelled.
        """
        self._forget_calls(agent_id)
        if self._is_main(agent_id):
            self._ui.stream.flush()
            self._ui.note("Interrupted.")
            return
        self._ui.note(f"{self._prefix(agent_id)}cancelled")

    def _handle_error(self, agent_id: int, event: Mapping[str, Any]) -> None:
        """Report a failed agent run.

        The message was already scrubbed of credentials by the orchestrator;
        it is shown verbatim so the user can act on it.

        Args:
            agent_id: Id of the agent whose run failed.
            event: ``error`` event carrying a ``message``.
        """
        message = str(event.get("message") or "")
        if self._is_main(agent_id):
            self._ui.error(f"Error: {message}")
            return
        self._ui.note(f"{self._prefix(agent_id)}error: {message}")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _is_main(self, agent_id: int) -> bool:
        """Return whether *agent_id* is the interactive main agent.

        Args:
            agent_id: Id carried by an event.

        Returns:
            ``True`` for the agent the user is talking to.
        """
        return agent_id == self._state.main_id

    def _root(self) -> Path:
        """Return the workspace root used to relativise displayed paths.

        Returns:
            The workspace path, or the current directory when the workspace
            object does not expose one.
        """
        path = getattr(self._state.workspace, "path", "") or "."
        return Path(path)

    def _agent_name(self, agent_id: int) -> str:
        """Look up a background agent's display name.

        Args:
            agent_id: Id carried by an event.

        Returns:
            The name from the last tree snapshot, falling back to
            ``agent #<id>`` when no snapshot mentions the agent yet.
        """
        for node in self._state.agents:
            if isinstance(node, Mapping) and node.get("id") == agent_id:
                name = node.get("name")
                if name:
                    return str(name)
        return f"agent #{agent_id}"

    def _prefix(self, agent_id: int) -> str:
        """Build the dim gutter that marks a background agent's lines.

        Args:
            agent_id: Id of the background agent.

        Returns:
            ``"↳ <name> "``, ready to be prepended to a rendered line.
        """
        return f"{AGENT_PREFIX}{self._agent_name(agent_id)} "

    def _forget_calls(self, agent_id: int) -> None:
        """Drop the remembered calls of one agent when its turn ends.

        A cancelled turn never reports its pending tool results, so the table
        would otherwise grow for the life of the process.

        Args:
            agent_id: Id of the agent whose turn just ended.
        """
        stale: List[Tuple[int, str]] = [
            key for key in self._calls if key[0] == agent_id
        ]
        for key in stale:
            del self._calls[key]


__all__ = ["AGENT_PREFIX", "EventPump"]
