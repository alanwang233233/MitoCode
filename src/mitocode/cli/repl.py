"""Interactive read-eval-print loop: turns, signals and exit for the CLI.

:class:`Repl` is the only component that decides when a turn ends, what
``Ctrl+C`` means and when the process stops.  Everything it needs is injected:
the shared :class:`~mitocode.cli.state.CliState`, the terminal surface
(:class:`~mitocode.cli.ui.CliUI`), the single
:class:`~mitocode.cli.pump.EventPump` task and the optional approver.

Three rules shape the loop.

**The pump outlives the turn.**  :class:`Repl` never creates, cancels or awaits
the pump task; the application owns it for the whole process.  A turn only
calls :meth:`EventPump.wait_idle` before printing its summary, which is what
keeps the last streamed token from being dropped when the prompt returns.

**Cancellation is cooperative.**  ``Ctrl+C`` during a turn marks the session
cancelled *and* cancels its orchestrator task.  The runtime then synthesises a
result for the in-flight tool, so the history stays valid and the user can keep
talking without ``/clear``.  ``Ctrl+C`` at the prompt is prompt_toolkit's
``KeyboardInterrupt``: the first one only warns, and a second one within
:data:`CTRL_C_WINDOW` exits.  ``Ctrl+D`` (``EOFError``) always exits.

**Input is replayed through one path.**  Optionally the ``--initial-prompt``
message is submitted first, then every prompt line is either empty (ignored),
a slash command (:func:`~mitocode.cli.commands.dispatch`) or a user turn.
A command or turn failure is logged and reported but never ends the process:
only ``/exit``, ``/quit``, EOF or the double ``Ctrl+C`` stop the loop, which
returns ``0``.  ``KeyboardInterrupt`` and ``CancelledError`` are not failures
and keep propagating.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from loguru import logger as loguru_logger
from prompt_toolkit.patch_stdout import patch_stdout

from mitocode.cli.commands import dispatch
from mitocode.cli.state import CliState, CommandContext

if TYPE_CHECKING:
    from mitocode.cli.approval import CliApprover
    from mitocode.cli.pump import EventPump
    from mitocode.cli.ui import CliUI

CTRL_C_WINDOW = 2.0
"""Seconds within which a second ``Ctrl+C`` means "exit"."""

CTRL_C_HINT = "Press Ctrl+C again to exit"
"""Line printed after the first ``Ctrl+C`` at the prompt."""


class Repl:  # pylint: disable=too-many-instance-attributes
    """Run the input loop and execute one turn at a time.

    One instance lives for the whole process.  It reads :attr:`CliState.main_id`
    afresh on every turn because ``/clear`` may have replaced the main session
    while a command ran, and it tracks only the *current* orchestrator task in
    :attr:`_active_turn` so :meth:`handle_sigint` has something to cancel.

    Attributes:
        state: Mutable process state the loop reads and updates.
        ui: Terminal surface input is read from and output is rendered through.
        pump: Event pump whose idle state gates the turn summary.
        approver: Approval object forwarded to slash commands, or ``None``.
        _active_turn: Orchestrator task of the running turn, or ``None``.
        _shutdown: Set once the loop must stop (second ``Ctrl+C`` or ``/exit``).
        _last_ctrl_c_at: Monotonic timestamp of the last ``Ctrl+C``; a value
            far in the past so the first press never exits.
        _cancel_requested: Whether :meth:`handle_sigint` -- not the caller --
            cancelled the current turn.  Only that cancellation is swallowed;
            any other one is a shutdown request and keeps unwinding.
    """

    def __init__(
        self,
        state: CliState,
        ui: "CliUI",
        pump: "EventPump",
        approver: Optional["CliApprover"] = None,
    ) -> None:
        """Wire the loop to its collaborators.

        Args:
            state: Process state holding the main id, arguments and counters.
            ui: Terminal surface used for input and output.
            pump: The process-wide event pump; this object never owns its task.
            approver: Approval object handed to slash commands, or ``None``.
        """
        self.state = state
        self.ui = ui
        self.pump = pump
        self.approver = approver
        self._active_turn: "Optional[asyncio.Task]" = None
        self._shutdown = asyncio.Event()
        self._last_ctrl_c_at = float("-inf")
        self._cancel_requested = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> int:
        """Read input and execute turns until the user exits.

        The whole loop is wrapped in
        :func:`prompt_toolkit.patch_stdout.patch_stdout` so a background agent's
        output printed while the user is typing is redrawn above the input line
        instead of shredding it.

        Returns:
            ``0`` for every clean exit path (EOF, ``/exit``, double ``Ctrl+C``).
        """
        initial = (self.state.args.initial_prompt or "").strip()

        with patch_stdout():
            if initial:
                await self.run_turn(initial)
            while not self._shutdown.is_set():
                try:
                    line = await self.ui.read_prompt()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    if self._note_ctrl_c():
                        break
                    continue
                await self._handle_line(line)
        return 0

    async def _handle_line(self, line: str) -> None:
        """Route one submitted line to a command or a turn.

        Blank lines are ignored: an accidental Enter must not spend a request.
        A line starting with ``/`` is a command, and only a command that asks to
        exit actually stops the loop; an unknown command already printed its
        error and the REPL continues.

        The method is total for ordinary failures: an exception raised by a
        command handler or by a turn is logged and reported through ``ui.error``
        instead of escaping :meth:`run`, because a failed request is not a reason
        to kill the CLI.  ``KeyboardInterrupt`` and ``CancelledError`` are
        ``BaseException`` and deliberately keep propagating.

        Args:
            line: Raw text submitted at the prompt.
        """
        text = line.strip()
        if not text:
            return
        if text.startswith("/"):
            ctx = CommandContext(self.state, self.ui, self.approver)
            try:
                result = await dispatch(text, ctx)
            except Exception as exc:  # pylint: disable=broad-except
                loguru_logger.exception("Slash command failed")
                self.ui.error(f"Command failed: {exc}")
                return
            if result is not None and result.exit_requested:
                self._shutdown.set()
            return
        try:
            await self.run_turn(text)
        except Exception as exc:  # pylint: disable=broad-except
            loguru_logger.exception("Turn failed")
            self.ui.error(f"Turn failed: {exc}")

    def _note_ctrl_c(self) -> bool:
        """Record an interrupt at the prompt and decide whether to exit.

        Returns:
            ``True`` when this press falls inside :data:`CTRL_C_WINDOW` of the
            previous one, which sets :attr:`_shutdown`; ``False`` when it is the
            first press, in which case the hint is printed.
        """
        now = time.monotonic()
        if now - self._last_ctrl_c_at <= CTRL_C_WINDOW:
            self._shutdown.set()
            return True
        self._last_ctrl_c_at = now
        self.ui.note(CTRL_C_HINT)
        return False

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------
    async def run_turn(self, text: str) -> None:
        """Submit *text* to the main session and wait for the turn to finish.

        Cancellation requested by :meth:`handle_sigint` is consumed here: the
        session has already been marked cancelled, so the turn ends quietly and
        the loop stays usable.  A cancellation of *this* coroutine, however, is
        a shutdown request and is re-raised after the summary.

        The method always ends by stopping the thinking spinner, waiting for
        the pump to drain and printing exactly one ``ui.finish_turn`` summary,
        whatever happened to the turn.

        Args:
            text: User message to append to the main session.

        Raises:
            asyncio.CancelledError: When the caller cancels the turn itself.
        """
        started = time.monotonic()
        tokens_before = self.state.estimate_tokens()
        calls_before = self.state.tool_call_count
        task: "Optional[asyncio.Task]" = None
        try:
            self.ui.start_thinking()
            try:
                self.state.main_session()
            except RuntimeError as exc:
                self.ui.error(str(exc))
                return

            task = await self.state.orchestrator.send_message(self.state.main_id, text)
            if task is None:
                self.ui.error(f"Main agent #{self.state.main_id} does not exist")
                return

            self._active_turn = task
            self._cancel_requested = False
            try:
                await task
            except asyncio.CancelledError:
                # Cancelling the awaiting coroutine also cancels the task it
                # awaits, so "the task is cancelled" alone does not say who did
                # it.  Only a cancellation requested by :meth:`handle_sigint` is
                # the user interrupting a turn; anything else is the caller
                # shutting this coroutine down and must keep unwinding.
                if not (self._active_turn is task and self._cancel_requested):
                    raise
        finally:
            self._active_turn = None
            self.ui.stop_thinking()
            await self.pump.wait_idle()
            tokens = max(0, self.state.estimate_tokens() - tokens_before)
            tool_calls = max(0, self.state.tool_call_count - calls_before)
            self.ui.finish_turn(time.monotonic() - started, tokens, tool_calls)
            self.state.turn_count += 1

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def handle_sigint(self) -> None:
        """Handle ``SIGINT`` delivered through the event loop's signal handler.

        A running turn is cancelled cooperatively: the session's cancel event
        stops the stream at its next checkpoint and the orchestrator task is
        cancelled so a long tool is interrupted immediately.  With no turn
        running -- the user is at the prompt -- the first press only records the
        time and a second press inside :data:`CTRL_C_WINDOW` sets
        :attr:`_shutdown`.
        """
        now = time.monotonic()
        task = self._active_turn
        if task is not None:
            try:
                self.state.main_session().cancel()
            except RuntimeError:
                pass
            self._cancel_requested = True
            task.cancel()
            self._last_ctrl_c_at = now
            return

        if now - self._last_ctrl_c_at <= CTRL_C_WINDOW:
            self._shutdown.set()
        self._last_ctrl_c_at = now


__all__ = ["CTRL_C_HINT", "CTRL_C_WINDOW", "Repl"]
