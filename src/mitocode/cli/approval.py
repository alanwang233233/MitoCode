"""Interactive approval for the MitoCode command line interface.

The backend's built-in tools only consult an approver when an operation escapes
the workspace root, so this module is the one place where a human is asked
"really?".  It implements the two-method-free :class:`Approver` protocol
(``async request(req) -> ApprovalDecision``) on top of the duck-typed
:class:`mitocode.cli.ui.CliUI` surface.

Four decisions shape the implementation.

**Policy first, dialog second.** :class:`ApprovalPolicy` mirrors the
``--permission-mode`` values verbatim, so ``CliArgs.permission_mode`` can be
passed straight in and the enum comparison still works.  ``allow`` and ``deny``
answer without touching the UI at all.

**The panel is the prompt message.** ``CliUI`` deliberately exposes no
"print a panel" method: everything that must appear above the input line is a
prompt message, which prompt_toolkit prints and then continues into the input
line.  :func:`format_permission_prompt` therefore composes the action, the
detail (verbatim — a truncated command is an unanswerable question) and the
reason into one multi-line message ending with :data:`PERMISSION_PROMPT`.

**One dialog at a time, and the cache is re-read inside the lock.** Two agents
can request approval concurrently, and both may arrive after a different
request has already been answered ``always``.  The lock serialises the dialogs;
re-checking the cache after acquiring it is what stops the second request from
asking about something the first one just approved.

**The detail never reaches the log.** The detail may be a command line carrying
a bearer token, so only the action and the reason are logged, at debug level.
The same reasoning narrows the "you are typing" notice to the action alone.

A refusal is a returned :class:`ApprovalDecision`, never an exception: the tool
call site turns ``DENY`` into a string the model can react to, while a raised
exception would abort the turn.  ``Ctrl+C`` and ``Ctrl+D`` inside the dialog are
converted to ``DENY`` for the same reason — an approval prompt must not be able
to cancel a turn.
"""

# Ellipsis bodies are required by typing and flagged by pylint below; keep both.
# pylint: disable=unnecessary-ellipsis

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Protocol, Set, Tuple, Union

from loguru import logger as loguru_logger

from mitocode.core.tool.builtin.approval import (
    ApprovalDecision,
    ApprovalRequest,
)

PERMISSION_PROMPT = "  Allow? [y] once  [n] no  [a] always this session "
"""The fixed answer vocabulary closing every permission panel."""

ALLOW_ONCE_ANSWERS: frozenset[str] = frozenset({"y", "yes"})
"""Answers that approve this request only."""

ALLOW_ALWAYS_ANSWERS: frozenset[str] = frozenset({"a", "always"})
"""Answers that approve this request and cache it for the session."""

TYPING_NOTICE_TEMPLATE = (
    "{action} needs permission · denied while you are typing · /agents to inspect"
)
"""Notice printed when a background request arrives during main input.

Only the action is interpolated: the detail may carry a credential and the
notice is printed on a line the user is currently typing on.
"""


class ApprovalPolicy(str, Enum):
    """How :class:`CliApprover` resolves a request before asking anyone.

    A ``str`` enum so the CLI's ``--permission-mode`` spelling can be passed to
    the constructor unchanged and compared to the enum members directly.
    Inherits ``str``, so external code may keep treating the value as a string.
    """

    ASK = "ask"
    ALLOW = "allow"
    DENY = "deny"


class ApprovalUI(Protocol):
    """The duck-typed slice of :class:`mitocode.cli.ui.CliUI` used here.

    Structural, exactly like :class:`mitocode.core.tool.builtin.approval.Approver`:
    any object with these attributes works, which is what lets tests inject a
    fake instead of a terminal. Not ``runtime_checkable`` — nothing here needs
    to test for it, and a promise of attribute presence would be weaker than the
    type checker's signature check.
    """

    prompt_active: bool
    """Whether the main REPL prompt is currently asking for input."""

    async def ask(self, message: str) -> str:
        """Ask the user one question.

        Args:
            message: Prompt shown above the input line.

        Returns:
            The submitted answer.

        Raises:
            KeyboardInterrupt: The user pressed Ctrl+C.
            EOFError: The user pressed Ctrl+D.
        """
        ...

    def pause_live(self) -> None:
        """Stop the transient spinner so the panel is not overwritten."""
        ...

    def resume_live(self) -> None:
        """Restart the spinner paused by :meth:`pause_live`."""
        ...

    def note(self, text: str) -> None:
        """Print one muted informational line.

        Args:
            text: Message shown to the user.
        """
        ...

    def error(self, text: str) -> None:
        """Print one error line.

        Args:
            text: Message shown to the user.
        """
        ...


def format_permission_prompt(req: ApprovalRequest) -> str:
    """Render the permission panel as the prompt message.

    Pure and side-effect free, so the REPL and the tests can render the same
    panel without a console. The detail is inserted with no truncation and no
    quoting: the user is being asked to approve exactly what was requested.

    Args:
        req: The operation awaiting a verdict.

    Returns:
        A multi-line message holding the action, detail and reason, closed by
        :data:`PERMISSION_PROMPT`.
    """
    return "\n".join(
        (
            "Permission required",
            "",
            f"  action   {req.action}",
            f"  detail   {req.detail}",
            f"  reason   {req.reason}",
            "",
            PERMISSION_PROMPT,
        )
    )


def _log_decision(
    req: ApprovalRequest, decision: ApprovalDecision, source: str
) -> None:
    """Record a verdict at debug level without the request's detail.

    Args:
        req: The request that was decided.
        decision: Verdict returned to the tool.
        source: How the verdict was reached, e.g. ``"policy"``, ``"cache"``,
            ``"answer"`` or ``"typing"``.
    """
    loguru_logger.debug(
        "Approval action={} reason={} source={} decision={}",
        req.action,
        req.reason,
        source,
        decision.value,
    )


class CliApprover:
    """Ask the terminal user before an out-of-workspace operation proceeds.

    One instance lives for the whole process and is installed in
    ``ToolContext.approver``. It is shared by the main agent and every
    background sub-agent, which is why every path funnels through one lock and
    one session cache.

    Attributes:
        ui: The prompt/notice surface, typically a
            :class:`mitocode.cli.ui.CliUI`.
        policy: The configured :class:`ApprovalPolicy`; ``ASK`` by default.
    """

    def __init__(
        self,
        ui: ApprovalUI,
        policy: Union[ApprovalPolicy, str] = ApprovalPolicy.ASK,
    ) -> None:
        """Initialise the approver.

        Args:
            ui: Prompt and notice surface used for dialogs.
            policy: ``ask``, ``allow`` or ``deny``. A plain string is accepted
                because that is how ``--permission-mode`` arrives from
                :class:`mitocode.cli.args.CliArgs`.

        Raises:
            ValueError: When *policy* is not one of the three accepted values.
        """
        self.ui = ui
        self.policy = ApprovalPolicy(policy)
        self._always: Set[Tuple[str, str]] = set()
        self._lock = asyncio.Lock()

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        """Return the verdict for *req*.

        The policy is applied without a dialog. Under ``ASK`` a request whose
        ``(action, detail)`` pair was already answered ``always`` is allowed
        silently; a request that arrives while the main prompt is active is
        denied with a notice instead of stealing the input line; everything
        else opens exactly one dialog.

        Args:
            req: The operation awaiting a verdict.

        Returns:
            ``ApprovalDecision.ALLOW`` or ``ApprovalDecision.DENY``. This method
            never raises for a refusal: the tool call site turns a denial into a
            refusal string for the model.
        """
        if self.policy is ApprovalPolicy.ALLOW:
            _log_decision(req, ApprovalDecision.ALLOW, "policy")
            return ApprovalDecision.ALLOW
        if self.policy is ApprovalPolicy.DENY:
            _log_decision(req, ApprovalDecision.DENY, "policy")
            return ApprovalDecision.DENY

        async with self._lock:
            return await self._ask_with_lock_held(req)

    def reset_cache(self) -> None:
        """Forget every ``always`` answer, as ``/permissions reset`` asks."""
        self._always.clear()

    def cached_requests(self) -> Set[Tuple[str, str]]:
        """Return the ``always``-approved requests of this session.

        Returns:
            A copy of the ``(action, detail)`` cache, so a caller such as
            ``/permissions`` can render or clear it without editing the state
            the approver relies on.
        """
        return set(self._always)

    async def _ask_with_lock_held(self, req: ApprovalRequest) -> ApprovalDecision:
        """Decide *req* while holding the serialisation lock.

        Args:
            req: The operation awaiting a verdict.

        Returns:
            The verdict for *req*, possibly served from the cache filled by an
            earlier request that was waiting on the same lock.
        """
        key = (req.action, req.detail)
        if key in self._always:
            _log_decision(req, ApprovalDecision.ALLOW, "cache")
            return ApprovalDecision.ALLOW

        if self.ui.prompt_active:
            self.ui.note(TYPING_NOTICE_TEMPLATE.format(action=req.action))
            _log_decision(req, ApprovalDecision.DENY, "typing")
            return ApprovalDecision.DENY

        decision, answer = await self._prompt(req)
        if decision is ApprovalDecision.ALLOW and answer in ALLOW_ALWAYS_ANSWERS:
            self._always.add(key)
        _log_decision(req, decision, "answer")
        return decision

    async def _prompt(self, req: ApprovalRequest) -> Tuple[ApprovalDecision, str]:
        """Show the permission panel and turn the answer into a verdict.

        The spinner is paused around the dialog and always resumed, including on
        an interrupt, so a refusal cannot leave the turn without a spinner.

        Args:
            req: The operation awaiting a verdict.

        Returns:
            The verdict and the normalised answer that produced it. The answer is
            empty when the dialog was interrupted.
        """
        self.ui.pause_live()
        try:
            try:
                answer = await self.ui.ask(format_permission_prompt(req))
            except (KeyboardInterrupt, EOFError):
                self.ui.note("Denied by user")
                return ApprovalDecision.DENY, ""
        finally:
            self.ui.resume_live()

        normalized = answer.strip().lower()
        if normalized in ALLOW_ONCE_ANSWERS or normalized in ALLOW_ALWAYS_ANSWERS:
            return ApprovalDecision.ALLOW, normalized

        self.ui.note("Denied by user")
        return ApprovalDecision.DENY, normalized


__all__ = [
    "ALLOW_ALWAYS_ANSWERS",
    "ALLOW_ONCE_ANSWERS",
    "ApprovalPolicy",
    "ApprovalUI",
    "CliApprover",
    "PERMISSION_PROMPT",
    "TYPING_NOTICE_TEMPLATE",
    "format_permission_prompt",
]
