"""Tests for the interactive CLI approval policy and approver.

Every test injects a fake UI, so no test needs a terminal, a prompt_toolkit
session or a running agent. The fake records what the approver asked, how many
dialogs overlapped, and whether the live spinner was paused and resumed.

The last two tests are the ones that encode the non-obvious contract: only one
permission dialog may exist at a time even when two agents request approval
concurrently, and the request's ``detail`` — which may be a command carrying a
credential — must never reach the log.
"""

import asyncio
import os
import sys
from typing import List

import pytest
from loguru import logger as loguru_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.approval import (
    PERMISSION_PROMPT,
    ApprovalPolicy,
    CliApprover,
    format_permission_prompt,
)
from mitocode.core.tool.builtin.approval import (
    ApprovalDecision,
    ApprovalRequest,
    Approver,
)


class _FakeUI:
    """Scripted stand-in for :class:`mitocode.cli.ui.CliUI`.

    ``ask`` pops the next scripted answer; an answer that is a
    :class:`BaseException` is raised instead of returned, which is how the
    tests reach the Ctrl+C and Ctrl+D paths without a terminal. ``hook`` is an
    optional coroutine awaited inside ``ask``, used to hold a dialog open while
    another request runs.

    Attributes:
        answers: Remaining scripted answers, popped from the front.
        prompt_active: Whether the main prompt is asking for input.
        notes: Everything passed to ``note`` or ``error``, in order.
        messages: Every message passed to ``ask``, in order.
        live: Net spinner state; ``pause_live`` decrements it, ``resume_live``
            increments it, so a balanced pair leaves it at zero.
        asks: Number of ``ask`` calls.
        max_active_asks: Highest number of overlapping ``ask`` calls observed.
        hook: Optional coroutine called as ``hook(ui, message)`` inside ``ask``.
    """

    def __init__(self, answers=(), *, prompt_active: bool = False) -> None:
        """Store the scripted answers.

        Args:
            answers: Values or exceptions returned/raised in order.
            prompt_active: Initial value of the main-prompt flag.
        """
        self.answers = list(answers)
        self.prompt_active = prompt_active
        self.notes: List[str] = []
        self.messages: List[str] = []
        self.live = 0
        self.asks = 0
        self.active_asks = 0
        self.max_active_asks = 0
        self.hook = None

    async def ask(self, message: str) -> str:
        """Record *message* and return the next scripted answer.

        Args:
            message: Prompt shown to the user.

        Returns:
            The next scripted answer.

        Raises:
            BaseException: When the scripted answer is an exception instance.
        """
        self.asks += 1
        self.messages.append(message)
        self.active_asks += 1
        self.max_active_asks = max(self.max_active_asks, self.active_asks)
        try:
            if self.hook is not None:
                await self.hook(self, message)
            answer = self.answers.pop(0)
        finally:
            self.active_asks -= 1
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def pause_live(self) -> None:
        """Stop the spinner, one step."""
        self.live -= 1

    def resume_live(self) -> None:
        """Restart the spinner, undoing one pause."""
        self.live += 1

    def note(self, text: str) -> None:
        """Record an informational line.

        Args:
            text: Message shown to the user.
        """
        self.notes.append(text)

    def error(self, text: str) -> None:
        """Record an error line, in the same list as notes.

        Args:
            text: Message shown to the user.
        """
        self.notes.append(text)


_REQ = ApprovalRequest("bash", "rm -rf /tmp/x", "outside workspace root")


def _joined_notes(ui: _FakeUI) -> str:
    """Join recorded notes, lowercased.

    Args:
        ui: Fake UI whose notes to inspect.

    Returns:
        All notes as one lowercase string for substring assertions.
    """
    return "\n".join(ui.notes).lower()


@pytest.mark.asyncio
async def test_always_caches_request():
    """Answering ``a`` allows every later identical request without asking."""
    ui = _FakeUI(["a", "unused"])
    approver = CliApprover(ui)

    assert await approver.request(_REQ) is ApprovalDecision.ALLOW
    assert await approver.request(_REQ) is ApprovalDecision.ALLOW

    assert len(ui.answers) == 1, "the second request must not consume an answer"
    assert ui.asks == 1, "the cached request must not open a second dialog"
    assert ui.live == 0, "pause_live and resume_live must balance"


@pytest.mark.asyncio
async def test_once_answer_allows_without_caching():
    """``yes`` allows this request but asks again next time."""
    ui = _FakeUI(["yes", "n"])
    approver = CliApprover(ui)

    assert await approver.request(_REQ) is ApprovalDecision.ALLOW
    assert await approver.request(_REQ) is ApprovalDecision.DENY
    assert approver.cached_requests() == set()
    assert ui.asks == 2, "a one-shot answer must not be remembered"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["n", "no", "", "  ", "later", "Y E S"])
async def test_other_answers_deny_and_say_so(answer: str):
    """Anything that is not y/yes/a/always is a denial with a visible note."""
    ui = _FakeUI([answer])
    decision = await CliApprover(ui).request(_REQ)

    assert decision is ApprovalDecision.DENY
    assert "denied" in _joined_notes(ui), "a refusal must be visible to the user"
    assert ui.live == 0, "the spinner must be resumed even on refusal"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, decision, cached",
    [
        ("y", ApprovalDecision.ALLOW, False),
        ("Y", ApprovalDecision.ALLOW, False),
        ("a", ApprovalDecision.ALLOW, True),
        ("always", ApprovalDecision.ALLOW, True),
    ],
)
async def test_answer_mapping(answer, decision, cached):
    """Lower-case the answer and map y/yes and a/always to allow."""
    ui = _FakeUI([answer])
    approver = CliApprover(ui)

    assert await approver.request(_REQ) is decision
    assert approver.cached_requests() == ({("bash", _REQ.detail)} if cached else set())


@pytest.mark.asyncio
async def test_deny_when_user_is_typing():
    """A background request while the prompt is active is refused, not queued."""
    ui = _FakeUI([], prompt_active=True)
    req = ApprovalRequest("read", "/etc/passwd", "outside")

    decision = await CliApprover(ui).request(req)

    assert decision is ApprovalDecision.DENY
    assert ui.asks == 0, "the main input line must never be pre-empted"
    assert ui.messages == [], "no dialog may be opened while the user is typing"
    assert ui.notes, "the refusal has to be explained"
    assert "denied" in ui.notes[0].lower()
    assert "read" in ui.notes[0].lower(), "the note names the action"
    assert req.detail not in ui.notes[0], (
        "the notice names the action only; the detail may contain a credential "
        "and already appears nowhere else"
    )


@pytest.mark.asyncio
async def test_policy_allow_and_deny_bypass_prompt():
    """The two fixed policies never open a dialog."""
    ui = _FakeUI([])

    assert (
        await CliApprover(ui, ApprovalPolicy.ALLOW).request(
            ApprovalRequest("bash", "x", "y")
        )
        is ApprovalDecision.ALLOW
    )
    assert (
        await CliApprover(ui, ApprovalPolicy.DENY).request(
            ApprovalRequest("bash", "x", "y")
        )
        is ApprovalDecision.DENY
    )
    assert ui.answers == []
    assert ui.asks == 0


@pytest.mark.asyncio
async def test_policy_accepts_the_string_from_cli_args():
    """``--permission-mode allow`` arrives as a plain string, not the enum."""
    ui = _FakeUI([])

    assert (
        await CliApprover(ui, "allow").request(_REQ) is ApprovalDecision.ALLOW
    ), "the constructor must normalise the CLI's string"
    assert ui.asks == 0


def test_policy_values_match_the_cli_modes():
    """The enum values are exactly the ``--permission-mode`` choices."""
    assert [policy.value for policy in ApprovalPolicy] == ["ask", "allow", "deny"]
    assert ApprovalPolicy.ASK == "ask", "a str enum compares to its CLI spelling"


@pytest.mark.asyncio
async def test_approver_satisfies_the_protocol():
    """The duck-typed seam still satisfies the backend protocol nominally."""
    assert isinstance(CliApprover(_FakeUI([])), Approver)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), EOFError()])
async def test_interrupt_denies_without_cancelling(interrupt):
    """Ctrl+C/Ctrl+D during the dialog is a denial, and the spinner resumes."""
    ui = _FakeUI([interrupt])
    decision = await CliApprover(ui).request(_REQ)

    assert decision is ApprovalDecision.DENY
    assert "denied" in _joined_notes(ui)
    assert ui.live == 0, "an interrupted dialog must still resume the spinner"


@pytest.mark.asyncio
async def test_prompt_shows_action_detail_and_reason_verbatim():
    """The dialog carries the three facts a human needs, unmodified."""
    detail = 'curl -H "Authorization: Bearer abc" https://example.invalid'
    req = ApprovalRequest("bash", detail, "outside workspace root")
    ui = _FakeUI(["n"])

    await CliApprover(ui).request(req)

    message = ui.messages[0]
    assert detail in message, "the detail is shown exactly as the tool sent it"
    assert req.reason in message
    assert req.action in message
    assert message.endswith(
        PERMISSION_PROMPT
    ), "the fixed answer vocabulary closes the panel"


def test_prompt_format_is_pure_and_stable():
    """The panel is a pure string, so the REPL can render it without a UI."""
    rendered = format_permission_prompt(_REQ)

    assert rendered.count("\n") >= 4, "the panel is a multi-line message"
    assert "Permission required" in rendered
    assert (
        _REQ.detail in rendered and _REQ.reason in rendered and _REQ.action in rendered
    )
    assert "y" in PERMISSION_PROMPT and "a" in PERMISSION_PROMPT


@pytest.mark.asyncio
async def test_cache_key_is_action_and_detail_exactly():
    """A different detail asks again: the cache is not per-action."""
    ui = _FakeUI(["a", "n"])
    approver = CliApprover(ui)

    first = ApprovalRequest("read", "/etc/passwd", "outside workspace root")
    second = ApprovalRequest("read", "/etc/hosts", "outside workspace root")

    assert await approver.request(first) is ApprovalDecision.ALLOW
    assert await approver.request(second) is ApprovalDecision.DENY
    assert approver.cached_requests() == {("read", "/etc/passwd")}
    assert ui.asks == 2


@pytest.mark.asyncio
async def test_cached_requests_is_a_copy_and_reset_clears_it():
    """Callers such as ``/permissions`` must not be able to edit the cache."""
    ui = _FakeUI(["a"])
    approver = CliApprover(ui)
    await approver.request(_REQ)

    snapshot = approver.cached_requests()
    snapshot.clear()
    assert approver.cached_requests() == {
        ("bash", _REQ.detail)
    }, "cached_requests must hand out a copy"

    approver.reset_cache()
    assert approver.cached_requests() == set()


@pytest.mark.asyncio
async def test_reset_cache_makes_the_same_request_ask_again():
    """After ``/permissions reset`` the identical request prompts once more."""
    ui = _FakeUI(["a", "n"])
    approver = CliApprover(ui)
    await approver.request(_REQ)

    approver.reset_cache()

    assert await approver.request(_REQ) is ApprovalDecision.DENY
    assert ui.asks == 2


@pytest.mark.asyncio
async def test_concurrent_requests_are_serialised_by_one_lock():
    """Two agents must never own a dialog at the same time."""
    ui = _FakeUI(["y", "y"])
    order: List[str] = []

    async def hook(fake_ui, message):
        """Hold the first dialog open so a second request has time to arrive."""
        order.append("enter")
        await asyncio.sleep(0.02)
        order.append("exit")

    ui.hook = hook
    approver = CliApprover(ui)
    first = ApprovalRequest("bash", "one", "outside workspace root")
    second = ApprovalRequest("read", "/etc/hosts", "outside workspace root")

    results = await asyncio.gather(approver.request(first), approver.request(second))

    assert results == [ApprovalDecision.ALLOW, ApprovalDecision.ALLOW]
    assert ui.max_active_asks == 1, "only one permission dialog may be open"
    assert order == [
        "enter",
        "exit",
        "enter",
        "exit",
    ], "the dialogs must not interleave"


@pytest.mark.asyncio
async def test_second_concurrent_request_rechecks_the_cache_inside_the_lock():
    """``always`` answered by the first request satisfies the queued second one."""
    ui = _FakeUI(["a", "n"])
    approver = CliApprover(ui)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hook(fake_ui, message):
        """Keep the first dialog open until the second request is waiting."""
        entered.set()
        await release.wait()

    ui.hook = hook
    first = asyncio.create_task(approver.request(_REQ))
    await entered.wait()
    second = asyncio.create_task(approver.request(_REQ))
    await asyncio.sleep(0)
    release.set()

    assert await first is ApprovalDecision.ALLOW
    assert (
        await second is ApprovalDecision.ALLOW
    ), "the queued request must see the cache the first one just filled"
    assert (
        ui.asks == 1
    ), "the cache is re-checked after the lock is acquired, not before"
    assert ui.answers == ["n"], "the second scripted answer must stay unused"


@pytest.mark.asyncio
async def test_detail_is_never_logged():
    """Debug logging names the action and reason but never the detail."""
    secret = "curl -H 'Authorization: Bearer sk-test-secret' https://example.invalid"
    req = ApprovalRequest("bash", secret, "outside workspace root")
    ui = _FakeUI(["y"])
    lines: List[str] = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        await CliApprover(ui).request(req)
    finally:
        loguru_logger.remove(sink)

    assert not [
        line for line in lines if secret in line
    ], f"the request detail may carry a credential and must never be logged, got {lines!r}"
    assert not [line for line in lines if "Bearer" in line]
    assert lines, "the decision itself is worth a debug line"
    assert any(
        req.action in line for line in lines
    ), "the action is the useful, harmless part of the record"
