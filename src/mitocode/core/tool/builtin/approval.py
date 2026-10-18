"""Approval protocol for the workspace-confined built-in tools.

Every built-in tool resolves its paths through one chokepoint that refuses to
act outside the workspace root.  What happens at that chokepoint for a request
that *does* escape is not a hard-coded policy in this module — it is delegated
to an :class:`Approver`, a one-method protocol, so that the shipped behaviour
(:class:`AlwaysAllow`) can later be swapped for a TUI confirmation dialog
without touching a single tool.

The seam is deliberately small:

* :class:`ApprovalDecision` — ``ALLOW`` or ``DENY``, nothing else.
* :class:`ApprovalRequest` — what is being attempted, in the three pieces a
  human needs in order to answer sensibly: the ``action`` (``"bash"``,
  ``"read"``, ``"write"``, ``"edit"``), the ``detail`` (the command, or the
  resolved absolute path), and the ``reason`` (why approval is needed).
* :class:`Approver` — ``async def request(req) -> ApprovalDecision``.

The method is ``async`` even though both shipped implementations never await
anything, because the real implementation will have to hand control to the
TUI's event loop and wait for a keystroke.  Making the protocol synchronous
now would mean changing the call sites later, at the exact moment someone is
adding a dialog.

A denial is reported to the caller as a refusal string rather than as an
exception, because the model can react to a refusal and a traceback is only
noise in a transcript.  Note that no timeout is built into the protocol: how
long to wait for a verdict is a property of the call site (``paths.resolve``
and ``shell_tools.Bash``), not of the approver, and only one of those call
sites ever has to wait on a human.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ApprovalDecision(Enum):
    """The answer an :class:`Approver` gives to an :class:`ApprovalRequest`.

    A two-valued enum rather than a boolean, so a future policy cannot smuggle
    in "allow but warn" by widening a ``bool``'s meaning, and so a printed
    decision (``ApprovalDecision.ALLOW``) reads as a decision rather than as a
    bare flag.  The string values are the wire/log form: they are what a
    refusal string quotes back to the model.
    """

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ApprovalRequest:
    """A single operation that escaped the workspace root and needs a verdict.

    Mutable rather than frozen because the request is a plain value object
    passed by reference to a human-facing dialog, which may want to annotate
    or normalise it (trimming a very long command, for instance) before
    showing it.

    Attributes:
        action: The kind of operation being attempted. One of ``"bash"``,
            ``"read"``, ``"write"`` or ``"edit"`` — the display vocabulary the
            built-in tools use, not a class name.
        detail: The concrete thing the operation would touch: the command line
            for ``Bash``, the resolved absolute path for the file tools.
        reason: Why approval is required at all, phrased for a human (for
            example ``"outside workspace root"``).
    """

    action: str
    detail: str
    reason: str


@runtime_checkable
class Approver(Protocol):  # pylint: disable=too-few-public-methods
    """Decides whether an out-of-root operation may proceed.

    Structural, not nominal: an approver is any object with a matching
    ``request``, so a test can pass a one-line stub, a lambda-backed object, or
    a future TUI dialog, without inheriting from anything here.  Marked
    ``runtime_checkable`` so ``isinstance(obj, Approver)`` is available as a
    cheap sanity check at wiring time.

    Only one method exists on purpose.  The protocol must stay the smallest
    surface that every future implementation — including an interactive one
    that has to render into another process's event loop — can actually honour.
    """

    async def request(  # pylint: disable=unnecessary-ellipsis
        self, req: ApprovalRequest
    ) -> ApprovalDecision:
        """Return the verdict for ``req``.

        Args:
            req: The operation awaiting a verdict.

        Returns:
            ``ApprovalDecision.ALLOW`` to let the operation proceed, or
            ``ApprovalDecision.DENY`` to refuse it.  Implementations return a
            refusal to the caller as a value; they do not raise.
        """
        ...


class AlwaysAllow:  # pylint: disable=too-few-public-methods
    """The shipped default approver: every request is allowed.

    Why this is the default rather than a prompt: the approval seam exists so
    that the *policy* is pluggable, not so that shipping built-in tools blocks
    on a UI decision.  Landing an interactive dialog as a prerequisite for
    ``Read``/``Write``/``Bash`` would couple eleven tools to a TUI that has not
    been designed yet, and would make the very first ``python -m mitocode``
    run interactive.  With ``AlwaysAllow`` installed the tools are fully usable
    today, the chokepoint is still exercised on every escape (so switching
    policy is a one-line change in ``ToolContext``), and the shipped behaviour
    of ``AlwaysAllow`` and of a future dialog that always says yes is
    observationally identical — which makes the swap safe to defer rather than
    urgent.

    Note that the approver is only consulted on *escape*.  Ordinary in-root
    reads and writes never ask, so a permissive default does not produce a
    dialog per keystroke.
    """

    async def request(  # pylint: disable=unused-argument
        self, req: ApprovalRequest
    ) -> ApprovalDecision:
        """Return ``ApprovalDecision.ALLOW`` without inspecting ``req``.

        Args:
            req: The operation awaiting a verdict.  Ignored: the point of this
                implementation is that the verdict does not depend on it.

        Returns:
            Always ``ApprovalDecision.ALLOW``.
        """
        return ApprovalDecision.ALLOW


class DenyAll:  # pylint: disable=too-few-public-methods
    """An approver that refuses every request.

    The test double.  Refusing everything makes the denial branch of every
    call site reachable in one line — a test can prove that a ``Bash`` escaping
    the root is actually blocked, and that the model receives a refusal string
    naming the request, without needing to mock an interactive dialog or stub a
    prompt that the shipped default never shows.  It is also the safe value to
    fall back to when an approver is missing or misconfigured.
    """

    async def request(  # pylint: disable=unused-argument
        self, req: ApprovalRequest
    ) -> ApprovalDecision:
        """Return ``ApprovalDecision.DENY`` without inspecting ``req``.

        Args:
            req: The operation awaiting a verdict.  Ignored: every request is
                refused.

        Returns:
            Always ``ApprovalDecision.DENY``.
        """
        return ApprovalDecision.DENY
