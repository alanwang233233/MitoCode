"""Everything a built-in tool needs from the world, in one place.

A tool function is registered as a bare coroutine and later called with
nothing but its model-supplied arguments, so the workspace it may touch, the
policy that may veto an escape, and the registry that tracks detached calls
have nowhere to arrive from.  :class:`ToolContext` is that somewhere: it is
built once at registration, handed to :func:`set_current`, and read back
inside each tool body by :func:`current`.

The context is process-wide rather than per-call, and that is a deliberate
trade.  Threading it through every tool signature would mean every tool
declares a first parameter that the model must never be asked to fill, and
the eleven tools in this package already take their arguments straight from
the model's function call.  The cost is that the module-level default is
global mutable state: a test that calls :func:`set_current` must reset it, or
every later test inherits a context pointing at some other test's temporary
directory.  :func:`set_current` therefore *replaces* the value rather than
mutating it, so a leaked context cannot be reshaped by whoever finds it, and
:func:`current` lazily creates and caches a default so a tool invoked with no
registration at all still works instead of raising ``AttributeError`` on
``None``.

:class:`~mitocode.core.tool.builtin.errors.tool_boundary` is the only other
thing every tool shares, and it lives in :mod:`errors` because it is about
failure, not about state.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, Tuple

from mitocode.core.tool.builtin.approval import AlwaysAllow, Approver
from mitocode.core.tool.builtin.background import BackgroundRegistry

if TYPE_CHECKING:
    # Imported for type checking only.  The runtime stays on duck typing so
    # that importing this module never drags in the tool registry or the
    # workspace manager -- both of which are considerably heavier than what a
    # tool body actually touches, and either of which could one day import
    # this module in return.
    from mitocode.core.tool.tool_manager import ToolManager
    from mitocode.core.workspace.workspace import Workspace

#: The id carried by ``current_agent_id`` when no agent is running a turn.
#:
#: ``-1`` rather than ``None`` because the variable is typed ``int`` for the
#: hot path -- comparing an int is cheaper than a branch on ``None`` -- and
#: because an id is always an int once one exists.  ``current_agent_context``
#: translates it back to ``None``, which is the value callers actually want,
#: so the sentinel never escapes this module.
UNKNOWN_AGENT_ID = -1

#: Id of the agent whose turn is currently executing the tool body.
#:
#: A :class:`~contextvars.ContextVar` rather than a plain global because the
#: event loop interleaves the orchestrator's own turn with any number of
#: background sub-agents, and a plain global would let a sub-agent's ``set``
#: be observed by the main turn that spawned it.  ``ToolContext.max_sub_agent_depth``
#: bounds the depth, but only the ContextVar keeps concurrent turns apart.
current_agent_id: ContextVar[int] = ContextVar(
    "current_agent_id", default=UNKNOWN_AGENT_ID
)


@dataclass
class ToolContext:
    """The state every built-in tool reads instead of reaching for globals.

    Attributes:
        workspace: The workspace whose root bounds every path tool.  ``None``
            means no boundary at all -- ``root()`` falls back to the process
            working directory and the approver is never consulted, because
            there is no root to escape from.  It is optional so a context can
            be built in a test or a smoke script without a workspace on disk.
        tool_manager: The registry the agent tools use to discover sibling
            agents' tools.  ``None`` until :func:`register_builtin_tools`
            installs it, which is why an agent tool must report a missing
            orchestrator rather than dereference it.
        approver: The policy consulted when an operation escapes the root.
            Defaults to :class:`~mitocode.core.tool.builtin.approval.AlwaysAllow`,
            the shipped permissive policy; a TUI dialog is a drop-in
            replacement.  Constructed per instance rather than shared, so a
            stateful approver added later cannot leak state between contexts.
        orchestrator: Duck-typed handle used to spawn and message sub-agents.
            ``Any`` rather than a concrete class: the built-in tools must not
            import the agent package, which would invert the one-way
            dependency ``tool_manager -> agent_session``.  It is attached after
            construction by :meth:`attach_orchestrator`.
        background: Registry of detached calls.  A fresh instance per context,
            built by ``default_factory`` rather than shared: handles are
            process-wide strings, so two contexts sharing one registry would
            let one agent collect or cancel the other's work.  The import of
            :mod:`~mitocode.core.tool.builtin.background` is at module level
            and therefore one-way -- that module imports *this* one only under
            ``TYPE_CHECKING``.
        max_output_chars: Soft ceiling on how much text a tool hands back to
            the model.  Not a hard one -- ``clamp``'s marker is added on top
            of the budget, so a clamped result can exceed it by the marker's
            length.  Deliberately soft: truncating is always preferable to
            dropping the request, and a model told a result was truncated can
            ask for the rest.
        max_sub_agent_depth: How many nested ``Agent`` calls are allowed.
            Bounds the recursion that an agent asking for an agent asking for
            an agent would otherwise create.
    """

    workspace: Optional[Workspace] = None
    tool_manager: Optional[ToolManager] = None
    approver: Approver = field(default_factory=AlwaysAllow)
    orchestrator: Any = None
    background: BackgroundRegistry = field(default_factory=BackgroundRegistry)
    max_output_chars: int = 30_000
    max_sub_agent_depth: int = 4

    def root(self) -> Path:
        """Return the directory every path argument is resolved against.

        Returns:
            ``workspace.path`` resolved, or the process working directory
            resolved when there is no workspace.  Resolved rather than used
            verbatim so that containment comparisons later in the package are
            between two fully symlink-expanded paths -- on macOS
            ``/var/folders/...`` and ``/private/var/folders/...`` are the same
            directory and a naive ``==`` would call a file in-root out-of-root.
        """
        if self.workspace is not None:
            return Path(self.workspace.path).resolve()
        return Path.cwd().resolve()

    def attach_orchestrator(self, orchestrator: Any) -> None:
        """Record the orchestrator the agent tools drive sub-agents through.

        Setter rather than a constructor argument because the orchestrator is
        built after the tool registry it registers tools into: passing it to
        ``__init__`` would mean constructing the context twice.

        Args:
            orchestrator: Any object exposing ``get_session`` and
                ``send_message``.  Not imported or type-checked, by design.
        """
        self.orchestrator = orchestrator


#: The registered current context.  ``None`` means "nothing registered", which
#: is distinct from "registered an empty context" -- the former takes
#: ``current()``'s lazy-default path, the latter does not.  Module-global
#: because every tool reads it without a parameter; see the module docstring
#: for why that trade is worth making.
_CURRENT: Optional[ToolContext] = None


def current_agent_context() -> Tuple[Optional[int], ToolContext]:
    """Return the id of the agent whose turn is running, and the context.

    Reads two independent pieces of state in one call because every tool that
    spawns a sub-agent needs both -- the id to stamp the launch with, and the
    context to launch it against -- and a caller that reads them separately
    can observe a mismatch if the turn ends in between.

    Returns:
        ``(agent_id, context)`` where *agent_id* is ``None`` when no agent
        turn is running (the ContextVar carries
        :data:`UNKNOWN_AGENT_ID`), and the registered :class:`ToolContext` or
        a lazily created default.
    """
    raw_id = current_agent_id.get()
    agent_id = None if raw_id == UNKNOWN_AGENT_ID else raw_id
    return agent_id, current()


def set_current(ctx: ToolContext) -> None:
    """Register the context every subsequent tool body will read.

    Replaces rather than mutates.  Mutating a shared context would mean a
    caller that wanted to point one tool at a different root silently retarget
    every tool that happens to be running, and a caller that wanted a fresh
    context would be unable to get one without the previous holder noticing.

    Args:
        ctx: The context to install.  Registering ``ToolContext()`` is
            different from registering nothing: it suppresses the lazy default
            in :func:`current`.
    """
    global _CURRENT  # pylint: disable=global-statement
    _CURRENT = ctx


def current() -> ToolContext:
    """Return the registered context, creating a default on first use.

    A tool is a plain coroutine called with only its model-supplied
    arguments, so nothing guarantees somebody ran :func:`set_current` first.
    Returning a usable default is what lets a tool be unit-tested, called from
    a REPL, or exercised by a smoke script without standing up a workspace and
    an orchestrator.

    Returns:
        The registered :class:`ToolContext`, or one created and cached on the
        first call so every later call observes the same instance.
    """
    global _CURRENT  # pylint: disable=global-statement
    if _CURRENT is None:
        _CURRENT = ToolContext()
    return _CURRENT


def clamp(text: str, limit: int) -> str:
    """Shorten *text* to roughly *limit* characters, keeping both ends.

    Head-only truncation is the wrong shape for tool output.  What a model
    loses first is the context that makes the rest interpretable -- the
    command that produced the output, the filename whose contents follow, the
    error whose tail names the line number -- and it loses the tail, which is
    where the conclusion of a long log or the final line of a file actually
    is.  Keeping half at each end preserves both the framing and the payoff,
    and the marker between them states plainly that the middle is gone, so the
    model narrows its next request instead of assuming it saw everything.

    Args:
        text: The text to shorten.  Returned unchanged when it already fits.
        limit: The character budget for the kept head and tail combined.

    Returns:
        *text* itself when ``len(text) <= limit``.  Otherwise the first
        ``limit // 2`` characters, a ``"[truncated: N characters omitted]"``
        marker, and the final ``limit - limit // 2`` characters.  The marker
        is added *on top of* the budget, so the result can exceed *limit* by
        the marker's length; ``limit`` bounds payload, not output size.  A
        non-positive *limit* yields the marker alone.
    """
    limit = max(limit, 0)
    if len(text) <= limit:
        return text

    head = limit // 2
    tail = limit - head
    dropped = len(text) - head - tail
    marker = f"\n\n[truncated: {dropped} characters omitted]\n\n"
    # `text[len(text) - tail:]` rather than `text[-tail:]`: with tail == 0 the
    # latter is `text[0:]` -- the whole string, the opposite of what an empty
    # tail means.
    return text[:head] + marker + text[len(text) - tail :]
