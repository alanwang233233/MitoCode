"""The three tools that let an agent create and observe other agents.

``Agent`` spawns a sub-agent and hands it a prompt, ``ListAgent`` renders the
agent tree as text, and ``QueryAgent`` answers one question about one agent
without handing back its whole transcript.  Together they are the substrate
multi-agent orchestration is built on, and every design decision below exists
because of how the orchestrator actually behaves rather than because of how it
would be pleasant to use.

**Nothing here imports the agent package at runtime.**  ``agent_session``
imports ``tool_manager``, so the dependency between the two packages is
one-way — ``tool_manager -> agent_session`` — and a runtime import back into
``agent_orchestrator``, ``agent_session`` or ``agent_state`` from the tool
package would close that into a cycle.  ``TYPE_CHECKING`` is not enough on its
own because some of the values are needed at *runtime*: a session's lifecycle
state is read as ``session.state.value`` and compared against a string literal
here, and the message blocks are private types of ``agent_session`` that cannot
be ``isinstance``-checked without importing them.  So the orchestrator is
duck-typed, ``AgentState`` is compared by value, and message blocks are
recognised by the ``role`` / ``message`` attributes they carry.  This is the
same constraint :mod:`~mitocode.core.tool.builtin.delivery_gate` documents, and
it is the same reason its ``RUNNING_STATE_VALUE`` is a duplicated literal.

**``Agent`` does not wait for the answer, and that is not an oversight.**
:meth:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator.send_message`
appends the prompt, marks the session running and returns the child's
``asyncio.Task``; awaiting that task here would hold the parent's turn for the
child's entire run.  Since a child may itself call ``Agent``, that wait is not
bounded by anything the parent can see, and the parent would be unable to do
anything else — including the ``QueryAgent`` calls it would need in order to
find out what it is waiting for.  So ``Agent`` creates, sends, and returns the
new id immediately, and the agent that wants the answer asks for it later.

**A missing orchestrator is a call-time error, not a registration-time one.**
The three tools are registered before any agent exists — there is nothing to
attach yet at that point — so refusing to register them would break every
setup that builds its tool registry ahead of its orchestrator.  The failure is
therefore detected by :func:`require_orchestrator` at the moment a tool is
actually used and answered with :data:`NO_ORCHESTRATOR`, a plain
``"Error: ..."`` sentence naming what is missing and what fixes it.  It is
*returned* rather than raised on purpose: the boundary renders every escaped
exception as ``"Error: <TypeName>: <message>"``, and an internal class name is
not a sentence the model can act on.  The boundary keeps that format for the
genuinely unexpected exceptions it is there to catch.

**A sub-agent inherits its parent's provider.**  A user who selected a model
for this conversation selected it for the work, and a sub-agent that quietly
fell back to the orchestrator default would silently run the whole delegation
on a different model.  The one case that genuinely has no provider to inherit —
a call made with no agent turn behind it at all — passes ``None``, which is not
the same thing: ``None`` is what tells
:meth:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator.create_session`
to use its own default.

**Depth is capped, and the cap counts ancestors.**  With ``parent_id`` chained
correctly, a model can fork without limit: each child can spawn its own
children, and each of those can too.  :func:`depth_of` walks the parent chain
up through ``get_session`` and refuses a spawn whose parent is already at
``ToolContext.max_sub_agent_depth``.  The walk keeps the ids it has visited,
so a malformed chain — a cycle, which nothing in this repository prevents and
which would otherwise spin forever — terminates instead of hanging the turn.
The cap alone is not enough, because ``parent_id`` is model-visible: a deep
agent could name a shallow ancestor and keep forking from there.  So a spawn
under an explicit parent is additionally confined to the caller's own subtree
(:func:`_within_caller_subtree`), which walks the same chain with the same
cycle guard and answers whether it lands on the caller.

**A name is always something the tree can render.**  ``AgentNode`` draws
whatever ``name`` holds, so an empty one is a blank node in the TUI with
nothing for the human to click.  ``name`` is therefore the first line of the
description truncated to :data:`MAX_NAME_CHARS` characters, falling back to
``sub-<id>`` when the description has no first line to take.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from loguru import logger as loguru_logger

from mitocode.core.tool.builtin.context import (
    ToolContext,
    clamp,
    current,
    current_agent_context,
)
from mitocode.core.tool.builtin.errors import tool_boundary

logger = loguru_logger.bind(name=__name__)

#: Longest name :func:`_derive_name` will give a sub-agent.  The name is a tree
#: label drawn by :class:`~mitocode.core.agent.agent_node.AgentNode`, not prose
#: the model reads: a model that wants the full text has the description, which
#: is kept intact and is what :func:`query_agent` shows.  Forty characters is
#: about what fits one line of the tree without pushing the rest of the row off
#: it.
MAX_NAME_CHARS = 40

#: Prefix of the generated fallback name.  Paired with the new agent's id, so
#: two sub-agents spawned with empty descriptions still have distinguishable
#: labels.
NAME_FALLBACK_PREFIX = "sub-"

#: Value of :attr:`~mitocode.core.agent.agent_state.AgentState` for a session
#: mid-turn, and the value :func:`query_agent` scans message blocks for.  The
#: block role and the state value are separate literals that happen to be the
#: same word; neither is imported, for the reason in the module docstring.
ASSISTANT_ROLE = "assistant"

#: Keys a tree snapshot may carry its children under.  ``AgentNode.to_dict()``
#: writes ``sub_agent`` while a hand-rolled snapshot may write
#: ``sub_agent_ids``; both are read so this module works against either shape
#: instead of rendering an empty child count for one of them.
CHILD_KEYS = ("sub_agent_ids", "sub_agent")

#: What every tool here answers with when no orchestrator is attached.
#:
#: A shared constant rather than three worded sentences, for the same reason
#: :mod:`~mitocode.core.tool.builtin.errors` has exactly one boundary: the
#: rendering must not be able to drift between tools.  It names the missing
#: thing, the method that supplies it, and that retrying will not help -- which
#: are the three things the model needs in order to tell a wiring mistake from
#: a request it should try again.
NO_ORCHESTRATOR = (
    "Error: no orchestrator is attached to the tool context, so agents cannot "
    "be created, listed or queried. This is a wiring problem rather than a bad "
    "request: attach one during startup with "
    "ToolContext.attach_orchestrator(orchestrator). Retrying will not help "
    "until it is attached."
)


def require_orchestrator(ctx: ToolContext) -> Optional[Any]:
    """Return the orchestrator on *ctx*, or ``None`` when none is attached.

    **This does not raise, and that is deliberate.**  The absent orchestrator is
    an anticipated configuration state — it is the *first* thing a user hits who
    starts the app before wiring one up — not a crash.  Raising here would send
    it through :func:`~mitocode.core.tool.builtin.errors.tool_boundary`, which
    renders every escaped exception as ``"Error: <TypeName>: <message>"``, so
    the single most likely configuration mistake would read to the model as
    ``Error: MitoCodeBuiltinToolError: ...``.  That names a Python class
    instead of the problem, and tells the model nothing it can branch on.  So
    the absence is *returned*, and each tool hands
    :data:`NO_ORCHESTRATOR` back to the model as a plain sentence.  The
    boundary keeps its type-name format, which is exactly right for the
    genuinely unexpected exceptions it is there to catch.

    Neither the registered agent tools nor this function can refuse the
    situation outright: registration happens before any agent exists, so there
    is nothing to attach at that point.

    Args:
        ctx: The context every built-in tool reads its world out of.  ``None``
            is accepted and reads as "no orchestrator".

    Returns:
        Whatever was attached as ``ctx.orchestrator``, or ``None``.  Returned
        untyped on purpose: the agent package cannot be imported here, so the
        surface this module actually uses — ``create_session``, ``get_session``,
        ``get_tree_snapshot`` and ``send_message`` — is duck-typed rather than
        checked.
    """
    return getattr(ctx, "orchestrator", None) if ctx is not None else None


def depth_of(ctx: ToolContext, agent_id: Optional[int]) -> int:
    """Return how many ancestors *agent_id* has, walking up the parent chain.

    The count is the number of ``Agent`` calls already made to reach this
    agent: the root has depth ``0``, its children ``1``, and so on.  It is
    what :func:`spawn_agent` compares against
    :attr:`~mitocode.core.tool.builtin.context.ToolContext.max_sub_agent_depth`
    to decide whether a further nesting is allowed.

    The walk is bounded by the set of ids it has already visited.  Nothing in
    the orchestrator prevents a cycle — ``parent_id`` is set by whoever creates
    a session, and a malformed or hand-built chain can point two agents at each
    other — so without the guard this would spin forever inside a single tool
    call and take the turn with it.  A cycle instead stops the walk, is logged,
    and yields the depth reached so far.

    Args:
        ctx: The context supplying the orchestrator to walk through.  A
            context with no orchestrator reports depth ``0`` rather than
            raising, so this function stays answerable on its own.
        agent_id: The agent to measure.  ``None`` — no owning turn — is depth
            ``0``, because there is no chain to walk.

    Returns:
        The number of ancestors found.  A ``parent_id`` pointing at a session
        the orchestrator does not have ends the walk where it stops: the
        chain is unreadable past that point and guessing further would be
        inventing a depth.
    """
    orchestrator = getattr(ctx, "orchestrator", None) if ctx is not None else None
    if orchestrator is None or agent_id is None:
        return 0

    depth = 0
    seen = {agent_id}
    walking = agent_id
    while True:
        session = orchestrator.get_session(walking)
        if session is None:
            return depth
        parent_id = getattr(session, "parent_id", None)
        if parent_id is None:
            return depth
        if parent_id in seen:
            # Malformed, and it should not happen: `parent_id` is written by
            # `create_session`, which only ever points a child at an existing
            # session.  Logged because a silent stop here reads like "this
            # agent is a root" and a wrong depth refusal is not something a
            # model can explain.
            logger.error(
                "Agent {} declares parent {}, which is already in its own "
                "ancestor chain {}; the parent chain is malformed and the "
                "depth of {} is reported as {}.",
                walking,
                parent_id,
                sorted(seen),
                agent_id,
                depth,
            )
            return depth
        seen.add(parent_id)
        walking = parent_id
        depth += 1


def _within_caller_subtree(orchestrator: Any, caller_id: int, agent_id: int) -> bool:
    """Report whether *agent_id* is *caller_id* or one of its descendants.

    The spawn-depth cap only means something if a deep agent cannot simply
    name a shallow ancestor as its parent and keep forking from there.  This
    walk is that second half of the rule: it climbs from *agent_id* through
    the parent chain and answers whether the chain lands on *caller_id* —
    which includes *agent_id* being the caller itself, the case an omitted
    ``parent_id`` resolves to anyway.

    The walk is bounded by the set of ids it has already visited, exactly as
    :func:`depth_of` is: nothing in the orchestrator prevents a cycle in the
    parent links, and an unguarded walk would spin inside a single tool call.
    A cycle stops the walk, is logged, and answers False — an unprovable
    parent is a refused parent.

    Args:
        orchestrator: The duck-typed handle exposing ``get_session``.
        caller_id: The agent whose turn is making the spawn.  Never ``None``:
            the caller filters the decision before this function is reached.
        agent_id: The requested parent to check.

    Returns:
        True when the climb from *agent_id* reaches *caller_id*, False when
        the chain ends at a root, an unknown session, or a cycle first.
    """
    seen = {agent_id}
    walking = agent_id
    while True:
        if walking == caller_id:
            return True
        session = orchestrator.get_session(walking)
        if session is None:
            return False
        parent_id = getattr(session, "parent_id", None)
        if parent_id is None:
            return False
        if parent_id in seen:
            logger.error(
                "Agent {} declares parent {}, which is already in its own "
                "ancestor chain {}; the parent chain is malformed and the "
                "subtree check for {} is answered as False.",
                walking,
                parent_id,
                sorted(seen),
                caller_id,
            )
            return False
        seen.add(parent_id)
        walking = parent_id


def _derive_name(description: Any, agent_id: Optional[int]) -> str:
    """Return the tree label for a sub-agent spawned from *description*.

    The first line is taken because a description is frequently a sentence or
    two and only the first line describes the agent to a human scanning a tree;
    it is truncated rather than wrapped because the tree draws one row per
    agent and a wrapping name pushes the rest of the row off it.

    Args:
        description: The description the model supplied.  Anything that is not
            a ``str`` is treated as absent rather than measured, so a model
            that sends the wrong type cannot crash name derivation.
        agent_id: The new agent's id, or ``None`` when it is not known yet —
            the id is allocated by the orchestrator, so the fallback name can
            only be built after the session exists.  Passing ``None`` returns
            an empty string, which the caller replaces once it knows the id.

    Returns:
        The first line of *description* clipped to :data:`MAX_NAME_CHARS`
        characters, or ``sub-<agent_id>`` when there is no first line and the
        id is known.  Empty only when there is no first line *and* no id.
    """
    text = description.strip() if isinstance(description, str) else ""
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line:
        return first_line[:MAX_NAME_CHARS]
    if agent_id is None:
        return ""
    return f"{NAME_FALLBACK_PREFIX}{agent_id}"


def _state_text(value: Any) -> str:
    """Return the ``"running"``-style state string behind *value*.

    Accepts an enum member, a bare string, or anything else, so neither this
    module nor a test double has to import
    :class:`~mitocode.core.agent.agent_state.AgentState` to answer it.

    Args:
        value: A session's ``state`` attribute, a snapshot's ``state`` entry,
            or ``None``.

    Returns:
        The enum's ``.value`` when there is one, the string itself when it
        already is one, and ``"unknown"`` otherwise.  ``"unknown"`` rather
        than an empty string because this value is rendered to the model, and
        a blank state reads like a tool that failed rather than one reporting
        something it could not identify.
    """
    candidate = getattr(value, "value", value)
    return candidate if isinstance(candidate, str) and candidate else "unknown"


def _child_ids(entry: Any) -> Sequence[Any]:
    """Return the child ids recorded in one tree-snapshot *entry*.

    Reads both keys in :data:`CHILD_KEYS` rather than one, because
    :meth:`~mitocode.core.agent.agent_node.AgentNode.to_dict` writes
    ``sub_agent`` and a snapshot built by hand elsewhere may write
    ``sub_agent_ids``.  Rendering a real agent as having no children because
    the key was spelled differently is a silent wrong answer, which is worse
    than reading two names.

    Args:
        entry: One item of ``get_tree_snapshot()``.  Anything that is not a
            mapping simply has no children.

    Returns:
        The recorded child ids, or an empty tuple when the entry carries none
        in either shape.
    """
    if not isinstance(entry, Mapping):
        return ()
    for key in CHILD_KEYS:
        value = entry.get(key)
        if isinstance(value, (list, tuple)):
            return value
    return ()


def _field(entry: Any, key: str, default: Any = "") -> Any:
    """Return ``entry[key]``, or *default* when *entry* cannot supply it.

    Args:
        entry: One item of ``get_tree_snapshot()``, expected to be a mapping.
        key: The key to read.
        default: What to return for a missing key or an entry that is not a
            mapping at all.

    Returns:
        The recorded value, or *default*.
    """
    if not isinstance(entry, Mapping):
        return default
    value = entry.get(key, default)
    return default if value is None else value


def _last_assistant_message(session: Any) -> str:
    """Return the text of the last assistant block on *session*.

    Scans ``session.messages`` backwards and stops at the **first** block whose
    ``role`` is ``"assistant"``, not the first one with text in it.  Falling
    through to an older block when the newest is empty would let the model read
    a stale answer as a current one; reporting nothing is the honest answer
    while a turn is still streaming.

    This does not call
    :meth:`~mitocode.core.agent.agent_session.AgentSession.build_standard_context`.
    That method converts the *entire* transcript — every block, every tool
    call, every tool result — into provider-shaped dicts, and this one question
    needs at most one block.  On a long-running agent that is the difference
    between a bounded read and a cost proportional to everything the agent has
    ever done.

    The blocks are private types of
    :mod:`~mitocode.core.agent.agent_session` and cannot be ``isinstance``-
    checked here without importing the agent package, so they are recognised by
    the ``role`` and ``message`` attributes they carry.  A tool-call block has
    no ``role`` at all and a tool-result block's ``role`` is ``"tool"``, so both
    are skipped by the same comparison.

    Args:
        session: Whatever ``orchestrator.get_session`` returned.  Its
            ``messages`` may be absent, empty, or hold any block shape.

    Returns:
        The last assistant message's text, or an empty string when the session
        has produced none yet or none is readable.
    """
    messages = getattr(session, "messages", None)
    if not isinstance(messages, (list, tuple)):
        return ""
    for block in reversed(messages):
        if getattr(block, "role", None) != ASSISTANT_ROLE:
            continue
        text = getattr(block, "message", None)
        return text if isinstance(text, str) else ""
    return ""


def _render_snapshot(entries: Sequence[Any]) -> str:
    """Render one line per tree-snapshot entry.

    Args:
        entries: The list ``get_tree_snapshot()`` returned.

    Returns:
        A header naming how many agents there are, followed by one line each
        carrying the id, the name, the state, the parent and the number of
        children -- the five things a model needs to decide whether to spawn,
        wait, or send work somewhere.
    """
    lines = [f"{len(entries)} agent{'s' if len(entries) != 1 else ''}:"]
    for entry in entries:
        children = _child_ids(entry)
        parent = _field(entry, "parent_id", None)
        name = _field(entry, "name") or "<unnamed>"
        state = _state_text(_field(entry, "state"))
        shown_parent = "none" if parent is None else parent
        plural = "" if len(children) == 1 else "ren"
        agent_id = _field(entry, "id", "?")
        lines.append(
            f"[{agent_id}] {name} ({state}) | parent {shown_parent} | "
            f"{len(children)} child{plural}"
        )
    return "\n".join(lines)


@tool_boundary
async def spawn_agent(
    description: str,
    prompt: str,
    parent_id: Optional[int] = None,
) -> str:
    """Create a sub-agent, send it *prompt*, and return its id immediately.

    This is the ``Agent`` tool.

    Does **not** wait for the sub-agent's answer.  The prompt is appended, the
    child starts its own turn on the same event loop, and this returns the new
    id so the parent can carry on and ask :func:`query_agent` later.  Awaiting
    the child's task instead would hold the parent's turn for the child's whole
    run — unbounded, since a child may spawn children of its own — and would
    take away the very tool the parent would need to check on it.

    The child hangs off the calling agent by default.  A model that omitted
    ``parent_id`` while running inside a sub-agent means "a child of me", and
    leaving it unset would attach the new agent to the root instead, which
    breaks the tree the TUI draws and the depth cap :func:`depth_of` enforces.

    Args:
        description: What this agent is for.  Its first line becomes the tree
            label; the whole string is kept as the session's description.
        prompt: The task to hand the new agent.  Sent as its first user
            message.
        parent_id: The agent this one belongs under.  ``None`` means the agent
            whose turn is running this call, and — with no agent turn running
            at all — no parent, so the new agent is a root.  An **explicit**
            parent must be the caller itself or inside the caller's subtree;
            anything else is refused, because a deep agent naming a shallow
            ancestor would step around the depth cap.  With no agent turn
            running there is no subtree to check against, so the explicit
            parent is honoured as given.

    Returns:
        The new agent's id and name in one sentence.  Every failure is an
        ``"Error: ..."`` string rather than an exception: an unknown parent, a
        depth cap reached, or anything the orchestrator or the boundary raises
        on the way.
    """
    caller_id, ctx = current_agent_context()
    orchestrator = require_orchestrator(ctx)
    if orchestrator is None:
        return NO_ORCHESTRATOR

    if parent_id is None:
        parent_id = caller_id

    if parent_id is not None:
        parent = orchestrator.get_session(parent_id)
        if parent is None:
            return (
                f"Error: no agent with id {parent_id}, so nothing can be "
                f"spawned under it. ListAgent shows the agents that exist."
            )
        if caller_id is not None and not _within_caller_subtree(
            orchestrator, caller_id, parent_id
        ):
            return (
                f"Error: agent {parent_id} is not you (agent {caller_id}) and "
                f"not inside your subtree, so you cannot spawn under it — "
                f"spawning under a shallower ancestor would step around the "
                f"nesting depth cap. Omit parent_id to spawn under yourself, "
                f"or name yourself or one of your descendants."
            )
        # Inherited, not re-created: a user who picked a model for this
        # conversation picked it for the delegated work too.  `None` here would
        # silently fall back to the orchestrator default.
        provider = getattr(parent, "provider", None)
    else:
        parent = None
        # `None` is what tells `create_session` to use its own default. Building
        # a provider here would hard-code a model this context never chose.
        provider = None

    depth = depth_of(ctx, parent_id)
    if depth >= ctx.max_sub_agent_depth:
        return (
            f"Error: refusing to spawn a sub-agent under agent {parent_id}. That "
            f"agent is already at nesting depth {depth}, and the limit is "
            f"{ctx.max_sub_agent_depth} nested Agent calls, so this would be "
            f"depth {depth + 1}. Do the work in this agent, or send it to an "
            f"existing agent instead of starting a deeper one."
        )

    session = await orchestrator.create_session(
        name=_derive_name(description, None),
        provider=provider,
        description=description,
        parent_id=parent_id,
    )

    agent_id = getattr(session, "agent_id", None)
    if agent_id is None:
        return (
            "Error: the orchestrator created a session that carries no agent "
            "id, so the prompt could not be delivered to it."
        )

    # The fallback name needs the id, which only exists now.  Assigned before
    # the send so the tree sync that `send_message` pushes already carries it;
    # the sync `create_session` pushed showed the placeholder.
    name = _derive_name(description, agent_id)
    if getattr(session, "name", None) != name:
        try:
            session.name = name
        except AttributeError:  # pragma: no cover - a frozen session double
            logger.error(
                "Session {} would not accept the generated name '{}'; it will "
                "appear in the tree with the name it was created with.",
                agent_id,
                name,
            )

    await orchestrator.send_message(agent_id, prompt)

    under = f" under agent {parent_id}" if parent_id is not None else " at the root"
    return (
        f"Created agent {agent_id} ({name}){under} and sent it the prompt. It "
        f"runs in the background: use QueryAgent({agent_id}) to ask what it "
        f"has produced."
    )


@tool_boundary
async def list_agents() -> str:
    """Render every agent and its state as text.  This is ``ListAgent``.

    Reads ``get_tree_snapshot()`` rather than walking sessions: the snapshot is
    the same pure data the TUI renders, so what the model is told and what the
    human sees cannot drift apart.

    Args:
        None.  The whole world comes from the current context.

    Returns:
        One line per agent with its id, name, state, parent and child count, or
        a sentence saying there are none.  An ``"Error: ..."`` string when no
        orchestrator is attached.
    """
    ctx = current()
    orchestrator = require_orchestrator(ctx)
    if orchestrator is None:
        return NO_ORCHESTRATOR

    entries = orchestrator.get_tree_snapshot()
    if not entries:
        return "There are no agents yet."

    if not isinstance(entries, (list, tuple)):
        return (
            f"Error: get_tree_snapshot() returned "
            f"{type(entries).__name__}; a list of agent records was expected."
        )

    return clamp(_render_snapshot(entries), ctx.max_output_chars)


@tool_boundary
async def query_agent(agent_id: int) -> str:
    """Report one agent's identity, state and last assistant message.

    This is ``QueryAgent``.  Answers one question about one agent — the usual
    one being "did the thing I delegated actually finish?" — so it reads only
    the last assistant block rather than the transcript.  See
    :func:`_last_assistant_message` for why
    :meth:`~mitocode.core.agent.agent_session.AgentSession.build_standard_context`
    is not used here.

    Args:
        agent_id: Id of the agent to report on, as returned by
            :func:`spawn_agent` or :func:`list_agents`.

    Returns:
        The id, name, description, state and last assistant message.  An
        ``"Error: ..."`` string for an unknown id or a missing orchestrator,
        never an exception: a model asking about an agent that finished and was
        cleaned up should read that it is gone, not watch the turn die.
    """
    ctx = current()
    orchestrator = require_orchestrator(ctx)
    if orchestrator is None:
        return NO_ORCHESTRATOR

    session = orchestrator.get_session(agent_id)
    if session is None:
        return (
            f"Error: no agent with id {agent_id}. ListAgent shows the agents "
            f"that currently exist."
        )

    name = getattr(session, "name", "") or "<unnamed>"
    description = getattr(session, "description", "") or ""
    lines = [
        f"Agent {agent_id} ({name}) " f"[{_state_text(getattr(session, 'state', ''))}]"
    ]
    lines.append(
        f"description: {description}" if description else "description: (none)"
    )

    message = _last_assistant_message(session)
    if message.strip():
        lines.append("last message:\n" + clamp(message, ctx.max_output_chars))
    else:
        lines.append(
            "last message: (none yet -- this agent has not produced an "
            "assistant message)"
        )

    return "\n".join(lines)
