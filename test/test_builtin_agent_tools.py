"""Unit tests for the built-in agent tools: spawn, list and query agents.

The orchestrator behind these tools is duck-typed on purpose — the tool
package must not import the agent package at runtime — so every double here
mimics the orchestrator's *shape* (``get_session``, ``create_session``,
``send_message``, ``get_tree_snapshot``) and none of them import or subclass
``AgentOrchestrator``, ``AgentSession`` or ``AgentState``.  A double that
subclassed the real classes would pass whenever the real class happens to
carry the attribute the implementation reads; the contract under test is only
"answer these method calls".
"""

import asyncio
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.core.tool.builtin.context as context_module
from mitocode.core.tool.builtin import agent_tools
from mitocode.core.tool.builtin.context import (
    ToolContext,
    current_agent_id,
    set_current,
)

#: Value the ContextVar carries when no agent is running a turn.
UNKNOWN_AGENT_ID = -1

#: Seconds a worker thread gets before a cycle test declares a hang.  A hang
#: here is the defect the cycle guard exists to prevent, so it must surface as
#: a failed test rather than a frozen suite.
DEADLINE_SECONDS = 10.0

#: The documented 'no assistant message yet' sentence of ``query_agent``.
NONE_YET = (
    "last message: (none yet -- this agent has not produced an assistant " "message)"
)

#: Sentinel for "this attribute does not exist on the block at all".
_UNSET = object()


# --- doubles ---------------------------------------------------------------


class _Block:
    """Duck-typed message block; ``role``/``message`` exist only when given.

    A tool-call block has no ``role`` attribute at all, a tool-result block
    carries ``role="tool"`` (and may still carry a message), and an assistant
    block may carry a role with no readable message — all three shapes appear
    in real transcripts, so all three appear here.
    """

    def __init__(self, message: Any = _UNSET, role: Any = _UNSET) -> None:
        if message is not _UNSET:
            self.message = message
        if role is not _UNSET:
            self.role = role


class _EnumState:
    """Stand-in for an ``AgentState`` member: the value lives on ``.value``."""

    def __init__(self, value: str) -> None:
        self.value = value


class _Session:
    """Duck-typed session carrying only the attributes the tools read."""

    def __init__(
        self,
        agent_id: int,
        name: str = "agent",
        state: Any = "idle",
        parent_id: Optional[int] = None,
        provider: Any = None,
        messages: Optional[List[Any]] = None,
        description: str = "",
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.state = state
        self.parent_id = parent_id
        self.provider = provider
        self.messages = [] if messages is None else messages
        self.description = description


class _FrozenNameSession:
    """A session whose ``name`` refuses assignment, as a frozen type would."""

    def __init__(self, agent_id: int, parent_id: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.provider = "inherited"
        self.description = ""
        self.messages = []
        self.state = "idle"

    @property
    def name(self) -> str:
        """The name the session was created with, never reassignable."""
        return "created-name"

    @name.setter
    def name(self, value: str) -> None:
        raise AttributeError("name is frozen")


class _Task:
    """The unawaited handle ``send_message`` hands back; records any await."""

    def __init__(self) -> None:
        self.was_awaited = False

    def __await__(self) -> Any:
        self.was_awaited = True
        return iter(())


class _Orchestrator:
    """Duck-typed orchestrator that records every call made against it."""

    def __init__(
        self,
        sessions: Optional[Dict[int, _Session]] = None,
        snapshot: Any = None,
    ) -> None:
        self.sessions = dict(sessions) if sessions else {}
        self.snapshot = snapshot
        self.created: List[Dict[str, Any]] = []
        self.sent: List[Tuple[int, str]] = []
        self.tasks: List[_Task] = []
        self.lookups: List[int] = []
        self._next_id = 100

    def get_session(self, agent_id: int) -> Optional[_Session]:
        self.lookups.append(agent_id)
        return self.sessions.get(agent_id)

    async def create_session(
        self,
        name: Any = None,
        provider: Any = None,
        description: Any = None,
        parent_id: Any = None,
    ) -> _Session:
        session = _Session(
            self._next_id,
            name=name,
            provider=provider,
            description=description,
            parent_id=parent_id,
        )
        self.sessions[session.agent_id] = session
        self.created.append(
            {
                "name": name,
                "provider": provider,
                "description": description,
                "parent_id": parent_id,
            }
        )
        self._next_id += 1
        return session

    async def send_message(self, agent_id: int, message: str) -> _Task:
        self.sent.append((agent_id, message))
        task = _Task()
        self.tasks.append(task)
        return task

    def get_tree_snapshot(self) -> Any:
        return self.snapshot


class _FrozenNameOrchestrator(_Orchestrator):
    """Creates sessions that refuse to have their generated name assigned."""

    async def create_session(
        self,
        name: Any = None,
        provider: Any = None,
        description: Any = None,
        parent_id: Any = None,
    ) -> _FrozenNameSession:
        session = _FrozenNameSession(self._next_id, parent_id=parent_id)
        self.sessions[session.agent_id] = session
        self.created.append(
            {
                "name": name,
                "provider": provider,
                "description": description,
                "parent_id": parent_id,
            }
        )
        self._next_id += 1
        return session


class _RaisingOrchestrator:
    """Orchestrator whose ``get_session`` raises — a programming error."""

    def get_session(self, agent_id: int) -> None:
        raise RuntimeError("boom")


class _IdlessSession:
    """A session the orchestrator created without an agent id on it."""

    def __init__(self) -> None:
        self.name = "created"
        self.state = "idle"
        self.messages = []


class _IdlessOrchestrator(_Orchestrator):
    """Creates sessions that carry no agent id at all."""

    async def create_session(
        self,
        name: Any = None,
        provider: Any = None,
        description: Any = None,
        parent_id: Any = None,
    ) -> _IdlessSession:
        self.created.append(
            {
                "name": name,
                "provider": provider,
                "description": description,
                "parent_id": parent_id,
            }
        )
        return _IdlessSession()


# --- fixtures and helpers ---------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_module_state():
    """Reset the context module's process-wide state around every test.

    ``_CURRENT`` and the ``current_agent_id`` ContextVar survive between test
    files in one pytest process, so a leaked registration would hand this
    file's tools another file's context (and vice versa).
    """
    context_module._CURRENT = None  # pylint: disable=protected-access
    token = current_agent_id.set(UNKNOWN_AGENT_ID)
    try:
        yield
    finally:
        current_agent_id.reset(token)
        context_module._CURRENT = None  # pylint: disable=protected-access


def _make_ctx(
    orchestrator: Any = None,
    max_sub_agent_depth: int = 4,
    max_output_chars: int = 30_000,
) -> ToolContext:
    """Build a ToolContext around *orchestrator* and register it as current."""
    ctx = ToolContext(
        orchestrator=orchestrator,
        max_sub_agent_depth=max_sub_agent_depth,
        max_output_chars=max_output_chars,
    )
    set_current(ctx)
    return ctx


def _cycle_sessions(cycle_len: int) -> Dict[int, _Session]:
    """Sessions 1..n whose parent links form the cycle 1 -> 2 -> ... -> n -> 1."""
    return {
        agent_id: _Session(agent_id, parent_id=(agent_id % cycle_len) + 1)
        for agent_id in range(1, cycle_len + 1)
    }


def _finish_within(call: Callable[[], Any], timeout: float = DEADLINE_SECONDS) -> Any:
    """Run *call* on a daemon worker thread; fail the test if it outlives it.

    The cycle guard being tested is a plain ``while`` loop that never awaits,
    so a regression would block the event loop itself — ``asyncio.wait_for``
    could never fire.  A worker thread turns that hang into a failed
    assertion instead of a frozen suite.
    """
    outcome: Dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # re-raised on the main thread below
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), (
        f"the parent-chain walk did not finish within {timeout}s; "
        "a cycle is spinning the turn instead of terminating"
    )
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _created_sentence(agent_id: int, name: str, under: str) -> str:
    """The documented success sentence of ``spawn_agent``, assembled by hand."""
    return (
        f"Created agent {agent_id} ({name}){under} and sent it the prompt. It "
        f"runs in the background: use QueryAgent({agent_id}) to ask what it "
        f"has produced."
    )


def _assert_plain_orchestrator_error(result: str) -> None:
    """The no-orchestrator answer is prose naming the wiring fix, no class name."""
    assert result.startswith("Error:"), f"expected an Error sentence: {result!r}"
    assert "no orchestrator" in result, f"must name the missing wiring: {result!r}"
    assert (
        "attach_orchestrator" in result
    ), f"must name the call that fixes it: {result!r}"
    assert (
        "Retrying will not help" in result
    ), f"must say retrying is pointless: {result!r}"
    assert (
        "MitoCodeBuiltinToolError" not in result
    ), f"a Python class name leaked into the model's answer: {result!r}"


def _query_orchestrator(messages: Any = None, **overrides: Any) -> _Orchestrator:
    """An orchestrator serving one session (id 3, 'writer') for query tests."""
    orch = _Orchestrator()
    session = _Session(
        3,
        name="writer",
        state="running",
        description="Writes drafts",
        messages=[] if messages is None else messages,
    )
    for key, value in overrides.items():
        setattr(session, key, value)
    orch.sessions[3] = session
    return orch


# --- require_orchestrator ----------------------------------------------------


def test_require_orchestrator_returns_the_attached_orchestrator():
    """An attached orchestrator is returned as-is, not copied or wrapped."""
    orch = _Orchestrator()
    ctx = _make_ctx(orch)
    assert (
        agent_tools.require_orchestrator(ctx) is orch
    ), "the duck-typed handle must come back untouched"


def test_require_orchestrator_absent_in_every_shape_returns_none():
    """None context, an attribute-less context, and an unattached context."""
    assert (
        agent_tools.require_orchestrator(None) is None
    ), "a None context reads as 'no orchestrator', never a crash"
    assert (
        agent_tools.require_orchestrator(object()) is None
    ), "a context without the attribute reads as 'no orchestrator'"
    assert (
        agent_tools.require_orchestrator(_make_ctx(None)) is None
    ), "an attached-None orchestrator reads as 'no orchestrator'"


# --- depth_of ----------------------------------------------------------------


def test_depth_of_counts_ancestors_up_the_parent_chain():
    """A 1 -> 2 -> 3 parent chain yields depths 0, 1 and 2."""
    orch = _Orchestrator(
        sessions={
            1: _Session(1, parent_id=None),
            2: _Session(2, parent_id=1),
            3: _Session(3, parent_id=2),
        }
    )
    ctx = _make_ctx(orch)
    assert agent_tools.depth_of(ctx, 1) == 0, "the root has no ancestors"
    assert agent_tools.depth_of(ctx, 2) == 1, "one hop up the chain"
    assert agent_tools.depth_of(ctx, 3) == 2, "two hops up the chain"


def test_depth_of_none_id_or_no_orchestrator_reports_zero():
    """A None id, a None context, and an unattached context all read as 0."""
    orch = _Orchestrator(sessions={5: _Session(5, parent_id=None)})
    assert agent_tools.depth_of(None, 5) == 0, "no context means no chain"
    assert (
        agent_tools.depth_of(_make_ctx(None), 5) == 0
    ), "no orchestrator means no walk is possible"
    assert (
        agent_tools.depth_of(_make_ctx(orch), None) == 0
    ), "no owning turn means no chain to walk"


def test_depth_of_dangling_parent_link_ends_the_walk():
    """A parent id the orchestrator does not have ends the walk where it stops."""
    orch = _Orchestrator(sessions={5: _Session(5, parent_id=99)})
    ctx = _make_ctx(orch)
    assert (
        agent_tools.depth_of(ctx, 5) == 1
    ), "the hop toward the missing parent 99 counts; nothing beyond it does"


@pytest.mark.parametrize("cycle_len,expected", [(1, 0), (2, 1), (3, 2)])
def test_depth_of_cyclic_parent_chain_terminates(cycle_len, expected):
    """A self-parent and 2-/3-cycles terminate at the depth already reached."""
    orch = _Orchestrator(sessions=_cycle_sessions(cycle_len))
    ctx = _make_ctx(orch)
    depth = _finish_within(lambda: agent_tools.depth_of(ctx, 1))
    assert (
        depth == expected
    ), f"a {cycle_len}-cycle must terminate having counted {expected} hops"


# --- _derive_name ------------------------------------------------------------


def test_derive_name_takes_the_stripped_first_line():
    """The first line wins, whitespace and all later lines dropped."""
    assert agent_tools._derive_name("Fix the bug\nsecond line\nthird", 5) == (
        "Fix the bug"
    ), "only the first line labels the tree row"
    assert (
        agent_tools._derive_name("   padded first\nsecond", 5) == "padded first"
    ), "the first line is stripped before it becomes the label"


def test_derive_name_truncates_the_first_line_to_forty_chars():
    """Forty characters fit; the forty-first and everything after is cut."""
    assert (
        agent_tools._derive_name("A" * 40, 7) == "A" * 40
    ), "a forty-character first line fits exactly"
    assert (
        agent_tools._derive_name("A" * 41, 7) == "A" * 40
    ), "one character over the limit is truncated to the limit"
    assert (
        agent_tools._derive_name("B" * 45 + "\nsecond", 7) == "B" * 40
    ), "truncation applies to the first line before the label is returned"


def test_derive_name_empty_description_falls_back_to_sub_id():
    """No first line means the fallback label, or '' while the id is unknown."""
    assert (
        agent_tools._derive_name("", 9) == "sub-9"
    ), "an empty description falls back to the id-based label"
    assert (
        agent_tools._derive_name("", None) == ""
    ), "with no id yet there is nothing to build the fallback from"


@pytest.mark.parametrize("description", [None, 42, ["line one"]])
def test_derive_name_non_string_description_counts_as_absent(description):
    """Anything that is not a str is treated as absent, never measured."""
    assert (
        agent_tools._derive_name(description, 9) == "sub-9"
    ), f"a {type(description).__name__} description must not crash or be measured"
    assert (
        agent_tools._derive_name(description, None) == ""
    ), "a non-str description with no id yields the empty label"


def test_derive_name_whitespace_only_description_counts_as_absent():
    """A first line of pure whitespace is no first line at all."""
    assert (
        agent_tools._derive_name("   \t  ", 9) == "sub-9"
    ), "whitespace-only descriptions fall back to the id-based label"


# --- spawn_agent -------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_agent_without_orchestrator_answers_a_plain_error_sentence():
    """No orchestrator is an 'Error:' sentence, never a raised class name."""
    _make_ctx(None)
    result = await agent_tools.spawn_agent("desc", "prompt")
    _assert_plain_orchestrator_error(result)

    context_module._CURRENT = None  # pylint: disable=protected-access
    # And with no context registered at all, the answer is still prose.
    result = await agent_tools.spawn_agent("desc", "prompt")
    _assert_plain_orchestrator_error(result)


@pytest.mark.asyncio
async def test_spawn_agent_creates_child_returns_id_sends_prompt_without_awaiting():
    """The child is created, the prompt queued, the id returned — task untouched."""
    orch = _Orchestrator(sessions={1: _Session(1, name="root", provider="prov-root")})
    _make_ctx(orch)

    result = await asyncio.wait_for(
        agent_tools.spawn_agent("Write tests", "please run the tests", parent_id=1),
        timeout=DEADLINE_SECONDS,
    )

    assert result == _created_sentence(
        100, "Write tests", " under agent 1"
    ), "the id, the derived name and the parent must all be in the answer"
    assert orch.created == [
        {
            "name": "Write tests",
            "provider": "prov-root",
            "description": "Write tests",
            "parent_id": 1,
        }
    ], "one session, created under the parent with the derived name"
    assert orch.sent == [(100, "please run the tests")], "the prompt must be sent"
    assert all(not task.was_awaited for task in orch.tasks), (
        "spawn must return before the child's task completes; awaiting it "
        "would hold the parent's turn for the child's whole run"
    )


@pytest.mark.asyncio
async def test_spawn_agent_child_inherits_provider_and_keeps_description():
    """The parent's provider is inherited; the description passes through whole."""
    orch = _Orchestrator()
    parent_provider = object()
    orch.sessions[1] = _Session(1, name="root", provider=parent_provider)
    _make_ctx(orch)

    description = "Fix the login bug\nand add regression tests"
    result = await agent_tools.spawn_agent(description, "go", parent_id=1)

    assert (
        orch.created[0]["provider"] is parent_provider
    ), "the child must run on the parent's provider, not the orchestrator default"
    assert (
        orch.created[0]["description"] == description
    ), "the whole description is kept; only the name is derived from line one"
    assert (
        orch.created[0]["name"] == "Fix the login bug"
    ), "the name handed to create_session is the first line"
    assert (
        orch.sessions[100].name == "Fix the login bug"
    ), "the session carries the derived name once the id exists"
    assert result == _created_sentence(
        100, "Fix the login bug", " under agent 1"
    ), "the answer names the child by its first line"


@pytest.mark.asyncio
async def test_spawn_agent_without_a_calling_agent_spawns_a_root():
    """No calling agent means a root child with provider None, at the root."""
    orch = _Orchestrator()
    _make_ctx(orch)

    result = await agent_tools.spawn_agent("Solo agent", "run alone")

    assert result == _created_sentence(
        100, "Solo agent", " at the root"
    ), "a parentless spawn is reported as created at the root"
    assert (
        orch.created[0]["parent_id"] is None
    ), "no calling agent means no parent is invented"
    assert (
        orch.created[0]["provider"] is None
    ), "None tells create_session to use its own default"
    assert orch.lookups == [], "a parentless spawn has no parent session to fetch"
    assert orch.sent == [(100, "run alone")], "the prompt still reaches the child"


@pytest.mark.asyncio
async def test_spawn_agent_defaults_parent_to_the_calling_agent():
    """parent_id omitted inside a turn means 'a child of me'."""
    orch = _Orchestrator(sessions={7: _Session(7, name="caller", provider="prov-7")})
    _make_ctx(orch)
    current_agent_id.set(7)

    result = await agent_tools.spawn_agent("Child", "go")

    assert result == _created_sentence(
        100, "Child", " under agent 7"
    ), "the calling agent from the context becomes the parent"
    assert (
        orch.created[0]["parent_id"] == 7
    ), "the default parent must come from the current agent context"
    assert (
        orch.created[0]["provider"] == "prov-7"
    ), "the inherited provider follows the resolved parent"
    assert orch.sent == [(100, "go")], "the prompt reaches the new child"


@pytest.mark.asyncio
async def test_spawn_agent_at_the_depth_cap_refuses_and_names_the_numbers():
    """The refusal names current depth, the limit, and the would-be depth."""
    orch = _Orchestrator(
        sessions={
            40: _Session(40, parent_id=None),
            50: _Session(50, parent_id=40),
            60: _Session(60, parent_id=50),
        }
    )
    _make_ctx(orch, max_sub_agent_depth=1)

    result = await agent_tools.spawn_agent("Too deep", "go", parent_id=60)

    assert result.startswith(
        "Error: refusing to spawn"
    ), f"a cap refusal is an Error sentence: {result!r}"
    assert "under agent 60" in result, f"the refused parent is named: {result!r}"
    assert "nesting depth 2" in result, f"the current depth is named: {result!r}"
    assert "limit is 1" in result, f"the limit is named: {result!r}"
    assert "would be depth 3" in result, f"the would-be depth is named: {result!r}"
    assert orch.created == [], "a refused spawn must not create anything"
    assert orch.sent == [], "a refused spawn must not send anything"


@pytest.mark.asyncio
async def test_spawn_agent_refuses_a_parent_outside_the_callers_subtree():
    """A deep agent cannot walk around the depth cap by naming a shallow parent.

    The caller at depth 1 names its own sibling branch as the parent.  The
    sibling exists, so the unknown-parent check passes; the subtree walk from
    it never lands on the caller, and the spawn is refused in prose with
    nothing created or sent.
    """
    orch = _Orchestrator(
        sessions={
            1: _Session(1, parent_id=None),
            2: _Session(2, parent_id=1),
            7: _Session(7, parent_id=1),
        }
    )
    _make_ctx(orch)
    current_agent_id.set(7)

    result = await agent_tools.spawn_agent("Climb", "go", parent_id=2)

    assert result.startswith(
        "Error: agent 2 is not you (agent 7)"
    ), f"the refusal names the requested parent and the caller: {result!r}"
    assert "depth cap" in result, f"the refusal says why the rule exists: {result!r}"
    assert (
        "Omit parent_id" in result
    ), f"the refusal names the allowed alternative: {result!r}"
    assert orch.created == [], "an out-of-subtree spawn must not create anything"
    assert orch.sent == [], "an out-of-subtree spawn must not send anything"


@pytest.mark.asyncio
async def test_spawn_agent_allows_a_parent_inside_the_callers_subtree():
    """Spawning under one of your own descendants still works."""
    orch = _Orchestrator(
        sessions={
            7: _Session(7, parent_id=None),
            8: _Session(8, parent_id=7),
        }
    )
    _make_ctx(orch)
    current_agent_id.set(7)

    result = await agent_tools.spawn_agent("Grandchild", "go", parent_id=8)

    assert result == _created_sentence(
        100, "Grandchild", " under agent 8"
    ), "a parent inside the caller's subtree is honoured"
    assert (
        orch.created[0]["parent_id"] == 8
    ), "the requested in-subtree parent reaches create_session"
    assert orch.sent == [(100, "go")], "the prompt is delivered as usual"


@pytest.mark.asyncio
async def test_spawn_agent_allows_an_explicit_parent_equal_to_the_caller():
    """Naming yourself explicitly is the same as omitting parent_id."""
    orch = _Orchestrator(sessions={7: _Session(7, parent_id=None)})
    _make_ctx(orch)
    current_agent_id.set(7)

    result = await agent_tools.spawn_agent("Self child", "go", parent_id=7)

    assert result == _created_sentence(
        100, "Self child", " under agent 7"
    ), "the caller itself is trivially inside its own subtree"
    assert (
        orch.created[0]["parent_id"] == 7
    ), "the explicit self-parent reaches create_session"


@pytest.mark.asyncio
async def test_spawn_agent_without_a_caller_honours_an_explicit_parent():
    """No agent turn means no subtree to check against; the parent is honoured.

    The REPL/test path has no caller, so there is no privilege boundary to
    enforce — the unknown-parent check still applies, as ever.
    """
    orch = _Orchestrator(sessions={5: _Session(5, parent_id=None)})
    _make_ctx(orch)

    result = await agent_tools.spawn_agent("Loose child", "go", parent_id=5)

    assert result == _created_sentence(
        100, "Loose child", " under agent 5"
    ), "with no calling agent, an explicit parent is honoured as before"
    assert (
        orch.created[0]["parent_id"] == 5
    ), "the explicit parent reaches create_session unchanged"


@pytest.mark.parametrize("cycle_len", [1, 2])
def test_subtree_walk_terminates_on_a_cyclic_chain(cycle_len):
    """A cycle above the named parent ends the walk in False, not a hang.

    The walk is the same unawaited ``while`` loop as :func:`depth_of`'s, so a
    regression would block the event loop itself; the daemon-thread deadline
    turns that into a failed assertion instead of a frozen suite.
    """
    orch = _Orchestrator(sessions=_cycle_sessions(cycle_len))

    reachable = _finish_within(lambda: agent_tools._within_caller_subtree(orch, 7, 1))

    assert reachable is False, (
        f"a caller outside a {cycle_len}-cycle is never reached by the walk; "
        "the visited-set guard must terminate it and answer False"
    )


@pytest.mark.parametrize("cycle_len", [1, 2, 3])
def test_spawn_agent_cyclic_parent_chain_terminates(cycle_len):
    """Spawning under a cyclic chain terminates and completes the spawn."""
    orch = _Orchestrator(sessions=_cycle_sessions(cycle_len))
    # A generous cap keeps every cyclic chain's depth under the limit, so a
    # terminating walk lets the spawn proceed; a hung walk hits the deadline.
    _make_ctx(orch, max_sub_agent_depth=10)

    result = _finish_within(
        lambda: asyncio.run(agent_tools.spawn_agent("Under cycle", "go", parent_id=1))
    )

    assert result == _created_sentence(
        100, "Under cycle", " under agent 1"
    ), f"a spawn under a {cycle_len}-cycle must terminate with the child created"
    assert orch.sent == [(100, "go")], "the prompt is sent once the walk ends"


@pytest.mark.asyncio
async def test_spawn_agent_unknown_parent_answers_error_and_creates_nothing():
    """A parent id with no session yields an Error sentence and no side effects."""
    orch = _Orchestrator()
    _make_ctx(orch)

    result = await agent_tools.spawn_agent("Orphan", "go", parent_id=99)

    assert result.startswith(
        "Error: no agent with id 99"
    ), f"the missing parent is named in an Error sentence: {result!r}"
    assert orch.created == [], "nothing is created under a parent that does not exist"
    assert orch.sent == [], "nothing is sent on behalf of a spawn that never happened"


@pytest.mark.asyncio
async def test_spawn_agent_empty_description_assigns_the_fallback_name():
    """An empty description names the session 'sub-<id>' once the id exists."""
    orch = _Orchestrator(sessions={1: _Session(1, name="root", provider="p")})
    _make_ctx(orch)

    result = await agent_tools.spawn_agent("", "go", parent_id=1)

    assert result == _created_sentence(
        100, "sub-100", " under agent 1"
    ), "the fallback label is built from the id the orchestrator allocated"
    assert (
        orch.sessions[100].name == "sub-100"
    ), "the placeholder name is corrected on the session before the send"


@pytest.mark.asyncio
async def test_spawn_agent_survives_a_session_that_refuses_the_name():
    """A session that will not take the generated name must not kill the spawn."""
    orch = _FrozenNameOrchestrator(sessions={1: _Session(1, name="root")})
    _make_ctx(orch)

    result = await agent_tools.spawn_agent("", "go", parent_id=1)

    assert result == _created_sentence(100, "sub-100", " under agent 1"), (
        "the spawn completes and reports the derived name even though the "
        "session refused to store it"
    )
    assert orch.sent == [(100, "go")], "the prompt is still delivered"


# --- list_agents -------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_agents_renders_both_child_keys_and_defaults():
    """Both child key spellings, enum state, None name and None parent render."""
    orch = _Orchestrator(
        snapshot=[
            {
                "id": 1,
                "name": "root",
                "state": "idle",
                "parent_id": None,
                "sub_agent_ids": [2],
            },
            {
                "id": 2,
                "name": None,
                "state": _EnumState("running"),
                "parent_id": 1,
                "sub_agent": [9, 10, 11],
            },
        ]
    )
    _make_ctx(orch)

    result = await agent_tools.list_agents()

    assert result == (
        "2 agents:\n"
        "[1] root (idle) | parent none | 1 child\n"
        "[2] <unnamed> (running) | parent 1 | 3 children"
    ), "one line per agent with id, name, state, parent and child count"


@pytest.mark.asyncio
async def test_list_agents_single_agent_singular_header():
    """One agent gets the singular header and an explicit zero child count."""
    orch = _Orchestrator(
        snapshot=[{"id": 5, "name": "solo", "state": "idle", "parent_id": None}]
    )
    _make_ctx(orch)

    result = await agent_tools.list_agents()

    assert (
        result == "1 agent:\n[5] solo (idle) | parent none | 0 children"
    ), "the singular header and the zero-children count are spelled out"


@pytest.mark.parametrize("snapshot", [[], None], ids=["empty-list", "none"])
@pytest.mark.asyncio
async def test_list_agents_empty_or_none_snapshot_reports_no_agents(snapshot):
    """A falsy snapshot of any shape renders the no-agents sentence."""
    orch = _Orchestrator(snapshot=snapshot)
    _make_ctx(orch)

    result = await agent_tools.list_agents()

    assert (
        result == "There are no agents yet."
    ), f"a {type(snapshot).__name__} snapshot reads as 'no agents yet'"


@pytest.mark.asyncio
async def test_list_agents_truthy_non_list_snapshot_names_the_type():
    """A truthy non-list snapshot is diagnosed, not rendered or swallowed."""
    orch = _Orchestrator(snapshot={"agents": 2})
    _make_ctx(orch)

    result = await agent_tools.list_agents()

    assert result == (
        "Error: get_tree_snapshot() returned dict; a list of agent records "
        "was expected."
    ), "the offending type is named so a wiring bug is diagnosable"


@pytest.mark.asyncio
async def test_list_agents_without_orchestrator_answers_a_plain_error_sentence():
    """No orchestrator is an 'Error:' sentence, never a raised class name."""
    _make_ctx(None)
    result = await agent_tools.list_agents()
    _assert_plain_orchestrator_error(result)

    context_module._CURRENT = None  # pylint: disable=protected-access
    # And with no context registered at all, the answer is still prose.
    result = await agent_tools.list_agents()
    _assert_plain_orchestrator_error(result)


@pytest.mark.asyncio
async def test_list_agents_missing_entry_keys_use_placeholders():
    """An entry with no keys renders the placeholders, never an exception."""
    orch = _Orchestrator(snapshot=[{}])
    _make_ctx(orch)

    result = await agent_tools.list_agents()

    assert (
        result == "1 agent:\n[?] <unnamed> (unknown) | parent none | 0 children"
    ), "every missing field falls back to its rendered placeholder"


@pytest.mark.asyncio
async def test_list_agents_non_mapping_entries_render_placeholders():
    """An entry that is not a mapping at all has no readable fields."""
    orch = _Orchestrator(snapshot=["junk"])
    _make_ctx(orch)

    result = await agent_tools.list_agents()

    assert (
        result == "1 agent:\n[?] <unnamed> (unknown) | parent none | 0 children"
    ), "a non-mapping entry is rendered from placeholders, not crashed on"


@pytest.mark.asyncio
async def test_spawn_agent_session_without_an_id_reports_the_error():
    """A created session with no agent id cannot receive the prompt."""
    orch = _IdlessOrchestrator()
    orch.sessions[1] = _Session(1, name="root")
    _make_ctx(orch)

    result = await agent_tools.spawn_agent("No id", "go", parent_id=1)

    assert result.startswith(
        "Error: the orchestrator created a session"
    ), f"the missing id is diagnosed in an Error sentence: {result!r}"
    assert "agent id" in result, f"the diagnosis names what is missing: {result!r}"
    assert orch.sent == [], "no prompt can be sent to a session with no id"


@pytest.mark.asyncio
async def test_list_agents_clamps_a_long_rendering():
    """A rendering longer than the budget is clamped, keeping both ends."""
    orch = _Orchestrator(
        snapshot=[{"id": 5, "name": "N" * 80, "state": "idle", "parent_id": None}]
    )
    _make_ctx(orch, max_output_chars=10)

    clamped = await agent_tools.list_agents()

    # The full rendering is 127 characters ("1 agent:\n[5] " + 80 + the
    # suffix), so a budget of 10 keeps 5 head characters, 5 tail characters
    # and reports the 117 dropped in between.
    assert (
        "[truncated: 117 characters omitted]" in clamped
    ), f"a long rendering must say how much was dropped: {clamped!r}"
    assert clamped.startswith("1 age"), "the head of the rendering is kept"
    assert clamped.endswith("ldren"), "the tail of the rendering is kept"
    assert "N" * 80 not in clamped, "the middle payload is genuinely dropped"


# --- query_agent -------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_agent_reports_the_full_shape():
    """Id, name, state, description and the last assistant message, in order."""
    orch = _query_orchestrator(
        messages=[_Block(role="assistant", message="Hello there")]
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result == (
        "Agent 3 (writer) [running]\n"
        "description: Writes drafts\n"
        "last message:\n"
        "Hello there"
    ), "the report is three lines: identity, description, last message"


@pytest.mark.asyncio
async def test_query_agent_scans_backwards_past_tool_blocks():
    """Tool-call and tool-result blocks are skipped; their text never leaks."""
    orch = _query_orchestrator(
        messages=[
            _Block(role="assistant", message="OLDER-ANSWER"),
            _Block(message="CALL-ONE"),
            _Block(role="tool", message="RESULT-ONE"),
            _Block(role="assistant", message="NEWEST-ANSWER"),
            _Block(message="CALL-TWO"),
            _Block(role="tool", message="RESULT-TWO"),
        ]
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result.endswith(
        "last message:\nNEWEST-ANSWER"
    ), f"the newest assistant block is reported: {result!r}"
    for leaked in ("OLDER-ANSWER", "CALL-ONE", "RESULT-ONE", "CALL-TWO", "RESULT-TWO"):
        assert (
            leaked not in result
        ), f"{leaked} must never reach the answer; only assistant text does"


@pytest.mark.asyncio
async def test_query_agent_reports_the_newest_assistant_message():
    """Between two assistant blocks, the later one is the answer."""
    orch = _query_orchestrator(
        messages=[
            _Block(role="assistant", message="older"),
            _Block(role="assistant", message="newer"),
        ]
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result.endswith(
        "last message:\nnewer"
    ), f"the scan runs backwards, so the newest block wins: {result!r}"
    assert "older" not in result, "an older assistant message must not be reported"


@pytest.mark.parametrize(
    "messages",
    [[], [_Block(message="CALL-NOISE"), _Block(role="tool", message="RESULT-NOISE")]],
    ids=["empty", "only-tool-blocks"],
)
@pytest.mark.asyncio
async def test_query_agent_without_assistant_blocks_reports_none_yet(messages):
    """No assistant block at all renders the 'none yet' sentence."""
    orch = _query_orchestrator(messages=messages)
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result == (
        "Agent 3 (writer) [running]\n" "description: Writes drafts\n" + NONE_YET
    ), "an agent with no assistant message yet says so honestly"


@pytest.mark.asyncio
async def test_query_agent_none_block_is_skipped():
    """A None block inside messages is skipped, not fatal."""
    orch = _query_orchestrator(
        messages=[_Block(role="assistant", message="real"), None]
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result.endswith(
        "last message:\nreal"
    ), f"a None block must not crash the scan or hide the answer: {result!r}"


@pytest.mark.parametrize(
    "messages",
    [
        [_Block(role="assistant", message="older text"), _Block(role="assistant")],
        [
            _Block(role="assistant", message="older text"),
            _Block(role="assistant", message=None),
        ],
    ],
    ids=["no-message-attr", "message-none"],
)
@pytest.mark.asyncio
async def test_query_agent_textless_last_assistant_stops_the_scan(messages):
    """A textless newest assistant block reports 'none yet', not older text."""
    orch = _query_orchestrator(messages=messages)
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result == (
        "Agent 3 (writer) [running]\n" "description: Writes drafts\n" + NONE_YET
    ), "the scan stops at the first assistant block even when it has no text"
    assert (
        "older text" not in result
    ), "falling through to an older block would dress a stale answer as current"


@pytest.mark.parametrize("messages", [None, "not-a-list"], ids=["none", "str"])
@pytest.mark.asyncio
async def test_query_agent_non_list_messages_count_as_none(messages):
    """A messages field that is not a list or tuple counts as no messages."""
    orch = _query_orchestrator(messages=messages)
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result == (
        "Agent 3 (writer) [running]\n" "description: Writes drafts\n" + NONE_YET
    ), "an unreadable transcript is reported as 'no message yet'"


@pytest.mark.asyncio
async def test_query_agent_unknown_id_answers_an_error_sentence():
    """An unregistered id yields an Error sentence naming the id."""
    orch = _query_orchestrator()
    _make_ctx(orch)

    result = await agent_tools.query_agent(99)

    assert result.startswith(
        "Error: no agent with id 99"
    ), f"the missing agent is named in an Error sentence: {result!r}"


@pytest.mark.asyncio
async def test_query_agent_without_orchestrator_answers_a_plain_error_sentence():
    """No orchestrator is an 'Error:' sentence, never a raised class name."""
    _make_ctx(None)
    result = await agent_tools.query_agent(3)
    _assert_plain_orchestrator_error(result)

    context_module._CURRENT = None  # pylint: disable=protected-access
    # And with no context registered at all, the answer is still prose.
    result = await agent_tools.query_agent(3)
    _assert_plain_orchestrator_error(result)


@pytest.mark.asyncio
async def test_query_agent_clamps_the_message_but_not_the_description():
    """The message is clamped to the budget; the description is kept whole."""
    orch = _query_orchestrator(
        messages=[_Block(role="assistant", message="M" * 200)],
        description="D" * 200,
    )
    _make_ctx(orch, max_output_chars=40)

    result = await agent_tools.query_agent(3)

    assert (
        "description: " + "D" * 200 in result
    ), "the description is reported in full — it is not clamped"
    assert (
        "M" * 20 + "\n\n[truncated: 160 characters omitted]\n\n" + "M" * 20 in result
    ), "the message keeps its head and tail around the truncation marker"
    assert (
        "[truncated: 160 characters omitted]" in result
    ), "the marker names how much was dropped"
    assert "M" * 200 not in result, "the middle of the message is genuinely dropped"


@pytest.mark.parametrize("description", ["", None], ids=["empty", "none"])
@pytest.mark.asyncio
async def test_query_agent_falsy_name_and_description_use_placeholders(description):
    """A falsy name renders '<unnamed>'; an absent description '(none)'."""
    orch = _query_orchestrator(
        messages=[_Block(role="assistant", message="hi")],
        name="",
        description=description,
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result == (
        "Agent 3 (<unnamed>) [running]\n" "description: (none)\n" "last message:\n" "hi"
    ), "blank identity fields render as explicit placeholders"


@pytest.mark.asyncio
async def test_query_agent_enum_state_renders_its_value():
    """An enum-like state renders through its .value attribute."""
    orch = _query_orchestrator(
        messages=[_Block(role="assistant", message="hi")],
        state=_EnumState("running"),
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result.startswith(
        "Agent 3 (writer) [running]"
    ), f"the enum member's value is what the model reads: {result!r}"


@pytest.mark.asyncio
async def test_query_agent_non_string_state_renders_unknown():
    """A state that is neither a string nor enum-like renders 'unknown'."""
    orch = _query_orchestrator(
        messages=[_Block(role="assistant", message="hi")], state=42
    )
    _make_ctx(orch)

    result = await agent_tools.query_agent(3)

    assert result.startswith(
        "Agent 3 (writer) [unknown]"
    ), f"an unreadable state is reported as 'unknown', never blank: {result!r}"


# --- error boundary ----------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_a_raising_get_session_keeps_the_class_name():
    """A genuine programming error is rendered with its exception type."""
    _make_ctx(_RaisingOrchestrator())

    result = await agent_tools.query_agent(1)

    assert (
        result == "Error: RuntimeError: boom"
    ), "the boundary format is exactly right for an unexpected exception"
