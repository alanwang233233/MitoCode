"""Tests for mitocode.core.agent.template.apply (replaying a template).

Applying a template is the step that turns three JSON blocks into the history a
provider will accept.  The tests here hold two contracts: the replayed blocks
land *before* the agent's first prompt and *after* (or, for a template that
declares its own persona, in place of) the session's system prompt, and the
resulting ``build_standard_context()`` output is a well-formed tool-call
transcript rather than a pile of dicts that merely look right.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import mitocode.core.agent.agent_orchestrator as orchestrator_module
from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.agent.agent_session import AgentSession, _PlainTextMessageBlock
from mitocode.core.agent.template.apply import apply_template
from mitocode.core.agent.template.template import AgentTemplate
from mitocode.core.chat_provider.scenario_provider import ScenarioProvider

BASE_PROMPT = "Base persona."


def _template(blocks, name="demo"):
    """Build an AgentTemplate carrying *blocks* without touching the disk."""
    return AgentTemplate(
        name=name,
        description="A demo template.",
        path=Path("/tmp") / name,
        template_path=Path("/tmp") / name / "template.json",
        source="test",
        session_autoload=tuple(blocks),
    )


def _text(role, message, reasoning=""):
    """One normalised text block."""
    return {
        "kind": "text",
        "role": role,
        "message": message,
        "reasoning_content": reasoning,
    }


def _call(name, call_id, arguments_string="{}"):
    """One normalised tool_call block."""
    return {
        "kind": "tool_call",
        "name": name,
        "id": call_id,
        "arguments_string": arguments_string,
        "parameter": {},
    }


def _result(call_id, content, success=True):
    """One normalised tool_result block."""
    return {
        "kind": "tool_result",
        "tool_call_id": call_id,
        "content": content,
        "success": success,
    }


def _session(system_prompt=BASE_PROMPT):
    """Build a real AgentSession with a scripted provider."""
    return AgentSession(provider=ScenarioProvider(), system_prompt=system_prompt)


def _roles(session):
    """Return the roles of the session's message blocks, in order."""
    return [getattr(block, "role", "tool_call") for block in session.messages]


# --- placement -------------------------------------------------------------


def test_blocks_are_replayed_after_the_default_system_prompt():
    """The template's context sits between the persona and the first task."""
    session = _session()

    apply_template(session, _template([_text("user", "earlier turn")]))

    assert _roles(session) == ["system", "user"]
    assert session.messages[1].message == "earlier turn"


def test_a_leading_system_block_replaces_the_default_prompt():
    """A template declaring a persona asserts the whole persona."""
    session = _session()

    apply_template(session, _template([_text("system", "Template persona.")]))

    assert _roles(session) == ["system"]
    assert session.messages[0].message == "Template persona."
    assert session.system_prompt == "Template persona."


def test_a_replacement_drops_every_default_system_block():
    """Replacement is by role, so a doubled default prompt cannot survive."""
    session = _session()
    session.messages.append(_PlainTextMessageBlock("system", "a second persona"))

    apply_template(session, _template([_text("system", "Template persona.")]))

    assert [block.message for block in session.messages if block.role == "system"] == [
        "Template persona."
    ]


def test_a_system_block_that_is_not_first_is_ordinary_history():
    """Only the first block can replace; later system text is just a message."""
    session = _session()

    apply_template(
        session,
        _template([_text("user", "task"), _text("system", "late persona")]),
    )

    assert _roles(session) == ["system", "user", "system"]
    assert session.messages[0].message == BASE_PROMPT


def test_a_template_with_no_blocks_leaves_the_session_untouched():
    """An empty template is a valid template and changes nothing."""
    session = _session()

    apply_template(session, _template([]))

    assert _roles(session) == ["system"]
    assert session.messages[0].message == BASE_PROMPT


# --- provider shape --------------------------------------------------------


def test_the_replayed_history_compiles_to_a_provider_shaped_transcript():
    """Call and result survive as one assistant tool_calls turn plus a tool turn."""
    session = _session()
    apply_template(
        session,
        _template(
            [
                _text("user", "read it"),
                _call("Read", "call_1", '{"file_path": "a.py"}'),
                _result("call_1", "print(1)"),
            ]
        ),
    )

    context = session.build_standard_context()

    assert [message["role"] for message in context] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assistant = context[2]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert (
        assistant["tool_calls"][0]["function"]["arguments"] == '{"file_path": "a.py"}'
    )
    assert context[3]["tool_call_id"] == "call_1"
    assert context[3]["content"] == "print(1)"


def test_reasoning_is_carried_onto_the_assistant_block():
    """A template may replay the reasoning that produced a turn."""
    session = _session()

    apply_template(
        session,
        _template([_text("assistant", "done", reasoning="because")]),
    )

    assert session.build_standard_context()[1]["reasoning_content"] == "because"


def test_a_tool_call_replayed_without_a_registered_tool_is_still_history():
    """Replay is data: it must not require the tool to exist in this session."""
    session = _session()

    apply_template(
        session,
        _template([_call("ToolThatVanished", "call_1"), _result("call_1", "ok")]),
    )

    assert (
        session.build_standard_context()[1]["tool_calls"][0]["function"]["name"]
        == "ToolThatVanished"
    )


# --- workspace substitution ------------------------------------------------


def test_the_workspace_placeholder_is_replaced_in_text_blocks(tmp_path):
    """A persona that names the workspace must name the *current* one."""
    session = _session()

    apply_template(
        session,
        _template([_text("system", "You work in {workspace}.")]),
        workspace=tmp_path,
    )

    assert session.messages[0].message == f"You work in {tmp_path}."


def test_other_braces_survive_verbatim(tmp_path):
    """JSON and code in a persona are not format strings."""
    session = _session()

    apply_template(
        session,
        _template([_text("system", 'Use {"a": 1} and {other}.')]),
        workspace=tmp_path,
    )

    assert session.messages[0].message == 'Use {"a": 1} and {other}.'


def test_without_a_workspace_the_placeholder_stays_visible(tmp_path):
    """Leaving it visible is honest; silently deleting it hides the omission."""
    session = _session()

    apply_template(session, _template([_text("system", "at {workspace}")]))

    assert session.messages[0].message == "at {workspace}"


# --- orchestrator wiring ---------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_seeds_the_new_session_from_the_template():
    """The orchestrator's default path applies the template to the new session."""
    orchestrator = AgentOrchestrator(
        default_provider=ScenarioProvider(),
        system_prompt=BASE_PROMPT,
        workspace=Path("/tmp/ws"),
    )

    session = await orchestrator.create_session(
        name="child",
        template=_template([_text("system", "Planner at {workspace}.")]),
    )

    assert _roles(session) == ["system"]
    # Compared through Path rather than as a literal: the same absolute path
    # renders with backslashes on Windows.
    assert session.messages[0].message == f"Planner at {Path('/tmp/ws')}."
    assert orchestrator.get_session(session.agent_id) is session


@pytest.mark.asyncio
async def test_the_template_is_applied_before_the_session_is_registered(monkeypatch):
    """No caller may observe a session that exists but has not been seeded."""
    orchestrator = AgentOrchestrator(
        default_provider=ScenarioProvider(), system_prompt=BASE_PROMPT
    )
    observed = {}

    def spy(session, template, workspace=None):
        observed["registered"] = orchestrator.get_session(session.agent_id)
        observed["messages"] = [block.message for block in session.messages]
        apply_template(session, template, workspace=workspace)

    monkeypatch.setattr(orchestrator_module, "apply_template", spy)

    await orchestrator.create_session(
        name="child", template=_template([_text("user", "context")])
    )

    assert observed["registered"] is None, (
        "the session must not be reachable from the registry while its history "
        "is still only the bare persona"
    )
    assert observed["messages"] == [BASE_PROMPT]


@pytest.mark.asyncio
async def test_create_session_without_a_template_is_unchanged():
    """The default path must stay byte-identical to the pre-template behaviour."""
    orchestrator = AgentOrchestrator(
        default_provider=ScenarioProvider(), system_prompt=BASE_PROMPT
    )

    session = await orchestrator.create_session(name="plain")

    assert _roles(session) == ["system"]
    assert session.messages[0].message == BASE_PROMPT


@pytest.mark.asyncio
async def test_a_seeded_child_still_joins_the_tree():
    """Template application must not disturb the parent/child wiring."""
    orchestrator = AgentOrchestrator(
        default_provider=ScenarioProvider(), system_prompt=BASE_PROMPT
    )
    parent = await orchestrator.create_session(name="parent")

    child = await orchestrator.create_session(
        name="child",
        parent_id=parent.agent_id,
        template=_template([_text("user", "context")]),
    )

    assert child.parent_id == parent.agent_id
    assert child.agent_id in parent.sub_agent_ids
    assert [entry["id"] for entry in orchestrator.get_tree_snapshot()] == [
        parent.agent_id,
        child.agent_id,
    ]


def test_every_replayed_block_is_marked_as_template_content():
    """The marker is what keeps a prepared turn out of `last message` answers."""
    session = _session()

    apply_template(
        session,
        _template(
            [_text("user", "ctx"), _call("Read", "call_1"), _result("call_1", "x")]
        ),
    )

    seeded = session.messages[1:]
    assert seeded, "the template's blocks must actually be appended"
    assert all(getattr(block, "from_template", False) for block in seeded)
    assert not getattr(
        session.messages[0], "from_template", False
    ), "the session's own persona is not template content"


def test_reusing_a_template_keeps_sessions_and_source_arguments_independent():
    """Mutable nested tool arguments belong to each newly seeded session."""
    import copy

    call = _call("Read", "read", '{"paths": ["a.py"]}')
    call["parameter"] = {"paths": ["a.py"]}
    template = _template([call, _result("read", "contents")])
    original = copy.deepcopy(template.session_autoload)
    first, second = _session(), _session()

    apply_template(first, template)
    apply_template(second, template)
    first.messages[1].parameter["paths"].append("changed.py")
    first.messages[2].content = "changed result"

    assert second.messages[1].parameter == {"paths": ["a.py"]}
    assert second.messages[2].content == "contents"
    assert template.session_autoload == original


def test_parallel_results_keep_their_ids_and_order_in_provider_context():
    """An out-of-order parallel result must still answer the right call."""
    session = _session()
    apply_template(
        session,
        _template(
            [
                _call("Read", "a"),
                _call("Glob", "b"),
                _result("b", "second", success=False),
                _result("a", "first"),
                _call("Grep", "c"),
                _result("c", "third"),
            ]
        ),
    )

    context = session.build_standard_context()

    assert [message["role"] for message in context] == [
        "system",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
    ]
    assert [call["id"] for call in context[1]["tool_calls"]] == ["a", "b"]
    assert [(context[i]["tool_call_id"], context[i]["content"]) for i in (2, 3, 5)] == [
        ("b", "second"),
        ("a", "first"),
        ("c", "third"),
    ]
    assert session.messages[3].success is False


def test_workspace_rendering_does_not_rewrite_tool_payloads(tmp_path):
    """Message text interpolates; reasoning and tool data stay verbatim."""
    arguments = '{"path": "{workspace}/a.py"}'
    session = _session()
    apply_template(
        session,
        _template(
            [
                _text("assistant", "at {workspace}", reasoning="inspect {workspace}"),
                _call("Read", "a", arguments),
                _result("a", "literal {workspace}"),
            ]
        ),
        workspace=tmp_path,
    )

    context = session.build_standard_context()

    assert context[1]["content"] == f"at {tmp_path}"
    assert context[1]["reasoning_content"] == "inspect {workspace}"
    assert context[2]["tool_calls"][0]["function"]["arguments"] == arguments
    assert context[3]["content"] == "literal {workspace}"


@pytest.mark.asyncio
async def test_failed_template_application_leaves_no_registered_child(monkeypatch):
    """A failed seed cannot leave a ghost agent in the registry or parent tree."""
    orchestrator = AgentOrchestrator(default_provider=ScenarioProvider())
    parent = await orchestrator.create_session(name="parent")
    before = orchestrator.get_tree_snapshot()

    def fail_apply(session, template, workspace=None):
        raise ValueError("cannot seed")

    monkeypatch.setattr(orchestrator_module, "apply_template", fail_apply)

    with pytest.raises(ValueError, match="cannot seed"):
        await orchestrator.create_session(
            name="child",
            parent_id=parent.agent_id,
            template=_template([_text("user", "context")]),
        )

    assert orchestrator.get_tree_snapshot() == before
    assert parent.sub_agent_ids == []


def test_replaying_one_template_into_two_sessions_does_not_share_mutable_blocks():
    template = _template(
        [
            _text("assistant", "prepared"),
            _call("Read", "call_1", '{"paths": ["original"]}'),
            _result("call_1", "original result"),
        ]
    )
    first, second = _session(), _session()
    apply_template(first, template)
    apply_template(second, template)

    first.messages[1].message = "changed"
    first.messages[2].parameter["paths"].append("changed")
    first.messages[3].content = "changed"

    assert second.messages[1].message == "prepared"
    assert second.messages[2].parameter == {"paths": ["original"]}
    assert second.build_standard_context()[-1]["content"] == "original result"
    assert template.session_autoload[0]["message"] == "prepared"
    assert template.session_autoload[1]["arguments_string"] == '{"paths": ["original"]}'


def test_workspace_substitution_is_limited_to_message_text(tmp_path):
    session = _session()
    apply_template(
        session,
        _template(
            [
                _text("assistant", "at {workspace}", reasoning="keep {workspace}"),
                _call("Read", "call_1", '{"path": "{workspace}"}'),
                _result("call_1", "literal {workspace}", success=False),
            ]
        ),
        workspace=tmp_path,
    )

    assert session.messages[1].message == f"at {tmp_path}"
    assert session.messages[1].reasoning_content == "keep {workspace}"
    assert session.messages[2].parameter == {"path": "{workspace}"}
    assert session.messages[3].success is False
    assert session.build_standard_context()[-1]["content"] == "literal {workspace}"


@pytest.mark.asyncio
async def test_failed_template_application_does_not_register_or_attach_child(
    monkeypatch,
):
    orchestrator = AgentOrchestrator(default_provider=ScenarioProvider())
    parent = await orchestrator.create_session(name="parent")
    before = orchestrator.get_tree_snapshot()

    def fail(session, template, workspace=None):
        raise ValueError("cannot seed history")

    monkeypatch.setattr(orchestrator_module, "apply_template", fail)
    with pytest.raises(ValueError, match="cannot seed history"):
        await orchestrator.create_session(
            name="child",
            parent_id=parent.agent_id,
            template=_template([_text("user", "seed")]),
        )

    assert orchestrator.get_tree_snapshot() == before
    assert parent.sub_agent_ids == []


@pytest.mark.parametrize(
    "blocks, expected",
    [
        ([], ""),
        ([_text("user", "context"), _text("system", "late")], ""),
        (
            [_text("system", "{workspace}: {workspace}; {other}")],
            "{workspace}: {workspace}; {other}",
        ),
    ],
)
def test_system_message_uses_only_a_leading_persona(blocks, expected):
    assert _template(blocks).system_message() == expected


def test_system_message_renders_all_workspace_occurrences(tmp_path):
    template = _template([_text("system", "{workspace}: {workspace}; {other}")])
    assert template.system_message(tmp_path) == f"{tmp_path}: {tmp_path}; {{other}}"


def test_reusing_a_template_creates_independent_message_blocks(tmp_path):
    template = _template(
        [
            _text("system", "Work at {workspace}."),
            _call("Read", "read", '{"path": "{workspace}/literal.py"}'),
            _result("read", "result from {workspace}", success=False),
        ]
    )
    first = _session()
    second = _session()

    apply_template(first, template, workspace=tmp_path / "first")
    apply_template(second, template, workspace=tmp_path / "second")

    assert first.system_prompt == f"Work at {tmp_path / 'first'}."
    assert second.system_prompt == f"Work at {tmp_path / 'second'}."
    assert all(a is not b for a, b in zip(first.messages, second.messages))
    first.messages[1].parameter["path"] = "changed"
    assert second.messages[1].parameter == {"path": "{workspace}/literal.py"}
    assert second.messages[2].success is False
    assert second.messages[2].content == "result from {workspace}"
    assert template.session_autoload[0]["message"] == "Work at {workspace}."
    assert (
        template.session_autoload[1]["arguments_string"]
        == '{"path": "{workspace}/literal.py"}'
    )


@pytest.mark.asyncio
async def test_template_failure_leaves_no_child_registered(monkeypatch):
    orchestrator = AgentOrchestrator(default_provider=ScenarioProvider())
    parent = await orchestrator.create_session(name="parent")
    before = orchestrator.get_tree_snapshot()

    def fail_apply(session, template, workspace=None):
        raise ValueError("synthetic replay failure")

    monkeypatch.setattr(orchestrator_module, "apply_template", fail_apply)
    with pytest.raises(ValueError, match="synthetic replay failure"):
        await orchestrator.create_session(
            name="child",
            parent_id=parent.agent_id,
            template=_template([_text("user", "context")]),
        )

    assert orchestrator.get_tree_snapshot() == before
    assert parent.sub_agent_ids == []


@pytest.mark.parametrize("leading_system", [False, True])
@pytest.mark.asyncio
async def test_template_persona_precedence_over_explicit_session_prompt(leading_system):
    orchestrator = AgentOrchestrator(
        default_provider=ScenarioProvider(),
        system_prompt=BASE_PROMPT,
    )
    session = await orchestrator.create_session(
        name="child",
        system_prompt="Explicit persona.",
        template=_template([_text("system" if leading_system else "user", "Seed.")]),
    )

    assert session.system_prompt == ("Seed." if leading_system else "Explicit persona.")
    assert session.messages[0].message == session.system_prompt
