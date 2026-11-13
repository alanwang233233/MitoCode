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
