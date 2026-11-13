"""Tests for mitocode.core.agent.template.template_json (template parsing).

A ``template.json`` is the declarative half of an agent template: a name, a
description and an optional ``session_autoload`` list of message blocks that is
replayed into a freshly created session.  The parser is the boundary between a
hand-written file and a provider request, so every rule it enforces is a
request the provider would otherwise reject -- an unpaired ``tool_call``, a
result answering nothing, a system block with no actor.

The pairing tests below are the important ones: hand-written templates
routinely omit ``tool_call.id``, and the parser is allowed to bind a result by
position *only* when the position is unambiguous.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mitocode.core.agent.template.template_json import parse_template_json
from mitocode.core.exception.mitocode_exception import (
    MitoCodeTemplateDefinitionError,
)


def _payload(**overrides):
    """Return a valid template mapping, with *overrides* applied."""
    payload = {
        "name": "demo",
        "description": "A demo template.",
    }
    payload.update(overrides)
    return payload


def _parse(**overrides):
    """Parse a valid template mapping with *overrides* applied."""
    return parse_template_json(json.dumps(_payload(**overrides)))


# --- the happy path --------------------------------------------------------


def test_a_minimal_template_needs_only_a_name_and_a_description():
    """No session_autoload means no seeded context, not an error."""
    document = _parse()

    assert document.name == "demo"
    assert document.description == "A demo template."
    assert document.session_autoload == ()


def test_an_empty_session_autoload_is_equivalent_to_an_absent_one():
    """An explicitly empty list is the same 'nothing to replay' state."""
    assert _parse(session_autoload=[]) == _parse()


def test_all_three_block_kinds_survive_a_round_trip():
    """Text, tool_call and tool_result blocks keep their fields and order."""
    document = _parse(
        session_autoload=[
            {"kind": "text", "role": "system", "message": "You plan."},
            {"kind": "text", "role": "user", "message": "Plan it."},
            {
                "kind": "tool_call",
                "name": "Read",
                "arguments_string": '{"file_path": "a.py"}',
            },
            {"kind": "tool_result", "tool_call_id": "call_1", "content": "print(1)"},
        ]
    )

    kinds = [block["kind"] for block in document.session_autoload]
    assert kinds == ["text", "text", "tool_call", "tool_result"]
    assert document.session_autoload[0]["role"] == "system"
    assert document.session_autoload[2]["name"] == "Read"
    assert document.session_autoload[2]["parameter"] == {"file_path": "a.py"}
    assert document.session_autoload[3]["content"] == "print(1)"
    assert document.session_autoload[3]["success"] is True


def test_reasoning_content_is_kept_when_present_and_defaulted_when_absent():
    """Reasoning is optional on text blocks and defaults to an empty string."""
    document = _parse(
        session_autoload=[
            {
                "kind": "text",
                "role": "assistant",
                "message": "a",
                "reasoning_content": "r",
            },
            {"kind": "text", "role": "assistant", "message": "b"},
        ]
    )

    assert document.session_autoload[0]["reasoning_content"] == "r"
    assert document.session_autoload[1]["reasoning_content"] == ""


def test_a_parameter_mapping_is_accepted_instead_of_arguments_string():
    """A hand-written template may pass the argument object directly."""
    document = _parse(
        session_autoload=[
            {"kind": "tool_call", "name": "Read", "parameter": {"file_path": "a.py"}},
            {"kind": "tool_result", "tool_call_id": "call_1", "content": "x"},
        ]
    )

    call = document.session_autoload[0]
    assert json.loads(call["arguments_string"]) == {"file_path": "a.py"}
    assert call["parameter"] == {"file_path": "a.py"}


def test_a_failed_tool_result_keeps_its_success_flag():
    """``success: false`` is data the model should see as a failed call."""
    document = _parse(
        session_autoload=[
            {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
            {
                "kind": "tool_result",
                "tool_call_id": "call_1",
                "content": "boom",
                "success": False,
            },
        ]
    )

    assert document.session_autoload[1]["success"] is False


# --- identity fields -------------------------------------------------------


def test_the_name_must_match_the_template_directory(tmp_path):
    """A template is addressed by its directory name, so the two cannot differ."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="directory name"):
        parse_template_json(json.dumps(_payload()), expected_name="other")


def test_the_name_must_be_a_lowercase_hyphenated_token():
    """Names are model-visible identifiers, not prose."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="'name'"):
        _parse(name="Demo Agent")


def test_underscores_are_accepted_in_a_name():
    """The templates that predate this format are named with underscores."""
    assert _parse(name="deep_research_agent").name == "deep_research_agent"


def test_a_doubled_or_leading_separator_is_refused():
    """``a__b`` and ``_ab`` are paths, not names."""
    for name in ("a__b", "_ab", "ab_", "a--b"):
        with pytest.raises(MitoCodeTemplateDefinitionError, match="'name'"):
            _parse(name=name)


def test_a_missing_name_is_refused():
    """``name`` is required: without it the template cannot be addressed."""
    payload = _payload()
    del payload["name"]
    with pytest.raises(MitoCodeTemplateDefinitionError, match="'name'"):
        parse_template_json(json.dumps(payload))


def test_a_missing_description_is_refused():
    """``description`` is what the model chooses a template by."""
    payload = _payload()
    del payload["description"]
    with pytest.raises(MitoCodeTemplateDefinitionError, match="'description'"):
        parse_template_json(json.dumps(payload))


def test_a_blank_description_is_refused():
    """Whitespace is not a description."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="'description'"):
        _parse(description="   \n  ")


def test_a_description_is_collapsed_to_one_line():
    """The catalogue renders one line per template, so newlines are folded."""
    assert _parse(description="line one\nline two").description == "line one line two"


def test_an_over_long_description_is_refused():
    """A description is a catalogue line, not a document."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="1024"):
        _parse(description="x" * 1025)


# --- document-level shape --------------------------------------------------


def test_text_that_is_not_json_is_refused():
    """The file must be JSON, not YAML or prose."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="JSON"):
        parse_template_json("name: demo")


def test_a_non_object_document_is_refused():
    """A bare list cannot carry a name."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="JSON object"):
        parse_template_json(json.dumps([_payload()]))


def test_non_string_content_is_refused():
    """The parser is called with decoded file text, never bytes."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="string"):
        parse_template_json(b'{"name": "demo"}')


def test_session_autoload_must_be_an_array():
    """A mapping is a single block, not a list of them."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="session_autoload"):
        _parse(session_autoload={"kind": "text", "role": "user", "message": "hi"})


def test_a_block_must_be_an_object():
    """A bare string cannot declare a kind."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="block 1"):
        _parse(session_autoload=["hello"])


def test_an_unknown_block_kind_is_refused():
    """Unknown kinds are refused rather than skipped: a partial context lies."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="kind"):
        _parse(session_autoload=[{"kind": "image", "url": "x"}])


# --- text blocks -----------------------------------------------------------


def test_a_text_block_needs_one_of_the_three_roles():
    """Only system, user and assistant messages exist."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="role"):
        _parse(session_autoload=[{"kind": "text", "role": "tool", "message": "x"}])


def test_a_text_block_needs_a_string_message():
    """A block with no text replays nothing."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="message"):
        _parse(session_autoload=[{"kind": "text", "role": "user"}])


def test_a_text_block_rejects_a_non_string_message():
    """A list of strings is not a message body."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="message"):
        _parse(session_autoload=[{"kind": "text", "role": "user", "message": ["a"]}])


# --- tool call blocks ------------------------------------------------------


def test_a_tool_call_needs_a_name():
    """A call with no tool name cannot be dispatched."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="name"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "arguments_string": "{}"},
                {"kind": "tool_result", "content": "x"},
            ]
        )


def test_arguments_string_must_be_a_json_object():
    """The provider parses the argument string, so it must already be an object."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="arguments_string"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read", "arguments_string": "not json"},
                {"kind": "tool_result", "content": "x"},
            ]
        )


def test_arguments_string_must_not_be_a_json_array():
    """A JSON array is valid JSON but not a valid argument object."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="arguments_string"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read", "arguments_string": "[1, 2]"},
                {"kind": "tool_result", "content": "x"},
            ]
        )


def test_a_tool_call_needs_arguments_or_a_parameter_mapping():
    """A call with neither has no arguments to send."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="arguments_string"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read"},
                {"kind": "tool_result", "content": "x"},
            ]
        )


# --- pairing --------------------------------------------------------------


def test_a_missing_call_id_is_generated_from_the_block_position():
    """Hand-written templates omit ids; the position is the identity."""
    document = _parse(
        session_autoload=[
            {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
            {"kind": "tool_result", "content": "x"},
        ]
    )

    call, result = document.session_autoload
    assert call["id"] == "call_1"
    assert result["tool_call_id"] == "call_1"


def test_the_generated_id_uses_the_block_number_not_a_running_count():
    """A call at block 3 is ``call_3``: the id names a position, not a counter."""
    document = _parse(
        session_autoload=[
            {"kind": "text", "role": "user", "message": "look"},
            {"kind": "text", "role": "assistant", "message": "looking"},
            {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
            {"kind": "tool_result", "content": "x"},
        ]
    )

    assert document.session_autoload[2]["id"] == "call_3"


def test_two_calls_may_not_share_an_id():
    """Two calls with one identity leave the provider no way to tell them apart."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="already used"):
        _parse(
            session_autoload=[
                {
                    "kind": "tool_call",
                    "id": "dup",
                    "name": "Read",
                    "arguments_string": "{}",
                },
                {
                    "kind": "tool_call",
                    "id": "dup",
                    "name": "Glob",
                    "arguments_string": "{}",
                },
                {"kind": "tool_result", "tool_call_id": "dup", "content": "a"},
                {"kind": "tool_result", "tool_call_id": "dup", "content": "b"},
            ]
        )


def test_a_generated_id_collision_is_refused():
    """A hand-picked ``call_1`` collides with the id block 1 would be given."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="already used"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
                {
                    "kind": "tool_call",
                    "id": "call_1",
                    "name": "Glob",
                    "arguments_string": "{}",
                },
                {"kind": "tool_result", "content": "a"},
                {"kind": "tool_result", "content": "b"},
            ]
        )


def test_a_result_naming_no_open_call_binds_to_the_only_open_one():
    """A single open call is unambiguous, so a stale id is corrected, not fatal."""
    document = _parse(
        session_autoload=[
            {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
            {
                "kind": "tool_result",
                "tool_call_id": "call_from_another_run",
                "content": "x",
            },
        ]
    )

    assert document.session_autoload[1]["tool_call_id"] == "call_1"


def test_a_result_naming_an_open_call_binds_to_that_call():
    """Ids win over position when the file names a call that is actually open."""
    document = _parse(
        session_autoload=[
            {"kind": "tool_call", "id": "a", "name": "Read", "arguments_string": "{}"},
            {"kind": "tool_call", "id": "b", "name": "Glob", "arguments_string": "{}"},
            {"kind": "tool_result", "tool_call_id": "b", "content": "second"},
            {"kind": "tool_result", "content": "first"},
        ]
    )

    assert document.session_autoload[2]["tool_call_id"] == "b"
    assert document.session_autoload[3]["tool_call_id"] == "a"


def test_a_result_is_refused_when_two_calls_are_open_and_the_id_is_unknown():
    """Guessing between two open calls would invent a transcript."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="tool_result"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
                {"kind": "tool_call", "name": "Glob", "arguments_string": "{}"},
                {"kind": "tool_result", "tool_call_id": "nope", "content": "x"},
            ]
        )


def test_a_result_with_no_open_call_is_refused():
    """A result answering nothing is a transcript the provider will reject."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="tool_result"):
        _parse(session_autoload=[{"kind": "tool_result", "content": "x"}])


def test_an_unanswered_tool_call_is_refused():
    """A trailing call with no result is a request the provider will reject."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="never answered"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
            ]
        )


def test_a_text_block_between_a_call_and_its_result_is_refused():
    """The provider requires tool results immediately after their call."""
    with pytest.raises(MitoCodeTemplateDefinitionError, match="block 2"):
        _parse(
            session_autoload=[
                {"kind": "tool_call", "name": "Read", "arguments_string": "{}"},
                {"kind": "text", "role": "user", "message": "too early"},
                {"kind": "tool_result", "content": "x"},
            ]
        )


def test_one_bad_block_rejects_the_whole_template():
    """Half a context is worse than none: nothing is salvaged per block."""
    with pytest.raises(MitoCodeTemplateDefinitionError) as error:
        _parse(
            session_autoload=[
                {"kind": "text", "role": "user", "message": "fine"},
                {"kind": "text", "role": "user"},
            ]
        )

    assert not hasattr(error.value, "partial")
