"""Tests for built-in tool registration."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.tool.builtin.register import register_builtin_tools
from mitocode.core.tool.builtin.context import ToolContext
from mitocode.core.tool.tool_manager import ToolManager

EXPECTED = {
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Bash",
    "Agent",
    "ListAgent",
    "QueryAgent",
    "Background",
    "BackgroundResult",
}


def test_registers_all_eleven_tools():
    manager = ToolManager()
    assert set(register_builtin_tools(manager)) == EXPECTED


def test_every_tool_has_an_agent_doc():
    manager = ToolManager()
    register_builtin_tools(manager)
    for name in EXPECTED:
        assert manager.get_spec(name).agent_doc


def test_schemas_are_well_formed():
    manager = ToolManager()
    register_builtin_tools(manager)
    for schema in manager.tool_schemas("agent"):
        fn = schema["function"]
        assert fn["name"] in EXPECTED
        assert fn["parameters"]["type"] == "object"


def test_read_schema_has_offset_and_limit_defaults():
    manager = ToolManager()
    register_builtin_tools(manager)
    props = manager.get_spec("Read").parameters_schema["properties"]
    assert props["offset"]["default"] == 0
    assert props["limit"]["default"] == 2000


def test_grep_schema_constrains_output_mode_with_an_enum():
    manager = ToolManager()
    register_builtin_tools(manager)
    props = manager.get_spec("Grep").parameters_schema["properties"]
    assert set(props["output_mode"]["enum"]) == {
        "files_with_matches",
        "content",
        "count",
    }


def test_bash_default_timeout_is_120000_ms():
    manager = ToolManager()
    register_builtin_tools(manager)
    assert (
        manager.get_spec("Bash").parameters_schema["properties"]["timeout"]["default"]
        == 120_000
    )


def test_registration_is_idempotent():
    manager = ToolManager()
    register_builtin_tools(manager)
    before = dict(manager.tools_map())
    register_builtin_tools(manager)
    assert manager.tools_map() == before  # second call changes nothing


def test_second_call_still_reports_all_eleven_attempted():
    manager = ToolManager()
    register_builtin_tools(manager)
    assert set(register_builtin_tools(manager)) == EXPECTED
    assert len(manager.tools_map()) == 11  # no duplicates, no growth


def test_registration_refuses_to_clobber_an_existing_registration():
    """A user tool named Read must survive; ours is skipped, not substituted."""
    manager = ToolManager()

    @manager.tool(name="Read")
    async def _user_read() -> str:
        return "mine"

    registered = register_builtin_tools(manager)
    assert set(registered) == EXPECTED  # attempted names are all returned
    assert manager.get_spec("Read").func.__name__ == "_user_read"
    assert "Write" in manager.tools_map()  # the other ten still registered


def test_register_with_context_installs_current():
    from mitocode.core.tool.builtin.context import current, set_current

    manager = ToolManager()
    ctx = ToolContext()
    names = register_builtin_tools(manager, context=ctx)
    assert set(names) == EXPECTED

    # context should be installed
    assert current() is ctx
    assert ctx.tool_manager is not None
