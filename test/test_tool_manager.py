import os
import subprocess
import sys
import unittest.mock as mock
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.agent.agent_session import AgentSession
from mitocode.backend.agent.session_factory import SessionFactory
from mitocode.backend.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.backend.tool.tool_manager import (
    ToolManager,
    declare_tool,
    _build_parameters_schema,
)

#: Imports tool_manager in a process where fastmcp cannot be resolved at all,
#: so the one-way tool_manager -> mcp -> fastmcp dependency can be checked for
#: real rather than by reading the source.  Run out-of-process because the test
#: session has already imported fastmcp.
_IMPORT_WITHOUT_FASTMCP = """
import importlib.abc
import sys


class _BlockFastMcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fastmcp" or fullname.startswith("fastmcp."):
            raise ImportError("fastmcp is not installed")
        return None


sys.meta_path.insert(0, _BlockFastMcp())
import mitocode.backend.tool.tool_manager

assert "fastmcp" not in sys.modules, "fastmcp was imported"
leaked = [m for m in sys.modules if m.startswith("mitocode.backend.mcp")]
assert not leaked, f"mcp package imported at runtime: {leaked}"
"""


def test_tool_manager_decorator_registration_and_docs():
    manager = ToolManager()

    @manager.tool(human_doc="给用户看的加法工具", agent_doc="Use to add two integers.")
    def add(a: int, b: int = 1):
        return a + b

    assert manager.tools_map()["add"](2, 3) == 5
    assert manager.human_docs()["add"] == "给用户看的加法工具"
    assert manager.agent_docs()["add"] == "Use to add two integers."


def test_declare_tool_metadata_can_be_read_by_manager():
    manager = ToolManager()

    @declare_tool(
        name="echo_text",
        human_doc="给人看：回显文本",
        agent_doc="Echo user text input.",
    )
    def echo(text: str):
        return text

    manager.register(echo)
    assert "echo_text" in manager.tools_map()
    assert manager.human_docs()["echo_text"] == "给人看：回显文本"
    assert manager.agent_docs()["echo_text"] == "Echo user text input."


def test_tool_schema_uses_agent_docs_and_signature_types():
    manager = ToolManager()

    @manager.tool(agent_doc="Search documents.")
    def search(keyword: str, top_k: int = 5):
        return f"{keyword}:{top_k}"

    schemas = manager.tool_schemas("agent")
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "Search documents."
    assert schema["function"]["parameters"]["properties"]["keyword"]["type"] == "string"
    assert schema["function"]["parameters"]["properties"]["top_k"]["type"] == "integer"
    assert "keyword" in schema["function"]["parameters"]["required"]


def test_agent_session_can_use_tool_manager():
    manager = ToolManager()

    @manager.tool(human_doc="给人看的文档", agent_doc="Agent doc")
    def ping():
        return "pong"

    session = AgentSession(
        provider=mock.MagicMock(spec=_SampleChatProvider), tool_manager=manager
    )
    assert "ping" in session.tools_map
    assert session.tools_map["ping"]() == "pong"
    assert session.tool_schemas[0]["function"]["name"] == "ping"
    assert session.get_tool_docs_for_human()["ping"] == "给人看的文档"
    assert session.get_tool_docs_for_agent()["ping"] == "Agent doc"


def test_session_factory_forwards_tool_manager():
    manager = ToolManager()

    @manager.tool(agent_doc="Agent can call this.")
    def noop():
        return None

    factory = SessionFactory(
        provider=mock.MagicMock(spec=_SampleChatProvider), tool_manager=manager
    )
    session = factory.create_session()
    assert "noop" in session.tools_map
    assert session.tool_schemas[0]["function"]["name"] == "noop"


def test_dict_reference_pollution():
    def test_tool(city: str = "Beijing", country: str = "China"):
        pass

    schema = _build_parameters_schema(test_tool)
    props = schema["properties"]
    assert props["city"]["default"] == "Beijing"
    assert props["country"]["default"] == "China"
    assert props["city"] is not props["country"]


def test_tool_manager_imports_without_fastmcp_installed():
    """tool_manager must stay importable where fastmcp is absent.

    Dependency runs one way — tool_manager -> mcp -> fastmcp — so the MCP
    reference in ``register_mcp_client``'s annotation must stay under
    ``TYPE_CHECKING`` and the module must duck-type the client at runtime.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(os.path.dirname(__file__), "..", "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_FASTMCP],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_register_accepts_explicit_parameters_schema():
    manager = ToolManager()
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}

    def get_weather(**kwargs):
        return kwargs

    spec = manager.register(get_weather, name="get_weather", parameters_schema=schema)
    assert spec.parameters_schema == schema
    assert manager.tool_schemas("agent")[0]["function"]["parameters"] == schema


def test_register_without_parameters_schema_still_derives_from_annotations():
    manager = ToolManager()

    def add(a: int, b: int):
        return a + b

    spec = manager.register(add)
    assert spec.parameters_schema["properties"]["a"]["type"] == "integer"


def test_unregister_removes_tool_and_reports_presence():
    manager = ToolManager()

    @manager.tool(agent_doc="noop")
    def noop():
        return None

    assert manager.unregister("noop") is True
    assert manager.unregister("noop") is False
    assert "noop" not in manager.tools_map()
    assert manager.tool_schemas("agent") == []


def _fake_mcp_client(server_name, tools, calls=None):
    """Duck-typed stand-in for McpClient, avoiding a real fastmcp dependency.

    Args:
        server_name: Value for the stand-in's ``server_name``.
        tools: Value for the stand-in's ``tools`` catalogue.
        calls: Optional list collecting one ``(tool_name, arguments)`` pair per
            invocation, so a test can observe what a registered tool dispatches.
    """

    async def call_tool(tool_name, arguments):
        if calls is not None:
            calls.append((tool_name, arguments))
        return f"{tool_name} got {sorted(arguments)}"

    return SimpleNamespace(server_name=server_name, tools=tools, call_tool=call_tool)


def _fake_tool(name, description="", input_schema=None):
    """Duck-typed stand-in for one MCP server tool."""
    return SimpleNamespace(
        name=name, description=description, input_schema=input_schema
    )


def test_build_mcp_tool_name_uses_double_underscore_namespace():
    assert (
        ToolManager.build_mcp_tool_name("weather", "get_forecast")
        == "mcp__weather__get_forecast"
    )


def test_register_mcp_client_registers_namespaced_tools():
    manager = ToolManager()
    client = _fake_mcp_client(
        "weather",
        [
            _fake_tool(
                "get_forecast",
                description="Get the forecast.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
    )

    specs = manager.register_mcp_client(client)
    assert [s.name for s in specs] == ["mcp__weather__get_forecast"]
    assert "mcp__weather__get_forecast" in manager.tools_map()
    assert manager.agent_docs()["mcp__weather__get_forecast"] == (
        "[weather] Get the forecast."
    )
    schema = manager.tool_schemas("agent")[0]["function"]["parameters"]
    assert schema["properties"]["city"]["type"] == "string"


def test_register_mcp_client_synthesizes_description_when_missing():
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description=None)])

    manager.register_mcp_client(client)
    doc = manager.agent_docs()["mcp__files__read"]
    assert "files" in doc
    assert "read" in doc


def test_register_mcp_client_normalizes_missing_input_schema():
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", input_schema=None)])

    manager.register_mcp_client(client)
    schema = manager.tool_schemas("agent")[0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"] == {}


def test_register_mcp_client_completes_a_partial_input_schema_without_mutating_it():
    manager = ToolManager()
    advertised = {"properties": {"path": {"type": "string"}}}
    client = _fake_mcp_client("files", [_fake_tool("read", input_schema=advertised)])

    manager.register_mcp_client(client)
    schema = manager.tool_schemas("agent")[0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"] == {"path": {"type": "string"}}
    # The server's own catalogue object must come back untouched.
    assert advertised == {"properties": {"path": {"type": "string"}}}


def test_register_mcp_client_skips_name_collision():
    manager = ToolManager()

    @manager.tool(agent_doc="built-in")
    def mcp__files__read():
        return "builtin"

    client = _fake_mcp_client("files", [_fake_tool("read", description="remote")])
    specs = manager.register_mcp_client(client)

    assert specs == []
    assert manager.agent_docs()["mcp__files__read"] == "built-in"


def test_register_mcp_client_sets_display_action_for_tui():
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="d")])

    spec = manager.register_mcp_client(client)[0]
    assert spec.display_action == "mcp:files"
    assert spec.human_doc == "[files] d"


@pytest.mark.asyncio
async def test_registered_mcp_tools_dispatch_to_their_own_server_tool():
    calls = []
    manager = ToolManager()
    client = _fake_mcp_client(
        "files",
        [_fake_tool("read", description="r"), _fake_tool("write", description="w")],
        calls,
    )
    manager.register_mcp_client(client)
    tools = manager.tools_map()

    assert await tools["mcp__files__read"](path="a.txt") == "read got ['path']"
    await tools["mcp__files__write"](path="b.txt", body="hi")

    assert calls == [
        ("read", {"path": "a.txt"}),
        ("write", {"path": "b.txt", "body": "hi"}),
    ]
