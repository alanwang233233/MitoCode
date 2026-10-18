import contextlib
import inspect
import os
import subprocess
import sys
import unittest.mock as mock
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pytest
from loguru import logger as loguru_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.agent.agent_session import AgentSession
from mitocode.core.agent.session_factory import SessionFactory
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.core.exception.mitocode_exception import MitoCodeToolDefinitionError
from mitocode.core.tool.tool_manager import (
    MAX_TOOL_NAME_LENGTH,
    MCP_TOOL_PREFIX,
    MCP_TOOL_SEPARATOR,
    ToolManager,
    declare_tool,
    _build_parameters_schema,
)

#: Imports tool_manager in a process where the MCP SDKs cannot be resolved at
#: all, then drives the MCP entry point, so the one-way
#: tool_manager -> mcp -> fastmcp dependency can be checked for real rather than by
#: reading the source.  Run out-of-process because the test session has already
#: imported fastmcp.
#:
#: Both halves matter.  The import-time assertion only sees imports executed while
#: ``tool_manager`` is being imported, so a lazy ``import`` inside a function body
#: sails past it; the registration below is what closes that hole, and the client
#: must advertise a tool for the registration loop to actually run.
_IMPORT_WITHOUT_MCP_SDK = """
import importlib.abc
import sys
from types import SimpleNamespace


class _BlockMcpSdk(importlib.abc.MetaPathFinder):
    \"\"\"Make both MCP SDK distributions unresolvable, however they are spelled.\"\"\"

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("fastmcp", "mcp"):
            raise ImportError(f"{fullname} is not installed")
        return None


class _DuckTypedClient:
    server_name = "files"
    tools = [SimpleNamespace(name="read", description="d", input_schema=None)]

    async def call_tool(self, tool_name, arguments):
        return ""


def leaked():
    return sorted(
        m
        for m in sys.modules
        if m.split(".")[0] in ("fastmcp", "mcp") or m.startswith("mitocode.core.mcp")
    )


sys.meta_path.insert(0, _BlockMcpSdk())
import mitocode.core.tool.tool_manager as tool_manager

assert not leaked(), f"mcp package imported at import time: {leaked()}"

# The advertised tool is what makes register_mcp_client's loop body run, which is
# exactly where a helpful-looking lazy import would hide.
registered = tool_manager.ToolManager().register_mcp_client(_DuckTypedClient())
assert [spec.name for spec in registered] == ["mcp__files__read"], (
    f"the loop body did not run, so this probe would prove nothing: {registered!r}"
)

assert not leaked(), f"register_mcp_client imported the mcp package: {leaked()}"
"""


def _run_import_probe(script, cwd):
    """Run *script* in a child interpreter that cannot resolve the MCP SDKs.

    ``PYTHONPATH`` is absolute and carries no empty trailing entry, and the child
    runs in an empty *cwd*: ``python -c`` puts the working directory at the front
    of ``sys.path`` and an empty ``PYTHONPATH`` element adds it again, so a
    ``cwd`` holding a ``mitocode/`` package would shadow the module under test
    and quietly turn the probe into a test of the decoy.

    Args:
        script: Source for the child interpreter.
        cwd: Directory the child runs in; must contain no importable package.

    Returns:
        The completed ``subprocess.CompletedProcess``.
    """
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([src_dir, env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        check=False,
    )


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


def test_tool_manager_imports_and_registers_without_the_mcp_sdk(tmp_path):
    """tool_manager must stay importable and usable without either MCP SDK.

    Dependency runs one way — tool_manager -> mcp -> fastmcp — so the MCP
    reference in ``register_mcp_client``'s annotation must stay under
    ``TYPE_CHECKING`` and the module must duck-type the client at runtime.

    The probe checks ``sys.modules`` twice: once after importing ``tool_manager``
    and once after registering a tool from a client.  A lazy ``import`` in a
    function body is invisible to the first check, which is why the second one
    drives the registration loop rather than merely calling the entry point.

    *tmp_path* is an empty directory so the child interpreter's ``sys.path``
    cannot pick up a stray ``mitocode/`` package and shadow the module under
    test.
    """
    result = _run_import_probe(_IMPORT_WITHOUT_MCP_SDK, tmp_path)
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


def _malformed_tool(name, description=123):
    """An advertised tool whose description is not a string.

    The type a server sends when it puts a number in a field the MCP schema
    says is a string.  It is what makes a catalogue unreadable part-way: the
    registry has to strip the description, and that step can only fail on a
    value that is not a string.

    Args:
        name: The tool's name as advertised.
        description: The undescribable value to carry.
    """
    return _fake_tool(name, description=description)


@contextlib.contextmanager
def _captured_logs():
    """Collect every loguru line emitted inside the block.

    Asserting against a patched ``logger`` proves the call was made; asserting
    against a real sink proves it reaches an operator's log file, which is the
    half that matters for a warning nobody is watching the console for.

    Yields:
        The list every emitted line is appended to.
    """
    lines = []
    sink_id = loguru_logger.add(lines.append, level="DEBUG")
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)


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


def test_mcp_tool_namespace_constants_are_the_documented_strings():
    """The prefix and separator are module-public; the composed name depends on them."""
    assert MCP_TOOL_PREFIX == "mcp__", "every remote tool must be recognisably remote"
    assert MCP_TOOL_SEPARATOR == "__", "server and tool names are joined by __"


def test_build_mcp_tool_name_joins_names_verbatim_without_escaping():
    """A server name holding the separator is not escaped, so composition is ambiguous."""
    composed = ToolManager.build_mcp_tool_name("my__files", "read")
    assert composed == "mcp__my__files__read", "names are concatenated as given"
    assert (
        ToolManager.build_mcp_tool_name("my", "files__read") == "mcp__my__files__read"
    ), "a different server/tool pair composes to the same string"


@pytest.mark.parametrize(
    "annotation, expected_type",
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (dict, "object"),
        (list, "array"),
        (type(None), "null"),
    ],
)
def test_build_parameters_schema_types_each_supported_scalar(annotation, expected_type):
    """Every supported base annotation maps to its documented JSON Schema type."""

    def probe(value: annotation):
        return value

    schema = _build_parameters_schema(probe)
    assert list(schema["properties"]) == ["value"], "one entry per keyword argument"
    assert schema["properties"]["value"] == {
        "type": expected_type
    }, f"{annotation!r} must render as {expected_type!r}"


def test_two_tools_with_the_same_annotation_get_independent_schema_dicts():
    """One tool's parameter schema must never be the same object as another's."""

    def first(text: str):
        return text

    def second(text: str):
        return text

    first_property = _build_parameters_schema(first)["properties"]["text"]
    second_property = _build_parameters_schema(second)["properties"]["text"]
    first_property["type"] = "integer"

    assert second_property == {
        "type": "string"
    }, "a schema fragment shared with the module's type map would be corrupted here"


def test_build_parameters_schema_types_an_unannotated_parameter_as_a_required_string():
    """An absent annotation means string, and the parameter stays required."""

    def unannotated(value):
        return value

    schema = _build_parameters_schema(unannotated)
    assert schema["properties"]["value"] == {
        "type": "string"
    }, "an un-annotated parameter is described as a string"
    assert schema["required"] == ["value"], "no default was given, so it is required"


def test_build_parameters_schema_renders_an_optional_parameter_as_a_two_branch_any_of():
    """``Optional[str]`` is a two-member union, so it is not a bare string."""

    def maybe(label: Optional[str]):
        return label

    schema = _build_parameters_schema(maybe)
    assert schema["properties"]["label"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }, "Optional[str] must offer both string and null"


def test_build_parameters_schema_renders_a_pep604_union_as_any_of():
    """``str | None`` must be treated exactly like ``Optional[str]``."""

    def maybe(label: str | None):
        return label

    schema = _build_parameters_schema(maybe)
    assert schema["properties"]["label"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }, "the PEP 604 union type is a union too"


def test_build_parameters_schema_renders_a_literal_as_an_enum_of_its_first_members_type():
    """The enum's JSON type is taken from its first member."""

    def choose(mode: Literal["read", "write"]):
        return mode

    schema = _build_parameters_schema(choose)
    assert schema["properties"]["mode"] == {
        "type": "string",
        "enum": ["read", "write"],
    }, "a Literal of strings is a string constrained to those values"


def test_build_parameters_schema_renders_an_empty_literal_as_a_string():
    """A Literal with no members has no member whose type could be sampled."""

    def choose(mode: Literal[()]):
        return mode

    schema = _build_parameters_schema(choose)
    assert schema["properties"]["mode"] == {
        "type": "string"
    }, "an empty enum falls back to the un-annotated default"


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (List[int], {"type": "array", "items": {"type": "integer"}}),
        (Tuple[int, str], {"type": "array", "items": {"type": "integer"}}),
        (Set[float], {"type": "array", "items": {"type": "number"}}),
    ],
)
def test_build_parameters_schema_types_a_parameterised_sequence_as_an_array(
    annotation, expected
):
    """A list, tuple or set annotation is an array described by its first item."""

    def collect(values: annotation):
        return values

    schema = _build_parameters_schema(collect)
    assert (
        schema["properties"]["values"] == expected
    ), f"{annotation!r} must render as {expected!r}"


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (list, {"type": "array"}),
        (List, {"type": "array"}),
        (dict, {"type": "object"}),
        (Dict, {"type": "object"}),
    ],
)
def test_build_parameters_schema_types_an_unparameterised_container_without_element_type(
    annotation, expected
):
    """A bare builtin and its bare ``typing`` alias must describe the same thing.

    Both spellings have no element or value type to advertise, so neither
    invents an ``items`` or ``additionalProperties`` key.
    """

    def collect(values: annotation):
        return values

    schema = _build_parameters_schema(collect)
    assert (
        schema["properties"]["values"] == expected
    ), f"{annotation!r} has no element type to advertise, so none is invented"


@pytest.mark.parametrize("annotation", [tuple, set])
def test_register_rejects_a_bare_tuple_or_set_annotation(annotation):
    """A bare ``list`` is supported but a bare ``tuple``/``set`` is not.

    The supported list in the docstring names str/int/float/bool/dict/list, so
    ``tuple`` and ``set`` are rejected even though their parameterised forms are
    accepted.
    """

    def collect(values: annotation):
        return values

    manager = ToolManager()
    with pytest.raises(MitoCodeToolDefinitionError, match="Unsupported type"):
        manager.register(collect, name="collect")

    assert (
        manager.tools_map() == {}
    ), "a rejected annotation must leave the registry untouched"


def test_build_parameters_schema_types_a_mapping_annotation_by_its_value_type():
    """The key type of a mapping annotation is dropped; the value type is kept."""

    def index(table: Dict[str, int]):
        return table

    schema = _build_parameters_schema(index)
    assert schema["properties"]["table"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }, "Dict[str, int] describes an object whose values are integers"


@pytest.mark.parametrize("annotation", [complex, Any])
def test_register_rejects_an_unsupported_annotation_without_registering(annotation):
    """An annotation outside the supported set fails loudly, leaving no half-built tool."""

    def broken(value: annotation):
        return value

    manager = ToolManager()
    with pytest.raises(
        MitoCodeToolDefinitionError, match="Unsupported type annotation"
    ):
        manager.register(broken, name="broken")

    assert manager.tools_map() == {}, "a failed derivation must not register a tool"


def test_register_rejects_a_positional_only_parameter_without_registering():
    """LLMs invoke tools by keyword, so a positional-only parameter is unrepresentable."""

    def broken(value: str, /):
        return value

    manager = ToolManager()
    with pytest.raises(
        MitoCodeToolDefinitionError, match="positional-only parameter 'value'"
    ):
        manager.register(broken, name="broken")

    assert (
        manager.tools_map() == {}
    ), "a rejected signature must leave the registry untouched"


def test_register_names_a_callable_object_by_its_class_in_the_rejection_message():
    """A tool without ``__name__`` is still named in the error, by its class."""

    class BrokenTool:
        def __call__(self, value: str, /):
            return value

    manager = ToolManager()
    with pytest.raises(MitoCodeToolDefinitionError, match="Tool function 'BrokenTool'"):
        manager.register(BrokenTool(), name="broken")

    assert (
        manager.tools_map() == {}
    ), "a rejected signature must leave the registry untouched"


def test_build_parameters_schema_omits_star_args_and_star_kwargs():
    """``*args``/``**kwargs`` are not keyword arguments a model could name."""

    def collect(required: int, *args, **kwargs):
        return required, args, kwargs

    schema = _build_parameters_schema(collect)
    assert list(schema["properties"]) == [
        "required"
    ], "only named parameters are described"
    assert schema["required"] == [
        "required"
    ], "the named parameter keeps its requiredness"


def test_build_parameters_schema_omits_the_required_key_when_nothing_is_required():
    """An all-defaults signature advertises defaults and no required list."""

    def defaulted(count: int = 0):
        return count

    schema = _build_parameters_schema(defaulted)
    assert "required" not in schema, "no required parameter means no required list"
    assert schema["properties"]["count"] == {
        "type": "integer",
        "default": 0,
    }, "the default is advertised so the model may omit the argument"


def test_build_parameters_schema_always_forbids_undeclared_arguments():
    """The top-level schema is always a closed object schema."""

    def simple(text: str):
        return text

    schema = _build_parameters_schema(simple)
    assert schema["type"] == "object", "a parameter list describes an object"
    assert (
        schema["additionalProperties"] is False
    ), "an undeclared argument cannot be passed by a model, so it is forbidden"


def test_declare_tool_without_a_name_falls_back_to_the_function_name():
    """An unnamed declaration keeps the decorated function's own name."""

    @declare_tool(human_doc="doc", agent_doc="agent")
    def fallback_name():
        return None

    manager = ToolManager()
    manager.register(fallback_name)

    assert list(manager.tools_map()) == [
        "fallback_name"
    ], "no name means the function's __name__"


def test_register_prefers_an_explicit_name_over_declare_tool_metadata():
    """The explicit ``name`` argument outranks the declared metadata name."""

    @declare_tool(name="meta_name")
    def conflict(text: str):
        return text

    manager = ToolManager()
    manager.register(conflict, name="explicit_name")

    assert list(manager.tools_map()) == [
        "explicit_name"
    ], "the explicit name is the one the model sees"


def test_register_prefers_explicit_docs_over_declare_tool_metadata():
    """The explicit doc arguments outrank the declared metadata docs."""

    @declare_tool(human_doc="meta human", agent_doc="meta agent")
    def conflict(text: str):
        return text

    manager = ToolManager()
    manager.register(conflict, human_doc="explicit human", agent_doc="explicit agent")

    assert manager.human_docs()["conflict"] == "explicit human", "explicit wins"
    assert manager.agent_docs()["conflict"] == "explicit agent", "explicit wins"


def test_register_falls_back_to_the_function_docstring_for_both_audiences():
    """With no metadata and no arguments, the function's docstring is the description."""

    def documented():
        """Search the workspace."""

    manager = ToolManager()
    manager.register(documented)

    assert manager.human_docs()["documented"] == "Search the workspace.", "docstring"
    assert manager.agent_docs()["documented"] == (
        "Search the workspace."
    ), "the agent description falls back to the human one"


def test_register_gives_an_undocumented_tool_empty_docs_for_both_audiences():
    """No docstring, no metadata and no arguments means no description at all."""

    def bare():
        return None

    manager = ToolManager()
    manager.register(bare)

    assert (
        manager.human_docs()["bare"] == ""
    ), "an absent docstring is not the string 'None'"
    assert manager.agent_docs()["bare"] == "", "and the agent gets nothing either"


def test_register_falls_back_to_the_human_doc_when_only_that_is_given():
    """An agent-only description may be omitted; the human one then serves both."""

    @declare_tool(human_doc="human only")
    def only_human():
        return None

    manager = ToolManager()
    manager.register(only_human)

    assert manager.human_docs()["only_human"] == "human only", "the declared human doc"
    assert manager.agent_docs()["only_human"] == (
        "human only"
    ), "the agent description defaults to the human description"


def test_register_reads_parameters_schema_from_declare_tool_metadata():
    """The documented metadata tier for ``parameters_schema`` does work.

    ``declare_tool`` writes only ``name``, ``human_doc`` and ``agent_doc``, so
    there is no public way to place ``parameters_schema`` into the metadata this
    tier reads; the private attribute has to be written by hand.  The tier is
    therefore honoured but not advertised as public API — see ``register``'s
    docstring.
    """
    manager = ToolManager()
    from_meta = {"type": "object", "properties": {"remote": {"type": "string"}}}

    @declare_tool(name="remote_call")
    def remote_call(**kwargs):
        return kwargs

    remote_call.__mitocode_tool_meta__["parameters_schema"] = from_meta
    spec = manager.register(remote_call)

    assert (
        spec.parameters_schema == from_meta
    ), "the metadata schema must outrank the derived empty one"
    assert (
        "remote" in spec.parameters_schema["properties"]
    ), "the server's shape survives"


def test_register_keeps_an_empty_parameters_schema_from_the_metadata_tier():
    """Tier two reads an empty schema like tier one does, not by truthiness.

    An explicit ``parameters_schema={}`` says "this tool takes no arguments",
    and that is a schema, not the absence of one.  Deriving one instead would
    advertise arguments the caller never agreed to, from a source that only
    looked like a default.
    """
    manager = ToolManager()

    @declare_tool(name="no_args")
    def no_args(**kwargs):
        return kwargs

    no_args.__mitocode_tool_meta__["parameters_schema"] = {}

    spec = manager.register(no_args)

    assert spec.parameters_schema == {}, "an empty schema from the metadata is kept"


def test_register_prefers_the_explicit_parameters_schema_over_declare_tool_metadata():
    """Tier one beats tier two: the explicit argument wins over the metadata."""

    @declare_tool(name="tiered")
    def tiered(**kwargs):
        return kwargs

    tiered.__mitocode_tool_meta__["parameters_schema"] = {
        "type": "object",
        "properties": {"from_meta": {"type": "string"}},
    }
    explicit = {"type": "object", "properties": {"explicit": {"type": "boolean"}}}

    manager = ToolManager()
    spec = manager.register(tiered, parameters_schema=explicit)

    assert spec.parameters_schema == explicit, "the explicit argument is the winner"
    assert (
        "from_meta" not in spec.parameters_schema["properties"]
    ), "the metadata tier must not be consulted at all"


def test_register_keeps_an_explicitly_empty_parameters_schema_instead_of_deriving_one():
    """An empty dict is a supplied schema, not an absent one.

    ``parameters_schema={}`` is how a caller says "this tool takes no
    arguments"; treating it as absent would advertise a derived schema instead.
    """

    def ping() -> str:
        return "pong"

    manager = ToolManager()
    spec = manager.register(ping, parameters_schema={})

    assert spec.parameters_schema == {}, "an explicit {} must be kept verbatim"


def test_register_replaces_a_registration_that_already_uses_the_name():
    """Registering the same name again overwrites rather than duplicating."""

    def first():
        return "first"

    def second():
        return "second"

    manager = ToolManager()
    manager.register(first, name="shared", agent_doc="first doc")
    spec = manager.register(second, name="shared", agent_doc="second doc")

    assert list(manager.tools_map()) == ["shared"], "one name, one entry"
    assert manager.tools_map()["shared"] is second, "the newest registration wins"
    assert spec.agent_doc == "second doc", "the description is replaced with it"


def test_register_defaults_both_display_hints_to_empty_strings():
    """Display hints are opt-in; without them the spec carries empty strings."""

    def plain(text: str):
        return text

    manager = ToolManager()
    spec = manager.register(plain)

    assert spec.display_action == "", "no display action was requested"
    assert spec.display_param == "", "no display parameter was requested"


def test_tool_decorator_returns_the_decorated_function_unchanged():
    """The decorator registers the function and hands the same object back."""

    def ping():
        return None

    manager = ToolManager()
    decorated = manager.tool(agent_doc="doc")(ping)

    assert decorated is ping, "registration must not wrap or replace the callable"


def test_tool_decorator_forwards_an_explicit_parameters_schema_to_register():
    """The decorator must pass ``parameters_schema`` through, not swallow it."""
    manager = ToolManager()
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}

    @manager.tool(name="get_weather", parameters_schema=schema)
    def get_weather(**kwargs):
        return kwargs

    assert (
        manager.get_spec("get_weather").parameters_schema == schema
    ), "the decorator forwards the schema to register()"


def test_get_spec_returns_none_for_a_name_that_was_never_registered():
    """An unknown name has no spec rather than raising."""
    manager = ToolManager()
    assert manager.get_spec("absent") is None, "a missing tool reports None"


def test_get_spec_returns_the_same_object_register_returned():
    """``get_spec`` exposes the registered object, not a copy of it."""

    def ping():
        return None

    manager = ToolManager()
    spec = manager.register(ping, name="ping")

    assert manager.get_spec("ping") is spec, "callers must see the stored spec"


def test_tool_schemas_defaults_to_the_agent_audience():
    """The default audience is the agent's."""
    manager = ToolManager()

    @manager.tool(human_doc="human only", agent_doc="agent only")
    def ping():
        return None

    assert manager.tool_schemas() == manager.tool_schemas(
        "agent"
    ), "the default audience is the agent's"
    assert (
        manager.tool_schemas()[0]["function"]["description"] == "agent only"
    ), "the agent's description is served by default"


def test_tool_schemas_serves_human_docs_for_the_human_audience():
    """``audience="human"`` switches the description to the human one."""
    manager = ToolManager()

    @manager.tool(human_doc="human only", agent_doc="agent only")
    def ping():
        return None

    assert manager.tool_schemas("human")[0]["function"]["description"] == (
        "human only"
    ), "the human audience gets the human description"


def test_tool_schemas_treats_an_unknown_audience_as_the_agent_one():
    """Only the literal ``"human"`` switches audience."""
    manager = ToolManager()

    @manager.tool(human_doc="human only", agent_doc="agent only")
    def ping():
        return None

    assert manager.tool_schemas("operator")[0]["function"]["description"] == (
        "agent only"
    ), "an unrecognised audience is served the agent description"


def test_register_mcp_client_registers_nothing_from_an_empty_catalogue():
    """A client that discovered no tools registers nothing and warns about nothing."""
    manager = ToolManager()
    client = _fake_mcp_client("files", [])

    with mock.patch("mitocode.core.tool.tool_manager.logger") as fake_logger:
        specs = manager.register_mcp_client(client)

    assert specs == [], "an empty catalogue yields no specs"
    assert manager.tools_map() == {}, "and leaves the registry untouched"
    assert not fake_logger.warning.called, "nothing collided, so nothing to warn about"


def test_register_mcp_client_returns_specs_in_catalogue_order():
    """The returned specs follow discovery order, not alphabetical order."""
    manager = ToolManager()
    client = _fake_mcp_client(
        "files",
        [
            _fake_tool("write", description="w"),
            _fake_tool("delete", description="d"),
            _fake_tool("read", description="r"),
        ],
    )

    specs = manager.register_mcp_client(client)

    assert [spec.name for spec in specs] == [
        "mcp__files__write",
        "mcp__files__delete",
        "mcp__files__read",
    ], "discovery order is the contract"
    assert list(manager.tools_map()) == [
        spec.name for spec in specs
    ], "the registry keeps the same order"


def test_register_mcp_client_namespaces_the_same_tool_name_per_server():
    """Two servers offering one tool name both register under their own namespace."""
    manager = ToolManager()
    alpha = _fake_mcp_client("alpha", [_fake_tool("search", description="from alpha")])
    beta = _fake_mcp_client("beta", [_fake_tool("search", description="from beta")])

    specs = manager.register_mcp_client(alpha) + manager.register_mcp_client(beta)

    assert [spec.name for spec in specs] == [
        "mcp__alpha__search",
        "mcp__beta__search",
    ], "the server name keeps two identical tool names apart"
    assert manager.agent_docs()["mcp__alpha__search"] == (
        "[alpha] from alpha"
    ), "each keeps its own description"


def test_register_mcp_client_warns_and_keeps_the_existing_tool_on_collision():
    """A skipped collision is logged, not silent, and the incumbent survives."""
    manager = ToolManager()

    @manager.tool(agent_doc="built-in")
    def mcp__files__read():
        return "builtin"

    client = _fake_mcp_client("files", [_fake_tool("read", description="remote")])
    with mock.patch("mitocode.core.tool.tool_manager.logger") as fake_logger:
        specs = manager.register_mcp_client(client)

    assert specs == [], "a colliding tool is not registered"
    assert manager.agent_docs()["mcp__files__read"] == "built-in", "the incumbent wins"
    fake_logger.warning.assert_called_once_with(
        "Skipping MCP tool 'mcp__files__read': name already registered"
    ), "the skip must be visible in the log"


def test_register_mcp_client_does_not_warn_when_every_name_is_free():
    """Only collisions warn; a clean registration is quiet."""
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="remote")])

    with mock.patch("mitocode.core.tool.tool_manager.logger") as fake_logger:
        specs = manager.register_mcp_client(client)

    assert len(specs) == 1, "the tool was registered"
    assert not fake_logger.warning.called, "a free name is not a warning"


def test_register_mcp_client_reaches_a_real_log_sink_when_it_skips_a_collision():
    """The skip is a log line an operator can find, not just a mock call.

    The ``logger.warning`` a skipped collision emits is the only trace that a
    server advertises a tool this harness could not register; asserting against
    a patched logger would pin the call site without proving anything reaches a
    sink.
    """
    manager = ToolManager()

    @manager.tool(agent_doc="built-in")
    def mcp__files__read():
        return "builtin"

    client = _fake_mcp_client("files", [_fake_tool("read", description="remote")])

    with _captured_logs() as lines:
        specs = manager.register_mcp_client(client)

    assert specs == [], "a colliding tool is not registered"
    warnings = [line for line in lines if "Skipping MCP tool" in line]
    assert len(warnings) == 1, f"expected one skipped collision, got {warnings}"
    assert "mcp__files__read" in warnings[0], "the skipped name is in the line"


def test_register_mcp_client_logs_nothing_when_no_name_collides():
    """A clean registration emits no log line at all."""
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="remote")])

    with _captured_logs() as lines:
        specs = manager.register_mcp_client(client)

    assert len(specs) == 1, "the tool was registered"
    assert lines == [], f"a registration with no collision is silent, got {lines}"


def test_register_mcp_client_publishes_nothing_when_the_catalogue_is_unreadable():
    """Registration is all-or-nothing, owned here rather than downstream.

    The catalogue's second tool carries a description that is not a string, so
    the read fails after the first tool would otherwise already be published.  A
    caller told only that "the registration failed" cannot name what to reclaim,
    so anything left behind would be a dispatchable tool nobody owns.  Driven
    here, in the registry's own suite: this property belongs to
    ``register_mcp_client`` and has to stay green even if the manager is never
    exercised against it.
    """
    manager = ToolManager()
    client = _fake_mcp_client(
        "weather",
        [
            _fake_tool("healthy", description="healthy on weather"),
            _malformed_tool("malformed"),
        ],
    )

    with pytest.raises(AttributeError):
        manager.register_mcp_client(client)

    assert manager.tools_map() == {}, "not even the healthy predecessor survives"
    assert manager.tool_schemas() == [], "so there is nothing to dispatch either"
    assert manager.get_spec("mcp__weather__healthy") is None


@pytest.mark.asyncio
async def test_register_mcp_client_keeps_the_first_of_two_clients_with_one_server_name():
    """A second client on the same server name never receives the dispatch."""
    first_calls = []
    second_calls = []
    first = _fake_mcp_client(
        "files", [_fake_tool("read", description="from first")], first_calls
    )
    second = _fake_mcp_client(
        "files", [_fake_tool("read", description="from second")], second_calls
    )
    manager = ToolManager()
    manager.register_mcp_client(first)
    manager.register_mcp_client(second)

    await manager.tools_map()["mcp__files__read"](path="a.txt")

    assert first_calls == [
        ("read", {"path": "a.txt"})
    ], "the first client keeps ownership of the composed name"
    assert second_calls == [], "the second client must never be dispatched to"


def test_register_mcp_client_synthesizes_an_exact_description_for_a_blank_description():
    """A whitespace-only description is replaced by the documented sentence."""

    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="  \n\t ")])

    manager.register_mcp_client(client)

    assert manager.human_docs()["mcp__files__read"] == (
        "[files] Tool 'read' provided by MCP server 'files'."
    ), "the synthesised description names the tool and the server"
    assert manager.agent_docs()["mcp__files__read"] == (
        "[files] Tool 'read' provided by MCP server 'files'."
    ), "both audiences get the same text"


def test_register_mcp_client_strips_surrounding_whitespace_from_a_description():
    """A real description keeps its content but loses its padding."""

    manager = ToolManager()
    client = _fake_mcp_client(
        "files", [_fake_tool("read", description="  Read a file.\n")]
    )

    manager.register_mcp_client(client)

    assert manager.agent_docs()["mcp__files__read"] == (
        "[files] Read a file."
    ), "the server's own description is used, trimmed"


def test_register_mcp_client_leaves_the_display_param_empty():
    """Only a display action is synthesised; there is no parameter to highlight."""

    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="d")])

    spec = manager.register_mcp_client(client)[0]

    assert (
        spec.display_param == ""
    ), "the module does not know which argument to highlight"


def test_register_mcp_client_preserves_a_complete_input_schema_verbatim():
    """A server that advertises a full schema gets it back unchanged."""
    manager = ToolManager()
    advertised = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    client = _fake_mcp_client("files", [_fake_tool("read", input_schema=advertised)])

    manager.register_mcp_client(client)

    schema = manager.tool_schemas("agent")[0]["function"]["parameters"]
    assert schema == advertised, "nothing is added to or removed from a complete schema"


def test_register_mcp_client_replaces_non_dict_properties_in_an_input_schema():
    """A ``properties`` value that is not a dict is replaced by an empty dict."""
    manager = ToolManager()
    client = _fake_mcp_client(
        "files",
        [_fake_tool("read", input_schema={"type": "object", "properties": "city"})],
    )

    manager.register_mcp_client(client)

    schema = manager.tool_schemas("agent")[0]["function"]["parameters"]
    assert schema["properties"] == {}, "the unusable value is dropped"
    assert schema["type"] == "object", "a usable type survives"


@pytest.mark.asyncio
async def test_registered_mcp_tool_returns_the_server_result_unchanged():
    """The generated callable hands the server's return value straight back."""

    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="r")])
    manager.register_mcp_client(client)

    result = await manager.tools_map()["mcp__files__read"](path="a.txt", limit=3)

    assert (
        result == "read got ['limit', 'path']"
    ), "the server's string is passed through"


def test_registered_mcp_tool_schema_hides_the_private_dispatch_bindings():
    """``_client``/``_tool_name`` exist on neither the callable nor the wire."""
    manager = ToolManager()
    client = _fake_mcp_client(
        "files",
        [
            _fake_tool(
                "read",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            )
        ],
    )

    spec = manager.register_mcp_client(client)[0]

    assert set(spec.parameters_schema["properties"]) == {
        "path"
    }, "the advertised arguments are exactly the server's"
    assert (
        "_client" not in spec.parameters_schema["properties"]
    ), "an internal binding must never be advertised to a model"
    assert (
        "_tool_name" not in spec.parameters_schema["properties"]
    ), "an internal binding must never be advertised to a model"


@pytest.mark.asyncio
async def test_registered_mcp_tool_exposes_no_dispatch_bindings_as_keywords():
    """The generated callable must accept nothing but the model's own arguments."""
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="r")])
    manager.register_mcp_client(client)

    parameters = inspect.signature(manager.tools_map()["mcp__files__read"]).parameters

    assert list(parameters) == [
        "kwargs"
    ], "the client and upstream tool name must be captured, never bound as keywords"


def test_register_mcp_client_warns_when_a_composed_name_is_too_long_for_a_model():
    """The composed name goes to the model verbatim, so its length is ours to warn about.

    The MCP specification imposes no name limit; the model API does, at 64
    characters.  A server name plus a tool name that together exceed it produce
    a tool that registers perfectly here and takes down the whole request at the
    provider.  The tool is still registered — truncating would rename it, and a
    renamed tool is a different tool — but the warning says why.
    """
    manager = ToolManager()
    long_server = "s" * 40
    long_tool = "t" * 40
    client = _fake_mcp_client(long_server, [_fake_tool(long_tool, description="d")])
    composed = ToolManager.build_mcp_tool_name(long_server, long_tool)

    with _captured_logs() as lines:
        specs = manager.register_mcp_client(client)

    assert len(composed) > MAX_TOOL_NAME_LENGTH, "the case is the one being tested"
    assert [spec.name for spec in specs] == [composed], "and the tool still registers"
    warnings = [line for line in lines if "characters, past the 64" in line]
    assert len(warnings) == 1, f"expected one length warning, got {warnings}"
    assert composed in warnings[0], "naming the composed name that is too long"


def test_register_mcp_client_does_not_warn_about_the_length_of_a_usable_name():
    """An ordinary namespaced name is well inside the limit and stays quiet."""
    manager = ToolManager()
    client = _fake_mcp_client("weather", [_fake_tool("get_forecast", description="d")])

    with _captured_logs() as lines:
        specs = manager.register_mcp_client(client)

    assert [spec.name for spec in specs] == ["mcp__weather__get_forecast"]
    assert lines == [], f"a 26-character name is not a warning, got {lines}"


@pytest.mark.asyncio
async def test_registered_mcp_tool_cannot_redirect_the_dispatch_with_a_tool_name_argument():
    """A ``_tool_name`` argument must be forwarded, never bind over the captured name.

    The defect this pins was ``async def _invoke(_client=client,
    _tool_name=tool.name, **kwargs)`` — the captured values sat in the same
    namespace as the model's keywords, so this argument silently redirected the
    call to a *different* tool on that server, bypassing the name-collision
    guard entirely.  ``_make_invoke`` closed it; the callable's signature is
    pinned separately by
    ``test_registered_mcp_tool_exposes_no_dispatch_bindings_as_keywords``.
    """
    calls = []
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="r")], calls)
    manager.register_mcp_client(client)

    await manager.tools_map()["mcp__files__read"](_tool_name="write", path="a.txt")

    assert calls == [
        ("read", {"_tool_name": "write", "path": "a.txt"})
    ], "the dispatch must stay on the registered tool, with _tool_name as a plain argument"


@pytest.mark.asyncio
async def test_registered_mcp_tool_cannot_redirect_the_dispatch_with_a_client_argument():
    """A ``_client`` argument must be forwarded, never replace the captured client.

    Same namespace collision as ``_tool_name``.  The defect raised
    ``AttributeError`` from the model's own argument before the server was ever
    reached; this asserts the dispatch contract instead of that incidental
    error message, so it also accepts a clearer error from a future regression.
    """
    calls = []
    manager = ToolManager()
    client = _fake_mcp_client("files", [_fake_tool("read", description="r")], calls)
    manager.register_mcp_client(client)

    await manager.tools_map()["mcp__files__read"](_client="not-a-client")

    assert calls == [
        ("read", {"_client": "not-a-client"})
    ], "the real client must still receive the call, with _client as a plain argument"
