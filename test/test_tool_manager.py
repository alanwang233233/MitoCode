import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.agent.agent_session import AgentSession
from mitocode.backend.agent.session_factory import SessionFactory
from mitocode.backend.tool.tools import ToolManager, declare_tool, _build_parameters_schema


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

    @declare_tool(name="echo_text", human_doc="给人看：回显文本", agent_doc="Echo user text input.")
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

    session = AgentSession(provider=mock.MagicMock(), tool_manager=manager)
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

    factory = SessionFactory(provider=mock.MagicMock(), tool_manager=manager)
    session = factory.create_session()
    assert "noop" in session.tools_map
    assert session.tool_schemas[0]["function"]["name"] == "noop"


def test_dict_reference_pollution():
    def test_tool(city: str = "Beijing", country: str = "China"):
        pass

    schema = _build_parameters_schema(test_tool)
    props = schema["properties"]
    print("\n--- 生成的 Schema 结果 ---")
    print(f"city 的 schema:   {props['city']}")
    print(f"country 的 schema: {props['country']}")
    print(f"两者是否是同一个对象？ {props['city'] is props['country']}")

    # 4. 断言验证（在原始 Bug 代码下，以下断言会失败）

    # 断言 A：city 的默认值应该是 "Beijing"
    # 【Bug 现象】：实际上这里会变成 "China"，因为被 country 覆盖了！
    assert props["city"]["default"] == "Beijing", \
        f"city 的默认值被污染了！实际是: {props['city'].get('default')}"

    # 断言 B：country 的默认值应该是 "China"
    assert props["country"]["default"] == "China"

    # 断言 C：这两个 schema 字典在内存中不能是同一个对象
    # 【Bug 现象】：实际上它们是同一个对象 (is 返回 True)
    assert props["city"] is not props["country"], \
        "字典引用污染！city 和 country 的 schema 指向了同一个内存地址。"
