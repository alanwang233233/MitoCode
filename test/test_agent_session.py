"""AgentSession 模块单元测试套件。

覆盖 src/mitocode/backend/agent/agent_session.py 的全部行为分支：
消息块三兄弟（文本/工具调用/工具结果）的构造与转换、AgentSession 构造校验、
托管/用户工具合并规则、文档查询委托、协作式取消、流式增量聚合、
标准上下文编译以及 TUI 节点快照映射。
"""

import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.agent.agent_session import (
    AgentSession,
    _MessageBlock,
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from mitocode.backend.agent.agent_state import AgentState
from mitocode.backend.tool.tool_manager import ToolManager

TOOL_NAME_WEATHER = "get_weather"
CALL_ID_PRIMARY = "call_001"
CALL_ID_SECONDARY = "call_002"
VALID_JSON_ARGS = '{"city": "Beijing"}'
INVALID_JSON_ARGS = "{这不是JSON"
LIST_JSON_ARGS = "[1, 2]"
LOGGER_PATCH_TARGET = "mitocode.backend.agent.agent_session.logger"


# ──────────────────────────────────────────────
# 辅助工厂函数
# ──────────────────────────────────────────────


def _noop_tool() -> str:
    """合并测试中复用的最小用户工具实现."""
    return "ok"


def _make_function_schema(name: str, description: str = "测试工具") -> dict:
    """构造一条 OpenAI function-calling 格式的工具 schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_manager(
    tools_map=None, schemas=None, human=None, agent=None
) -> mock.MagicMock:
    """构造已配置返回值的 MagicMock(spec=ToolManager)，隔离真实注册中心."""
    manager = mock.MagicMock(spec=ToolManager)
    manager.tools_map.return_value = dict(tools_map or {})
    manager.tool_schemas.return_value = list(schemas or [])
    manager.human_docs.return_value = dict(human or {})
    manager.agent_docs.return_value = dict(agent or {})
    return manager


def _finished_tool_call(name: str, call_id: str, arguments: str):
    """构造一个已完成 finish() 的工具调用消息块."""
    block = _ToolCallMessageBlock(name, call_id)
    block.append_argument_chunk(arguments)
    block.finish()
    return block


def _expected_tool_call_entry(name: str, call_id: str, arguments: str) -> dict:
    """构造 build_standard_context 中工具调用的期望标准格式."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _joined_logger_message(call_args) -> str:
    """把 mock 日志调用的全部实参拼接成单个字符串便于包含性断言."""
    parts = [str(arg) for arg in call_args.args]
    parts += [f"{key}={value}" for key, value in call_args.kwargs.items()]
    return " ".join(parts)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def mock_provider() -> mock.MagicMock:
    """提供确定性 Mock LLM provider，隔离真实聊天后端."""
    return mock.MagicMock(name="mock_provider")


@pytest.fixture
def make_session(mock_provider):
    """会话工厂夹具：默认注入 Mock provider，kwargs 可覆盖全部设置."""

    def _make(**settings) -> AgentSession:
        settings.setdefault("provider", mock_provider)
        return AgentSession(**settings)

    return _make


@pytest.fixture
def session(make_session) -> AgentSession:
    """默认配置（仅 Mock provider）的 AgentSession 实例."""
    return make_session()


# ──────────────────────────────────────────────
# _MessageBlock 基类
# ──────────────────────────────────────────────


class TestMessageBlockBase:
    """_MessageBlock 基类初始状态测试."""

    def test_init_finished_defaults_false(self):
        """构造基类消息块 → finished 初始为 False."""
        # Act
        block = _MessageBlock()
        # Assert
        assert block.finished is False, "基类消息块初始 finished 应为 False"


# ──────────────────────────────────────────────
# _PlainTextMessageBlock
# ──────────────────────────────────────────────


class TestPlainTextMessageBlock:
    """用户/助手文本消息块的构造与流式追加行为测试."""

    @pytest.mark.parametrize(
        "role, message, expected_message",
        [
            ("user", "你好", "你好"),
            ("assistant", "", ""),
        ],
        ids=["带初始消息", "缺省消息为空串"],
    )
    def test_init_stores_role_and_message_and_finishes_immediately(
        self, role, message, expected_message
    ):
        """构造文本块 → 立即保存角色与消息且 finished 即刻为 True."""
        # Act
        block = _PlainTextMessageBlock(role, message)
        # Assert
        assert block.role == role, "应保存构造传入的角色"
        assert block.message == expected_message, "应保存构造传入的消息内容"
        assert block.finished is True, "文本块构造完成后应立即视为已完成"

    def test_append_accumulates_delta_in_order(self):
        """连续多次 append → 增量按顺序累积到 message 末尾."""
        # Arrange
        block = _PlainTextMessageBlock("assistant", "你")
        # Act
        block.append("好")
        block.append("，世界")
        # Assert
        assert block.message == "你好，世界", "append 应把增量顺序拼接到消息末尾"


# ──────────────────────────────────────────────
# _ToolCallMessageBlock
# ──────────────────────────────────────────────


class TestToolCallMessageBlock:
    """助手工具调用块的参数累积、finish 三路与标准格式转换测试."""

    def test_init_defaults_finished_false_and_empty_fields(self):
        """构造工具调用块 → finished 为 False 且各字段保持初始值."""
        # Act
        block = _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY)
        # Assert
        assert block.name == TOOL_NAME_WEATHER, "应保存工具名"
        assert block.id == CALL_ID_PRIMARY, "应保存调用 id"
        assert block.finished is False, "初始不应处于完成状态"
        assert block.arguments_string == "", "初始参数原始串应为空"
        assert block.parameter == {}, "初始 parameter 应为空字典"
        assert block.parse_error is None, "初始不应有解析错误"

    def test_append_argument_chunk_accumulates_raw_fragments(self):
        """分片追加参数片段 → 原始 JSON 片段按序累积且不触发解析."""
        # Arrange
        block = _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY)
        # Act
        block.append_argument_chunk('{"city": ')
        block.append_argument_chunk('"Beijing"}')
        # Assert
        assert (
            block.arguments_string == VALID_JSON_ARGS
        ), "append_argument_chunk 应原样累积 JSON 片段"
        assert block.finished is False, "仅追加片段不应改变完成状态"

    def test_finish_valid_json_parses_parameter_without_error(self):
        """finish 收到合法 JSON 对象 → 解析进 parameter 且无 parse_error 与日志."""
        # Arrange
        block = _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY)
        block.append_argument_chunk(VALID_JSON_ARGS)
        # Act
        with mock.patch(LOGGER_PATCH_TARGET) as fake_logger:
            block.finish()
            # Assert
            assert block.finished is True, "finish 后必须置位 finished"
            assert block.parameter == {
                "city": "Beijing"
            }, "合法 JSON 应被解析为字典存入 parameter"
            assert block.parse_error is None, "成功路径不应产生 parse_error"
            fake_logger.error.assert_not_called()

    def test_finish_empty_arguments_sets_empty_parameter(self):
        """finish 收到空参数串 → parameter 回落为空字典且不报错."""
        # Arrange
        block = _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY)
        # Act
        with mock.patch(LOGGER_PATCH_TARGET) as fake_logger:
            block.finish()
            # Assert
            assert block.finished is True, "finish 后必须置位 finished"
            assert block.parameter == {}, "空参数串时 parameter 应为空字典"
            assert block.parse_error is None, "空参数串不算解析失败"
            fake_logger.error.assert_not_called()

    def test_finish_invalid_json_sets_parse_error_and_logs_tool_name(self):
        """finish 收到非法 JSON → 参数回落空字典并记录含工具名的错误日志."""
        # Arrange
        tool_name = "broken_tool"
        block = _ToolCallMessageBlock(tool_name, CALL_ID_PRIMARY)
        block.append_argument_chunk(INVALID_JSON_ARGS)
        # Act
        with mock.patch(LOGGER_PATCH_TARGET) as fake_logger:
            block.finish()
            # Assert
            assert block.finished is True, "即使解析失败也必须置位 finished"
            assert block.parameter == {}, "解析失败时 parameter 应回落为空字典"
            assert block.parse_error is not None, "解析失败时应记录非空的错误信息"
            assert (
                "JSON parsing failed" in block.parse_error
            ), "parse_error 应包含固定前缀 JSON parsing failed"
            fake_logger.error.assert_called_once()
            logged = _joined_logger_message(fake_logger.error.call_args)
            assert tool_name in logged, "错误日志信息中应包含工具名"

    def test_finish_non_dict_json_keeps_parsed_value_without_error(self):
        """finish 收到合法但非对象的 JSON（如数组）→ 按现行为原样保留不报错."""
        # Arrange
        block = _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY)
        block.append_argument_chunk(LIST_JSON_ARGS)
        # Act
        with mock.patch(LOGGER_PATCH_TARGET) as fake_logger:
            block.finish()
            # Assert
            assert block.parameter == [1, 2], "非对象 JSON 应按现行为原样存入"
            assert block.parse_error is None, "可解析的 JSON 不应视为失败"
            fake_logger.error.assert_not_called()

    def test_to_standard_unfinished_returns_empty_dict(self):
        """未完成的工具调用块转标准格式 → 返回空字典."""
        # Arrange
        block = _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY)
        block.append_argument_chunk(VALID_JSON_ARGS)
        # Act
        result = block.to_standard()
        # Assert
        assert result == {}, "未完成时 to_standard 应返回空字典"

    def test_to_standard_finished_returns_openai_format_with_raw_arguments(self):
        """完成的工具调用块转标准格式 → 返回保留原始参数串的标准结构."""
        # Arrange
        block = _finished_tool_call(TOOL_NAME_WEATHER, CALL_ID_PRIMARY, VALID_JSON_ARGS)
        expected = _expected_tool_call_entry(
            TOOL_NAME_WEATHER, CALL_ID_PRIMARY, VALID_JSON_ARGS
        )
        # Act
        result = block.to_standard()
        # Assert
        assert result == expected, "完成后应返回 id/type/function 三键标准格式"
        assert (
            result["function"]["arguments"] == VALID_JSON_ARGS
        ), "arguments 应保留原始字符串而非解析后的对象"


# ──────────────────────────────────────────────
# _ToolResultMessageBlock
# ──────────────────────────────────────────────


class TestToolResultMessageBlock:
    """本地工具执行结果块的初始化行为测试."""

    def test_init_stores_result_fields_and_finishes_immediately(self):
        """构造结果块 → 保存 id 与内容、角色为 tool 且立即完成."""
        # Act
        block = _ToolResultMessageBlock(CALL_ID_PRIMARY, "28℃ 晴")
        # Assert
        assert block.role == "tool", "结果块角色应为 tool"
        assert block.tool_call_id == CALL_ID_PRIMARY, "应保存工具调用 id"
        assert block.content == "28℃ 晴", "应保存执行结果内容"
        assert block.finished is True, "结果块构造后应立即视为已完成"


# ──────────────────────────────────────────────
# AgentSession.factory
# ──────────────────────────────────────────────


class TestAgentSessionFactory:
    """staticmethod factory 的实例创建与设置转发测试."""

    def test_factory_returns_new_session_instance(self, mock_provider):
        """factory 仅传 provider → 返回全新的 AgentSession 实例."""
        # Act
        made = AgentSession.factory(provider=mock_provider)
        # Assert
        assert isinstance(made, AgentSession), "factory 应返回 AgentSession 实例"
        assert made.provider is mock_provider, "provider 应被原样保存"

    @pytest.mark.parametrize(
        "settings, attr, expected",
        [
            ({"agent_id": 42}, "agent_id", 42),
            ({"name": "FactoryBot"}, "name", "FactoryBot"),
            ({"description": "由工厂创建"}, "description", "由工厂创建"),
            ({"parent_id": 7}, "parent_id", 7),
        ],
        ids=["转发agent_id", "转发name", "转发description", "转发parent_id"],
    )
    def test_factory_forwards_settings_to_created_session(
        self, mock_provider, settings, attr, expected
    ):
        """factory 透传各设置键 → 对应属性逐一按入参保存."""
        # Act
        made = AgentSession.factory(provider=mock_provider, **settings)
        # Assert
        assert getattr(made, attr) == expected, f"设置 {attr} 应被 factory 转发"


# ──────────────────────────────────────────────
# AgentSession.__init__ 校验
# ──────────────────────────────────────────────


class TestAgentSessionInitValidation:
    """AgentSession 构造参数校验的异常路径测试."""

    @pytest.mark.parametrize(
        "settings",
        [{}, {"provider": None}, {"provider": ""}],
        ids=["完全未提供provider", "provider为None", "provider为空串"],
    )
    def test_init_missing_provider_raises_value_error(self, settings):
        """provider 缺失或 falsy → 抛出带固定文案的 ValueError."""
        # Act / Assert
        with pytest.raises(ValueError, match=r"Provider is required\."):
            AgentSession(**settings)

    @pytest.mark.parametrize(
        "bad_manager",
        [object(), "not_a_manager"],
        ids=["普通object实例", "字符串"],
    )
    def test_init_invalid_tool_manager_type_raises_type_error(
        self, mock_provider, bad_manager
    ):
        """tool_manager 非 None 但类型不符 → 抛出 TypeError."""
        # Act / Assert
        with pytest.raises(TypeError, match=r"(?i)tool[_ ]?manager"):
            AgentSession(provider=mock_provider, tool_manager=bad_manager)


# ──────────────────────────────────────────────
# AgentSession.__init__ 默认值与自定义设置
# ──────────────────────────────────────────────


class TestAgentSessionInitDefaults:
    """AgentSession 构造后的默认属性矩阵与自定义设置存储测试."""

    def test_init_default_attribute_values(self, session, mock_provider):
        """仅传 provider 构造 → 全部固定属性保持契约默认值."""
        # Assert
        assert session.provider is mock_provider, "应原样保存传入的 provider"
        assert session.tools_map == {}, "默认 tools_map 应为空字典"
        assert session.tool_schemas == [], "默认 tool_schemas 应为空列表"
        assert session.tool_manager is None, "默认无工具管理器"
        assert session.on_end is None, "默认 on_end 回调应为 None"
        assert session.agent_id == -1, "默认 agent_id 应为 -1"
        assert session.name == "", "默认 name 应为空串"
        assert session.description == "", "默认 description 应为空串"
        assert session.parent_id is None, "默认 parent_id 应为 None"
        assert session.sub_agent_ids == [], "默认 sub_agent_ids 应为空列表"
        assert session.state == AgentState.DEFAULT, "默认状态应为 DEFAULT(idle)"
        assert session.messages == [], "默认消息历史应为空列表"

    def test_init_custom_settings_stored_on_attributes(self, make_session):
        """传入完整自定义设置 → 各属性逐一按入参保存."""
        # Arrange
        custom_on_end = lambda: None  # noqa: E731
        # Act
        made = make_session(
            agent_id=5,
            name="Worker",
            description="干活的代理",
            parent_id=3,
            on_end=custom_on_end,
        )
        # Assert
        assert made.agent_id == 5, "agent_id 应按入参保存"
        assert made.name == "Worker", "name 应按入参保存"
        assert made.description == "干活的代理", "description 应按入参保存"
        assert made.parent_id == 3, "parent_id 应按入参保存"
        assert made.on_end is custom_on_end, "on_end 应按入参原样保存"


# ──────────────────────────────────────────────
# 托管 / 用户 工具合并
# ──────────────────────────────────────────────


class TestAgentSessionToolMerge:
    """tool_manager 提供时的 tools_map 与 tool_schemas 合并规则测试."""

    def test_init_merges_tools_user_overrides_managed_same_name(self, make_session):
        """用户与托管存在同名工具 → 用户函数覆盖托管函数且其余托管键保留."""
        # Arrange
        managed_fn = mock.MagicMock(name="managed_run")
        user_fn = mock.MagicMock(name="user_run")
        manager = _make_manager(
            tools_map={"run": managed_fn, "managed_only": _noop_tool}
        )
        # Act
        made = make_session(tool_manager=manager, tools_map={"run": user_fn})
        # Assert
        assert made.tools_map["run"] is user_fn, "同名工具应以用户提供的函数为准"
        assert made.tools_map["managed_only"] is _noop_tool, "仅托管的工具应保留"

    def test_init_merges_schemas_user_named_overrides_managed(self, make_session):
        """用户与托管存在同名 schema → 用户条目覆盖且总数去重."""
        # Arrange
        managed_shared = _make_function_schema("shared", description="托管版")
        managed_keep = _make_function_schema("keep", description="仅托管")
        user_shared = _make_function_schema("shared", description="用户版")
        user_extra = _make_function_schema("extra", description="仅用户")
        manager = _make_manager(schemas=[managed_shared, managed_keep])
        # Act
        made = make_session(
            tool_manager=manager, tool_schemas=[user_shared, user_extra]
        )
        # Assert
        assert len(made.tool_schemas) == 3, "同名覆盖后总数应为去重后的 3 条"
        by_name = {entry["function"]["name"]: entry for entry in made.tool_schemas}
        assert by_name["shared"] is user_shared, "同名 schema 应以用户条目为准"
        assert by_name["keep"] is managed_keep, "仅托管的条目应保留"
        assert by_name["extra"] is user_extra, "仅用户的条目应并入"

    @pytest.mark.parametrize(
        "bad_schema",
        [
            {"type": "function"},
            {"type": "function", "function": {"description": "缺少名称"}},
        ],
        ids=["缺function键", "缺function-name键"],
    )
    def test_init_skips_managed_schemas_missing_name(self, make_session, bad_schema):
        """托管 schema 缺 function 键或缺名称 → 该坏条目被跳过."""
        # Arrange
        good_schema = _make_function_schema("good_tool")
        manager = _make_manager(schemas=[bad_schema, good_schema])
        # Act
        made = make_session(tool_manager=manager)
        # Assert
        assert len(made.tool_schemas) == 1, "坏条目应被剔除只留好条目"
        assert made.tool_schemas[0] is good_schema, "留下的应是合法托管条目"

    def test_init_keeps_unnamed_user_schema_by_identity(self, make_session):
        """用户 schema 无 function.name → 以对象身份保留在最终列表中."""
        # Arrange
        anonymous = {"type": "function", "function": {"description": "无名条目"}}
        named = _make_function_schema("named_user_tool")
        manager = _make_manager(schemas=[_make_function_schema("managed_a")])
        # Act
        made = make_session(tool_manager=manager, tool_schemas=[named, anonymous])
        # Assert
        assert len(made.tool_schemas) == 3, "无名条目也应计入最终列表"
        assert any(
            entry is anonymous for entry in made.tool_schemas
        ), "无名用户 schema 应以原对象身份出现在结果中"

    def test_init_without_manager_uses_user_collections_directly(self, make_session):
        """未提供 manager → 直接采用用户提供的 tools_map 与 tool_schemas."""
        # Arrange
        user_tools = {"local": _noop_tool}
        user_schemas = [_make_function_schema("local_tool")]
        # Act
        made = make_session(tools_map=user_tools, tool_schemas=user_schemas)
        # Assert
        assert made.tools_map == {"local": _noop_tool}, "应直接使用用户工具表"
        assert made.tool_schemas == user_schemas, "应直接使用用户 schema 列表"

    def test_init_with_real_tool_manager_registers_tools(self, make_session):
        """注入真实 ToolManager 注册的工具 → 会话合并出可调用工具与对应 schema."""
        # Arrange
        manager = ToolManager()

        @manager.tool(human_doc="给人看的文档", agent_doc="Agent doc")
        def ping():
            return "pong"

        # Act
        made = make_session(tool_manager=manager)
        # Assert
        assert "ping" in made.tools_map, "托管工具应进入会话工具表"
        assert made.tools_map["ping"]() == "pong", "托管工具应可正常调用"
        assert (
            made.tool_schemas[0]["function"]["name"] == "ping"
        ), "托管 schema 应进入会话 schema 列表"


# ──────────────────────────────────────────────
# 文档查询委托
# ──────────────────────────────────────────────


class TestAgentSessionDocs:
    """人类/代理文档查询的委托与降级行为测试."""

    def test_get_tool_docs_for_human_without_manager_returns_empty_dict(self, session):
        """无 manager 时查询人类文档 → 返回空字典."""
        # Act
        docs = session.get_tool_docs_for_human()
        # Assert
        assert docs == {}, "无 manager 时人类文档应为空字典"

    def test_get_tool_docs_for_human_delegates_to_manager(self, make_session):
        """有 manager 时查询人类文档 → 委托 manager.human_docs() 恰好一次."""
        # Arrange
        manager = _make_manager(human={"search": "给人看的搜索说明"})
        made = make_session(tool_manager=manager)
        # Act
        docs = made.get_tool_docs_for_human()
        # Assert
        assert docs == {"search": "给人看的搜索说明"}, "应返回 manager 的人类文档"
        manager.human_docs.assert_called_once()

    def test_get_tool_docs_for_agent_without_manager_returns_empty_dict(self, session):
        """无 manager 时查询代理文档 → 返回空字典."""
        # Act
        docs = session.get_tool_docs_for_agent()
        # Assert
        assert docs == {}, "无 manager 时代理文档应为空字典"

    def test_get_tool_docs_for_agent_delegates_to_manager(self, make_session):
        """有 manager 时查询代理文档 → 委托 manager.agent_docs() 恰好一次."""
        # Arrange
        manager = _make_manager(agent={"search": "Use to search."})
        made = make_session(tool_manager=manager)
        # Act
        docs = made.get_tool_docs_for_agent()
        # Assert
        assert docs == {"search": "Use to search."}, "应返回 manager 的代理文档"
        manager.agent_docs.assert_called_once()


# ──────────────────────────────────────────────
# 协作式取消
# ──────────────────────────────────────────────


class TestAgentSessionCancellation:
    """协作式取消标记的状态转换测试."""

    def test_is_cancelled_initially_false_and_cancel_marks_true(self, session):
        """初始未取消 → cancel 后变为已取消且重复取消保持 True."""
        # Assert
        assert session.is_cancelled() is False, "初始不应处于取消状态"
        # Act
        session.cancel()
        # Assert
        assert session.is_cancelled() is True, "取消后应处于取消状态"
        # Act（幂等性）
        session.cancel()
        # Assert
        assert session.is_cancelled() is True, "重复取消应保持取消状态"

    def test_cancel_sets_internal_event(self, session):
        """cancel → 内部 asyncio.Event 被置位（供运行时等待）."""
        # Act
        session.cancel()
        # Assert
        assert session._cancel_event.is_set(), "内部取消事件应被置位"

    def test_append_user_prompt_clears_cancellation_flag(self, session):
        """取消后再发送用户输入 → 取消标记被清除."""
        # Arrange
        session.cancel()
        assert session.is_cancelled() is True, "前置：应先处于取消状态"
        # Act
        session.append_user_prompt("重新开始")
        # Assert
        assert session.is_cancelled() is False, "新用户输入应重置取消标记"

    def test_append_user_prompt_appends_finished_user_block(self, session):
        """发送用户输入 → 追加 role=user 且立即完成的文本块."""
        # Act
        session.append_user_prompt("第一句")
        session.append_user_prompt("第二句")
        # Assert
        assert len(session.messages) == 2, "两条输入应产生两个消息块"
        assert all(
            isinstance(block, _PlainTextMessageBlock) for block in session.messages
        ), "用户输入应对应文本消息块"
        assert all(
            block.role == "user" and block.finished for block in session.messages
        ), "用户文本块角色应为 user 且立即完成"


# ──────────────────────────────────────────────
# append_assistant_delta
# ──────────────────────────────────────────────


class TestAgentSessionAssistantDelta:
    """助手流式增量的新建分支与累积分支测试."""

    def test_append_assistant_delta_empty_history_creates_assistant_block(
        self, session
    ):
        """历史为空时收到增量 → 新建含该增量的 assistant 文本块."""
        # Act
        session.append_assistant_delta("Hello")
        # Assert
        assert len(session.messages) == 1, "空历史应新建一个块"
        assert isinstance(
            session.messages[0], _PlainTextMessageBlock
        ), "新建块应为文本块"
        assert session.messages[0].message == "Hello", "增量内容应写入新块"

    @pytest.mark.parametrize(
        "tail_factory",
        [
            lambda: _ToolResultMessageBlock(CALL_ID_PRIMARY, "执行结果"),
            lambda: _ToolCallMessageBlock(TOOL_NAME_WEATHER, CALL_ID_PRIMARY),
        ],
        ids=["尾块为工具结果块", "尾块为工具调用块"],
    )
    def test_append_assistant_delta_after_non_text_block_creates_new_block(
        self, session, tail_factory
    ):
        """最后一块不是文本块时收到增量 → 新建独立 assistant 文本块."""
        # Arrange
        session.messages.append(tail_factory())
        blocks_before = len(session.messages)
        # Act
        session.append_assistant_delta("已处理完毕")
        # Assert
        assert len(session.messages) == blocks_before + 1, "非文本块之后应新建文本块"
        assert isinstance(
            session.messages[-1], _PlainTextMessageBlock
        ), "新建块应为 assistant 文本块"
        assert session.messages[-1].role == "assistant", "新块角色应为 assistant"

    def test_append_assistant_delta_after_user_block_creates_new_block(self, session):
        """最后一块是 user 文本块时收到增量 → 角色不同应新建 assistant 块."""
        # Arrange
        session.append_user_prompt("问题")
        blocks_before = len(session.messages)
        # Act
        session.append_assistant_delta("回答")
        # Assert
        assert len(session.messages) == blocks_before + 1, "user 块之后应新建块"
        assert session.build_standard_context() == [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ], "上下文应保持 user 与 assistant 两条独立消息"

    def test_append_assistant_delta_consecutive_deltas_accumulate_one_block(
        self, session
    ):
        """连续 assistant 增量 → 不新建块而是累积进同一块."""
        # Act
        session.append_assistant_delta("你")
        session.append_assistant_delta("好")
        # Assert
        assert len(session.messages) == 1, "连续增量应复用同一块"
        assert session.messages[0].message == "你好", "增量应按序拼接"


# ──────────────────────────────────────────────
# build_standard_context
# ──────────────────────────────────────────────


class TestAgentSessionStandardContext:
    """标准上下文构建：跳过、映射与 tool_calls 聚合冲刷测试."""

    def test_build_standard_context_empty_messages_returns_empty_list(self, session):
        """历史为空 → 返回空列表."""
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == [], "空历史的上下文应为空列表"

    @pytest.mark.parametrize(
        "unfinished_factory",
        [
            lambda: _ToolCallMessageBlock("pending_tool", CALL_ID_PRIMARY),
            lambda: _make_unfinished_plain_text(),
        ],
        ids=["未完成工具调用块被跳过", "未完成文本块被跳过"],
    )
    def test_build_standard_context_skips_unfinished_blocks(
        self, session, unfinished_factory
    ):
        """历史中存在未完成块 → 未完成块不计入上下文."""
        # Arrange
        session.messages.append(unfinished_factory())
        session.messages.append(_PlainTextMessageBlock("user", "可见消息"))
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == [
            {"role": "user", "content": "可见消息"}
        ], "上下文只应包含已完成的消息块"

    def test_build_standard_context_converts_plain_text_blocks_in_order(self, session):
        """纯文本历史 → 按顺序映射为 role/content 字典."""
        # Arrange
        session.messages.append(_PlainTextMessageBlock("user", "在吗"))
        session.messages.append(_PlainTextMessageBlock("assistant", "在的"))
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == [
            {"role": "user", "content": "在吗"},
            {"role": "assistant", "content": "在的"},
        ], "文本块应按原顺序映射为标准消息"

    def test_build_standard_context_flushes_pending_tool_calls_before_text(
        self, session
    ):
        """待定工具调用后遇到文本块 → 先冲刷为 assistant/tool_calls 再放文本."""
        # Arrange
        session.messages.append(_PlainTextMessageBlock("user", "北京天气如何"))
        session.messages.append(
            _finished_tool_call(TOOL_NAME_WEATHER, CALL_ID_PRIMARY, VALID_JSON_ARGS)
        )
        session.messages.append(_PlainTextMessageBlock("assistant", "今天晴"))
        expected = [
            {"role": "user", "content": "北京天气如何"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _expected_tool_call_entry(
                        TOOL_NAME_WEATHER, CALL_ID_PRIMARY, VALID_JSON_ARGS
                    )
                ],
            },
            {"role": "assistant", "content": "今天晴"},
        ]
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == expected, "文本块前应先冲刷待定的工具调用"

    def test_build_standard_context_flushes_pending_tool_calls_before_result(
        self, session
    ):
        """待定工具调用后遇到结果块 → 先冲刷再放工具结果."""
        # Arrange
        session.messages.append(
            _finished_tool_call("query_temp", CALL_ID_PRIMARY, VALID_JSON_ARGS)
        )
        session.messages.append(_ToolResultMessageBlock(CALL_ID_PRIMARY, "28℃"))
        expected = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _expected_tool_call_entry(
                        "query_temp", CALL_ID_PRIMARY, VALID_JSON_ARGS
                    )
                ],
            },
            {"role": "tool", "tool_call_id": CALL_ID_PRIMARY, "content": "28℃"},
        ]
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == expected, "结果块前应先冲刷待定的工具调用"

    def test_build_standard_context_aggregates_multiple_tool_calls_in_one_flush(
        self, session
    ):
        """连续多个待定工具调用 → 聚合进同一条 assistant 消息并保持顺序."""
        # Arrange
        session.messages.append(_finished_tool_call("tool_a", "call_a", '{"x": 1}'))
        session.messages.append(_finished_tool_call("tool_b", "call_b", '{"y": 2}'))
        session.messages.append(_ToolResultMessageBlock("call_a", "A 完成"))
        expected = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _expected_tool_call_entry("tool_a", "call_a", '{"x": 1}'),
                    _expected_tool_call_entry("tool_b", "call_b", '{"y": 2}'),
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "A 完成"},
        ]
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == expected, "多个工具调用应聚合为单次冲刷且顺序一致"

    def test_build_standard_context_flushes_trailing_tool_calls_at_end(self, session):
        """结尾残留待定工具调用 → 最后统一冲刷为 assistant/tool_calls."""
        # Arrange
        session.messages.append(
            _finished_tool_call("final_tool", CALL_ID_PRIMARY, '{"k": "v"}')
        )
        expected = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _expected_tool_call_entry(
                        "final_tool", CALL_ID_PRIMARY, '{"k": "v"}'
                    )
                ],
            }
        ]
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == expected, "结尾残留的工具调用应在最后冲刷输出"

    def test_build_standard_context_tool_result_without_pending_calls_appends_directly(
        self, session
    ):
        """无前置待定工具调用的结果块 → 直接追加不触发空冲刷."""
        # Arrange
        session.messages.append(_ToolResultMessageBlock(CALL_ID_PRIMARY, "直接结果"))
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == [
            {"role": "tool", "tool_call_id": CALL_ID_PRIMARY, "content": "直接结果"}
        ], "无待定调用时结果块应直接映射为 tool 消息"

    def test_build_standard_context_ignores_finished_unknown_block_types(self, session):
        """已完成但类型未知（非三种标准块）的块 → 被静默忽略不计入上下文."""
        # Arrange

        class _UnknownBlock(_MessageBlock):
            """测试专用的未知消息块子类."""

            def __init__(self):
                super().__init__()
                self.finished = True

        session.messages.append(_UnknownBlock())
        session.messages.append(_PlainTextMessageBlock("user", "可见消息"))
        # Act
        context = session.build_standard_context()
        # Assert
        assert context == [
            {"role": "user", "content": "可见消息"}
        ], "未知类型的已完成块不应出现在上下文中"


# ──────────────────────────────────────────────
# to_agent_node 快照
# ──────────────────────────────────────────────


class TestAgentSessionNodeSnapshot:
    """to_agent_node 快照字段映射与防御性拷贝测试."""

    def test_to_agent_node_maps_session_fields_to_node(self, make_session):
        """自定义设置下生成快照 → 各字段一一映射且 state 取枚举值字符串."""
        # Arrange
        made = make_session(agent_id=7, name="Scout", description="侦察兵", parent_id=2)
        made.sub_agent_ids.extend([10, 11])
        made.state = AgentState.RUNNING
        # Act
        node = made.to_agent_node()
        # Assert
        assert node.id == 7, "快照 id 应来自 agent_id"
        assert node.name == "Scout", "快照 name 应一一对应"
        assert node.description == "侦察兵", "快照 description 应一一对应"
        assert node.parent_id == 2, "快照 parent_id 应一一对应"
        assert node.sub_agent_ids == [10, 11], "快照应包含全部子代理 id"
        assert node.state == "running", "快照 state 应取枚举值字符串"

    def test_to_agent_node_defensive_copy_of_sub_agent_ids(self, session):
        """修改快照或会话任一侧子代理列表 → 另一侧不受影响."""
        # Arrange
        session.sub_agent_ids.extend([5, 6])
        # Act
        node = session.to_agent_node()
        node.sub_agent_ids.append(7)
        # Assert（快照侧修改不影响会话）
        assert session.sub_agent_ids == [5, 6], "修改快照不应影响会话原列表"
        # Act（会话侧修改不影响既有快照）
        session.sub_agent_ids.append(8)
        # Assert
        assert node.sub_agent_ids == [5, 6, 7], "修改会话不应影响既有快照"


def _make_unfinished_plain_text() -> _PlainTextMessageBlock:
    """构造一个被人为置为未完成的文本块，用于验证跳过逻辑."""
    block = _PlainTextMessageBlock("assistant", "半截输出")
    block.finished = False
    return block
