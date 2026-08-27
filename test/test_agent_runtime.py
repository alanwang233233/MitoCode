"""AgentRuntime 模块单元测试套件。

覆盖 src/mitocode/backend/agent/agent_runtime.py 的全部行为分支：
_invoke_tool / _invoke_on_end 的协程与线程分发规则、run() 主循环的
文本流式事件、按 index 聚合并行工具调用、分片 id/name 同步、
工具执行五分支（解析失败/未注册/TypeError/异常/成功）、on_end 回调时序、
协作式取消（启动前/流中）以及无 finish_reason 时的回环行为。
"""

import os
import sys
import threading
import unittest.mock as mock
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.agent.agent_runtime import AgentRuntime
from mitocode.backend.agent.agent_session import AgentSession
from mitocode.backend.chat_provider.sample_chat_provider import _SampleChatProvider

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

DEFAULT_USER_PROMPT = "北京天气如何"
TOOL_NAME_WEATHER = "get_weather"
TOOL_NAME_ECHO = "echo"
CALL_ID_PRIMARY = "call_001"
CALL_ID_FRAGMENT_BASE = "call_9"
CALL_ID_FRAGMENT_TAIL = "xyz"
CALL_ID_FRAGMENT_FULL = "call_9xyz"
ARG_CHUNK_FIRST = '{"city": '
ARG_CHUNK_SECOND = '"Beijing"}'
EXPECTED_WEATHER_PARAMS = {"city": "Beijing"}
REASON_TOOL_CALLS = "tool_calls"
REASON_STOP = "stop"
STANDARD_REASONS = ["stop", "length", "content_filter"]
PARSE_ERROR_PREFIX = "Invalid JSON Format. "
PARSE_ERROR_MARKER = "JSON parsing failed"

WEATHER_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME_WEATHER,
        "description": "查询天气",
        "parameters": {"type": "object", "properties": {}},
    },
}


# ──────────────────────────────────────────────
# 流式 chunk 辅助类
# ──────────────────────────────────────────────


class _FunctionPayload:
    """模拟 OpenAI chunk 中 tool_call.function 载荷，避免 MagicMock(name=) 陷阱."""

    def __init__(self, name: Optional[str] = None, arguments: Optional[str] = None):
        self.name = name
        self.arguments = arguments


class _ToolCallChunk:
    """模拟流式 tool_call 片段：index 必填，id/name/arguments 分片可选."""

    def __init__(
        self,
        index: int,
        call_id: Optional[str] = None,
        name: Optional[str] = None,
        arguments: Optional[str] = None,
    ):
        self.index = index
        self.id = call_id
        self.function: Optional[_FunctionPayload] = None
        if name is not None or arguments is not None:
            self.function = _FunctionPayload(name=name, arguments=arguments)


class _Chunk:
    """模拟 LLM 流式响应块；choices=[] 或缺失 choices 属性均可显式构造."""

    def __init__(
        self,
        content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[_ToolCallChunk]] = None,
        choices: Optional[List[Any]] = None,
    ):
        if choices is not None:
            self.choices: Any = choices
        else:
            self.choices = [
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ]


class _ScriptedProvider(_SampleChatProvider):
    """按脚本逐轮回放 chunk 列表，并记录每次 chat 调用的 tools/messages 入参."""

    def __init__(self, rounds: List[List[_Chunk]]):
        self._rounds: List[List[_Chunk]] = [list(r) for r in rounds]
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, tools=None, messages=None, **kwargs):
        self.calls.append({"tools": tools, "messages": messages})
        chunks = self._rounds.pop(0)

        async def _stream():
            for chunk in chunks:
                yield chunk

        return _stream()

    async def get_model_list(self):
        return []

    async def init(self):
        pass

    @classmethod
    def get_settings_schema(cls):
        return {}

    @classmethod
    def check_settings_schema(cls, settings):
        return True


class _CancelMidStreamProvider(_SampleChatProvider):
    """首个 chunk 输出后立即触发 session.cancel() 的受控 provider."""

    def __init__(self):
        self.session: Optional[AgentSession] = None
        self.chat_count = 0

    async def chat(self, tools=None, messages=None, **kwargs):
        self.chat_count += 1
        session = self.session

        async def _stream():
            yield _Chunk(content="第一句")
            session.cancel()
            yield _Chunk(content="不应出现")
            yield _Chunk(finish_reason=REASON_STOP)

        return _stream()

    async def get_model_list(self):
        return []

    async def init(self):
        pass

    @classmethod
    def get_settings_schema(cls):
        return {}

    @classmethod
    def check_settings_schema(cls, settings):
        return True


def _counting_tool(exc: Optional[Exception] = None) -> Tuple[Callable, List[dict]]:
    """构造带调用记录的工具：正常返回 'ok'；注入 exc 时抛出该异常."""

    calls: List[dict] = []

    def _tool(**kwargs):
        calls.append(kwargs)
        if exc is not None:
            raise exc
        return "ok"

    return _tool, calls


async def _drain_events(gen: AsyncGenerator) -> List[Dict[str, Any]]:
    """收集 run() 生成的全部事件为列表."""
    events: List[Dict[str, Any]] = []
    async for event in gen:
        events.append(event)
    return events


def _events_of_type(
    events: List[Dict[str, Any]], event_name: str
) -> List[Dict[str, Any]]:
    """过滤指定类型的事件子集."""
    return [e for e in events if e["event"] == event_name]


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def make_session():
    """会话工厂夹具：注入 provider 并预置用户提示词，kwargs 可覆盖全部设置."""

    def _make(provider, **settings) -> AgentSession:
        settings.setdefault("provider", provider)
        session = AgentSession(**settings)
        session.append_user_prompt(DEFAULT_USER_PROMPT)
        return session

    return _make


# ──────────────────────────────────────────────
# _invoke_tool 分发
# ──────────────────────────────────────────────


class TestInvokeToolDispatch:
    """_invoke_tool 对协程函数/同步函数/返回协程的同步函数三种分发规则测试."""

    @pytest.mark.asyncio
    async def test_invoke_tool_coroutine_function_awaits_directly_with_kwargs(self):
        """协程工具函数 → 直接 await 并展开 kwargs 返回其结果."""
        # Arrange
        received = {}

        async def probe(city: str) -> str:
            received.update(city=city)
            return f"{city}:晴"

        # Act
        result = await AgentRuntime._invoke_tool(probe, {"city": "北京"})

        # Assert
        assert result == "北京:晴", "协程函数结果应被直接 await 返回"
        assert received == {"city": "北京"}, "kwargs 应原样展开传给工具"

    @pytest.mark.asyncio
    async def test_invoke_tool_sync_function_runs_in_worker_thread_returns_value(self):
        """同步工具函数 → 经工作线程执行并返回普通值（不在事件循环线程内运行）."""
        # Arrange
        main_thread_id = threading.get_ident()
        tool_thread_ids = []

        def add(a: int, b: int) -> int:
            tool_thread_ids.append(threading.get_ident())
            return a + b

        # Act
        result = await AgentRuntime._invoke_tool(add, {"a": 2, "b": 40})

        # Assert
        assert result == 42, "同步函数返回值应原样透出"
        assert tool_thread_ids[0] != main_thread_id, "同步函数应在独立工作线程中执行"

    @pytest.mark.asyncio
    async def test_invoke_tool_sync_function_returning_coroutine_inner_result_awaited(
        self,
    ):
        """同步函数返回裸协程 → 内层协程被二次 await 而非原样返回协程对象."""
        # Arrange
        executed = []

        def outer():
            async def inner():
                executed.append(True)
                return "inner-ok"

            return inner()

        # Act
        result = await AgentRuntime._invoke_tool(outer, {})

        # Assert
        assert result == "inner-ok", "内层协程的返回值应被取出"
        assert executed == [True], "返回的内层协程应确实被执行过"


# ──────────────────────────────────────────────
# run(): provider 交互
# ──────────────────────────────────────────────


class TestRunProviderInteraction:
    """run() 与 provider.chat 的交互契约：tools 传参规则与异常传播."""

    @pytest.mark.asyncio
    async def test_run_empty_tool_schemas_passes_tools_none_to_chat(self, make_session):
        """无 tool_schemas → chat 收到 tools=None 且 messages 为标准上下文."""
        # Arrange
        provider = _ScriptedProvider([[_Chunk(finish_reason=REASON_STOP)]])
        session = make_session(provider)

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        assert len(provider.calls) == 1, "chat 应恰好调用一次"
        assert provider.calls[0]["tools"] is None, "空 schemas 应转换为 tools=None"
        assert provider.calls[0]["messages"] == [
            {"role": "user", "content": DEFAULT_USER_PROMPT}
        ], "messages 应为编译后的标准上下文"
        assert events[-1]["event"] == "system.finished.standard", "应以标准结束收尾"

    @pytest.mark.asyncio
    async def test_run_nonempty_tool_schemas_passed_through_to_chat(self, make_session):
        """有 tool_schemas → 原样传递给 chat 的 tools 参数而非 None."""
        # Arrange
        provider = _ScriptedProvider([[_Chunk(finish_reason=REASON_STOP)]])
        session = make_session(provider, tool_schemas=[WEATHER_SCHEMA])

        # Act
        await _drain_events(AgentRuntime.run(session))

        # Assert
        assert provider.calls[0]["tools"] == [
            WEATHER_SCHEMA
        ], "非空 schemas 应原样透传给 chat"

    @pytest.mark.asyncio
    async def test_run_provider_chat_exception_propagates_to_caller(self, make_session):
        """provider.chat 抛异常 → 异常向上传播给 run() 的调用方."""
        # Arrange
        provider = mock.MagicMock(spec=_SampleChatProvider)
        provider.chat = mock.AsyncMock(side_effect=RuntimeError("llm down"))
        session = make_session(provider)

        # Act & Assert
        with pytest.raises(RuntimeError, match="llm down"):
            await _drain_events(AgentRuntime.run(session))


# ──────────────────────────────────────────────
# run(): 文本流式
# ──────────────────────────────────────────────


class TestRunTextStreaming:
    """run() 文本增量处理：仅 truthy content 产事件并聚合成单条助手消息."""

    @pytest.mark.asyncio
    async def test_run_text_stream_emits_events_only_for_truthy_content(
        self, make_session
    ):
        """混合 truthy/falsy content → 仅非空内容产生 agent.text 事件."""
        # Arrange
        chunks = [
            _Chunk(content="你好"),
            _Chunk(content=""),
            _Chunk(content=None),
            _Chunk(finish_reason=REASON_STOP),
        ]
        provider = _ScriptedProvider([chunks])
        session = make_session(provider)

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        assert events == [
            {"event": "agent.text", "delta": "你好"},
            {"event": "system.finished.standard", "reason": REASON_STOP},
        ], "空串/None 内容不应产生 agent.text 事件"

    @pytest.mark.asyncio
    async def test_run_appends_assistant_deltas_into_single_message_block(
        self, make_session
    ):
        """多个文本增量 → 聚合进同一条 assistant 消息而非多条."""
        # Arrange
        chunks = [
            _Chunk(content="你好"),
            _Chunk(content="世界"),
            _Chunk(finish_reason=REASON_STOP),
        ]
        provider = _ScriptedProvider([chunks])
        session = make_session(provider)

        # Act
        await _drain_events(AgentRuntime.run(session))

        # Assert
        assistant_blocks = [
            m for m in session.messages if getattr(m, "role", "") == "assistant"
        ]
        assert len(assistant_blocks) == 1, "连续增量应复用同一条 assistant 消息块"
        assert assistant_blocks[0].message == "你好世界", "增量应按顺序拼接"

    @pytest.mark.parametrize(
        "skipped_chunk",
        [
            pytest.param(_Chunk(choices=[]), id="empty-choices-list"),
            pytest.param(SimpleNamespace(), id="missing-choices-attribute"),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_chunk_without_choices_is_skipped_silently(
        self, make_session, skipped_chunk
    ):
        """无 choices 的异常 chunk → 静默跳过且不影响后续 chunk 处理."""
        # Arrange
        chunks = [
            _Chunk(content="A"),
            skipped_chunk,
            _Chunk(content="B"),
            _Chunk(finish_reason=REASON_STOP),
        ]
        provider = _ScriptedProvider([chunks])
        session = make_session(provider)

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        text_events = _events_of_type(events, "agent.text")
        assert [e["delta"] for e in text_events] == [
            "A",
            "B",
        ], "空 choices 块应被跳过，前后正常块均保留"


# ──────────────────────────────────────────────
# run(): 单工具完整循环
# ──────────────────────────────────────────────


class TestRunSingleToolLoop:
    """run() 单工具完整链路：header→delta→回环执行→result→二轮 stop."""

    @pytest.mark.asyncio
    async def test_run_single_tool_round_loops_back_then_finishes_standard(
        self, make_session
    ):
        """单工具两轮对话 → 事件序列精确符合协议且回环上下文含工具结果."""
        # Arrange
        weather_tool = mock.MagicMock(return_value="Beijing:晴")
        rounds = [
            [
                _Chunk(content="查一下"),
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(
                            0, call_id=CALL_ID_PRIMARY, name=TOOL_NAME_WEATHER
                        )
                    ]
                ),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments=ARG_CHUNK_FIRST)]),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments=ARG_CHUNK_SECOND)]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [
                _Chunk(content="好了"),
                _Chunk(finish_reason=REASON_STOP),
            ],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={TOOL_NAME_WEATHER: weather_tool})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        expected = [
            {"event": "agent.text", "delta": "查一下"},
            {
                "event": "agent.tool_call.header",
                "id": CALL_ID_PRIMARY,
                "name": TOOL_NAME_WEATHER,
            },
            {
                "event": "agent.tool_call.delta",
                "id": CALL_ID_PRIMARY,
                "delta": ARG_CHUNK_FIRST,
            },
            {
                "event": "agent.tool_call.delta",
                "id": CALL_ID_PRIMARY,
                "delta": ARG_CHUNK_SECOND,
            },
            {"event": "system.finished.tool_calls"},
            {
                "event": "agent.tool_call.execution",
                "name": TOOL_NAME_WEATHER,
                "args": EXPECTED_WEATHER_PARAMS,
            },
            {
                "event": "agent.tool_call.result",
                "id": CALL_ID_PRIMARY,
                "success": True,
                "result": "Beijing:晴",
            },
            {"event": "agent.text", "delta": "好了"},
            {"event": "system.finished.standard", "reason": REASON_STOP},
        ]
        assert events == expected, "完整单工具链路的事件序列应精确匹配协议"
        assert len(provider.calls) == 2, "工具轮后应回环再次调用 chat"
        second_context = provider.calls[1]["messages"]
        assert second_context[-1] == {
            "role": "tool",
            "tool_call_id": CALL_ID_PRIMARY,
            "content": "Beijing:晴",
        }, "第二轮上下文应包含工具结果消息"
        assert (
            second_context[2]["tool_calls"][0]["function"]["arguments"]
            == '{"city": "Beijing"}'
        ), "第二轮上下文应包含聚合后的 assistant 工具调用"
        weather_tool.assert_called_once_with(**EXPECTED_WEATHER_PARAMS)

    @pytest.mark.parametrize(
        "return_value, expected_str",
        [
            pytest.param(42, "42", id="int"),
            pytest.param({"temp": 28}, "{'temp': 28}", id="dict"),
            pytest.param(None, "None", id="none"),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_success_results_are_stringified_verbatim(
        self, make_session, return_value, expected_str
    ):
        """工具各种返回值 → 以 str() 原样字符串化进入 result 事件."""
        # Arrange
        tool = mock.MagicMock(return_value=return_value)
        rounds = [
            [
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(0, call_id=CALL_ID_PRIMARY, name=TOOL_NAME_ECHO)
                    ]
                ),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={TOOL_NAME_ECHO: tool})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        result_events = _events_of_type(events, "agent.tool_call.result")
        assert len(result_events) == 1, "应恰好产生一条 result 事件"
        assert result_events[0]["success"] is True, "成功路径 success 应为 True"
        assert (
            result_events[0]["result"] == expected_str
        ), f"返回值 {return_value!r} 应以 str() 形式透出"


# ──────────────────────────────────────────────
# run(): 多工具并行聚合
# ──────────────────────────────────────────────


class TestRunParallelTools:
    """run() 按 index 聚合并行工具调用并按创建顺序串行执行."""

    @pytest.mark.asyncio
    async def test_run_parallel_tools_aggregated_by_index_executed_in_creation_order(
        self, make_session
    ):
        """两个并行工具跨 chunk 交错到达 → 按首见 index 顺序聚合与执行."""
        # Arrange
        tool_a = mock.MagicMock(return_value="a1")
        tool_b = mock.MagicMock(return_value="b20")
        rounds = [
            [
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(0, call_id="call_a", name="tool_a"),
                        _ToolCallChunk(1, call_id="call_b", name="tool_b"),
                    ]
                ),
                _Chunk(tool_calls=[_ToolCallChunk(1, arguments='{"n": 2}')]),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments='{"n": 1}')]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={"tool_a": tool_a, "tool_b": tool_b})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        headers = [
            (e["name"], e["id"])
            for e in _events_of_type(events, "agent.tool_call.header")
        ]
        assert headers == [
            ("tool_a", "call_a"),
            ("tool_b", "call_b"),
        ], "header 应按 index 首见顺序各发一条"
        deltas = [
            (e["id"], e["delta"])
            for e in _events_of_type(events, "agent.tool_call.delta")
        ]
        assert deltas == [
            ("call_b", '{"n": 2}'),
            ("call_a", '{"n": 1}'),
        ], "delta 事件应对应各自 index 的参数分片"
        executions = [
            (e["name"], e["args"])
            for e in _events_of_type(events, "agent.tool_call.execution")
        ]
        assert executions == [
            ("tool_a", {"n": 1}),
            ("tool_b", {"n": 2}),
        ], "execution 应按创建顺序（index 0 先于 index 1）"
        results = [
            (e["id"], e["success"], e["result"])
            for e in _events_of_type(events, "agent.tool_call.result")
        ]
        assert results == [
            ("call_a", True, "a1"),
            ("call_b", True, "b20"),
        ], "每个工具调用应有对应成功 result"


# ──────────────────────────────────────────────
# run(): 分片 id / name
# ──────────────────────────────────────────────


class TestRunFragmentedChunks:
    """run() 处理跨 chunk 分片的 id 与 name：单 header、累积同步已建块."""

    @pytest.mark.asyncio
    async def test_run_fragmented_name_yields_single_header_and_merged_execution(
        self, make_session
    ):
        """name 分两个 chunk 到达 → 仅首个片段发 header，执行用合并后全名命中注册表."""
        # Arrange
        weather_tool = mock.MagicMock(return_value="ok")
        rounds = [
            [
                _Chunk(tool_calls=[_ToolCallChunk(0, call_id="call_n", name="get_")]),
                _Chunk(tool_calls=[_ToolCallChunk(0, name="weather")]),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments='{"city": "x"}')]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={TOOL_NAME_WEATHER: weather_tool})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        headers = _events_of_type(events, "agent.tool_call.header")
        assert len(headers) == 1, "name 分片只应产生一个 header"
        assert headers[0]["name"] == "get_", "header 携带首见名片段"
        executions = _events_of_type(events, "agent.tool_call.execution")
        assert (
            executions[0]["name"] == TOOL_NAME_WEATHER
        ), "执行时应使用跨片合并后的完整名称"
        results = _events_of_type(events, "agent.tool_call.result")
        assert results[0]["success"] is True, "合并名称应命中注册表成功执行"

    @pytest.mark.asyncio
    async def test_run_fragmented_id_accumulates_into_delta_and_final_block_id(
        self, make_session
    ):
        """id 分两个 chunk 到达 → delta 事件携带累积值，最终 block.id 被同步更新."""
        # Arrange
        echo_tool = mock.MagicMock(return_value="value:7")
        rounds = [
            [
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(
                            0, call_id=CALL_ID_FRAGMENT_BASE, name=TOOL_NAME_ECHO
                        )
                    ]
                ),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments='{"x": ')]),
                _Chunk(tool_calls=[_ToolCallChunk(0, call_id=CALL_ID_FRAGMENT_TAIL)]),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments="7}")]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={TOOL_NAME_ECHO: echo_tool})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        deltas = [
            (e["id"], e["delta"])
            for e in _events_of_type(events, "agent.tool_call.delta")
        ]
        assert deltas == [
            (CALL_ID_FRAGMENT_BASE, '{"x": '),
            (CALL_ID_FRAGMENT_FULL, "7}"),
        ], "后续 delta 事件的 id 应反映跨片累积值"
        results = _events_of_type(events, "agent.tool_call.result")
        assert (
            results[0]["id"] == CALL_ID_FRAGMENT_FULL
        ), "已建块的 id 应被后续 id 分片同步更新"
        assert results[0]["success"] is True, "分片 id 工具仍应成功执行"


# ──────────────────────────────────────────────
# run(): 参数先于 header
# ──────────────────────────────────────────────


class TestRunArgumentsBeforeHeader:
    """run() 处理参数分片先于 name 到达的退化场景."""

    @pytest.mark.asyncio
    async def test_run_arguments_before_header_skips_block_append_but_emits_event(
        self, make_session
    ):
        """参数分片先到而该 index 无活动块 → 仍发 delta 事件但不追加进块."""
        # Arrange
        echo_tool = mock.MagicMock(return_value={})
        rounds = [
            [
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(0, call_id="call_x", arguments='{"x": 1}')
                    ]
                ),
                _Chunk(tool_calls=[_ToolCallChunk(0, name=TOOL_NAME_ECHO)]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={TOOL_NAME_ECHO: echo_tool})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        deltas = _events_of_type(events, "agent.tool_call.delta")
        assert deltas == [
            {"event": "agent.tool_call.delta", "id": "call_x", "delta": '{"x": 1}'}
        ], "无活动块时参数分片仍应产出 delta 事件"
        executions = _events_of_type(events, "agent.tool_call.execution")
        assert executions[0]["args"] == {}, "早到的参数不应出现在最终 parameter 中"
        results = _events_of_type(events, "agent.tool_call.result")
        assert results[0]["success"] is True, "空参工具应照常执行成功"


# ──────────────────────────────────────────────
# run(): 工具执行失败路径
# ──────────────────────────────────────────────


class TestRunToolFailurePaths:
    """run() 工具执行的四种失败分支及错误文案契约."""

    @pytest.mark.asyncio
    async def test_run_parse_error_reports_invalid_json_without_invoking_tool(
        self, make_session
    ):
        """参数非法 JSON → 报 Invalid JSON Format 且完全不调用工具."""
        # Arrange
        echo_tool = mock.MagicMock(return_value="x")
        rounds = [
            [
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(0, call_id=CALL_ID_PRIMARY, name=TOOL_NAME_ECHO)
                    ]
                ),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments="{这不是JSON")]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map={TOOL_NAME_ECHO: echo_tool})

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        results = _events_of_type(events, "agent.tool_call.result")
        assert results[0]["success"] is False, "解析失败应为失败结果"
        assert results[0]["result"].startswith(
            PARSE_ERROR_PREFIX
        ), "错误文案应以 'Invalid JSON Format. ' 开头"
        assert (
            PARSE_ERROR_MARKER in results[0]["result"]
        ), "文案应包含底层 JSON 解析失败标记"
        assert echo_tool.call_count == 0, "解析失败时不应调用任何工具"
        tool_message = provider.calls[1]["messages"][-1]
        assert (
            tool_message["content"] == results[0]["result"]
        ), "失败文案应作为 tool 消息回传 LLM"

    @pytest.mark.parametrize(
        "tool_name, tool_factory, args_json, expected_result",
        [
            pytest.param(
                "ghost_tool",
                lambda: (None, None),
                '{"a": 1}',
                "Error: Tool 'ghost_tool' is not registered.",
                id="unregistered-tool",
            ),
            pytest.param(
                "rigid",
                lambda: _counting_tool(TypeError("unexpected kwarg")),
                "{}",
                "Invalid argument. unexpected kwarg",
                id="type-error",
            ),
            pytest.param(
                "bomb",
                lambda: _counting_tool(RuntimeError("boom")),
                "{}",
                "Error executing tool: boom",
                id="generic-exception",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_failure_scenarios_report_exact_error_message(
        self, make_session, tool_name, tool_factory, args_json, expected_result
    ):
        """未注册/TypeError/一般异常 → result 事件 success=False 且文案精确匹配."""
        # Arrange
        tool, calls = tool_factory()
        tools_map = {} if tool is None else {tool_name: tool}
        rounds = [
            [
                _Chunk(
                    tool_calls=[
                        _ToolCallChunk(0, call_id=CALL_ID_PRIMARY, name=tool_name)
                    ]
                ),
                _Chunk(tool_calls=[_ToolCallChunk(0, arguments=args_json)]),
                _Chunk(finish_reason=REASON_TOOL_CALLS),
            ],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider, tools_map=tools_map)

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        results = _events_of_type(events, "agent.tool_call.result")
        assert len(results) == 1, "应恰好产生一条失败 result 事件"
        assert results[0] == {
            "event": "agent.tool_call.result",
            "id": CALL_ID_PRIMARY,
            "success": False,
            "result": expected_result,
        }, f"失败文案应精确为: {expected_result}"
        assert len(provider.calls) == 2, "失败后仍应回环把结果提交给 LLM"
        if tool is not None:
            assert len(calls) == 1, "注册过的工具应被尝试调用一次"


# ──────────────────────────────────────────────
# run(): 终止
# ──────────────────────────────────────────────


class TestRunTerminationAndOnEnd:
    """run() 的终止原因透传以及未知 finish_reason 的回环行为."""

    @pytest.mark.parametrize("reason", STANDARD_REASONS)
    @pytest.mark.asyncio
    async def test_run_terminal_reasons_propagate_to_finished_event(
        self, make_session, reason
    ):
        """三种标准终止原因 → reason 原样透传进 system.finished.standard 事件."""
        # Arrange
        provider = _ScriptedProvider([[_Chunk(finish_reason=reason)]])
        session = make_session(provider)

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        assert events[-1] == {
            "event": "system.finished.standard",
            "reason": reason,
        }, f"终止原因 {reason} 应原样透传"

    @pytest.mark.asyncio
    async def test_run_unknown_finish_reason_loops_back_to_chat_until_stop(
        self, make_session
    ):
        """首轮无 finish_reason → 不产终态事件而是回环再次请求直至 stop."""
        # Arrange
        rounds = [
            [_Chunk(content="片段")],
            [_Chunk(finish_reason=REASON_STOP)],
        ]
        provider = _ScriptedProvider(rounds)
        session = make_session(provider)

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        assert _events_of_type(events, "agent.text") == [
            {"event": "agent.text", "delta": "片段"}
        ], "首轮仅产生文本事件"
        assert events[-1] == {
            "event": "system.finished.standard",
            "reason": REASON_STOP,
        }, "第二轮 stop 后应以标准结束收尾"
        assert len(provider.calls) == 2, "未知 finish_reason 应回环重新调用 chat"


# ──────────────────────────────────────────────
# run(): 协作式取消
# ──────────────────────────────────────────────


class TestRunCancellation:
    """run() 在启动前与流中被取消时的终止行为."""

    @pytest.mark.asyncio
    async def test_run_cancelled_before_start_emits_single_cancelled_event_never_chats(
        self, make_session
    ):
        """启动前已取消 → 仅产出一条 system.cancelled 且从不调用 chat."""
        # Arrange
        provider = _ScriptedProvider([])
        session = make_session(provider)
        session.cancel()

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        assert events == [{"event": "system.cancelled"}], "应只产出一条取消事件"
        assert provider.calls == [], "取消后不应发起任何 chat 请求"

    @pytest.mark.asyncio
    async def test_run_cancel_mid_stream_stops_with_single_cancelled_event(
        self, make_session
    ):
        """流中取消 → 立即产出单条取消事件且后续 chunk 完全不被消费."""
        # Arrange
        provider = _CancelMidStreamProvider()
        session = make_session(provider)
        provider.session = session

        # Act
        events = await _drain_events(AgentRuntime.run(session))

        # Assert
        assert events == [
            {"event": "agent.text", "delta": "第一句"},
            {"event": "system.cancelled"},
        ], "取消后应只有首条文本加一条取消事件"
        assert provider.chat_count == 1, "chat 只应调用一次"
        assert all(
            e["event"] != "system.finished.standard" for e in events
        ), "取消路径不应产出标准结束事件"
