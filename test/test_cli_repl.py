"""cli.repl / cli.rendering / cli.prompt_io 单元测试。

REPL 行为全部通过注入的脚本化 input_reader 与替身 orchestrator 驱动：
发送/排队/排空、Esc 中断、/exit 退出、Ctrl+C 防御、关停清理；渲染器
单独以记录 Console 断言流式文本、工具链路与子 Agent 标签。
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli import prompt_io
from mitocode.cli.prompt_io import run_with_prompt_toolkit
from mitocode.cli.rendering import EventRenderer
from mitocode.cli.repl import CliRepl
from mitocode.core.agent.agent_state import AgentState


class _FakeSession:
    """记录 cancel 调用的 session 替身."""

    def __init__(self, state=AgentState.FINISHED, name="Main"):
        self.state = state
        self.name = name
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


class _FakeOrchestrator:
    """REPL 依赖面的 orchestrator 替身（事件队列可预填充）."""

    def __init__(self):
        self.event_queue = asyncio.Queue()
        self.session = _FakeSession()
        self.sent = []
        self.cancelled = []

    def get_session(self, agent_id):
        return self.session

    async def send_message(self, agent_id, text):
        self.sent.append((agent_id, text))

    async def cancel_agent(self, agent_id, cascade=True):
        self.cancelled.append(agent_id)

    def get_tree_snapshot(self):
        return [{"id": 0}, {"id": 1}]

    def set_default_provider(self, provider):
        pass


def _reader(*lines):
    """脚本化输入端口：依次返回各行，耗尽后抛 EOFError."""
    lines = list(lines)

    async def _read(prompt):
        if not lines:
            raise EOFError
        return lines.pop(0)

    return _read


def _make_repl(orchestrator, reader, console=None):
    return CliRepl(orchestrator, 0, reader, console or Console(record=True, width=200))


# ──────────────────────────────────────────────
# 主循环：发送与退出
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plain_text_is_sent_to_main_agent():
    """普通文本 → send_message 到 main agent."""
    orch = _FakeOrchestrator()

    await _make_repl(orch, _reader("hello")).run()

    assert orch.sent == [(0, "hello")]


@pytest.mark.asyncio
async def test_exit_command_stops_loop_and_cancels_agents():
    """/exit → 循环结束，关停时取消全部 agent."""
    orch = _FakeOrchestrator()

    await _make_repl(orch, _reader("/exit")).run()

    assert orch.sent == []
    assert orch.cancelled == [0, 1]


@pytest.mark.asyncio
async def test_unknown_command_is_not_sent_as_chat():
    """未知斜杠命令 → 提示且不发送."""
    orch = _FakeOrchestrator()

    await _make_repl(orch, _reader("/nope")).run()

    assert orch.sent == []


@pytest.mark.asyncio
async def test_empty_lines_are_skipped():
    """空行 → 不发送不报错."""
    orch = _FakeOrchestrator()

    await _make_repl(orch, _reader("   ", "")).run()

    assert orch.sent == []


# ──────────────────────────────────────────────
# 排队与排空
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_while_running_is_queued():
    """RUNNING 中输入 → 入队提示，不发送."""
    orch = _FakeOrchestrator()
    orch.session.state = AgentState.RUNNING
    repl = _make_repl(orch, _reader("first", "second"))

    await repl.run()

    assert orch.sent == []
    assert repl._queued == ["first", "second"]
    assert "Queued" in repl.console.export_text()


@pytest.mark.asyncio
async def test_maybe_drain_sends_queue_when_idle():
    """空闲时 _maybe_drain → 队列依次发送并清空."""
    orch = _FakeOrchestrator()
    repl = _make_repl(orch, _reader())
    repl._queued = ["a", "b"]

    await repl._maybe_drain()

    assert orch.sent == [(0, "a"), (0, "b")]
    assert repl._queued == []


@pytest.mark.asyncio
async def test_maybe_drain_skips_while_running():
    """RUNNING 中 _maybe_drain → 不发送."""
    orch = _FakeOrchestrator()
    orch.session.state = AgentState.RUNNING
    repl = _make_repl(orch, _reader())
    repl._queued = ["a"]

    await repl._maybe_drain()

    assert orch.sent == []
    assert repl._queued == ["a"]


@pytest.mark.asyncio
async def test_sync_event_drains_queue():
    """sync 事件（状态已离开 RUNNING）→ 消费者自动排空队列."""
    orch = _FakeOrchestrator()
    orch.event_queue.put_nowait((0, {"type": "sync", "agents": []}))
    drained = asyncio.Event()

    async def fake_send(agent_id, text):
        orch.sent.append((agent_id, text))
        drained.set()

    orch.send_message = fake_send
    repl = _make_repl(orch, _reader())
    repl._queued = ["queued-msg"]

    async def waiting_reader(prompt):
        await drained.wait()
        raise EOFError

    repl.input_reader = waiting_reader
    await repl.run()

    assert orch.sent == [(0, "queued-msg")]


@pytest.mark.asyncio
async def test_queued_events_are_rendered():
    """消费者把队列中的 token/tool 事件渲染到 console."""
    orch = _FakeOrchestrator()
    orch.event_queue.put_nowait((0, {"type": "token", "agent_id": 0, "content": "hi"}))
    orch.event_queue.put_nowait(
        (0, {"type": "tool_result", "agent_id": 0, "result": "done", "success": True})
    )
    rendered = asyncio.Event()

    real_render = EventRenderer.render

    def spy_render(self, event):
        real_render(self, event)
        if event.get("type") == "tool_result":
            rendered.set()

    EventRenderer.render = spy_render
    try:

        async def waiting_reader(prompt):
            await rendered.wait()
            raise EOFError

        repl = _make_repl(orch, waiting_reader)
        await repl.run()
        output = repl.console.export_text()
    finally:
        EventRenderer.render = real_render

    assert "hi" in output
    assert "done" in output


# ──────────────────────────────────────────────
# 中断与防御
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_interrupt_cancels_running_session():
    """RUNNING 时 request_interrupt → session.cancel + 提示."""
    orch = _FakeOrchestrator()
    orch.session.state = AgentState.RUNNING
    repl = _make_repl(orch, _reader())

    repl.request_interrupt()

    assert orch.session.cancelled == 1
    assert "Interrupted" in repl.console.export_text()


@pytest.mark.asyncio
async def test_request_interrupt_ignores_idle_session():
    """空闲时 request_interrupt → 无操作无提示."""
    orch = _FakeOrchestrator()
    repl = _make_repl(orch, _reader())

    repl.request_interrupt()

    assert orch.session.cancelled == 0


def test_request_exit_sets_flag():
    """request_exit → 置退出标记."""
    repl = _make_repl(_FakeOrchestrator(), _reader())

    repl.request_exit()

    assert repl._exit_requested is True


@pytest.mark.asyncio
async def test_keyboard_interrupt_shows_hint_and_continues():
    """输入端口抛 KeyboardInterrupt → 打印提示继续，直到 EOF 退出."""
    orch = _FakeOrchestrator()
    lines = ["boom"]

    async def reader(prompt):
        if lines:
            value = lines.pop(0)
            raise KeyboardInterrupt
        raise EOFError

    repl = _make_repl(orch, reader)

    await repl.run()

    assert "Press Ctrl+D" in repl.console.export_text()


# ──────────────────────────────────────────────
# EventRenderer
# ──────────────────────────────────────────────


def _render_console(events, main_agent_id=0):
    orch = _FakeOrchestrator()
    orch.session.name = "Main"
    console = Console(record=True, width=200)
    renderer = EventRenderer(console, orch, main_agent_id)
    for event in events:
        renderer.render(event)
    return console


def test_renderer_streams_tokens_then_closes():
    """连续 token 流式拼接，turn_end 结束文本段."""
    console = _render_console(
        [
            {"type": "token", "agent_id": 0, "content": "Hel"},
            {"type": "token", "agent_id": 0, "content": "lo"},
            {"type": "turn_end", "agent_id": 0, "reason": "stop"},
        ]
    )
    output = console.export_text()
    assert "Hello" in output


def test_renderer_tool_call_full_chain():
    """工具链路：header 名称/描述 → execution 参数 → result 成败."""
    console = _render_console(
        [
            {
                "type": "tool_call",
                "agent_id": 0,
                "action": "bash",
                "description": "Run a command",
            },
            {
                "type": "tool_execution",
                "agent_id": 0,
                "name": "Bash",
                "args": {"command": "ls"},
            },
            {
                "type": "tool_result",
                "agent_id": 0,
                "result": "file.txt",
                "success": True,
            },
            {"type": "tool_call", "agent_id": 0, "action": "x", "description": ""},
            {"type": "tool_result", "agent_id": 0, "result": "bad", "success": False},
        ]
    )
    output = console.export_text()
    assert "bash" in output
    assert "Run a command" in output
    assert '"command": "ls"' in output
    assert "file.txt" in output
    assert "bad" in output


def test_renderer_labels_sub_agent():
    """子 Agent 事件 → 前缀 #id name 标签."""
    orch = _FakeOrchestrator()
    sub = _FakeSession(name="Worker")
    orch.session = sub
    console = Console(record=True, width=200)
    renderer = EventRenderer(console, orch, main_agent_id=0)
    renderer.render({"type": "token", "agent_id": 3, "content": "part"})
    renderer.render({"type": "turn_end", "agent_id": 3, "reason": "stop"})

    output = console.export_text()
    assert "#3 Worker" in output
    assert "part" in output


# ──────────────────────────────────────────────
# prompt_io 薄壳
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_io_wires_reader_and_runs(monkeypatch):
    """run_with_prompt_toolkit → 注入 reader、跑完 REPL、关停 agent."""
    orch = _FakeOrchestrator()

    class _FakePromptSession:
        def __init__(self, *args, **kwargs):
            pass

        async def prompt_async(self, text):
            raise EOFError

    monkeypatch.setattr(prompt_io, "PromptSession", _FakePromptSession)
    repl = CliRepl(orch, 0, _reader(), Console(record=True, width=200))

    await run_with_prompt_toolkit(repl)

    assert orch.cancelled == [0, 1]


# ──────────────────────────────────────────────
# 审批仲裁（ask_approval 路由）
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pending_approval_routes_next_line():
    """审批挂起时输入的下一行 → 路由给审批 Future，不作为聊天发送."""
    orch = _FakeOrchestrator()
    lines = ["trigger", "1"]
    answers = []

    async def reader(prompt):
        if not lines:
            raise EOFError
        value = lines.pop(0)
        if value == "1":
            # 等审批 waiter 注册后再把答案交给主循环（避免竞态）
            while repl._approval_waiter is None:
                await asyncio.sleep(0)
        return value

    repl = _make_repl(orch, reader)

    async def approval_flow():
        answer = await repl.ask_approval("Choice [1-3]: ")
        answers.append(answer)

    flow = asyncio.get_running_loop().create_task(approval_flow())
    await repl.run()
    await flow

    assert answers == ["1"]
    assert orch.sent == [(0, "trigger")]  # "1" 没被当作聊天/命令处理
