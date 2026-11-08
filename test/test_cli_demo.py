"""Tests for the CLI demo player (``cli/demo.py``).

Covers the playback controls, the key sources and the read-only replay loop
of ``/demo``.  No test touches a real terminal: keys come from
:class:`FakeKeySource`, delays are injected away and UI output is read from a
``StringIO`` console.
"""

import asyncio
import os
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from rich.console import Console

from mitocode.cli.demo import (
    DEMO_SPEEDS,
    DEMO_TICK,
    DemoControls,
    DemoPlayer,
    FakeKeySource,
    TerminalKeySource,
    _map_key,
)
from mitocode.cli.theme import MITOCODE_THEME
from mitocode.cli.ui import CliUI

# noinspection PyProtectedMember
from mitocode.core.agent.agent_session import (
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)

ERROR_COLOR_ANSI = "\x1b[38;2;224;108;117m"
"""Truecolor escape for the theme's ``mito.error`` colour (#E06C75)."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ui(history_path: Path = Path("cli-history")) -> "tuple[CliUI, StringIO]":
    """Build a :class:`CliUI` writing into a ``StringIO`` console.

    Args:
        history_path: Prompt history file handed to :class:`CliUI`.

    Returns:
        ``(ui, buffer)`` for assertions.
    """
    buffer = StringIO()
    console = Console(
        file=buffer, width=200, force_terminal=False, theme=MITOCODE_THEME
    )
    ui = CliUI(console, history_path=history_path)
    return ui, buffer


# ---------------------------------------------------------------------------
# Task 16: DemoControls
# ---------------------------------------------------------------------------


def test_demo_speeds_constant_and_default_index_is_1x():
    """速度档位常量固定，默认索引 1 即 1x."""
    assert DEMO_SPEEDS == (0.5, 1, 2, 4, 8, 16)
    controls = DemoControls()
    assert controls.speed_index == 1
    assert controls.speed == 1


def test_demo_controls_defaults():
    """初始状态：未暂停、未退出、默认 1x."""
    controls = DemoControls()
    assert controls.paused is False
    assert controls.quit_requested is False
    assert controls.speed == 1


def test_demo_controls_right_moves_up_and_stops_at_boundary():
    """right 逐档升速，到 16x 后不越界."""
    controls = DemoControls()
    controls.handle_key("right")
    assert controls.speed == 2
    controls.handle_key("right")
    assert controls.speed == 4
    for _ in range(4):
        controls.handle_key("right")
    assert controls.speed == 16


def test_demo_controls_left_moves_down_and_stops_at_boundary():
    """left 逐档降速，到 0.5x 后不越界."""
    controls = DemoControls()
    controls.handle_key("left")
    assert controls.speed == 0.5
    controls.handle_key("left")
    assert controls.speed == 0.5


def test_demo_controls_space_toggles_pause():
    """space 在暂停与继续之间切换."""
    controls = DemoControls()
    controls.handle_key("space")
    assert controls.paused is True
    controls.handle_key("space")
    assert controls.paused is False


def test_demo_controls_quit_keys_stop_playback():
    """q、esc、ctrl+c 都置退出标记."""
    for key in ("q", "esc", "ctrl+c"):
        controls = DemoControls()
        controls.handle_key(key)
        assert controls.quit_requested is True, key


def test_demo_controls_ignores_unknown_keys():
    """未知按键（含 None）不改变任何状态."""
    controls = DemoControls()
    for key in (None, "x", "Q", "enter"):
        controls.handle_key(key)
    assert controls.speed == 1
    assert controls.paused is False
    assert controls.quit_requested is False


def test_demo_controls_change_note_reports_speed_change():
    """调速后 change_note 返回一次 Speed note，再次调用为空."""
    controls = DemoControls()
    controls.handle_key("right")
    assert controls.change_note() == "Speed 2x"
    assert controls.change_note() == ""


def test_demo_controls_change_note_reports_half_speed():
    """0.5x 档的 note 显示为 Speed 0.5x."""
    controls = DemoControls()
    controls.handle_key("left")
    assert controls.speed == 0.5
    assert controls.change_note() == "Speed 0.5x"


def test_demo_controls_change_note_reports_pause_and_resume():
    """space 暂停/继续各产生一条 note，各消费一次."""
    controls = DemoControls()
    controls.handle_key("space")
    assert controls.change_note() == "Paused · space to resume"
    controls.handle_key("space")
    assert controls.change_note() == "Resumed."
    assert controls.change_note() == ""


def test_demo_controls_change_note_empty_without_change():
    """到边界调速无变化时不产生 note."""
    controls = DemoControls()
    for _ in range(5):  # 1x -> 16x，最后一次在边界上无变化
        controls.handle_key("right")
        assert controls.change_note() != "" or controls.speed == 16
    assert controls.speed == 16
    controls.handle_key("right")  # boundary, no change
    assert controls.change_note() == ""
    controls2 = DemoControls()
    controls2.handle_key("q")  # quit produces no note
    assert controls2.change_note() == ""


# ---------------------------------------------------------------------------
# Task 17: KeySource protocol, TerminalKeySource, FakeKeySource
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\x1b[D", "left"),
        ("\x1b[C", "right"),
        (" ", "space"),
        ("q", "q"),
        ("\x1b", "esc"),
        ("\x03", "ctrl+c"),
    ],
)
def test_terminal_key_map_translates_known_raw_keys(raw, expected):
    """终端原始字节序列按约定映射为规范键名."""
    assert _map_key(raw) == expected


@pytest.mark.parametrize("raw", ["", "x", "Q", "\x1b[A", "\x1bOB", "\n", "\x1b[Dx"])
def test_terminal_key_map_returns_none_for_unknown_raw_keys(raw):
    """未知输入（含空串、其他转义序列）映射为 None."""
    assert _map_key(raw) is None


def test_fake_key_source_yields_keys_in_order_then_none():
    """FakeKeySource 按预排顺序吐键，耗尽后恒返回 None."""
    source = FakeKeySource(["right", " ", "q"])
    assert source.poll(0.0) == "right"
    assert source.poll(0.0) == " "
    assert source.poll(0.0) == "q"
    assert source.poll(0.0) is None
    assert source.poll(0.0) is None


def test_fake_key_source_empty_sequence_returns_none_immediately():
    """空预排序列直接返回 None."""
    assert FakeKeySource([]).poll(0.5) is None


def test_terminal_key_source_close_is_idempotent_without_tty():
    """无 TTY 环境下构造与重复 close 都不抛异常."""
    source = TerminalKeySource()
    source.close()
    source.close()


# ---------------------------------------------------------------------------
# Task 18: DemoPlayer + DemoTextRenderer
# ---------------------------------------------------------------------------


class _CountingOrchestrator:
    """Fake orchestrator that counts every ``send_message`` call."""

    def __init__(self) -> None:
        """Start with a zero call counter."""
        self.send_message_calls = 0

    async def send_message(self, *args: Any, **kwargs: Any) -> None:
        """Count one call.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        self.send_message_calls += 1


class _FakeToolManager:
    """Fake registry exposing ``display_param`` for one tool name."""

    def get_spec(self, name: str) -> Any:
        """Return a spec with ``display_param`` for ``Widget``.

        Args:
            name: Tool name looked up.

        Returns:
            A spec-like object, or ``None`` for unknown names.
        """
        if name == "Widget":
            return SimpleNamespace(display_param="target")
        return None


def _zero_sleeper(delays: List[float]):
    """Build an async sleeper that records delays and never waits.

    Args:
        delays: List the observed delays are appended to.

    Returns:
        An async callable matching the ``sleeper`` signature.
    """

    async def sleeper(delay: float) -> None:
        """Record one delay without sleeping.

        Args:
            delay: Seconds the production sleeper would wait.
        """
        delays.append(delay)

    return sleeper


def _tool_call_block(call_id: str, arguments: str) -> "_ToolCallMessageBlock":
    """Build a finished tool call block.

    Args:
        call_id: Tool call id used for result pairing.
        arguments: Raw JSON argument string streamed into the block.

    Returns:
        The finished block.
    """
    block = _ToolCallMessageBlock(
        "Read" if "file_path" in arguments else "Widget", call_id
    )
    block.append_argument_chunk(arguments)
    block.finish()
    return block


def _player(
    ui: CliUI,
    blocks: List[Any],
    keys: List[str],
    *,
    title: str = "T",
    root: Any = None,
    tool_manager: Any = None,
    sleeper: Any = None,
    orchestrator: Any = None,
    key_source: Any = None,
):
    """Build a :class:`DemoPlayer` over a fake session.

    Args:
        ui: UI whose console receives the replay.
        blocks: Message blocks of the fake session.
        keys: Scripted key sequence for :class:`FakeKeySource`.
        title: Session title shown in the header note.
        root: Workspace root passed through to the formatters.
        tool_manager: Optional registry for ``display_param`` lookups.
        sleeper: Optional async sleeper override.
        orchestrator: Optional orchestrator stored on the fake session.
        key_source: Optional pre-built key source overriding *keys*.

    Returns:
        ``(player, session)`` so tests can assert read-only behaviour.
    """
    session = SimpleNamespace(messages=blocks, orchestrator=orchestrator)
    if key_source is None:
        # The player now holds the console behind a "press any key" prompt
        # after playback; append the confirmation key so tests return
        # immediately instead of exhausting the poll budget.
        keys = list(keys) + ["x"]
        key_source = FakeKeySource(keys)
    player = DemoPlayer(
        ui,
        session,
        title=title,
        key_source=key_source,
        tool_manager=tool_manager,
        root=root,
        sleeper=sleeper,
    )
    return player, session


_MD_SAMPLE = (
    "## Plan\n"
    "\n"
    "Use **bold** and `code` words.\n"
    "\n"
    "```python\n"
    "print(1)\n"
    "```\n"
)
"""Assistant sample exercising headings, inline markdown and a code fence."""


def test_render_session_history_renders_markdown(tmp_path):
    """回显与实时对话一致：标题/行内 markdown/代码围栏都被渲染."""
    from mitocode.cli.demo import render_session_history

    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("assistant", _MD_SAMPLE)]
    session = SimpleNamespace(messages=blocks)
    render_session_history(ui, session, title="T", root=tmp_path)
    out = buffer.getvalue()
    assert "##" not in out, "标题标记被渲染掉"
    assert "Plan" in out
    assert "**" not in out, "行内加粗标记被渲染掉"
    assert "bold" in out and "code" in out
    assert "```" not in out, "围栏标记被渲染掉"
    assert "print(1)" in out
    assert "python" in out, "围栏语言标记按实时规则显示"
    assert len(session.messages) == 1


@pytest.mark.asyncio
async def test_demo_player_renders_markdown(tmp_path):
    """/demo 回放同样渲染 markdown，而不是原样输出."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("assistant", _MD_SAMPLE)]
    player, session = _player(ui, blocks, [], root=tmp_path)
    await player.run()
    out = buffer.getvalue()
    assert "##" not in out, "标题标记被渲染掉"
    assert "Plan" in out
    assert "**" not in out, "行内加粗标记被渲染掉"
    assert "bold" in out and "code" in out
    assert "```" not in out, "围栏标记被渲染掉"
    assert "print(1)" in out
    assert "python" in out, "围栏语言标记按实时规则显示"
    assert len(session.messages) == 1


@pytest.mark.asyncio
async def test_demo_player_replays_user_assistant_reasoning_and_tools(tmp_path):
    """user 一次性、reasoning dim、assistant 分块、工具头与结果按序重建."""
    ui, buffer = _ui(tmp_path / "history")
    reasoning = _PlainTextMessageBlock("assistant", "Answer body")
    reasoning.reasoning_content = "thinking hard"
    read_output = "\n".join(f"line {index}" for index in range(120))
    blocks = [
        _PlainTextMessageBlock("user", "hello demo"),
        reasoning,
        _tool_call_block("call_1", '{"file_path": "/tmp/demo/a.py"}'),
        _ToolResultMessageBlock("call_1", read_output),
    ]
    player, _session = _player(
        ui, blocks, [], root=tmp_path, tool_manager=_FakeToolManager()
    )
    await player.run()
    output = buffer.getvalue()
    assert "❯ hello demo" in output
    assert "thinking hard" in output
    assert "Answer body" in output
    assert "● Read" in output
    assert "/tmp/demo/a.py" in output.replace("\\", "/")
    assert "⎿ 120 lines" in output
    # reasoning dim 行在 assistant 正文之前
    assert output.index("thinking hard") < output.index("Answer body")


@pytest.mark.asyncio
async def test_demo_player_error_result_gets_error_style(tmp_path):
    """运行时报失败的结果保留原文并套用 mito.error 颜色."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=200,
        force_terminal=True,
        color_system="truecolor",
        theme=MITOCODE_THEME,
    )
    ui = CliUI(console, history_path=tmp_path / "history")
    blocks = [
        _ToolResultMessageBlock("call_9", "error: something broke badly", success=False)
    ]
    player, _session = _player(ui, blocks, [], root=tmp_path)
    await player.run()
    output = buffer.getvalue()
    assert "error: something broke badly" in output
    assert ERROR_COLOR_ANSI in output


@pytest.mark.asyncio
async def test_demo_player_skips_system_and_unfinished_tool_call(tmp_path):
    """system 块与未 finished 的 tool_call 块都不出现在画面上."""
    ui, buffer = _ui(tmp_path / "history")
    unfinished = _ToolCallMessageBlock("Bash", "call_2")
    unfinished.append_argument_chunk('{"command": "rm -rf /"}')
    blocks = [
        _PlainTextMessageBlock("system", "SECRET SYSTEM PROMPT"),
        _PlainTextMessageBlock("user", "hi"),
        unfinished,
    ]
    player, _session = _player(ui, blocks, [], root=tmp_path)
    await player.run()
    output = buffer.getvalue()
    assert "SECRET SYSTEM PROMPT" not in output
    assert "Bash" not in output
    assert "rm -rf" not in output


@pytest.mark.asyncio
async def test_demo_player_result_without_paired_header_still_prints(tmp_path):
    """找不到配对工具头的 tool_result 照常打印."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_ToolResultMessageBlock("call_404", "orphan output")]
    player, _session = _player(ui, blocks, [], root=tmp_path)
    await player.run()
    assert "⎿" in buffer.getvalue()
    assert "orphan output" in buffer.getvalue()


@pytest.mark.parametrize("key", ["q", "esc", "ctrl+c"])
@pytest.mark.asyncio
async def test_demo_player_quit_keys_end_playback_with_finished_note(tmp_path, key):
    """q/esc/ctrl+c 提前结束并打印 Demo finished."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("assistant", "a" * 40)]
    player, _session = _player(ui, blocks, [key], root=tmp_path)
    await player.run()
    assert "Demo finished." in buffer.getvalue()


@pytest.mark.asyncio
async def test_demo_player_is_read_only(tmp_path):
    """回放不调用 orchestrator、不修改 session.messages."""
    ui, _buffer = _ui(tmp_path / "history")
    blocks = [
        _PlainTextMessageBlock("user", "hi"),
        _PlainTextMessageBlock("assistant", "a" * 40),
    ]
    orchestrator = _CountingOrchestrator()
    player, session = _player(ui, blocks, [], root=tmp_path, orchestrator=orchestrator)
    await player.run()
    assert orchestrator.send_message_calls == 0
    assert session.messages is blocks
    assert len(session.messages) == 2


@pytest.mark.asyncio
async def test_demo_player_header_note_shows_title_and_hints(tmp_path):
    """头部 note 含标题、默认 1x 与按键提示."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("user", "hi")]
    player, _session = _player(ui, blocks, [], title="My Demo", root=tmp_path)
    await player.run()
    assert 'Demo · "My Demo" · 1x · ←/→ speed · space pause · q quit' in (
        buffer.getvalue()
    )


@pytest.mark.asyncio
async def test_demo_player_right_key_prints_speed_note_and_updates_tick(tmp_path):
    """right 调速打印 Speed 2x，播放 poll 按 DEMO_TICK/2 计时."""
    from mitocode.cli.demo import _ANY_KEY_POLL

    class _LateConfirmSource(FakeKeySource):
        """Fake source whose confirmation key appears only after playback."""

        def __init__(self, keys, after_polls):
            super().__init__(keys)
            self._after_polls = after_polls
            self._poll_count = 0

        def poll(self, timeout):
            self._poll_count += 1
            self.timeouts.append(timeout)
            if self._keys:
                return self._keys.pop(0)
            if self._poll_count > self._after_polls:
                return "x"
            return None

    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("assistant", "a" * 64)]
    keys = _LateConfirmSource(["right"], after_polls=4)
    player, _session = _player(ui, blocks, [], root=tmp_path, key_source=keys)
    await player.run()
    output = buffer.getvalue()
    assert "Speed 2x" in output
    # 4 个 chunk 的播放 poll：第一个按默认 1x，其余按 2x；最后一个是
    # "press any key" 确认 poll（超时 _ANY_KEY_POLL）。
    assert keys.timeouts[0] == DEMO_TICK / 1
    assert keys.timeouts[1:4] == [DEMO_TICK / 2] * 3
    assert keys.timeouts[4] == _ANY_KEY_POLL


@pytest.mark.asyncio
async def test_demo_player_space_toggles_pause_with_notes(tmp_path):
    """space 暂停与恢复各打印一条 note，且回放能继续完成."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("assistant", "a" * 40)]
    player, _session = _player(ui, blocks, ["space", "space"], root=tmp_path)
    await player.run()
    output = buffer.getvalue()
    assert "Paused · space to resume" in output
    assert "Resumed." in output
    assert "Demo finished." in output
    assert "a" * 40 in output


@pytest.mark.asyncio
async def test_demo_player_closes_key_source_once(tmp_path):
    """回放结束后 close 恰好调用一次."""

    class _ClosableFake(FakeKeySource):
        """Fake that counts ``close`` calls."""

        def __init__(self, keys: List[str]) -> None:
            """Store the sequence and start the counter.

            Args:
                keys: Key names to hand out in order.
            """
            super().__init__(keys)
            self.close_calls = 0

        def close(self) -> None:
            """Count one close call."""
            self.close_calls += 1

    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("user", "hi")]
    session = SimpleNamespace(messages=blocks)
    closable = _ClosableFake([])
    player = DemoPlayer(ui, session, title="T", key_source=closable, root=tmp_path)
    await player.run()
    assert closable.close_calls == 1
    assert "Demo finished." in buffer.getvalue()


# ---------------------------------------------------------------------------
# Restored-session echo (startup history replay)
# ---------------------------------------------------------------------------


def test_render_session_history_prints_full_transcript(tmp_path):
    """回显渲染 user/assistant/推理/工具头/工具结果，头部含标题与消息数."""
    from mitocode.cli.demo import render_session_history

    ui, buffer = _ui(tmp_path / "history")
    call = _ToolCallMessageBlock("Read", "call_1")
    call.append_argument_chunk('{"file_path": "src/a.py"}')
    call.finish()
    blocks = [
        _PlainTextMessageBlock("system", "You are MitoCode."),
        _PlainTextMessageBlock("user", "hello there"),
        _PlainTextMessageBlock("assistant", "I will read the file."),
        blocks_reasoning_assistant("Thinking hard", "And here is the answer."),
        call,
        _ToolResultMessageBlock("call_1", "120 lines"),
    ]
    session = SimpleNamespace(messages=blocks)
    render_session_history(
        ui, session, title="Old chat", tool_manager=None, root=tmp_path
    )
    out = buffer.getvalue()
    assert 'Restored · "Old chat"' in out
    assert "❯ hello there" in out
    assert "I will read the file." in out
    assert "Thinking hard" in out
    assert "And here is the answer." in out
    assert "Read" in out
    # Read 结果按 format_result 的既有规则折叠为行数预览。
    assert "⎿ 1 lines" in out
    assert "You are MitoCode." not in out, "system 块不回显"
    assert len(session.messages) == 6, "回显不修改 messages"


def blocks_reasoning_assistant(reasoning: str, message: str):
    """Build an assistant block carrying reasoning plus text.

    Args:
        reasoning: Reasoning text replayed before the answer.
        message: The assistant answer text.

    Returns:
        A ``_PlainTextMessageBlock`` with ``reasoning_content`` set.
    """
    block = _PlainTextMessageBlock("assistant", message)
    block.reasoning_content = reasoning
    return block


def test_render_session_history_skips_unfinished_and_mismatched(tmp_path):
    """未 finished 的工具调用跳过；无配对头的结果仍打印."""
    from mitocode.cli.demo import render_session_history

    ui, buffer = _ui(tmp_path / "history")
    unfinished = _ToolCallMessageBlock("Bash", "call_x")
    orphan_result = _ToolResultMessageBlock("call_lost", "orphan output")
    blocks = [
        _PlainTextMessageBlock("user", "go"),
        unfinished,
        orphan_result,
    ]
    session = SimpleNamespace(messages=blocks)
    render_session_history(ui, session, title="T", root=tmp_path)
    out = buffer.getvalue()
    assert "Bash" not in out, "未完成的工具调用不打印"
    assert "orphan output" in out
    assert "❯ go" in out


def test_render_session_history_empty_history_prints_only_header(tmp_path):
    """空历史（仅 system 块）只打印头部 note，不抛异常."""
    from mitocode.cli.demo import render_session_history

    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("system", "You are MitoCode.")]
    session = SimpleNamespace(messages=blocks)
    render_session_history(ui, session, title="Empty", root=tmp_path)
    out = buffer.getvalue()
    assert 'Restored · "Empty"' in out
    assert "❯" not in out


@pytest.mark.asyncio
async def test_demo_player_waits_for_any_key_after_finish(tmp_path):
    """回放结束后提示按任意键，拿到按键才返回."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("user", "hi")]
    keys = FakeKeySource(["x"])
    player, _session = _player(ui, blocks, [], root=tmp_path, key_source=keys)
    await player.run()
    output = buffer.getvalue()
    assert "Press any key to return" in output
    assert keys._keys == [], "等待阶段消费了确认按键"


@pytest.mark.asyncio
async def test_demo_player_closes_key_source_when_playback_raises(tmp_path):
    """回放中抛异常时也必须 close，否则终端停在 raw 模式（无回显、Ctrl+C 失效）."""

    class _ClosableFake(FakeKeySource):
        """Fake that counts ``close`` calls."""

        def __init__(self, keys: List[str]) -> None:
            super().__init__(keys)
            self.close_calls = 0

        def close(self) -> None:
            """Count one close call."""
            self.close_calls += 1

    class _ExplodingUI:
        """UI whose note() raises the way a broken renderer would."""

        verbose = False

        class console:
            @staticmethod
            def print(*args, **kwargs):
                return None

        def note(self, message):
            """Fail on the first note."""
            raise RuntimeError("render boom")

    closable = _ClosableFake([])
    session = SimpleNamespace(messages=[_PlainTextMessageBlock("user", "hi")])
    player = DemoPlayer(
        _ExplodingUI(), session, title="T", key_source=closable, root=tmp_path
    )

    with pytest.raises(RuntimeError):
        await player.run()

    assert closable.close_calls == 1, (
        "the key source must be closed even when playback fails; otherwise the "
        "terminal keeps the raw attributes the demo installed"
    )


@pytest.mark.asyncio
async def test_demo_player_closes_key_source_when_cancelled(tmp_path):
    """回放被取消时也必须 close."""

    class _ClosableFake(FakeKeySource):
        """Fake that counts ``close`` calls."""

        def __init__(self, keys: List[str]) -> None:
            super().__init__(keys)
            self.close_calls = 0

        def close(self) -> None:
            """Count one close call."""
            self.close_calls += 1

    closable = _ClosableFake([])
    session = SimpleNamespace(messages=[_PlainTextMessageBlock("user", "hi")])
    player = DemoPlayer(
        _ui(tmp_path / "history")[0],
        session,
        title="T",
        key_source=closable,
        root=tmp_path,
    )

    task = asyncio.ensure_future(player.run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (
        closable.close_calls == 1
    ), "a cancelled playback must still restore the terminal"


def test_demo_module_imports_without_termios(monkeypatch):
    """没有 termios 的平台（如 Windows）上 CLI 必须仍能导入.

    ``mitocode.cli.app`` imports this module unconditionally, so a
    top-level ``import termios`` made the whole CLI unusable on Windows --
    even for ``--help`` and ``--version``, which never touch the demo.
    """
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def _no_termios(name, *args, **kwargs):
        if name == "termios":
            raise ImportError("No module named 'termios'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_termios)
    monkeypatch.delitem(sys.modules, "mitocode.cli.demo", raising=False)

    module = importlib.import_module("mitocode.cli.demo")

    assert module is not None
    # The key source must degrade instead of raising.
    source = module.TerminalKeySource()
    source.close()
    source.close()


def test_terminal_key_source_without_termios_yields_no_keys(monkeypatch):
    """没有 termios 时按键源不得抛异常，只是不产出按键."""
    import mitocode.cli.demo as demo_module

    monkeypatch.setattr(demo_module, "termios", None, raising=False)

    source = demo_module.TerminalKeySource()
    assert source.poll(0.01) is None
    source.close()
