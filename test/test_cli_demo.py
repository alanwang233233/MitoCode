"""Tests for the CLI demo player (``cli/demo.py``).

Covers the playback controls, the key sources and the read-only replay loop
of ``/demo``.  No test touches a real terminal: keys come from
:class:`FakeKeySource`, delays are injected away and UI output is read from a
``StringIO`` console.
"""

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
    DemoTextRenderer,
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


def test_demo_speeds_constant_and_default_index_is_4x():
    """速度档位常量固定，默认索引 3 即 4x."""
    assert DEMO_SPEEDS == (0.5, 1, 2, 4, 8, 16)
    controls = DemoControls()
    assert controls.speed_index == 3
    assert controls.speed == 4


def test_demo_controls_defaults():
    """初始状态：未暂停、未退出、默认速度."""
    controls = DemoControls()
    assert controls.paused is False
    assert controls.quit_requested is False
    assert controls.speed == 4


def test_demo_controls_right_moves_up_and_stops_at_boundary():
    """right 逐档升速，到 16x 后不越界."""
    controls = DemoControls()
    controls.handle_key("right")
    assert controls.speed == 8
    controls.handle_key("right")
    assert controls.speed == 16
    controls.handle_key("right")
    controls.handle_key("right")
    assert controls.speed == 16


def test_demo_controls_left_moves_down_and_stops_at_boundary():
    """left 逐档降速，到 0.5x 后不越界."""
    controls = DemoControls()
    controls.handle_key("left")
    assert controls.speed == 2
    controls.handle_key("left")
    controls.handle_key("left")
    assert controls.speed == 0.5
    controls.handle_key("left")
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
    assert controls.speed == 4
    assert controls.paused is False
    assert controls.quit_requested is False


def test_demo_controls_change_note_reports_speed_change():
    """调速后 change_note 返回一次 Speed note，再次调用为空."""
    controls = DemoControls()
    controls.handle_key("right")
    assert controls.change_note() == "Speed 8x"
    assert controls.change_note() == ""


def test_demo_controls_change_note_reports_half_speed():
    """0.5x 档的 note 显示为 Speed 0.5x."""
    controls = DemoControls()
    for _ in range(3):
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
    controls.handle_key("right")  # 8
    assert controls.change_note() != ""
    controls.handle_key("right")  # 16
    assert controls.change_note() != ""
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
    player = DemoPlayer(
        ui,
        session,
        title=title,
        key_source=key_source or FakeKeySource(list(keys)),
        tool_manager=tool_manager,
        root=root,
        sleeper=sleeper,
    )
    return player, session


def test_demo_text_renderer_prints_chunks_without_trailing_newline():
    """chunk 连续打印且不自动换行."""
    buffer = StringIO()
    console = Console(file=buffer, width=200, theme=MITOCODE_THEME)
    renderer = DemoTextRenderer(console)
    renderer.print_chunk("hello ")
    renderer.print_chunk("world")
    assert buffer.getvalue() == "hello world"


def test_demo_text_renderer_finish_adds_newline_once():
    """finish 补一次换行；以换行结尾的文本不再补."""
    buffer = StringIO()
    console = Console(file=buffer, width=200, theme=MITOCODE_THEME)
    renderer = DemoTextRenderer(console)
    renderer.print_chunk("hi")
    renderer.finish()
    assert buffer.getvalue() == "hi\n"
    renderer.finish()
    assert buffer.getvalue() == "hi\n"

    buffer2 = StringIO()
    renderer2 = DemoTextRenderer(Console(file=buffer2, width=200, theme=MITOCODE_THEME))
    renderer2.print_chunk("hi\n")
    renderer2.finish()
    assert buffer2.getvalue() == "hi\n"


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
    assert "/tmp/demo/a.py" in output
    assert "⎿ 120 lines" in output
    # reasoning dim 行在 assistant 正文之前
    assert output.index("thinking hard") < output.index("Answer body")


@pytest.mark.asyncio
async def test_demo_player_error_result_gets_error_style(tmp_path):
    """error: 开头的结果内容保留并套用 mito.error 颜色."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=200,
        force_terminal=True,
        color_system="truecolor",
        theme=MITOCODE_THEME,
    )
    ui = CliUI(console, history_path=tmp_path / "history")
    blocks = [_ToolResultMessageBlock("call_9", "error: something broke badly")]
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
    """头部 note 含标题、默认 4x 与按键提示."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("user", "hi")]
    player, _session = _player(ui, blocks, [], title="My Demo", root=tmp_path)
    await player.run()
    assert 'Demo · "My Demo" · 4x · ←/→ speed · space pause · q quit' in (
        buffer.getvalue()
    )


@pytest.mark.asyncio
async def test_demo_player_right_key_prints_speed_note_and_updates_tick(tmp_path):
    """right 调速打印 Speed 8x，后续 poll 收到 DEMO_TICK/8 的超时."""
    ui, buffer = _ui(tmp_path / "history")
    blocks = [_PlainTextMessageBlock("assistant", "a" * 64)]
    keys = FakeKeySource(["right"])
    player, _session = _player(ui, blocks, [], root=tmp_path, key_source=keys)
    await player.run()
    output = buffer.getvalue()
    assert "Speed 8x" in output
    # 第一次 poll 发生在按键生效前，仍按默认 4x 计时
    assert keys.timeouts[0] == DEMO_TICK / 4
    assert keys.timeouts[1:] == [DEMO_TICK / 8] * (len(keys.timeouts) - 1)


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
