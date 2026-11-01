"""Tests for the interactive CLI shell: banner, tool lines, spinners and input.

Every test drives a :class:`rich.console.Console` that writes into a
``StringIO``, and input goes through an injected fake prompt session, so no test
ever needs a real terminal. Spinner animation is exercised for lifecycle
correctness only; its frames are never asserted because Rich suppresses live
output on a non-interactive console.
"""

import os
import sys
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from rich.console import Console
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.format import format_tool_call
from mitocode.cli.stream import StreamingRenderer
from mitocode.cli.theme import MITOCODE_THEME
from mitocode.cli.ui import (
    DEFAULT_HISTORY_PATH,
    PROMPT_MESSAGE,
    SLASH_COMMANDS,
    BannerInfo,
    CliUI,
)


class _FakePromptSession:
    """Stand-in for ``PromptSession`` that replays scripted answers.

    Answers may be exceptions, which are raised instead of returned so tests can
    prove the UI propagates them untouched.

    Attributes:
        answers: Remaining answers, popped from the front.
        messages: Messages passed to each ``prompt_async`` call.
    """

    def __init__(self, answers):
        """Store the scripted answers.

        Args:
            answers: Values or exceptions returned/raised in order.
        """
        self.answers = list(answers)
        self.messages = []

    async def prompt_async(self, message="", **kwargs):
        """Record *message* and return the next scripted answer.

        Args:
            message: Prompt shown to the user.
            **kwargs: Ignored prompt options.

        Returns:
            The next scripted answer.
        """
        self.messages.append(message)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _ui(**kwargs) -> tuple[CliUI, StringIO]:
    """Build a UI writing to an in-memory console.

    Args:
        **kwargs: Extra ``CliUI`` keyword arguments.

    Returns:
        The UI and the buffer holding everything it has printed.
    """
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=100,
        force_terminal=False,
        theme=MITOCODE_THEME,
    )
    return CliUI(console, **kwargs), buffer


def test_banner_and_tool_lines_render(tmp_path):
    ui, buffer = _ui()
    ui.show_banner(
        BannerInfo("0.1.0", "model-x", tmp_path, False, 11, {"fs": "connected"})
    )
    ui.begin_tool(
        format_tool_call("Read", {"file_path": "a.py"}, "file_path", tmp_path)
    )
    ui.finish_tool(Text("12 lines"))
    text = buffer.getvalue()
    assert "model-x" in text and "fs" in text and "Read  a.py" in text
    assert "12 lines" in text


def test_banner_shows_version_workspace_tools_and_initialized_state(tmp_path):
    ui, buffer = _ui()
    ui.show_banner(BannerInfo("0.1.0", "model-x", tmp_path, True, 3, {}))
    text = buffer.getvalue()
    assert "◆ MitoCode" in text and "v0.1.0" in text
    assert "workspace" in text and tmp_path.name in text
    assert "3 built-in" in text and "0 MCP" in text
    assert "none" in text
    assert "(not initialized)" not in text


def test_banner_marks_uninitialized_workspace(tmp_path):
    ui, buffer = _ui()
    ui.show_banner(BannerInfo("0.1.0", "m", tmp_path, False, 1, {}))
    assert "(not initialized)" in buffer.getvalue()


def test_banner_lists_every_mcp_state(tmp_path):
    ui, buffer = _ui()
    servers = {"github": "connected", "filesystem": "failed", "local": "starting"}
    ui.show_banner(BannerInfo("0.1.0", "m", tmp_path, True, 1, servers))
    text = buffer.getvalue()
    assert "● github connected" in text
    assert "● filesystem failed" in text
    assert "● local starting" in text
    assert "3 MCP" in text


def test_clear_screen_before_banner_and_resume_without_pause():
    ui, buffer = _ui()
    ui.clear_screen()
    assert "MitoCode" in buffer.getvalue()
    ui.pause_live()
    ui.resume_live()
    ui.stop_thinking()


def test_tool_lines_accept_background_prefix(tmp_path):
    ui, buffer = _ui()
    ui.begin_tool(
        format_tool_call("Read", {"file_path": "a.py"}, "file_path", tmp_path),
        prefix="↳ sub ",
    )
    ui.finish_tool(Text("3 lines"), prefix="↳ sub ")
    text = buffer.getvalue()
    assert "● ↳ sub Read  a.py" in text
    assert "  ⎿ ↳ sub 3 lines" in text


def test_begin_tool_flushes_buffered_assistant_text_first(tmp_path):
    ui, buffer = _ui()
    ui.stream.feed("partial answer")
    ui.begin_tool(
        format_tool_call("Read", {"file_path": "a.py"}, "file_path", tmp_path)
    )
    text = buffer.getvalue()
    assert text.index("partial answer") < text.index("● Read  a.py")


def test_note_warn_and_error_render_plain_text():
    ui, buffer = _ui()
    ui.note("Interrupted.")
    ui.warn("model is not advertised")
    ui.error("Error: boom")
    text = buffer.getvalue()
    assert "Interrupted." in text
    assert "model is not advertised" in text
    assert "Error: boom" in text


def test_finish_turn_uses_elapsed_and_token_formatters():
    ui, buffer = _ui()
    ui.finish_turn(12.0, 1800, 3)
    assert "✻ Done in 12s · ~1.8k tokens · 3 tool calls" in buffer.getvalue()


def test_clear_screen_prints_one_line_status(tmp_path):
    ui, buffer = _ui()
    ui.show_banner(BannerInfo("0.1.0", "model-x", tmp_path, True, 1, {}))
    ui.clear_screen()
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert lines[-1].strip().startswith("model-x · ")
    assert lines[-1].strip().endswith(tmp_path.name)


def test_constructor_defaults_and_public_attributes(tmp_path):
    session = _FakePromptSession([])
    ui, _ = _ui(verbose=True, prompt_session=session, history_path=tmp_path / "h")
    assert ui.verbose is True
    assert ui.prompt_active is False
    assert isinstance(ui.stream, StreamingRenderer)
    assert DEFAULT_HISTORY_PATH == Path.home() / ".mitocode" / "history"


@pytest.mark.asyncio
async def test_ask_tracks_prompt_active_and_propagates_eof():
    session = _FakePromptSession(["y", EOFError()])
    ui, _ = _ui(prompt_session=session)
    assert await ui.ask("Allow? ") == "y"
    assert ui.prompt_active is False
    with pytest.raises(EOFError):
        await ui.read_prompt()
    assert ui.prompt_active is False


@pytest.mark.asyncio
async def test_prompt_composes_banner_tool_wrapping_and_spinner_lifecycle():
    ui, buffer = _ui()

    async def exercise() -> None:
        ui.start_thinking()
        ui.start_thinking()
        ui.stop_thinking()
        ui.begin_tool(Text("Read  a.py"))
        ui.pause_live()
        ui.resume_live()
        ui.finish_tool(Text("12 lines"))
        ui.start_thinking()
        ui.pause_live()
        ui.resume_live()
        ui.stop_thinking()

    await exercise()
    text = buffer.getvalue()
    assert "● Read  a.py" in text and "  ⎿ 12 lines" in text


@pytest.mark.asyncio
async def test_read_prompt_uses_prompt_message_and_propagates_interrupt():
    session = _FakePromptSession(["", KeyboardInterrupt()])
    ui, _ = _ui(prompt_session=session)
    assert await ui.read_prompt() == ""
    assert session.messages == [PROMPT_MESSAGE]
    with pytest.raises(KeyboardInterrupt):
        await ui.read_prompt()
    assert ui.prompt_active is False


class _FakeBuffer:
    """Minimal buffer that records the edits a key binding performs.

    Attributes:
        text: Current buffer contents.
        accepted: Whether ``validate_and_handle`` was called.
        deleted: Number of characters removed by the last deletion.
    """

    def __init__(self, text=""):
        """Initialise the buffer with *text*.

        Args:
            text: Starting contents.
        """
        self.text = text
        self.accepted = False
        self.deleted = 0

    def delete_before_cursor(self, count):
        """Remove *count* characters before the cursor.

        Args:
            count: Number of characters to drop.
        """
        self.deleted = count
        self.text = self.text[:-count] if count else self.text

    def insert_text(self, text):
        """Append *text* at the cursor.

        Args:
            text: Text to insert.
        """
        self.text += text

    def validate_and_handle(self):
        """Record that the line was submitted."""
        self.accepted = True


class _FakeEvent:
    """Key press event carrying a fake buffer and app.

    Attributes:
        current_buffer: Buffer the handler edits.
        app: Object with the ``invalidate`` callback the handlers may call.
    """

    def __init__(self, buffer):
        """Wrap *buffer* in a fake prompt_toolkit event.

        Args:
            buffer: Buffer the handler should act on.
        """
        self.current_buffer = buffer
        self.app = _FakeApp()


class _FakeApp:
    """Prompt application stub recording redraw requests.

    Attributes:
        invalidated: Number of times a redraw was requested.
    """

    def __init__(self):
        """Initialise with no redraw requests."""
        self.invalidated = 0

    def invalidate(self):
        """Record one redraw request."""
        self.invalidated += 1


def _binding_handler(bindings, keys):
    """Return the handler registered for a key sequence.

    Args:
        bindings: KeyBindings returned by ``CliUI._prompt_key_bindings``.
        keys: Tuple of string key names, such as ``("c-m",)`` for Enter.

    Returns:
        The handler callable for *keys*.
    """
    for binding in bindings.bindings:
        names = tuple(getattr(key, "value", str(key)) for key in binding.keys)
        if names == keys:
            return binding.handler
    raise AssertionError(f"no binding for {keys}")


def test_enter_binding_continues_after_backslash_and_accepts_otherwise():
    ui, _ = _ui()
    bindings = ui._prompt_key_bindings()
    enter = _binding_handler(bindings, ("c-m",))

    continued = _FakeBuffer("first line\\")
    enter(_FakeEvent(continued))
    assert continued.text == "first line\n"
    assert continued.deleted == 1 and continued.accepted is False

    submitted = _FakeBuffer("hello")
    enter(_FakeEvent(submitted))
    assert submitted.accepted is True
    assert submitted.text == "hello"


def test_enter_binding_submits_a_real_prompt_toolkit_buffer():
    """Enter must reach the accept handler of a real ``Buffer``.

    A fake buffer cannot catch a mistyped prompt_toolkit method name, so this
    regression test drives the Enter binding with a genuine
    ``prompt_toolkit.buffer.Buffer`` wrapped in a minimal event.
    """
    ui, _ = _ui()
    accepted = []
    real_buffer = Buffer(accept_handler=lambda buf: accepted.append(buf.text))
    real_buffer.text = "ship it"

    _binding_handler(ui._prompt_key_bindings(), ("c-m",))(_FakeEvent(real_buffer))

    assert accepted == ["ship it"]


def test_alt_enter_and_ctrl_j_insert_newlines_and_ctrl_l_clears():
    ui, buffer = _ui()
    bindings = ui._prompt_key_bindings()

    for keys in (("escape", "c-m"), ("c-j",)):
        target = _FakeBuffer("abc")
        _binding_handler(bindings, keys)(_FakeEvent(target))
        assert target.text == "abc\n"

    event = _FakeEvent(_FakeBuffer())
    _binding_handler(bindings, ("c-l",))(event)
    assert "MitoCode" in buffer.getvalue()
    assert event.app.invalidated == 1


def test_build_prompt_session_sets_history_completion_and_toolbar(tmp_path):
    ui, _ = _ui()
    ui.show_banner(BannerInfo("0.1.0", "model-x", tmp_path, True, 1, {}))
    history_path = tmp_path / ".mitocode" / "history"
    session = ui.build_prompt_session(history_path)
    assert isinstance(session, PromptSession)
    assert history_path.parent.is_dir()
    assert history_path.parent.stat().st_mode & 0o077 == 0
    assert "/help" in session.completer.words
    toolbar = session.bottom_toolbar
    rendered = toolbar() if callable(toolbar) else toolbar
    assert "model-x" in str(rendered) and "/help" in str(rendered)


# ---------------------------------------------------------------------------
# Reasoning live region
# ---------------------------------------------------------------------------


def test_feed_reasoning_opens_region_and_inherits_thinking_clock():
    """首个 reasoning delta 将 thinking spinner 切换为 reasoning 且保留计时起点."""
    ui, _ = _ui()
    ui.start_thinking()
    started = ui._live_started
    ui.feed_reasoning("step one")
    assert ui._live_kind == "reasoning"
    assert ui._live_started == started
    ui.end_reasoning()


def test_end_reasoning_prints_one_summary_and_is_idempotent():
    """收起恰好打印一行 Thought for 摘要；重复调用不再打印且缓冲清空."""
    ui, buffer = _ui()
    ui.feed_reasoning("hmm")
    ui.feed_reasoning(" more")
    ui.end_reasoning()
    ui.end_reasoning()
    assert buffer.getvalue().count("Thought for") == 1
    assert ui._reasoning_text == ""
    assert ui._live_kind is None


def test_empty_reasoning_never_opens_region_or_prints_summary():
    """空 delta 完全被忽略：不开区域、不打印摘要."""
    ui, buffer = _ui()
    ui.feed_reasoning("")
    assert ui._live_kind is None
    ui.end_reasoning()
    assert "Thought for" not in buffer.getvalue()


def test_reasoning_buffer_is_bounded():
    """缓冲最多保留最近 2000 字符."""
    ui, _ = _ui()
    ui.feed_reasoning("x" * 3000)
    assert len(ui._reasoning_text) == 2000
    ui.end_reasoning()


def test_reasoning_region_without_prior_thinking_gets_fresh_clock():
    """无 thinking spinner 时 reasoning 区域自起计时."""
    ui, _ = _ui()
    ui.feed_reasoning("direct reasoning, no thinking spinner")
    assert ui._live_kind == "reasoning"
    assert ui._live_started is not None
    ui.end_reasoning()


def test_background_begin_tool_keeps_main_thinking_and_reasoning_region():
    """后台工具头不抢占主会话的 thinking/reasoning 区域；空闲时仍给 spinner."""
    ui, _ = _ui()

    ui.feed_reasoning("thinking hard")
    ui.begin_tool(_tool_line(), prefix="↳ bg ")
    assert ui._live_kind == "reasoning"
    ui.end_reasoning()

    ui.start_thinking()
    ui.begin_tool(_tool_line(), prefix="↳ bg ")
    assert ui._live_kind == "thinking"
    ui.stop_thinking()

    ui.begin_tool(_tool_line(), prefix="↳ bg ")
    assert ui._live_kind == "tool"
    ui.finish_tool(Text("ok"))
    assert ui._live_kind is None


def _tool_line() -> Text:
    """Build a minimal tool header for begin_tool calls."""
    return Text("Read a.py")


def test_slash_commands_include_effort():
    """/effort 出现在输入补全命令列表中."""
    assert "effort" in SLASH_COMMANDS
