"""Tests for the CLI runtime state and the slash-command registry.

Every test drives the real :class:`CliUI` with a ``StringIO`` console and a
scripted prompt session, so nothing here needs a terminal. Orchestrators,
providers, MCP managers and approvers are small fakes defined in this file;
tool registration uses the real :class:`ToolManager`.
"""

import os
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.args import CliArgs
from mitocode.cli.commands import COMMANDS, Command, CommandResult, dispatch
from mitocode.cli.state import CliState, CommandContext
from mitocode.cli.theme import MITOCODE_THEME
from mitocode.cli.ui import CliUI
from mitocode.core.skills.skill import SkillRoot
from mitocode.core.skills.skills_manager import SkillsManager
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace

EXPECTED_COMMANDS = {
    "help",
    "status",
    "model",
    "effort",
    "clear",
    "agents",
    "tools",
    "skills",
    "mcp",
    "permissions",
    "verbose",
    "cost",
    "log",
    "exit",
    "quit",
}


class _FakePromptSession:
    """Replay scripted answers through the ``CliUI.ask`` seam."""

    def __init__(self, answers):
        """Store the answers to return in order.

        Args:
            answers: Answers returned by successive ``prompt_async`` calls.
        """
        self.answers = list(answers)
        self.messages = []

    async def prompt_async(self, message="", **kwargs):
        """Record the prompt and return the next scripted answer.

        Args:
            message: Prompt text shown to the user.
            **kwargs: Ignored prompt options.

        Returns:
            The next scripted answer.
        """
        self.messages.append(message)
        return self.answers.pop(0)


class _FakeOrchestrator:
    """Track session creation and destruction without running agents."""

    def __init__(self, sessions=None, snapshot=None, next_id=7):
        """Build the fake with an optional session map.

        Args:
            sessions: Agent id to session mapping. Defaults to a bare ``Main``.
            snapshot: Tree snapshot returned by ``get_tree_snapshot``.
            next_id: Id returned by ``create_agent``.
        """
        self.sessions = sessions or {0: SimpleNamespace(agent_id=0)}
        self.snapshot = snapshot or [
            {
                "id": 0,
                "name": "Main",
                "state": "idle",
                "parent_id": None,
                "sub_agent_ids": [],
            }
        ]
        self.next_id = next_id
        self.destroyed = []
        self.created = 0
        self.created_kwargs = []

    def get_tree_snapshot(self):
        """Return the configured tree snapshot.

        Returns:
            The list of agent dictionaries.
        """
        return self.snapshot

    def get_session(self, agent_id):
        """Return the session registered under *agent_id*.

        Args:
            agent_id: Numeric agent id.

        Returns:
            The session or ``None``.
        """
        return self.sessions.get(agent_id)

    async def destroy_agent(self, agent_id):
        """Record and forget one session.

        Args:
            agent_id: Session being destroyed.
        """
        self.destroyed.append(agent_id)
        self.sessions.pop(agent_id, None)

    async def create_agent(self, **kwargs):
        """Record a creation request and return the next id.

        Args:
            **kwargs: Session options such as ``name`` and ``description``.

        Returns:
            The configured next agent id.
        """
        self.created += 1
        self.created_kwargs.append(kwargs)
        return self.next_id


class _FakeSession:
    """A session exposing just the attributes the commands read."""

    def __init__(self, messages=None, name="Main"):
        """Store the message history and display name.

        Args:
            messages: Internal message blocks to expose.
            name: Session display name.
        """
        self.messages = list(messages or [])
        self.name = name
        self.refresh_calls = 0

    def refresh_tools(self):
        """Count one tool-registry refresh."""
        self.refresh_calls += 1


class _FakeMcp:
    """Record refreshes and expose a fixed status mapping."""

    def __init__(self, statuses=None):
        """Store the status snapshot.

        Args:
            statuses: Server name to status object mapping.
        """
        self.refreshed = []
        self._statuses = statuses or {}

    async def refresh(self, name=None):
        """Record a refresh request.

        Args:
            name: Server to refresh, or ``None`` for all.
        """
        self.refreshed.append(name)

    def status(self):
        """Return the configured status mapping.

        Returns:
            Mapping of server name to status object.
        """
        return self._statuses


class _FakeApprover:
    """Stand-in approver exposing the cache controls commands use."""

    policy = SimpleNamespace(value="ask")

    def __init__(self):
        """Start with one cached always-allow entry."""
        self.cached = [("Bash", "ls")]
        self.reset_calls = 0

    def cached_requests(self):
        """Return the cached requests.

        Returns:
            A set of ``(action, detail)`` pairs.
        """
        return set(self.cached)

    def reset_cache(self):
        """Clear the cache and count the call."""
        self.reset_calls += 1
        self.cached.clear()


class _AsyncWorkspace:
    """In-memory workspace whose dump is a coroutine."""

    def __init__(self, path):
        """Store the paths.

        Args:
            path: Workspace root.
        """
        self.path = Path(path)
        self.config_path = self.path / ".mitocode"
        self.dump_calls = 0
        self.reload_calls = 0
        self.dump_result = True

    async def dump(self):
        """Record and report a successful dump.

        Returns:
            The configured result.
        """
        self.dump_calls += 1
        return self.dump_result

    async def reload(self):
        """Record a reload.

        Returns:
            ``True``.
        """
        self.reload_calls += 1
        return True


def _read_tool(file_path: str) -> str:
    """Representative built-in file-reading tool.

    Args:
        file_path: Path requested by the model.

    Returns:
        An empty file body.
    """
    return ""


def _ctx(  # pylint: disable=too-many-arguments
    tmp_path,
    *,
    answers=None,
    orchestrator=None,
    workspace=None,
    mcp_manager=None,
    approver=None,
    skills_manager=None,
):
    """Build a command context over an in-memory console.

    Args:
        tmp_path: Temporary workspace root from pytest.
        answers: Scripted answers for ``CliUI.ask``.
        orchestrator: Orchestrator fake; a default one is built when omitted.
        workspace: Workspace fake or a real in-memory ``Workspace``.
        mcp_manager: MCP fake, or ``None`` when MCP is disabled.
        approver: Approver fake, or ``None``.
        skills_manager: Skills manager fake or real, or ``None`` when the
            skills feature is absent.

    Returns:
        ``(CommandContext, StringIO)`` where the buffer holds printed output.
    """
    buffer = StringIO()
    console = Console(
        file=buffer, width=200, force_terminal=False, theme=MITOCODE_THEME
    )
    prompt = _FakePromptSession(answers or [])
    ui = CliUI(console, prompt_session=prompt, history_path=tmp_path / "h")
    provider = SimpleNamespace(
        default_model="model-x", model_list=["model-x", "model-y"]
    )
    tool_manager = ToolManager()
    tool_manager.register(
        _read_tool, name="Read", human_doc="Read a file from the workspace"
    )
    active_orchestrator = orchestrator or _FakeOrchestrator()
    active_workspace = workspace or _AsyncWorkspace(tmp_path)
    state = CliState(
        args=CliArgs(workspace=tmp_path),
        workspace=active_workspace,
        tool_manager=tool_manager,
        provider=provider,
        orchestrator=active_orchestrator,
        mcp_manager=mcp_manager,
        skills_manager=skills_manager,
        ui=ui,
        approver=approver,
        log_path=tmp_path / "cli.log",
        main_id=0,
    )
    return CommandContext(state=state, ui=ui, approver=approver), buffer


def test_command_registry_is_complete():
    assert set(COMMANDS) == EXPECTED_COMMANDS
    for name, entry in COMMANDS.items():
        assert entry.name == name
        assert entry.usage.startswith("/")
        assert entry.description
        assert callable(entry.handler)


@pytest.mark.asyncio
async def test_non_slash_returns_none_and_unknown_does_not_exit(tmp_path):
    ctx, buffer = _ctx(tmp_path)

    assert (await dispatch("hello", ctx)) is None

    result = await dispatch("/nope", ctx)
    assert isinstance(result, CommandResult)
    assert result.exit_requested is False
    assert "Unknown command" in buffer.getvalue()
    assert "nope" in buffer.getvalue()


@pytest.mark.asyncio
async def test_exit_and_quit_request_exit(tmp_path):
    ctx, _ = _ctx(tmp_path)

    assert (await dispatch("/exit", ctx)).exit_requested is True
    assert (await dispatch("/quit", ctx)).exit_requested is True


@pytest.mark.asyncio
async def test_handler_failure_is_contained_by_dispatch(tmp_path, monkeypatch):
    ctx, buffer = _ctx(tmp_path)

    async def _boom(context, args):
        raise RuntimeError("boom")

    monkeypatch.setitem(COMMANDS, "boom", Command("boom", "/boom", "Explode.", _boom))

    result = await dispatch("/boom", ctx)

    assert isinstance(result, CommandResult)
    assert result.exit_requested is False
    assert "Command failed: boom" in buffer.getvalue()


@pytest.mark.asyncio
async def test_clear_destroys_and_recreates(tmp_path):
    ctx, buffer = _ctx(tmp_path)
    ctx.state.turn_count = 4
    ctx.state.tool_call_count = 3
    ctx.state.total_chars = 100

    await dispatch("/clear", ctx)

    assert ctx.state.orchestrator.destroyed == [0]
    assert ctx.state.main_id == 7
    assert ctx.state.turn_count == 0
    assert ctx.state.tool_call_count == 0
    assert ctx.state.total_chars == 0
    assert "New session #7" in buffer.getvalue()


@pytest.mark.asyncio
async def test_tools_and_status_include_registry_and_model(tmp_path):
    ctx, buffer = _ctx(tmp_path)

    await dispatch("/tools", ctx)
    await dispatch("/status", ctx)

    text = buffer.getvalue()
    assert "Read" in text and "model-x" in text and "Main" in text
    assert "Read a file" in text


def _write_skill(root, name, description="A test skill."):
    """Create one skill directory under *root* with minimal valid frontmatter."""
    entry = root / name
    entry.mkdir(parents=True)
    (entry / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody.\n",
        encoding="utf-8",
    )
    return entry


async def _discovered_manager(root):
    """Build a real manager over *root* and run one discovery."""
    manager = SkillsManager([SkillRoot("workspace", root)])
    await manager.discover()
    return manager


@pytest.mark.asyncio
async def test_skills_command_lists_the_catalogue(tmp_path):
    """Each discovered skill appears with its name, source and description."""
    root = tmp_path / "skills"
    _write_skill(root, "alpha", "Do alpha things.")
    ctx, buffer = _ctx(tmp_path, skills_manager=await _discovered_manager(root))

    await dispatch("/skills", ctx)

    text = buffer.getvalue()
    assert "alpha" in text
    assert "Do alpha things." in text
    assert "workspace" in text


@pytest.mark.asyncio
async def test_skills_command_without_manager_says_unavailable(tmp_path):
    """No wired skills manager is reported in prose, not an exception."""
    ctx, buffer = _ctx(tmp_path)

    await dispatch("/skills", ctx)

    assert "not available" in buffer.getvalue()


@pytest.mark.asyncio
async def test_skills_command_refresh_rescans_before_listing(tmp_path):
    """``/skills refresh`` re-scans, so skills added after startup appear."""
    root = tmp_path / "skills"
    root.mkdir(parents=True)
    manager = SkillsManager([SkillRoot("workspace", root)])
    await manager.discover()
    ctx, buffer = _ctx(tmp_path, skills_manager=manager)

    await dispatch("/skills", ctx)
    assert "alpha" not in buffer.getvalue()

    _write_skill(root, "alpha", "Arrived late.")
    await dispatch("/skills refresh", ctx)
    assert "alpha" in buffer.getvalue()
    assert "Arrived late." in buffer.getvalue()


@pytest.mark.asyncio
async def test_skills_command_rejects_unknown_argument(tmp_path):
    """An unrecognised argument gets the usage line."""
    ctx, buffer = _ctx(tmp_path, skills_manager=SkillsManager([]))

    await dispatch("/skills bogus", ctx)

    assert "Usage: /skills [refresh]" in buffer.getvalue()


@pytest.mark.asyncio
async def test_skills_command_refresh_rejects_extra_arguments(tmp_path):
    """Arguments after ``refresh`` get the usage line, not silent ignoring."""
    ctx, buffer = _ctx(tmp_path, skills_manager=SkillsManager([]))

    await dispatch("/skills refresh extra", ctx)

    assert "Usage: /skills [refresh]" in buffer.getvalue()


@pytest.mark.asyncio
async def test_skills_command_warns_about_skipped_entries(tmp_path):
    """Discovery issues surface as a warning next to the table."""
    root = tmp_path / "skills"
    (root / "not-a-skill").mkdir(parents=True)
    _write_skill(root, "alpha")
    ctx, buffer = _ctx(tmp_path, skills_manager=await _discovered_manager(root))

    await dispatch("/skills", ctx)

    text = buffer.getvalue()
    assert "skipped" in text
    assert "alpha" in text


@pytest.mark.asyncio
async def test_skills_command_refresh_failure_is_reported_not_raised(tmp_path):
    """A failing re-scan is reported through the UI and the REPL survives."""

    class _ExplodingManager:
        """Skills-manager stand-in whose discover() always raises."""

        async def discover(self):
            """Raise the scripted failure."""
            raise RuntimeError("scan exploded")

        def list_skills(self):
            """Report an empty catalogue."""
            return ()

        def issues(self):
            """Report no issues."""
            return ()

    ctx, buffer = _ctx(tmp_path, skills_manager=_ExplodingManager())

    await dispatch("/skills refresh", ctx)

    assert "Skills refresh failed" in buffer.getvalue()


@pytest.mark.asyncio
async def test_agents_render_tree_and_last_assistant_message(tmp_path):
    messages = [
        SimpleNamespace(role="assistant", message="first answer"),
        SimpleNamespace(role="user", message="question"),
        SimpleNamespace(role="assistant", message="second answer"),
        SimpleNamespace(role="tool", message="tool output"),
    ]
    session = _FakeSession(messages=messages)
    orchestrator = _FakeOrchestrator(sessions={0: session})
    ctx, buffer = _ctx(tmp_path, orchestrator=orchestrator)

    await dispatch("/agents", ctx)
    await dispatch("/agents 0", ctx)

    text = buffer.getvalue()
    assert "#0" in text and "Main" in text and "idle" in text
    assert "second answer" in text
    assert "first answer" not in text.split("second answer")[0]


@pytest.mark.asyncio
async def test_agents_unknown_id_warns_without_crashing(tmp_path):
    ctx, buffer = _ctx(tmp_path)

    await dispatch("/agents 99", ctx)
    await dispatch("/agents not-a-number", ctx)

    text = buffer.getvalue()
    assert "99" in text
    assert "not-a-number" in text


@pytest.mark.asyncio
async def test_model_switches_from_argument_and_from_prompt(tmp_path):
    ctx, buffer = _ctx(tmp_path, answers=["2"])

    await dispatch("/model model-y", ctx)
    assert ctx.state.provider.default_model == "model-y"

    ctx.state.provider.default_model = "model-x"
    await dispatch("/model", ctx)
    assert ctx.state.provider.default_model == "model-y"

    text = buffer.getvalue()
    assert "1. model-x" in text and "2. model-y" in text
    prompt = getattr(ctx.ui, "_prompt_session")
    assert prompt.messages[-1] == "  Model number or name: "


@pytest.mark.asyncio
async def test_model_rejects_bad_number_and_empty_answer(tmp_path):
    ctx, buffer = _ctx(tmp_path, answers=["9", ""])

    await dispatch("/model", ctx)
    await dispatch("/model", ctx)

    assert ctx.state.provider.default_model == "model-x"
    text = buffer.getvalue()
    assert "out of range" in text.lower()


@pytest.mark.asyncio
async def test_model_switch_registers_unlisted_model(tmp_path):
    """An unlisted name is added to ``model_list`` so ``chat()`` accepts it."""
    ctx, buffer = _ctx(tmp_path)

    await dispatch("/model model-z", ctx)
    await dispatch("/model model-z", ctx)

    provider = ctx.state.provider
    assert provider.default_model == "model-z"
    assert provider.model_list == ["model-x", "model-y", "model-z"]
    assert buffer.getvalue().count("not in the provider's model list") == 1
    assert "Model set to model-z." in buffer.getvalue()


@pytest.mark.asyncio
async def test_model_switch_from_empty_or_missing_list_registers_choice(tmp_path):
    """A choice against an empty or missing list becomes the whole list."""
    for starting_list in ([], None):
        ctx, buffer = _ctx(tmp_path)
        ctx.state.provider.model_list = starting_list

        await dispatch("/model foo", ctx)

        assert ctx.state.provider.default_model == "foo"
        assert ctx.state.provider.model_list == ["foo"]
        assert "not in the provider's model list" in buffer.getvalue()


@pytest.mark.asyncio
async def test_model_number_pick_leaves_listed_models_unchanged(tmp_path):
    """Picking a listed number never rewrites ``model_list``."""
    ctx, _ = _ctx(tmp_path, answers=["2"])

    await dispatch("/model", ctx)

    assert ctx.state.provider.default_model == "model-y"
    assert ctx.state.provider.model_list == ["model-x", "model-y"]


@pytest.mark.asyncio
async def test_mcp_refresh_calls_manager_and_every_session(tmp_path):
    session = _FakeSession()
    orchestrator = _FakeOrchestrator(sessions={0: session})
    manager = _FakeMcp()
    ctx, buffer = _ctx(tmp_path, orchestrator=orchestrator, mcp_manager=manager)

    await dispatch("/mcp refresh fs", ctx)

    assert manager.refreshed == ["fs"]
    assert session.refresh_calls == 1
    assert "refreshed" in buffer.getvalue().lower()


@pytest.mark.asyncio
async def test_mcp_status_and_disabled_are_not_fatal(tmp_path):
    status = SimpleNamespace(state="connected", error="", tool_count=3)
    ctx, buffer = _ctx(tmp_path, mcp_manager=_FakeMcp({"fs": status}))
    await dispatch("/mcp", ctx)
    assert "fs" in buffer.getvalue() and "connected" in buffer.getvalue()

    disabled, disabled_buffer = _ctx(tmp_path)
    await dispatch("/mcp", disabled)
    await dispatch("/mcp refresh", disabled)
    assert "No MCP" in disabled_buffer.getvalue()


@pytest.mark.asyncio
async def test_permissions_show_policy_and_reset_cache(tmp_path):
    approver = _FakeApprover()
    ctx, buffer = _ctx(tmp_path, approver=approver)

    await dispatch("/permissions", ctx)
    assert "ask" in buffer.getvalue()
    assert "Bash" in buffer.getvalue()

    await dispatch("/permissions reset", ctx)
    assert approver.reset_calls == 1
    assert approver.cached == []


@pytest.mark.asyncio
async def test_verbose_toggles_state_and_ui(tmp_path):
    ctx, buffer = _ctx(tmp_path)

    await dispatch("/verbose", ctx)
    assert ctx.state.verbose is True and ctx.ui.verbose is True

    await dispatch("/verbose", ctx)
    assert ctx.state.verbose is False and ctx.ui.verbose is False
    assert "Verbose" in buffer.getvalue()


@pytest.mark.asyncio
async def test_help_shows_usage_brackets_verbatim(tmp_path):
    """Table cells render usage brackets instead of parsing them as markup."""
    ctx, buffer = _ctx(tmp_path)

    await dispatch("/help", ctx)

    text = buffer.getvalue()
    assert "/model [name]" in text
    assert "/mcp [refresh [name]]" in text
    assert "/permissions [reset]" in text
    assert "Command failed" not in text


@pytest.mark.asyncio
async def test_status_renders_bracketed_model_verbatim(tmp_path):
    """A model name holding a closing bracket is not parsed as markup."""
    ctx, buffer = _ctx(tmp_path)
    ctx.state.provider.default_model = "model[/x]"

    await dispatch("/status", ctx)

    text = buffer.getvalue()
    assert "model[/x]" in text
    assert "Command failed" not in text


@pytest.mark.asyncio
async def test_permissions_renders_bracketed_cache_entries_verbatim(tmp_path):
    """A cached command containing ``[/tmp/x]`` shows verbatim, without raising."""
    approver = _FakeApprover()
    approver.cached = [("Bash", 'echo "[/tmp/x]"')]
    ctx, buffer = _ctx(tmp_path, approver=approver)

    await dispatch("/permissions", ctx)

    text = buffer.getvalue()
    assert "[/tmp/x]" in text
    assert "Command failed" not in text


@pytest.mark.asyncio
async def test_cost_and_log_report_session_facts(tmp_path):
    session = _FakeSession(
        messages=[SimpleNamespace(message="abcdefgh", content="", parameter={})]
    )
    ctx, buffer = _ctx(tmp_path, orchestrator=_FakeOrchestrator(sessions={0: session}))

    await dispatch("/cost", ctx)
    await dispatch("/log", ctx)

    text = buffer.getvalue()
    assert "Approximate tokens: ~2" in text
    assert str(tmp_path / "cli.log") in text


@pytest.mark.asyncio
async def test_estimate_tokens_counts_message_and_tool_text(tmp_path):
    messages = [
        SimpleNamespace(message="abcdefgh", content="", parameter={}),
        SimpleNamespace(message="", content="1234", parameter={}),
        SimpleNamespace(message="", content="", parameter={"path": "ab"}),
        SimpleNamespace(
            message="", content="", parameter={}, reasoning_content="12345678"
        ),
    ]
    session = _FakeSession(messages=messages)
    ctx = _ctx(tmp_path, orchestrator=_FakeOrchestrator(sessions={0: session}))[0]

    assert ctx.state.estimate_tokens() == 8


@pytest.mark.asyncio
async def test_skills_command_warns_on_issues_even_with_empty_catalogue(tmp_path):
    """A root holding only broken entries still shows the skipped warning."""
    root = tmp_path / "skills"
    (root / "broken").mkdir(parents=True)
    ctx, buffer = _ctx(tmp_path, skills_manager=await _discovered_manager(root))

    await dispatch("/skills", ctx)

    text = buffer.getvalue()
    assert "No skills installed" in text
    assert "skipped" in text


# ---------------------------------------------------------------------------
# /effort and the /status effort row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_effort_show_set_warn_and_clear(tmp_path):
    """/effort 显示当前值、设置档位、警告放行未知档位、default 清除."""
    ctx, buffer = _ctx(tmp_path, answers=[])
    ctx.state.provider.reasoning_effort = "high"

    await dispatch("/effort", ctx)
    assert "high" in buffer.getvalue()

    await dispatch("/effort low", ctx)
    assert ctx.state.provider.reasoning_effort == "low"

    await dispatch("/effort ultra", ctx)
    assert ctx.state.provider.reasoning_effort == "ultra"
    assert "not a known tier" in buffer.getvalue()

    await dispatch("/effort default", ctx)
    assert ctx.state.provider.reasoning_effort is None


@pytest.mark.asyncio
async def test_status_shows_effort_row(tmp_path):
    """/status 表格包含 effort 行；未设置时按规格显示 default (not sent)."""
    ctx, buffer = _ctx(tmp_path, answers=[])
    ctx.state.provider.reasoning_effort = "high"
    await dispatch("/status", ctx)
    assert "effort" in buffer.getvalue()
    assert "high" in buffer.getvalue()

    ctx2, buffer2 = _ctx(tmp_path, answers=[])
    await dispatch("/status", ctx2)
    assert "not sent" in buffer2.getvalue()
