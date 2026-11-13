"""Tests for the CLI startup assembly in :mod:`mitocode.cli.bootstrap`.

The provider, the MCP manager and the orchestrator are replaced by recording
fakes, so these tests stay offline and can assert the *order* of the steps that
the design fixes: MCP must start before the first agent session exists, because
``AgentSession`` snapshots the tool registry in its constructor.
"""

import asyncio
import json
import os
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.cli.bootstrap as bootstrap_module
from mitocode.cli.approval import CliApprover
from mitocode.cli.args import CliArgs
from mitocode.cli.bootstrap import (
    GENERAL_TEMPLATE_NAME,
    MINIMAL_PROMPT,
    Bootstrapped,
    bootstrap,
    build_system_prompt,
    build_template_manager,
    template_roots,
)
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.cli.ui import CliUI
from mitocode.core.agent.template import AgentTemplate, builtin_templates_dir
from mitocode.core.config.manager import ConfigManager, default_global_dir
from mitocode.core.skills.skill import Skill, SkillRoot
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace

ORDERS: list[str] = []


class _FakeProvider:
    """Recording stand-in for :class:`ChatProviderOpenAI`.

    ``factory`` records the call, then returns an instance holding the keyword
    arguments so tests can inspect what the bootstrap passed to the provider.
    """

    def __init__(self, **kwargs):
        self.default_model = kwargs.get("default_model", "")
        self.model_list = list(kwargs.get("default_model_list") or [])
        self.settings = kwargs.get("settings", {})
        self.stream = kwargs.get("stream", False)

    @classmethod
    async def factory(cls, **kwargs):
        ORDERS.append("provider.factory")
        return cls(**kwargs)


class _FakeMcp:
    """Recording stand-in for :class:`McpManager`."""

    instances: list = []

    def __init__(self, config, *, tool_manager):
        self.config = config
        self.tool_manager = tool_manager
        _FakeMcp.instances.append(self)

    async def start(self):
        ORDERS.append("mcp.start")

    async def stop(self):
        ORDERS.append("mcp.stop")

    def status(self):
        return {}


class _FakeOrchestrator:
    """Recording stand-in for :class:`AgentOrchestrator`."""

    def __init__(self, **kwargs):
        ORDERS.append("orchestrator")
        self.kw = kwargs

    async def create_session(self, **kwargs):
        ORDERS.append("create_session")
        self.session_kwargs = kwargs
        return SimpleNamespace(agent_id=0, **kwargs)


class _RecordingUI(CliUI):
    """A real :class:`CliUI` that also keeps its warnings and errors."""

    def __init__(self, console: Console) -> None:
        super().__init__(console, history_path=Path("cli-history"))
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        super().warn(text)

    def error(self, text: str) -> None:
        self.errors.append(text)
        super().error(text)


def _ui() -> _RecordingUI:
    """Build a recording UI whose console writes into a buffer.

    Returns:
        The UI, with the ``StringIO`` buffer attached as ``stream_buffer``.
    """
    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False)
    ui = _RecordingUI(console)
    ui.stream_buffer = buffer  # type: ignore[attr-defined]
    return ui


def _args(**over) -> CliArgs:
    """Build :class:`CliArgs` with every startup field defaulted.

    Args:
        **over: Fields to override.

    Returns:
        The arguments.
    """
    base = dict(
        workspace=Path("."),
        permission_mode="ask",
        mcp_config=None,
        no_mcp=False,
        verbose=False,
        debug=False,
        initial_prompt=None,
        show_help=False,
        show_version=False,
    )
    base.update(over)
    return CliArgs(**base)


def _provider_args(root: Path, provider=None, **over) -> CliArgs:
    """Write the credential file the CLI now requires and build the arguments.

    Also writes a minimal workspace configuration, because :func:`bootstrap`
    cannot run without one.

    Args:
        root: Workspace root receiving
            ``.mitocode/secret/provider.json`` and ``.mitocode/config.json``.
        provider: Fields to merge over the default
            ``{"api_key": "k", "model": "m"}`` credential document.
        **over: Fields to override on the arguments.

    Returns:
        The arguments pointing at *root*.
    """
    payload = {"api_key": "k", "model": "m"}
    payload.update(provider or {})
    path = root / ".mitocode" / "secret" / "provider.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    (path.parent.parent / "config.json").write_text(
        json.dumps({"version": 2, "name": root.name, "description": ""}),
        encoding="utf-8",
    )
    base = dict(workspace=root)
    base.update(over)
    return CliArgs(**base)


def _patch(
    monkeypatch,
    *,
    provider=_FakeProvider,
    mcp=_FakeMcp,
    global_dir: Path = Path("/mitocode-test-global"),
):
    """Install the recording fakes on the bootstrap module.

    The global configuration directory is redirected as well: the template
    roots include it, so a developer machine carrying a ``general`` template
    in ``~/.config/.mitocode`` would otherwise change the prompt these tests
    assert on.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        provider: Replacement for ``ChatProviderOpenAI``.
        mcp: Replacement for ``McpManager``.
        global_dir: Directory the global template root resolves to.  The
            default does not exist, which is the realistic case for CI.
    """
    monkeypatch.setattr(bootstrap_module, "ChatProviderOpenAI", provider)
    monkeypatch.setattr(bootstrap_module, "McpManager", mcp)
    monkeypatch.setattr(bootstrap_module, "AgentOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        bootstrap_module,
        "ConfigManager",
        lambda path: ConfigManager(path, global_dir=global_dir),
    )


def _write_mcp(root: Path, payload: str) -> Path:
    """Write the default MCP configuration file under *root*.

    Args:
        root: Workspace root.
        payload: File contents.

    Returns:
        The path that was written.
    """
    path = root / ".mitocode" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _general(message="{persona}"):
    """Build the ``general`` template the default persona is read from.

    Args:
        message: The system block's text.  The default keeps a workspace slot so
            substitution is still exercised.
    """
    path = Path("/templates") / GENERAL_TEMPLATE_NAME
    return AgentTemplate(
        name=GENERAL_TEMPLATE_NAME,
        description="The default persona.",
        path=path,
        template_path=path / "template.json",
        source="builtin",
        session_autoload=(
            (
                {
                    "kind": "text",
                    "role": "system",
                    "message": "You are MitoCode, an interactive coding agent running "
                    "in the user's terminal.\nYou help with engineering tasks in the "
                    "workspace at {workspace}.",
                }
                if message == "{persona}"
                else {
                    "kind": "text",
                    "role": "system",
                    "message": message,
                }
            ),
        ),
    )


def test_the_default_persona_comes_from_the_general_template(tmp_path):
    """The persona is data, not a constant: a user can shadow the template."""
    template = _general("Custom persona at {workspace}.")

    prompt = build_system_prompt(tmp_path, templates=(template,))

    assert prompt.startswith("Custom persona at ")
    assert str(tmp_path.resolve()) in prompt
    assert "{workspace}" not in prompt


def _persona_text(tmp_path, template=None):
    """Render the persona a ``general`` template would contribute at *tmp_path*.

    Args:
        tmp_path: Workspace root substituted into the persona.
        template: The template to render; a default ``general`` when omitted.
    """
    template = _general() if template is None else template
    return (
        template.session_autoload[0]["message"].replace(
            "{workspace}", str(Path(tmp_path).resolve())
        )
        + "\n"
    )


def test_a_template_catalogue_adds_no_skills_section(tmp_path):
    """Templates are not skills: their section must not imply the Skill tool."""
    prompt = build_system_prompt(tmp_path, templates=(_general(),))

    assert prompt.startswith(_persona_text(tmp_path))
    assert "## Skills" not in prompt


def test_build_system_prompt_with_skills_appends_the_catalogue(tmp_path):
    """A non-empty catalogue appends a section instructing Skill-tool use."""
    prompt = build_system_prompt(
        tmp_path,
        skills=(_skill("alpha", "Do alpha things."),),
        templates=(_general(),),
    )

    assert prompt.startswith(_persona_text(tmp_path))
    assert "## Skills" in prompt
    assert "- alpha: Do alpha things." in prompt
    assert "Skill tool" in prompt
    assert prompt.index("## Skills") < prompt.index("## Agent Templates")
    assert prompt.count(str(tmp_path.resolve())) == 1


def test_build_system_prompt_with_templates_appends_the_catalogue(tmp_path):
    """The model learns which template names the Agent tool accepts."""
    reviewer = AgentTemplate(
        name="reviewer",
        description="Reviews a change.",
        path=Path("/templates/reviewer"),
        template_path=Path("/templates/reviewer/template.json"),
        source="builtin",
    )

    prompt = build_system_prompt(tmp_path, templates=(_general(), reviewer))

    assert "## Agent Templates" in prompt
    assert "- reviewer: Reviews a change." in prompt
    assert "template" in prompt


def test_build_system_prompt_without_templates_has_no_template_section(tmp_path):
    """No template means no section and the minimal fallback persona."""
    prompt = build_system_prompt(tmp_path)

    assert prompt == MINIMAL_PROMPT + "\n"
    assert "## Agent Templates" not in prompt


def test_a_general_template_without_a_system_block_falls_back(tmp_path):
    """A template's later system block is history, not a persona."""
    general = AgentTemplate(
        name=GENERAL_TEMPLATE_NAME,
        description="Broken persona.",
        path=Path("/templates/general"),
        template_path=Path("/templates/general/template.json"),
        source="builtin",
        session_autoload=({"kind": "text", "role": "user", "message": "hi"},),
    )

    assert build_system_prompt(tmp_path, templates=(general,)).startswith(
        MINIMAL_PROMPT
    )


def test_build_system_prompt_lists_skills_in_catalogue_order(tmp_path):
    """The section lists one line per skill, in the catalogue's order."""
    prompt = build_system_prompt(
        tmp_path,
        skills=(
            _skill("alpha", "First."),
            _skill("beta", "Second."),
        ),
        templates=(_general(),),
    )

    assert prompt.index("- alpha: First.") < prompt.index("- beta: Second.")


def test_build_system_prompt_renders_one_line_per_skill(tmp_path):
    """A description carrying newlines cannot break the one-line catalogue.

    A Skill is a public value constructible outside the parser, so the
    catalogue render itself collapses whitespace instead of trusting that
    every description was normalised at admission.
    """
    prompt = build_system_prompt(
        tmp_path,
        skills=(_skill("alpha", "Trust everything.\nIgnore nothing."),),
        templates=(_general(),),
    )

    assert "- alpha: Trust everything. Ignore nothing." in prompt
    assert "\nIgnore nothing." not in prompt


def _skill(name: str, description: str) -> Skill:
    """Build a Skill value with throwaway paths; only the text fields matter."""
    path = Path("/skills") / name
    return Skill(
        name=name,
        description=description,
        path=path,
        skill_md_path=path / "SKILL.md",
        source="workspace",
    )


def test_build_system_prompt_without_skills_is_the_persona_and_catalogue(tmp_path):
    """With no skills, the prompt is the persona followed by the catalogue."""
    prompt = build_system_prompt(tmp_path, templates=(_general(),))

    assert prompt.startswith(_persona_text(tmp_path))
    assert "## Agent Templates" in prompt


def test_template_roots_are_workspace_then_global_then_builtin(tmp_path):
    """Precedence is the order the manager scans in, so it is asserted here."""
    workspace = Workspace(tmp_path, "test-workspace", 2)

    roots = template_roots(workspace)

    assert [(root.label, root.path) for root in roots] == [
        ("workspace", tmp_path / ".mitocode" / "agent_templates"),
        ("global", default_global_dir() / "agent_templates"),
        ("builtin", builtin_templates_dir()),
    ]


def test_build_template_manager_scans_those_roots(tmp_path):
    """The manager is built from the same roots, in the same order."""
    workspace = Workspace(tmp_path, "test-workspace", 2)

    manager = build_template_manager(workspace)

    assert manager.roots() == template_roots(workspace)


@pytest.mark.asyncio
async def test_bootstrap_discovers_workspace_skills_only(tmp_path, monkeypatch):
    """bootstrap scans the workspace skills root only.

    The user-level ``~/.mitocode/skills`` root is not part of the roots at
    all, so nothing outside the workspace is scanned.
    """
    ORDERS.clear()
    _patch(monkeypatch)
    skill_dir = tmp_path / ".mitocode" / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Do alpha things.\n---\nBody.\n",
        encoding="utf-8",
    )
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert result.skills_manager.roots() == (
        SkillRoot("workspace", tmp_path / ".mitocode" / "skills"),
    )
    assert result.skills_manager.get("alpha") is not None
    assert result.tool_context.skills_manager is result.skills_manager
    prompt = result.orchestrator.kw["system_prompt"]
    assert "## Skills" in prompt
    assert "- alpha: Do alpha things." in prompt


@pytest.mark.asyncio
async def test_bootstrap_without_any_skills_leaves_the_prompt_alone(
    tmp_path, monkeypatch
):
    """With no skill installed anywhere, no skills section is added."""
    ORDERS.clear()
    _patch(monkeypatch)
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert "## Skills" not in result.orchestrator.kw["system_prompt"]
    assert result.orchestrator.kw["system_prompt"].startswith("You are MitoCode")
    assert result.skills_manager.list_skills() == ()


@pytest.mark.asyncio
async def test_bootstrap_discovers_templates_and_attaches_them(tmp_path, monkeypatch):
    """The workspace catalogue reaches both the tool context and the prompt."""
    ORDERS.clear()
    _patch(monkeypatch)
    entry = tmp_path / ".mitocode" / "agent_templates" / "planner"
    entry.mkdir(parents=True)
    (entry / "template.json").write_text(
        json.dumps(
            {
                "name": "planner",
                "description": "Plans before acting.",
                "session_autoload": [
                    {"kind": "text", "role": "system", "message": "You plan."}
                ],
            }
        ),
        encoding="utf-8",
    )
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    manager = result.tool_context.template_manager
    assert manager is not None
    assert manager.get("planner") is not None, "the workspace template is discovered"
    assert manager.get(GENERAL_TEMPLATE_NAME) is not None, (
        "the packaged general template must survive a workspace override that "
        "does not name it"
    )
    prompt = result.orchestrator.kw["system_prompt"]
    assert "## Agent Templates" in prompt
    assert "- planner: Plans before acting." in prompt
    assert Path(result.orchestrator.kw["workspace"]).resolve() == tmp_path.resolve()
    assert manager.issues() == (), "the packaged templates must load without issues"


@pytest.mark.asyncio
async def test_bootstrap_reads_the_global_template_root(tmp_path, monkeypatch):
    """The user-level copy is scanned, and reports itself as the global root."""
    ORDERS.clear()
    global_dir = tmp_path / "global-config"
    _patch(monkeypatch, global_dir=global_dir)
    entry = global_dir / "agent_templates" / "globetrotter"
    entry.mkdir(parents=True)
    (entry / "template.json").write_text(
        json.dumps({"name": "globetrotter", "description": "Machine-wide."}),
        encoding="utf-8",
    )
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    template = result.tool_context.template_manager.get("globetrotter")
    assert template is not None
    assert template.source == "global"
    assert "- globetrotter: Machine-wide." in result.orchestrator.kw["system_prompt"]


@pytest.mark.asyncio
async def test_mcp_starts_before_session_and_model_is_union(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path, provider={"model": "my-model"}),
        ui,
        CliApprover(ui),
    )

    assert ORDERS.index("provider.factory") < ORDERS.index("mcp.start")
    assert ORDERS.index("mcp.start") < ORDERS.index("create_session")
    assert "my-model" in result.provider.model_list
    assert result.orchestrator.kw["system_prompt"].startswith("You are MitoCode")
    assert result.orchestrator.kw["system_prompt"].count(str(tmp_path.resolve())) == 1
    assert result.orchestrator.kw["tool_manager"] is result.tool_manager
    assert result.orchestrator.kw["default_provider"] is result.provider
    assert result.tool_context.orchestrator is result.orchestrator
    assert result.orchestrator.session_kwargs == {
        "name": "Main",
        "description": "Interactive session",
    }
    assert result.mcp_manager is not None
    assert result.mcp_error == ""
    assert result.provider_config.model == "my-model"
    assert result.provider.settings["api_key"] == "k"
    assert result.provider.settings["timeout"] == 60.0
    assert isinstance(result, Bootstrapped)


@pytest.mark.asyncio
async def test_provider_base_url_never_falls_through_to_the_environment(
    tmp_path, monkeypatch
):
    """An omitted base_url must not let the OpenAI SDK read OPENAI_BASE_URL."""
    ORDERS.clear()
    _patch(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://ambient.invalid/v1")
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert (
        result.provider.settings["base_url"]
        == bootstrap_module.DEFAULT_PROVIDER_BASE_URL
    )


@pytest.mark.asyncio
async def test_explicit_provider_base_url_is_passed_through(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path, provider={"base_url": "http://workspace.invalid/v1"}),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["base_url"] == "http://workspace.invalid/v1"


@pytest.mark.asyncio
async def test_model_absent_from_live_list_is_appended_with_warning(
    tmp_path, monkeypatch
):
    ORDERS.clear()

    class _EndpointProvider(_FakeProvider):
        @classmethod
        async def factory(cls, **kwargs):
            ORDERS.append("provider.factory")
            instance = cls(**kwargs)
            instance.model_list = ["other-model"]
            return instance

    _patch(monkeypatch, provider=_EndpointProvider)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path, provider={"model": "my-model"}),
        ui,
        CliApprover(ui),
    )

    assert result.provider.model_list == ["other-model", "my-model"]
    assert any("not advertised" in warning for warning in ui.warnings)


@pytest.mark.asyncio
async def test_model_present_in_live_list_is_not_duplicated(tmp_path, monkeypatch):
    ORDERS.clear()

    class _ListingProvider(_FakeProvider):
        @classmethod
        async def factory(cls, **kwargs):
            ORDERS.append("provider.factory")
            instance = cls(**kwargs)
            instance.model_list = ["my-model"]
            return instance

    _patch(monkeypatch, provider=_ListingProvider)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path, provider={"model": "my-model"}),
        ui,
        CliApprover(ui),
    )

    assert result.provider.model_list == ["my-model"]
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_broken_default_mcp_is_non_fatal(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, "{}")
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert result.mcp_manager is None
    assert result.mcp_error
    assert ui.warnings


@pytest.mark.asyncio
async def test_explicit_broken_mcp_is_fatal(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    explicit = tmp_path / "servers.json"
    explicit.write_text("{", encoding="utf-8")
    ui = _ui()

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await bootstrap(
            _provider_args(tmp_path, mcp_config=explicit),
            ui,
            CliApprover(ui),
        )

    assert "explicit" in str(excinfo.value)
    assert not any(order == "mcp.start" for order in ORDERS)


@pytest.mark.asyncio
async def test_no_mcp_flag_creates_no_manager(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path, no_mcp=True),
        ui,
        CliApprover(ui),
    )

    assert result.mcp_manager is None
    assert result.mcp_error == ""
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_missing_default_mcp_creates_no_manager(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert result.mcp_manager is None
    assert result.mcp_error == ""


@pytest.mark.asyncio
async def test_provider_failure_becomes_config_error_with_scrubbed_key(
    tmp_path, monkeypatch
):
    ORDERS.clear()

    class _FailingProvider:
        @classmethod
        async def factory(cls, **kwargs):
            raise RuntimeError(f"refused credential {kwargs['settings']['api_key']}")

    _patch(monkeypatch, provider=_FailingProvider)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await bootstrap(
            _provider_args(tmp_path, provider={"api_key": "super-secret"}),
            ui,
            CliApprover(ui),
        )

    message = str(excinfo.value)
    assert "super-secret" not in message
    assert "***" in message
    assert not any(order == "mcp.start" for order in ORDERS)


@pytest.mark.asyncio
async def test_session_failure_after_mcp_start_stops_the_manager(tmp_path, monkeypatch):
    """A failure after ``mcp.start()`` must not orphan the MCP servers.

    ``create_session`` is the last startup step and the first one that can fail
    while a live manager exists; the manager reference is otherwise lost with
    the exception.
    """
    ORDERS.clear()

    class _ExplodingOrchestrator(_FakeOrchestrator):
        """Orchestrator whose first session cannot be created."""

        async def create_session(self, **kwargs):
            ORDERS.append("create_session")
            raise RuntimeError("session creation failed")

    _patch(monkeypatch)
    monkeypatch.setattr(bootstrap_module, "AgentOrchestrator", _ExplodingOrchestrator)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    with pytest.raises(RuntimeError, match="session creation failed"):
        await bootstrap(
            _provider_args(tmp_path),
            ui,
            CliApprover(ui),
        )

    assert ORDERS.count("mcp.stop") == 1
    assert ORDERS.index("mcp.start") < ORDERS.index("mcp.stop")
    assert ORDERS.index("create_session") < ORDERS.index("mcp.stop")
    assert _FakeMcp.instances[-1].config is not None


# ──────────────────────────────────────────────
# _start_mcp: start() 失败时不得遗留 manager
# ──────────────────────────────────────────────


def _workspace(root: Path) -> Workspace:
    """Build a workspace rooted at *root* for direct ``_start_mcp`` calls.

    Args:
        root: Directory the default MCP configuration is read from.

    Returns:
        The workspace object.
    """
    return Workspace(root, "test-workspace", 1)


class _FailingStartMcp:
    """MCP manager stand-in whose ``start()`` raises a scripted exception.

    Attributes:
        failure: Exception instance raised by ``start()``.
    """

    def __init__(self, config, *, tool_manager):
        self.config = config
        self.tool_manager = tool_manager

    async def start(self):
        """Raise the scripted failure after recording the start."""
        ORDERS.append("mcp.start")
        raise _FailingStartMcp.failure

    async def stop(self):
        """Record a stop without touching real subprocesses."""
        ORDERS.append("mcp.stop")


@pytest.mark.asyncio
async def test_start_mcp_cancelled_stops_manager_and_propagates(tmp_path, monkeypatch):
    """start() 被取消 → 先 shielded stop 掉 manager，CancelledError 原样传播."""
    ORDERS.clear()
    _FailingStartMcp.failure = asyncio.CancelledError()
    monkeypatch.setattr(bootstrap_module, "McpManager", _FailingStartMcp)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    with pytest.raises(asyncio.CancelledError):
        await bootstrap_module._start_mcp(
            _args(workspace=tmp_path), _workspace(tmp_path), ToolManager(), ui
        )

    assert ORDERS.count("mcp.stop") == 1, "取消传播前必须 stop 已启动的 manager"
    assert ORDERS.index("mcp.start") < ORDERS.index("mcp.stop")


@pytest.mark.asyncio
async def test_start_mcp_runtime_error_stops_manager_and_propagates(
    tmp_path, monkeypatch
):
    """start() 抛 RuntimeError → 先 stop manager，异常原样传播."""
    ORDERS.clear()
    _FailingStartMcp.failure = RuntimeError("connect exploded")
    monkeypatch.setattr(bootstrap_module, "McpManager", _FailingStartMcp)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    with pytest.raises(RuntimeError, match="connect exploded"):
        await bootstrap_module._start_mcp(
            _args(workspace=tmp_path), _workspace(tmp_path), ToolManager(), ui
        )

    assert ORDERS.count("mcp.stop") == 1, "异常传播前必须 stop 已启动的 manager"
    assert ORDERS.index("mcp.start") < ORDERS.index("mcp.stop")


@pytest.mark.asyncio
async def test_start_mcp_success_path_returns_manager_without_stop(
    tmp_path, monkeypatch
):
    """start() 成功 → 返回 (manager, "") 且不触发任何 stop."""
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    manager, error = await bootstrap_module._start_mcp(
        _args(workspace=tmp_path), _workspace(tmp_path), ToolManager(), ui
    )

    assert isinstance(manager, _FakeMcp), "成功路径应返回启动的 manager 本身"
    assert error == ""
    assert ORDERS == ["mcp.start"], "成功路径不应有 stop 或其它记录"


@pytest.mark.asyncio
async def test_config_error_path_never_constructs_or_stops_a_manager(
    tmp_path, monkeypatch
):
    """默认 MCP 配置损坏 → 仅警告禁用，不构造 manager 也不调用 stop."""
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, "{ not json")
    ui = _ui()
    known = len(_FakeMcp.instances)

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert result.mcp_manager is None
    assert result.mcp_error
    assert len(_FakeMcp.instances) == known, "配置错误路径不应构造任何 manager"
    assert "mcp.stop" not in ORDERS, "不存在 manager 时不得调用 stop"


@pytest.mark.asyncio
async def test_explicit_config_error_propagates_without_stopping(tmp_path, monkeypatch):
    """显式 MCP 配置损坏 → 启动失败传播，且不构造、不 stop 任何 manager."""
    ORDERS.clear()
    _patch(monkeypatch)
    explicit = tmp_path / "servers.json"
    explicit.write_text("{", encoding="utf-8")
    ui = _ui()
    known = len(_FakeMcp.instances)

    with pytest.raises(MitoCodeCliConfigError):
        await bootstrap(
            _provider_args(tmp_path, mcp_config=explicit),
            ui,
            CliApprover(ui),
        )

    assert len(_FakeMcp.instances) == known, "配置错误路径不应构造任何 manager"
    assert "mcp.stop" not in ORDERS, "不存在 manager 时不得调用 stop"


# ──────────────────────────────────────────────
# bootstrap: API key 交给 orchestrator 用于字面抹除
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_passes_provider_api_key_for_redaction(tmp_path, monkeypatch):
    """bootstrap 把解析出的 API key 作为 redact_secrets 传给 orchestrator."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert result.orchestrator.kw["redact_secrets"] == (
        "k",
    ), "orchestrator 必须拿到字面 key 才能在错误事件与日志中抹除它"


# ──────────────────────────────────────────────
# bootstrap(restore=...): 启动时恢复已保存会话
# ──────────────────────────────────────────────


def _restore_record() -> dict:
    """Build a minimal valid session record for restore tests.

    Returns:
        A schema-shaped record with one main agent carrying one user message.
    """
    return {
        "schema_version": 1,
        "session_id": "20261107T000000Z-abcdef01",
        "title": "Old session",
        "created_at": "2026-11-07T00:00:00Z",
        "updated_at": "2026-11-07T00:00:00Z",
        "workspace": "/tmp/w",
        "model": "m",
        "main_agent_id": 0,
        "agents": [
            {
                "agent_id": 0,
                "name": "Main",
                "description": "Interactive session",
                "parent_id": None,
                "system_prompt": "restored prompt",
                "messages": [
                    {
                        "kind": "text",
                        "role": "user",
                        "message": "hi",
                        "reasoning_content": "",
                    }
                ],
            }
        ],
    }


class _RestoreOrchestrator:
    """Recording stand-in for the orchestrator's reset/restore/create calls.

    Attributes:
        reset_calls: Number of ``reset_sessions`` invocations.
        restore_calls: Lists of agent records passed to ``restore_tree``.
        created: Keyword arguments of every ``create_session`` call.
    """

    def __init__(self, restore_error=None):
        self.reset_calls = 0
        self.restore_calls = []
        self.created = []
        self.restore_error = restore_error

    async def reset_sessions(self):
        """Count one registry reset."""
        self.reset_calls += 1

    async def restore_tree(self, agents):
        """Rebuild fake sessions, or raise the scripted error.

        Args:
            agents: Agent records from the session document.

        Returns:
            Sessions keyed by their original agent id.

        Raises:
            Exception: The scripted ``restore_error``, when set.
        """
        self.restore_calls.append(list(agents))
        if self.restore_error is not None:
            raise self.restore_error
        sessions = {}
        for agent in agents:
            sessions[agent["agent_id"]] = SimpleNamespace(
                agent_id=agent["agent_id"], system_prompt="", messages=[]
            )
        return sessions

    async def create_session(self, **kwargs):
        """Record a creation request and return a bare main session.

        Args:
            **kwargs: Session options.

        Returns:
            A namespace carrying the agent id and the options.
        """
        self.created.append(kwargs)
        return SimpleNamespace(agent_id=0, **kwargs)


def _patch_restore(monkeypatch, restore_error=None) -> _RestoreOrchestrator:
    """Install the restore-recording orchestrator on the bootstrap module.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        restore_error: Exception ``restore_tree`` should raise, or ``None``.

    Returns:
        The orchestrator instance bootstrap will receive.
    """
    orchestrator = _RestoreOrchestrator(restore_error)
    monkeypatch.setattr(bootstrap_module, "ChatProviderOpenAI", _FakeProvider)
    monkeypatch.setattr(bootstrap_module, "McpManager", _FakeMcp)
    monkeypatch.setattr(
        bootstrap_module, "AgentOrchestrator", lambda **kw: orchestrator
    )
    # Same isolation as `_patch`: the global template root must not reach the
    # developer's real configuration directory.
    monkeypatch.setattr(
        bootstrap_module,
        "ConfigManager",
        lambda path: ConfigManager(path, global_dir=Path("/mitocode-test-global")),
    )
    return orchestrator


@pytest.mark.asyncio
async def test_bootstrap_restore_rebuilds_the_saved_tree(tmp_path, monkeypatch):
    """A valid record restores ids, messages and the system prompt."""
    ORDERS.clear()
    orchestrator = _patch_restore(monkeypatch)
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path), ui, CliApprover(ui), restore=_restore_record()
    )

    assert result.restore_succeeded is True
    assert result.restore_error == ""
    assert result.main_session.agent_id == 0
    assert result.main_session.system_prompt == "restored prompt"
    assert [block.message for block in result.main_session.messages] == ["hi"]
    assert orchestrator.reset_calls == 1, "恢复前必须先清空注册表"
    assert orchestrator.created == [], "恢复成功时不得再新建 main"


@pytest.mark.asyncio
async def test_bootstrap_restore_failure_falls_back_to_a_new_session(
    tmp_path, monkeypatch
):
    """A failing restore warns nobody, cleans the registry and creates Main."""
    ORDERS.clear()
    orchestrator = _patch_restore(
        monkeypatch, restore_error=ValueError("duplicate agent id")
    )
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path), ui, CliApprover(ui), restore=_restore_record()
    )

    assert result.restore_succeeded is False
    assert result.restore_error == "duplicate agent id"
    assert len(orchestrator.created) == 1, "回退路径必须新建一个 main"
    assert orchestrator.reset_calls == 2, "失败后必须再清一次注册表"


@pytest.mark.asyncio
async def test_bootstrap_without_restore_reports_no_restore(tmp_path, monkeypatch):
    """The default path creates the session and reports no restore attempt."""
    ORDERS.clear()
    _patch(monkeypatch)
    ui = _ui()

    result = await bootstrap(_provider_args(tmp_path), ui, CliApprover(ui))

    assert result.restore_succeeded is False
    assert result.restore_error == ""
    assert result.main_session.agent_id == 0


@pytest.mark.asyncio
async def test_unknown_effort_tier_warns_at_startup(tmp_path, monkeypatch):
    """未知 effort 档位启动时警告一次，且原值进入 provider settings."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _provider_args(
            tmp_path, provider={"model": "my-model", "reasoning_effort": "ultra"}
        ),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["reasoning_effort"] == "ultra"
    assert any("reasoning_effort" in warning for warning in ui.warnings)


@pytest.mark.asyncio
async def test_known_effort_tier_is_silent(tmp_path, monkeypatch):
    """已知档位不警告，settings 正常携带."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _provider_args(
            tmp_path, provider={"model": "my-model", "reasoning_effort": "high"}
        ),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["reasoning_effort"] == "high"
    assert not any("reasoning_effort" in warning for warning in ui.warnings)


@pytest.mark.asyncio
async def test_unset_effort_passes_none_in_settings(tmp_path, monkeypatch):
    """未配置 effort 时 settings 携带 None，且不警告."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _provider_args(tmp_path, provider={"model": "my-model"}),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["reasoning_effort"] is None
    assert not any("reasoning_effort" in warning for warning in ui.warnings)
