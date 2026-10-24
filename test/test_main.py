"""Tests for the Typer CLI entry point in :mod:`mitocode.__main__`.

The TUI launch path is exercised end to end with the real workspace, tool
registry, provider construction and agent creation; only the Textual
application itself and ``time.sleep`` are replaced, since a terminal and a
three second wait have no place in an offline test suite.
"""

import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from mitocode import __main__ as cli
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.core.tool.builtin.context import ToolContext, set_current
from mitocode.i18n import set_locale

runner = CliRunner()

#: Every environment variable the CLI layer reads, cleared between tests.
CLI_ENV_VARS = (
    "MITOCODE_LANG",
    "LC_ALL",
    "LC_MESSAGES",
    "LANG",
    "MITOCODE_API_KEY",
    "OPENAI_API_KEY",
    "MITOCODE_BASE_URL",
    "OPENAI_BASE_URL",
    "MITOCODE_MODEL",
    "MITOCODE_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """Snapshot and restore process-global state around each test.

    ``load_dotenv``, the translator and the built-in tool context all mutate
    state that ``monkeypatch`` does not track, so the environment is
    snapshotted before the test and restored after it.  CLI-relevant
    variables are cleared up front so the local shell's ``LANG`` or API keys
    cannot leak into assertions.
    """
    saved_env = dict(os.environ)
    set_locale(None)
    set_current(ToolContext())
    for name in CLI_ENV_VARS:
        os.environ.pop(name, None)
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    set_locale(None)
    set_current(ToolContext())


class _RecordingProvider(_SampleChatProvider):
    """Provider double that records its factory arguments and instance."""

    instances: List["_RecordingProvider"] = []

    @classmethod
    async def factory(cls, **kwargs) -> "_RecordingProvider":
        """Mirror the base factory: construct, await init, record."""
        instance = cls(**kwargs)
        await instance.init()
        return instance

    @classmethod
    def get_settings_schema(cls) -> Dict:
        """Mirror the OpenAI provider's schema shape."""
        return {"type": "object", "required": ["api_key"]}

    @classmethod
    def check_settings_schema(cls, settings: Any) -> bool:
        """Require a non-blank ``api_key`` string like the real provider."""
        if not isinstance(settings, dict):
            return False
        api_key = settings.get("api_key", "")
        return isinstance(api_key, str) and len(api_key.strip()) > 0

    async def init(self) -> None:
        """Record the instance instead of opening an API connection."""
        _RecordingProvider.instances.append(self)

    async def get_model_list(self) -> List[str]:
        """Return no models; nothing in the launch path reads the list."""
        return []

    async def chat(self, **kwargs: Any) -> AsyncGenerator | Dict:
        """Yield nothing; the launch path never sends a completion."""
        yield {}


@pytest.fixture(autouse=True)
def _stub_tui_boundaries(monkeypatch):
    """Replace the OpenAI provider, the Textual app and ``time.sleep``.

    Returns:
        A dict with the installed doubles: ``frontend`` (the
        :class:`MagicMock` standing in for :class:`VibeFrontend`) and
        ``slept`` (the list of values passed to the sleep stub).
    """
    _RecordingProvider.instances = []
    monkeypatch.setattr(cli, "ChatProviderOpenAI", _RecordingProvider)
    frontend_cls = MagicMock(name="VibeFrontend")
    monkeypatch.setattr(cli, "VibeFrontend", frontend_cls)
    slept: List[Any] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))
    return {"frontend": frontend_cls, "slept": slept}


def test_tui_launches_with_env_key(tmp_path, monkeypatch, _stub_tui_boundaries):
    """`tui` builds provider, workspace and Main agent, then runs the TUI."""
    monkeypatch.chdir(tmp_path)
    os.environ["MITOCODE_API_KEY"] = "env-key"

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    assert "deprecated" in result.output
    provider = _RecordingProvider.instances[0]
    assert provider.settings["api_key"] == "env-key"
    assert provider.settings["base_url"] == cli.DEFAULT_BASE_URL
    assert provider.settings["timeout"] == cli.DEFAULT_TIMEOUT_SECONDS
    assert provider.default_model == cli.DEFAULT_MODEL
    assert (tmp_path / ".mitocode" / "config.json").is_file()
    assert _stub_tui_boundaries["slept"] == [3]
    _stub_tui_boundaries["frontend"].return_value.run.assert_called_once_with()


def test_tui_reads_api_key_from_env_file(tmp_path, monkeypatch):
    """A .env file in the current directory supplies the provider key."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MITOCODE_API_KEY=from-file\n")

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    provider = _RecordingProvider.instances[0]
    assert provider.settings["api_key"] == "from-file"
    assert "Loaded provider settings" in result.output


def test_tui_environment_wins_over_env_file(tmp_path, monkeypatch):
    """Variables already in the environment take precedence over .env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MITOCODE_API_KEY=from-file\n")
    os.environ["MITOCODE_API_KEY"] = "from-env"

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    assert _RecordingProvider.instances[0].settings["api_key"] == "from-env"


def test_tui_openai_env_vars_are_accepted(tmp_path, monkeypatch):
    """The de-facto OPENAI_* spellings configure the provider too."""
    monkeypatch.chdir(tmp_path)
    os.environ["OPENAI_API_KEY"] = "openai-key"
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    provider = _RecordingProvider.instances[0]
    assert provider.settings["api_key"] == "openai-key"
    assert provider.settings["base_url"] == "https://api.openai.com/v1"


def test_tui_model_and_timeout_overrides(tmp_path, monkeypatch):
    """MITOCODE_MODEL and MITOCODE_TIMEOUT reach the provider settings."""
    monkeypatch.chdir(tmp_path)
    os.environ["MITOCODE_API_KEY"] = "k"
    os.environ["MITOCODE_MODEL"] = "custom/model"
    os.environ["MITOCODE_TIMEOUT"] = "90"

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    provider = _RecordingProvider.instances[0]
    assert provider.default_model == "custom/model"
    assert provider.settings["timeout"] == 90


def test_tui_invalid_timeout_falls_back_to_default(tmp_path, monkeypatch):
    """A non-integer timeout is reported and the default is used."""
    monkeypatch.chdir(tmp_path)
    os.environ["MITOCODE_API_KEY"] = "k"
    os.environ["MITOCODE_TIMEOUT"] = "soon"

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    assert "Invalid MITOCODE_TIMEOUT" in result.output
    assert (
        _RecordingProvider.instances[0].settings["timeout"]
        == cli.DEFAULT_TIMEOUT_SECONDS
    )


def test_tui_without_api_key_exits_with_error(
    tmp_path, monkeypatch, _stub_tui_boundaries
):
    """No key anywhere: the command explains where to put one and exits 1."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 1
    assert "MITOCODE_API_KEY" in result.output
    assert _RecordingProvider.instances == []
    _stub_tui_boundaries["frontend"].assert_not_called()


def test_tui_corrupt_workspace_config_surfaces_error(
    tmp_path, monkeypatch, _stub_tui_boundaries
):
    """A broken workspace config is reported, not silently recreated."""
    monkeypatch.chdir(tmp_path)
    os.environ["MITOCODE_API_KEY"] = "k"
    config_dir = tmp_path / ".mitocode"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("not json")

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 1
    assert "Failed to launch MitoCode" in result.output
    assert (config_dir / "config.json").is_file()
    _stub_tui_boundaries["frontend"].assert_not_called()


def test_tui_zh_locale_localizes_output(tmp_path, monkeypatch):
    """MITOCODE_LANG=zh renders the localized messages."""
    monkeypatch.chdir(tmp_path)
    os.environ["MITOCODE_LANG"] = "zh_CN.UTF-8"

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 1
    assert "未找到" in result.output
    assert "MITOCODE_API_KEY" in result.output


def test_tui_grace_wait_happens_before_launch(
    tmp_path, monkeypatch, _stub_tui_boundaries
):
    """The deprecation notice and wait precede the Textual app running."""
    monkeypatch.chdir(tmp_path)
    os.environ["MITOCODE_API_KEY"] = "k"
    slept = _stub_tui_boundaries["slept"]
    frontend = _stub_tui_boundaries["frontend"]
    frontend.return_value.run.side_effect = lambda: slept.append("run")

    result = runner.invoke(cli.app, ["tui"])

    assert result.exit_code == 0
    assert slept == [3, "run"]


def test_cli_placeholder_reports_not_implemented():
    """`cli` is a placeholder that says so and exits non-zero."""
    result = runner.invoke(cli.app, ["cli"])

    assert result.exit_code == 1
    assert "'cli' is not implemented yet." in result.output


def test_webui_placeholder_reports_not_implemented():
    """`webui` is a placeholder that says so and exits non-zero."""
    result = runner.invoke(cli.app, ["webui"])

    assert result.exit_code == 1
    assert "'webui' is not implemented yet." in result.output


def test_bare_invocation_lists_commands():
    """Invoking without a subcommand lists the available commands."""
    result = runner.invoke(cli.app, [])

    assert "tui" in result.output
    assert "cli" in result.output
    assert "webui" in result.output
