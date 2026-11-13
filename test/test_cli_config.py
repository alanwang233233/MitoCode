"""Tests for provider credential loading and interactive setup in the CLI."""

import dataclasses
import json
import os
import stat
import sys
from io import StringIO
from pathlib import Path

import pytest
from loguru import logger as loguru_logger
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.cli.config as config_module

from mitocode.cli.theme import MITOCODE_THEME

from mitocode.cli.config import (
    DEFAULT_TIMEOUT,
    KNOWN_REASONING_EFFORTS,
    ProviderConfig,
    ask_provider_settings,
    resolve_provider_config,
    secret_path,
    write_secret_file,
)
from mitocode.cli.errors import MitoCodeCliConfigError


def _write_secret(workspace: Path, payload: str) -> Path:
    path = secret_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolated_global_config(tmp_path, monkeypatch):
    """Keep a developer's real global credential out of the test run."""
    monkeypatch.setattr(
        "mitocode.core.config.manager.default_global_dir",
        lambda **_: tmp_path / "global-config",
    )


def _console() -> tuple[Console, StringIO]:
    buffer = StringIO()
    return (
        Console(file=buffer, width=100, force_terminal=False, theme=MITOCODE_THEME),
        buffer,
    )


def _fake_asker(answers):
    """Build a scripted line reader and record its calls."""
    calls: list[dict] = []
    sequence = iter(answers)

    def ask(prompt: str, *, console: Console, password: bool = False) -> str:
        calls.append({"prompt": prompt, "password": password})
        return next(sequence)

    return ask, calls


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_secret_file_is_the_only_source(tmp_path):
    _write_secret(
        tmp_path,
        json.dumps(
            {
                "api_key": "secret-key",
                "model": "secret-model",
                "base_url": "http://endpoint",
                "timeout": 30,
                "reasoning_effort": "HIGH",
            }
        ),
    )

    cfg = resolve_provider_config(tmp_path)

    assert cfg == ProviderConfig(
        api_key="secret-key",
        base_url="http://endpoint",
        model="secret-model",
        timeout=30.0,
        reasoning_effort="high",
    )


def test_environment_variables_are_ignored(tmp_path, monkeypatch):
    """A populated process environment must not rescue a missing file."""
    monkeypatch.setenv("MITOCODE_API_KEY", "from-env")
    monkeypatch.setenv("MITOCODE_MODEL", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert str(secret_path(tmp_path)) in str(info.value)
    assert "from-env" not in str(info.value)


def test_dotenv_file_is_ignored(tmp_path):
    (tmp_path / ".env").write_text(
        "MITOCODE_API_KEY=k\nMITOCODE_MODEL=m\n", encoding="utf-8"
    )

    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(tmp_path)


def test_default_timeout_is_applied(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m"}')

    cfg = resolve_provider_config(tmp_path)

    assert cfg.timeout == DEFAULT_TIMEOUT
    assert isinstance(cfg.timeout, float)
    assert cfg.base_url is None
    assert cfg.reasoning_effort is None


def test_scalar_timeout_is_coerced_to_float(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m","timeout":"12.5"}')

    assert resolve_provider_config(tmp_path).timeout == 12.5


@pytest.mark.parametrize("value", ["0", "-1", "soon", "Infinity"])
def test_invalid_timeout_raises_and_names_the_file(tmp_path, value):
    _write_secret(
        tmp_path, json.dumps({"api_key": "k", "model": "m", "timeout": value})
    )

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert str(secret_path(tmp_path)) in str(info.value)


def test_boolean_timeout_is_ignored(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m","timeout":true}')

    assert resolve_provider_config(tmp_path).timeout == DEFAULT_TIMEOUT


def test_blank_optional_timeout_uses_the_default(tmp_path):
    """A whitespace-only optional field is absent, like the written file's omission."""
    _write_secret(tmp_path, '{"api_key":"k","model":"m","timeout":"   "}')

    assert resolve_provider_config(tmp_path).timeout == DEFAULT_TIMEOUT


def test_blank_fields_are_missing(tmp_path):
    _write_secret(tmp_path, '{"api_key":"   ","model":"m"}')

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert "api_key" in str(info.value)


def test_non_string_base_url_is_ignored(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m","base_url":42}')

    assert resolve_provider_config(tmp_path).base_url is None


def test_missing_fields_error_names_only_the_missing_key(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k"}')

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    headline = str(info.value).splitlines()[0]
    assert "model" in headline
    assert "api_key" not in headline


def test_repr_hides_api_key(tmp_path):
    _write_secret(tmp_path, '{"api_key":"super-secret","model":"m"}')

    cfg = resolve_provider_config(tmp_path)

    assert "super-secret" not in repr(cfg)


def test_secret_api_key_never_appears_in_errors(tmp_path):
    _write_secret(tmp_path, '{"api_key":"super-secret"}')

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert "super-secret" not in str(info.value)


def test_missing_file_error_mentions_the_path_and_fields(tmp_path):
    path = secret_path(tmp_path)

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    message = str(info.value)
    assert str(path) in message
    assert "api_key" in message and "model" in message


def test_malformed_json_is_fatal_and_names_the_file(tmp_path):
    _write_secret(tmp_path, "{not json")

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert str(secret_path(tmp_path)) in str(info.value)
    assert "JSON" in str(info.value)


def test_non_object_root_is_fatal(tmp_path):
    _write_secret(tmp_path, '["api_key", "model"]')

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert "JSON object" in str(info.value)


def test_unreadable_secret_file_is_fatal(tmp_path):
    secret_path(tmp_path).mkdir(parents=True)

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert str(secret_path(tmp_path)) in str(info.value)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs extra privileges")
def test_dangling_secret_symlink_is_fatal_and_preserved(tmp_path):
    path = secret_path(tmp_path)
    path.parent.mkdir(parents=True)
    target = tmp_path / "missing-provider.json"
    path.symlink_to(target)

    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(tmp_path)

    assert str(path) in str(info.value)
    assert path.is_symlink()
    assert not target.exists()


def test_provider_config_is_frozen(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m"}')

    cfg = resolve_provider_config(tmp_path)

    assert isinstance(cfg, ProviderConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.api_key = "other"


def test_provider_config_api_key_field_hides_from_repr():
    api_key_field = {f.name: f for f in dataclasses.fields(ProviderConfig)}["api_key"]

    assert api_key_field.repr is False


# ---------------------------------------------------------------------------
# reasoning_effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", sorted(KNOWN_REASONING_EFFORTS))
def test_known_tier_is_lowercased_and_does_not_warn(tmp_path, tier):
    records: list = []
    handler_id = loguru_logger.add(records.append, level="WARNING")
    try:
        _write_secret(
            tmp_path,
            json.dumps(
                {"api_key": "k", "model": "m", "reasoning_effort": tier.upper()}
            ),
        )
        cfg = resolve_provider_config(tmp_path)
    finally:
        loguru_logger.remove(handler_id)

    assert cfg.reasoning_effort == tier
    assert not any("reasoning_effort" in str(record) for record in records)


def test_unknown_tier_warns_but_is_accepted(tmp_path):
    records: list = []
    handler_id = loguru_logger.add(records.append, level="WARNING")
    try:
        _write_secret(
            tmp_path,
            '{"api_key":"k","model":"m","reasoning_effort":"ultra"}',
        )
        cfg = resolve_provider_config(tmp_path)
    finally:
        loguru_logger.remove(handler_id)

    assert cfg.reasoning_effort == "ultra"
    assert any("reasoning_effort" in str(record) for record in records)


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------


def test_prompt_collects_required_and_optional_settings():
    console, buffer = _console()
    ask, calls = _fake_asker(["key", "model-x", "http://endpoint", "12", "HIGH"])

    values = ask_provider_settings(console, ask=ask)

    assert values == {
        "api_key": "key",
        "model": "model-x",
        "base_url": "http://endpoint",
        "timeout": 12.0,
        "reasoning_effort": "high",
    }
    assert calls[0]["password"] is True
    assert any(not call["password"] for call in calls[1:])
    assert "key" not in buffer.getvalue()


def test_prompt_skips_blank_optional_fields(tmp_path):
    console, _ = _console()
    ask, _calls = _fake_asker(["key", "model-x", "", "", ""])

    values = ask_provider_settings(console, ask=ask)

    assert values == {"api_key": "key", "model": "model-x"}


def test_prompt_reasks_blank_required_fields():
    console, buffer = _console()
    ask, _calls = _fake_asker(["", "key", "", "model-x", "", "", ""])

    values = ask_provider_settings(console, ask=ask)

    assert values == {"api_key": "key", "model": "model-x"}
    assert "cannot be blank" in buffer.getvalue()


def test_prompt_reasks_invalid_timeout():
    console, buffer = _console()
    ask, _calls = _fake_asker(["key", "m", "", "0", "soon", "30", ""])

    values = ask_provider_settings(console, ask=ask)

    assert values["timeout"] == 30.0
    assert "greater than zero" in buffer.getvalue()
    assert "must be a number" in buffer.getvalue()


def test_prompt_accepts_unknown_effort_with_a_warning():
    console, buffer = _console()
    ask, _calls = _fake_asker(["key", "m", "", "", "ultra"])

    values = ask_provider_settings(console, ask=ask)

    assert values["reasoning_effort"] == "ultra"
    assert "not a known effort tier" in buffer.getvalue()


# ---------------------------------------------------------------------------
# Writing the file
# ---------------------------------------------------------------------------


def test_write_secret_file_is_private_and_drops_blank_values(tmp_path):
    path = write_secret_file(
        tmp_path,
        {
            "api_key": "k",
            "model": "m",
            "base_url": "",
            "timeout": 30.0,
            "reasoning_effort": None,
        },
    )

    assert path == secret_path(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {"api_key": "k", "model": "m", "timeout": 30.0}
    # Windows has no mode bits: os.chmod there only toggles the read-only
    # attribute, and access is decided by the directory's ACL.
    if os.name != "nt":
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not path.with_name(path.name + ".tmp").exists()


def test_written_file_round_trips_through_resolver(tmp_path):
    write_secret_file(tmp_path, {"api_key": "k", "model": "m"})

    cfg = resolve_provider_config(tmp_path)

    assert (cfg.api_key, cfg.model) == ("k", "m")


def test_write_failure_is_reported_without_values(tmp_path):
    (tmp_path / ".mitocode").mkdir()
    (tmp_path / ".mitocode" / "secret").write_text(
        "blocks the directory", encoding="utf-8"
    )

    with pytest.raises(MitoCodeCliConfigError) as info:
        write_secret_file(tmp_path, {"api_key": "k", "model": "m"})

    assert str(secret_path(tmp_path)) in str(info.value)


def test_write_secret_file_refuses_to_overwrite_existing(tmp_path):
    """Even a malformed credential file must survive a second setup attempt."""
    path = _write_secret(tmp_path, '{"api_key": "old-secret"')
    original = path.read_text(encoding="utf-8")

    with pytest.raises(MitoCodeCliConfigError) as info:
        write_secret_file(tmp_path, {"api_key": "new-secret", "model": "m"})

    assert path.read_text(encoding="utf-8") == original
    assert "new-secret" not in str(info.value)
    assert "old-secret" not in str(info.value)


def test_write_failure_leaves_no_temporary_file(tmp_path, monkeypatch):
    """A failed atomic move must not strand a secret-bearing temporary file."""

    def _fail_replace(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(config_module.os, "replace", _fail_replace)

    with pytest.raises(MitoCodeCliConfigError):
        write_secret_file(tmp_path, {"api_key": "k", "model": "m"})

    secret_dir = secret_path(tmp_path).parent
    assert list(secret_dir.glob("*.tmp")) == []
    assert not secret_path(tmp_path).exists()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs extra privileges")
def test_write_secret_file_refuses_a_secret_dir_outside_the_workspace(tmp_path):
    """.mitocode/secret 指向工作区外时，密钥不得被写到工作区外."""
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    (tmp_path / ".mitocode").mkdir()
    (tmp_path / ".mitocode" / "secret").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MitoCodeCliConfigError) as info:
        write_secret_file(tmp_path, {"api_key": "sk-leaked", "model": "m"})

    # The key must not have landed outside the workspace.
    assert not (outside / "provider.json").exists()
    assert "sk-leaked" not in str(info.value)
    assert str(outside) in str(info.value)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs extra privileges")
def test_write_secret_file_refuses_a_secret_file_outside_the_workspace(tmp_path):
    """provider.json 指向工作区外的符号链接时不得把密钥写到那里."""
    outside = tmp_path.parent / f"outside-file-{tmp_path.name}"
    outside.mkdir()
    target = outside / "provider.json"
    (tmp_path / ".mitocode").mkdir()
    (tmp_path / ".mitocode" / "secret").mkdir()
    secret_path(tmp_path).symlink_to(target)

    with pytest.raises(MitoCodeCliConfigError) as info:
        write_secret_file(tmp_path, {"api_key": "sk-leaked", "model": "m"})

    assert not target.exists()
    assert "sk-leaked" not in str(info.value)


def test_provider_config_falls_back_to_a_global_copy(tmp_path):
    """A workspace without its own credential can use the user-level copy."""
    global_dir = tmp_path / "global"
    global_secret = global_dir / "secret" / "provider.json"
    global_secret.parent.mkdir(parents=True)
    global_secret.write_text(
        json.dumps({"api_key": "global-key", "model": "global-model"}),
        encoding="utf-8",
    )

    cfg = resolve_provider_config(tmp_path, global_dir=global_dir)

    assert cfg.api_key == "global-key"
    assert cfg.model == "global-model"


def test_project_provider_config_wins_over_the_global_copy(tmp_path):
    """The project credential overrides a user-level one."""
    global_dir = tmp_path / "global"
    global_secret = global_dir / "secret" / "provider.json"
    global_secret.parent.mkdir(parents=True)
    global_secret.write_text(
        json.dumps({"api_key": "global-key", "model": "global-model"}),
        encoding="utf-8",
    )
    _write_secret(tmp_path, json.dumps({"api_key": "project-key", "model": "m"}))

    cfg = resolve_provider_config(tmp_path, global_dir=global_dir)

    assert cfg.api_key == "project-key"
