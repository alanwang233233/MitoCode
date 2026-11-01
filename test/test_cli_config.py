"""Tests for provider configuration resolution in the MitoCode CLI."""

import dataclasses
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.args import CliArgs
from mitocode.cli.config import (
    DEFAULT_TIMEOUT,
    DOTENV_FILENAME,
    SECRET_RELATIVE_PATH,
    ProviderConfig,
    dotenv_values,
    resolve_provider_config,
)
from mitocode.cli.errors import MitoCodeCliConfigError


def _args(**over):
    base = dict(
        workspace=Path("."),
        model=None,
        api_key=None,
        base_url=None,
        timeout=None,
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


def _write_secret(workspace: Path, payload: str) -> Path:
    path = workspace / SECRET_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def test_precedence_flags_over_env_over_dotenv_over_secret(tmp_path):
    (tmp_path / ".env").write_text(
        "MITOCODE_API_KEY=from-dotenv\nMITOCODE_MODEL=dotenv-model\n", encoding="utf-8"
    )
    _write_secret(
        tmp_path,
        '{"api_key":"secret","base_url":"http://secret","model":"secret-m","timeout":7}',
    )
    cfg = resolve_provider_config(
        _args(api_key="flag", model="flag-m", base_url="http://flag"),
        tmp_path,
        environ={"MITOCODE_API_KEY": "env", "MITOCODE_MODEL": "env-m"},
    )
    assert (cfg.api_key, cfg.model, cfg.base_url, cfg.timeout) == (
        "flag",
        "flag-m",
        "http://flag",
        7.0,
    )


def test_precedence_env_over_dotenv_over_secret(tmp_path):
    (tmp_path / ".env").write_text(
        "MITOCODE_API_KEY=from-dotenv\nMITOCODE_MODEL=dotenv-model\n"
        "MITOCODE_TIMEOUT=5\n",
        encoding="utf-8",
    )
    _write_secret(tmp_path, '{"api_key":"secret","model":"secret-m","timeout":7}')
    cfg = resolve_provider_config(
        _args(),
        tmp_path,
        environ={"MITOCODE_API_KEY": "env", "MITOCODE_MODEL": "env-m"},
    )
    assert (cfg.api_key, cfg.model, cfg.base_url, cfg.timeout) == (
        "env",
        "env-m",
        None,
        5.0,
    )


def test_precedence_dotenv_over_secret_within_workspace_layers(tmp_path):
    (tmp_path / ".env").write_text(
        "MITOCODE_API_KEY=from-dotenv\nMITOCODE_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    _write_secret(
        tmp_path,
        '{"api_key":"secret","base_url":"http://secret","model":"secret-m","timeout":7}',
    )
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert (cfg.api_key, cfg.model, cfg.base_url, cfg.timeout) == (
        "from-dotenv",
        "dotenv-model",
        "http://secret",
        7.0,
    )


def test_workspace_base_url_with_environment_key_is_rejected(tmp_path):
    (tmp_path / ".env").write_text(
        "MITOCODE_BASE_URL=http://dotenv\n", encoding="utf-8"
    )
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(
            _args(),
            tmp_path,
            environ={"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"},
        )
    message = str(info.value)
    assert DOTENV_FILENAME in message
    assert "MITOCODE_BASE_URL" in message
    assert "http://dotenv" not in message


def test_secret_base_url_with_environment_key_is_rejected(tmp_path):
    _write_secret(tmp_path, '{"base_url":"http://secret"}')
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(
            _args(),
            tmp_path,
            environ={"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"},
        )
    message = str(info.value)
    assert str(SECRET_RELATIVE_PATH) in message
    assert "http://secret" not in message


def test_flag_key_with_workspace_base_url_is_rejected(tmp_path):
    _write_secret(tmp_path, '{"base_url":"http://secret"}')
    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(_args(api_key="k", model="m"), tmp_path, environ={})


def test_workspace_key_pairs_with_workspace_base_url(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m","base_url":"http://secret"}')
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert cfg.base_url == "http://secret"
    assert cfg.source == "secret"


def test_workspace_key_pairs_with_workspace_base_url_in_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "MITOCODE_API_KEY=k\nMITOCODE_MODEL=m\nMITOCODE_BASE_URL=http://dotenv\n",
        encoding="utf-8",
    )
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert cfg.base_url == "http://dotenv"


def test_workspace_key_pairs_with_environment_base_url(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m"}')
    cfg = resolve_provider_config(
        _args(), tmp_path, environ={"MITOCODE_BASE_URL": "http://env"}
    )
    assert cfg.base_url == "http://env"


def test_environment_key_pairs_with_environment_base_url(tmp_path):
    cfg = resolve_provider_config(
        _args(),
        tmp_path,
        environ={
            "MITOCODE_API_KEY": "k",
            "MITOCODE_MODEL": "m",
            "MITOCODE_BASE_URL": "http://env",
        },
    )
    assert cfg.base_url == "http://env"


def test_flag_base_url_pairs_with_environment_key(tmp_path):
    cfg = resolve_provider_config(
        _args(base_url="http://flag"),
        tmp_path,
        environ={"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"},
    )
    assert cfg.base_url == "http://flag"


def test_missing_credentials_names_keys_not_values(tmp_path):
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(_args(), tmp_path, environ={})
    message = str(info.value)
    assert "MITOCODE_API_KEY" in message and "MITOCODE_MODEL" in message
    assert "sk-" not in message and "super-secret" not in message


def test_repr_hides_api_key(tmp_path):
    cfg = resolve_provider_config(
        _args(api_key="super-secret", model="m"), tmp_path, environ={}
    )
    assert "super-secret" not in repr(cfg)


def test_timeout_must_be_positive(tmp_path):
    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(
            _args(timeout=0.0, api_key="k", model="m"), tmp_path, environ={}
        )


def test_missing_error_mentions_local_file_locations(tmp_path):
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(_args(), tmp_path, environ={})
    message = str(info.value)
    assert DOTENV_FILENAME in message
    assert str(SECRET_RELATIVE_PATH) in message


def test_missing_error_names_only_the_missing_key(tmp_path):
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(_args(api_key="k"), tmp_path, environ={})
    message = str(info.value)
    headline = message.splitlines()[0]
    assert "model" in headline
    assert "api_key" not in headline
    assert "MITOCODE_MODEL" in message


@pytest.mark.parametrize(
    ("overrides", "environ", "dotenv_text", "secret_text", "expected"),
    [
        ({"api_key": "k", "model": "m"}, {}, "", None, "flags"),
        ({}, {"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"}, "", None, "environment"),
        (
            {},
            {},
            "MITOCODE_API_KEY=k\nMITOCODE_MODEL=m\n",
            None,
            "dotenv",
        ),
        ({}, {}, "", '{"api_key":"k","model":"m"}', "secret"),
    ],
)
def test_source_records_winning_layer(
    tmp_path, overrides, environ, dotenv_text, secret_text, expected
):
    if dotenv_text:
        (tmp_path / ".env").write_text(dotenv_text, encoding="utf-8")
    if secret_text:
        _write_secret(tmp_path, secret_text)
    cfg = resolve_provider_config(_args(**overrides), tmp_path, environ=environ)
    assert cfg.source == expected
    assert cfg.api_key == "k"


def test_default_timeout_is_applied(tmp_path):
    cfg = resolve_provider_config(_args(api_key="k", model="m"), tmp_path, environ={})
    assert cfg.timeout == DEFAULT_TIMEOUT
    assert isinstance(cfg.timeout, float)


def test_timeout_strings_are_coerced_to_float(tmp_path):
    cfg = resolve_provider_config(
        _args(api_key="k", model="m"), tmp_path, environ={"MITOCODE_TIMEOUT": "12.5"}
    )
    assert cfg.timeout == 12.5


@pytest.mark.parametrize("value", ["0", "-1", "soon"])
def test_invalid_environment_timeout_raises(tmp_path, value):
    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(
            _args(api_key="k", model="m"),
            tmp_path,
            environ={"MITOCODE_TIMEOUT": value},
        )


def test_blank_timeout_environment_value_uses_default(tmp_path):
    cfg = resolve_provider_config(
        _args(api_key="k", model="m"), tmp_path, environ={"MITOCODE_TIMEOUT": ""}
    )
    assert cfg.timeout == DEFAULT_TIMEOUT


def test_openai_aliases_are_honoured_in_environment(tmp_path):
    cfg = resolve_provider_config(
        _args(),
        tmp_path,
        environ={
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "http://openai",
            "MITOCODE_MODEL": "m",
        },
    )
    assert (cfg.api_key, cfg.base_url, cfg.model) == (
        "openai-key",
        "http://openai",
        "m",
    )


def test_openai_aliases_are_honoured_in_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=openai-key\nOPENAI_BASE_URL=http://openai\n"
        "MITOCODE_MODEL=m\n",
        encoding="utf-8",
    )
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert (cfg.api_key, cfg.base_url) == ("openai-key", "http://openai")


def test_blank_values_do_not_win(tmp_path):
    _write_secret(
        tmp_path, '{"api_key":"secret-key","model":"secret-m","base_url":"  "}'
    )
    cfg = resolve_provider_config(
        _args(api_key="", base_url=" "),
        tmp_path,
        environ={"MITOCODE_API_KEY": "  "},
    )
    assert cfg.api_key == "secret-key"
    assert cfg.model == "secret-m"
    assert cfg.base_url is None


def test_secret_file_alone_is_enough(tmp_path):
    _write_secret(tmp_path, '{"api_key":"secret-key","model":"secret-m","timeout":30}')
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert (cfg.api_key, cfg.model, cfg.timeout) == ("secret-key", "secret-m", 30.0)


def test_unparsable_secret_file_is_treated_as_absent(tmp_path):
    _write_secret(tmp_path, "{not json")
    cfg = resolve_provider_config(
        _args(), tmp_path, environ={"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"}
    )
    assert (cfg.api_key, cfg.model) == ("k", "m")
    assert cfg.source == "environment"


def test_unparsable_secret_file_does_not_raise_when_env_is_empty(tmp_path):
    _write_secret(tmp_path, "{not json")
    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(_args(), tmp_path, environ={})


def test_secret_file_with_non_object_root_is_ignored(tmp_path):
    _write_secret(tmp_path, '["api_key", "model"]')
    cfg = resolve_provider_config(
        _args(), tmp_path, environ={"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"}
    )
    assert cfg.source == "environment"


def test_secret_file_field_types_are_normalised(tmp_path):
    _write_secret(
        tmp_path, '{"api_key": ["not", "a", "string"], "model": "m", "timeout": "9"}'
    )
    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(_args(), tmp_path, environ={})

    _write_secret(tmp_path, '{"api_key":"k","model":"m","timeout":"9","base_url":42}')
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert cfg.timeout == 9.0
    assert cfg.base_url is None


def test_secret_boolean_timeout_is_ignored(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m","timeout":true}')
    cfg = resolve_provider_config(_args(), tmp_path, environ={})
    assert cfg.timeout == DEFAULT_TIMEOUT


def test_unreadable_secret_file_is_treated_as_absent(tmp_path):
    (tmp_path / SECRET_RELATIVE_PATH).mkdir(parents=True)
    cfg = resolve_provider_config(
        _args(), tmp_path, environ={"MITOCODE_API_KEY": "k", "MITOCODE_MODEL": "m"}
    )
    assert cfg.source == "environment"


def test_invalid_flag_timeout_error_mentions_the_flag(tmp_path):
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(
            _args(api_key="k", model="m", timeout=-5.0), tmp_path, environ={}
        )
    assert "--timeout" in str(info.value)


def test_invalid_environment_timeout_error_mentions_the_variable(tmp_path):
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(
            _args(api_key="k", model="m"),
            tmp_path,
            environ={"MITOCODE_TIMEOUT": "-2"},
        )
    assert "MITOCODE_TIMEOUT" in str(info.value)


def test_invalid_dotenv_timeout_error_mentions_the_file(tmp_path):
    (tmp_path / DOTENV_FILENAME).write_text("MITOCODE_TIMEOUT=-2\n", encoding="utf-8")
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(_args(api_key="k", model="m"), tmp_path, environ={})
    assert DOTENV_FILENAME in str(info.value)


def test_invalid_secret_timeout_error_mentions_the_file(tmp_path):
    _write_secret(tmp_path, '{"api_key":"k","model":"m","timeout":-2}')
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(_args(), tmp_path, environ={})
    assert str(SECRET_RELATIVE_PATH) in str(info.value)


def test_dotenv_values_missing_file_returns_empty(tmp_path):
    assert not dotenv_values(tmp_path / "nope.env")
    assert dotenv_values(tmp_path) == {}


def test_dotenv_values_undecodable_file_is_treated_as_absent(tmp_path):
    (tmp_path / DOTENV_FILENAME).write_bytes(b"\xff\xfeMITOCODE_API_KEY=k\n")
    assert dotenv_values(tmp_path / DOTENV_FILENAME) == {}

    with pytest.raises(MitoCodeCliConfigError):
        resolve_provider_config(_args(), tmp_path, environ={})


def test_dotenv_values_does_not_mutate_environ(tmp_path, monkeypatch):
    monkeypatch.delenv("MITOCODE_MODEL", raising=False)
    (tmp_path / DOTENV_FILENAME).write_text(
        "MITOCODE_MODEL=from-file\n", encoding="utf-8"
    )
    before = dict(os.environ)

    resolved = dotenv_values(tmp_path / DOTENV_FILENAME)

    assert resolved == {"MITOCODE_MODEL": "from-file"}
    assert dict(os.environ) == before
    assert "MITOCODE_MODEL" not in os.environ


def test_resolve_does_not_mutate_environ(tmp_path, monkeypatch):
    monkeypatch.delenv("MITOCODE_API_KEY", raising=False)
    (tmp_path / DOTENV_FILENAME).write_text(
        "MITOCODE_API_KEY=from-file\nMITOCODE_MODEL=m\n", encoding="utf-8"
    )
    before = dict(os.environ)

    resolve_provider_config(_args(), tmp_path, environ={})

    assert dict(os.environ) == before


def test_environment_defaults_to_process_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("MITOCODE_API_KEY", "from-process")
    monkeypatch.setenv("MITOCODE_MODEL", "process-model")
    cfg = resolve_provider_config(_args(), tmp_path)
    assert (cfg.api_key, cfg.model) == ("from-process", "process-model")


def test_provider_config_is_frozen(tmp_path):
    cfg = resolve_provider_config(_args(api_key="k", model="m"), tmp_path, environ={})
    assert isinstance(cfg, ProviderConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.api_key = "other"


def test_provider_config_api_key_field_hides_from_repr():
    api_key_field = {f.name: f for f in dataclasses.fields(ProviderConfig)}["api_key"]
    assert api_key_field.repr is False


def test_secret_api_key_never_appears_in_errors(tmp_path):
    _write_secret(tmp_path, '{"api_key":"super-secret"}')
    with pytest.raises(MitoCodeCliConfigError) as info:
        resolve_provider_config(_args(), tmp_path, environ={})
    message = str(info.value)
    assert "super-secret" not in message
    assert "MITOCODE_MODEL" in message


def test_blank_workspace_base_url_falls_through_to_environment(tmp_path):
    """A blank workspace base_url loses the layer contest, so env wins legally."""
    _write_secret(
        tmp_path, '{"api_key":"secret-key","model":"secret-m","base_url":"  "}'
    )
    cfg = resolve_provider_config(
        _args(), tmp_path, environ={"MITOCODE_BASE_URL": "http://env"}
    )
    assert cfg.base_url == "http://env"
    assert cfg.source == "secret"
