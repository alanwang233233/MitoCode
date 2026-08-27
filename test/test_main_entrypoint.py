import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.__main__ import (  # noqa: E402
    _default_provider_kind,
    _init_workspace,
    _parse_provider_settings,
)


def test_parse_provider_settings_accepts_empty():
    assert _parse_provider_settings("") == {}


def test_parse_provider_settings_rejects_non_object():
    with pytest.raises(ValueError):
        _parse_provider_settings('["bad"]')


def test_init_workspace_creates_then_loads(tmp_path):
    root = tmp_path / "ws"
    ws1 = _init_workspace(str(root), "demo")
    assert ws1.config["name"] == "demo"
    assert ws1.config_file.exists()

    ws2 = _init_workspace(str(root), "ignored")
    assert ws2.config["name"] == "demo"


def test_default_provider_kind_uses_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _default_provider_kind() == "scenario"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert _default_provider_kind() == "openai"
