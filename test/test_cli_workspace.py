"""Tests for workspace resolution at CLI startup."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.cli.bootstrap as bootstrap
from mitocode import APP_VERSION
from mitocode.cli.bootstrap import WorkspaceResolution, resolve_workspace
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.core.exception.mitocode_exception import MitoCodeMigrateError
from mitocode.core.workspace.workspace import Workspace


def _write_config(root: Path, payload: str) -> Path:
    config = root / ".mitocode" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(payload, encoding="utf-8")
    return config


@pytest.mark.asyncio
async def test_missing_workspace_uses_in_memory_config(tmp_path):
    result = await resolve_workspace(tmp_path)

    assert isinstance(result, WorkspaceResolution)
    assert result.initialized is False
    assert result.migrated_from is None
    assert isinstance(result.workspace, Workspace)
    assert result.workspace.path == tmp_path
    assert result.workspace.config["version"] == APP_VERSION
    assert result.workspace.config["name"] == tmp_path.name
    assert not (tmp_path / ".mitocode").exists()


@pytest.mark.asyncio
async def test_current_workspace_loads_without_migration(tmp_path):
    _write_config(
        tmp_path, json.dumps({"version": APP_VERSION, "name": "x", "description": ""})
    )

    result = await resolve_workspace(tmp_path)

    assert result.initialized is True
    assert result.migrated_from is None
    assert result.workspace.config["version"] == APP_VERSION
    assert result.workspace.config["name"] == "x"


@pytest.mark.asyncio
async def test_outdated_workspace_is_migrated(tmp_path):
    config = _write_config(
        tmp_path, json.dumps({"version": 1, "name": "x", "description": ""})
    )

    result = await resolve_workspace(tmp_path)

    assert result.initialized is True
    assert result.migrated_from == 1
    assert result.workspace.config["version"] == APP_VERSION
    assert json.loads(config.read_text(encoding="utf-8"))["version"] == APP_VERSION


@pytest.mark.asyncio
async def test_newer_workspace_is_fatal(tmp_path):
    _write_config(tmp_path, json.dumps({"version": 99, "name": "x", "description": ""}))

    with pytest.raises(MitoCodeCliConfigError):
        await resolve_workspace(tmp_path)


@pytest.mark.asyncio
async def test_invalid_workspace_json_is_fatal(tmp_path):
    _write_config(tmp_path, "{nope")

    with pytest.raises(MitoCodeCliConfigError):
        await resolve_workspace(tmp_path)


@pytest.mark.asyncio
async def test_config_without_version_is_fatal(tmp_path):
    _write_config(tmp_path, json.dumps({"name": "x", "description": ""}))

    with pytest.raises(MitoCodeCliConfigError):
        await resolve_workspace(tmp_path)


@pytest.mark.asyncio
async def test_error_message_names_path_but_not_file_contents(tmp_path):
    _write_config(tmp_path, '{"version": 99, "name": "confidential-name"}')

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await resolve_workspace(tmp_path)

    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "confidential-name" not in message


@pytest.mark.asyncio
async def test_invalid_json_message_names_path_but_not_contents(tmp_path):
    config = _write_config(tmp_path, '{"secret-value": nope')

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await resolve_workspace(tmp_path)

    message = str(excinfo.value)
    assert str(config) in message
    assert "secret-value" not in message


@pytest.mark.asyncio
async def test_missing_workspace_does_not_touch_disk(tmp_path):
    await resolve_workspace(tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_config_with_invalid_utf8_is_fatal(tmp_path):
    config = tmp_path / ".mitocode" / "config.json"
    config.parent.mkdir()
    config.write_bytes(b'{"version": 2, "name": "\xff\xfe"}')

    with pytest.raises(MitoCodeCliConfigError):
        await resolve_workspace(tmp_path)


@pytest.mark.asyncio
async def test_migration_failure_is_fatal(tmp_path, monkeypatch):
    _write_config(tmp_path, json.dumps({"version": 1, "name": "x", "description": ""}))

    class _FailingMigration:
        @staticmethod
        def migrate(path):
            raise MitoCodeMigrateError("boom")

    monkeypatch.setattr(bootstrap, "BasicMigration", _FailingMigration)

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await resolve_workspace(tmp_path)

    assert str(tmp_path) in str(excinfo.value)
