"""Tests for workspace resolution at CLI startup."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.cli.bootstrap as bootstrap
from mitocode import APP_VERSION
from mitocode.cli.bootstrap import (
    WorkspaceResolution,
    ensure_workspace,
    resolve_workspace,
)
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.core.exception.mitocode_exception import MitoCodeMigrateError
from mitocode.core.workspace.workspace import Workspace


def _write_config(root: Path, payload: str) -> Path:
    config = root / ".mitocode" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(payload, encoding="utf-8")
    return config


@pytest.mark.asyncio
async def test_missing_workspace_is_fatal(tmp_path):
    with pytest.raises(MitoCodeCliConfigError) as info:
        await resolve_workspace(tmp_path)

    assert str(tmp_path) in str(info.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_current_workspace_loads_without_migration(tmp_path):
    _write_config(
        tmp_path, json.dumps({"version": APP_VERSION, "name": "x", "description": ""})
    )

    result = await resolve_workspace(tmp_path)

    assert isinstance(result, WorkspaceResolution)
    assert result.migrated_from is None
    assert result.workspace.config["version"] == APP_VERSION
    assert result.workspace.config["name"] == "x"


@pytest.mark.asyncio
async def test_outdated_workspace_is_migrated(tmp_path):
    config = _write_config(
        tmp_path, json.dumps({"version": 1, "name": "x", "description": ""})
    )

    result = await resolve_workspace(tmp_path)

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
    with pytest.raises(MitoCodeCliConfigError):
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


# ---------------------------------------------------------------------------
# ensure_workspace (the pre-logging confirmation step)
# ---------------------------------------------------------------------------


def test_ensure_workspace_keeps_an_existing_configuration(tmp_path):
    _write_config(tmp_path, json.dumps({"version": APP_VERSION, "name": "x"}))
    confirmed: list[Path] = []

    def _accept(path: Path) -> bool:
        confirmed.append(path)
        return True

    assert ensure_workspace(tmp_path, confirm=_accept) is True
    assert confirmed == []


def test_ensure_workspace_declined_leaves_no_files(tmp_path):
    confirmed: list[Path] = []

    def _decline(path: Path) -> bool:
        confirmed.append(path)
        return False

    assert ensure_workspace(tmp_path, confirm=_decline) is False
    assert confirmed == [tmp_path]
    assert not (tmp_path / ".mitocode").exists()


def test_ensure_workspace_creates_private_configuration(tmp_path):
    confirmed: list[Path] = []

    def _accept(path: Path) -> bool:
        confirmed.append(path)
        return True

    assert ensure_workspace(tmp_path, confirm=_accept) is True
    assert confirmed == [tmp_path]

    config = tmp_path / ".mitocode" / "config.json"
    document = json.loads(config.read_text(encoding="utf-8"))
    assert document["name"] == tmp_path.name
    assert document["version"] == APP_VERSION
    assert stat.S_IMODE((tmp_path / ".mitocode").stat().st_mode) == 0o700
    assert Workspace(tmp_path, tmp_path.name, APP_VERSION).config_file == config


def test_ensure_workspace_names_a_relative_workspace(tmp_path, monkeypatch):
    """``-w .`` must still store a non-empty directory name in the config."""
    monkeypatch.chdir(tmp_path)

    assert ensure_workspace(Path("."), confirm=lambda path: True) is True

    document = json.loads(
        (tmp_path / ".mitocode" / "config.json").read_text(encoding="utf-8")
    )
    assert document["name"] == tmp_path.name
    assert document["version"] == APP_VERSION


def test_ensure_workspace_preserves_existing_subdirectories(tmp_path):
    skills = tmp_path / ".mitocode" / "skills" / "alpha"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")

    assert ensure_workspace(tmp_path, confirm=lambda path: True) is True

    assert (skills / "SKILL.md").is_file()
    assert (tmp_path / ".mitocode" / "config.json").is_file()


def test_ensure_workspace_prompts_when_config_is_missing_beside_other_state(tmp_path):
    secret = tmp_path / ".mitocode" / "secret" / "provider.json"
    secret.parent.mkdir(parents=True)
    secret.write_text('{"api_key":"k","model":"m"}', encoding="utf-8")
    confirmed: list[Path] = []

    assert ensure_workspace(
        tmp_path, confirm=lambda path: confirmed.append(path) or True
    )
    assert confirmed == [tmp_path]
    assert secret.is_file()


def test_ensure_workspace_rejects_a_missing_directory(tmp_path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises(MitoCodeCliConfigError) as info:
        ensure_workspace(missing, confirm=lambda path: True)

    assert str(missing) in str(info.value)


def test_ensure_workspace_rejects_a_file_instead_of_a_directory(tmp_path):
    workspace_file = tmp_path / "not-a-directory"
    workspace_file.write_text("file", encoding="utf-8")

    with pytest.raises(MitoCodeCliConfigError) as info:
        ensure_workspace(workspace_file, confirm=lambda path: True)

    assert str(workspace_file) in str(info.value)
    assert not (tmp_path / ".mitocode").exists()


def test_ensure_workspace_fails_when_mitocode_is_a_file(tmp_path):
    (tmp_path / ".mitocode").write_text("file", encoding="utf-8")

    with pytest.raises(MitoCodeCliConfigError) as info:
        ensure_workspace(tmp_path, confirm=lambda path: True)

    assert str(tmp_path / ".mitocode") in str(info.value)
    assert (tmp_path / ".mitocode").is_file()
    assert not (tmp_path / ".mitocode" / "config.json").exists()


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="the read-only directory rule is POSIX-only, and root can write one anyway",
)
def test_ensure_workspace_fails_when_the_directory_is_not_writable(tmp_path):
    os.chmod(tmp_path, 0o500)
    try:
        with pytest.raises(MitoCodeCliConfigError):
            ensure_workspace(tmp_path, confirm=lambda path: True)
    finally:
        os.chmod(tmp_path, 0o700)

    assert not (tmp_path / ".mitocode").exists()
