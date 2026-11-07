"""Tests for MCP configuration loading at CLI startup."""

import json

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode import APP_VERSION
from mitocode.cli.args import CliArgs
from mitocode.cli.bootstrap import load_mcp_servers, resolve_workspace
from mitocode.core.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.core.mcp.mcp_config import McpServerKind
from mitocode.core.workspace.workspace import Workspace


def _args(**over):
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


def _write_config(root: Path, payload: str) -> Path:
    """Write a minimal workspace configuration for *root*."""
    path = root / ".mitocode" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _workspace(root: Path) -> Workspace:
    return Workspace(root, "x", APP_VERSION)


def _write_mcp(root: Path, payload: str) -> Path:
    path = root / ".mitocode" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def test_no_mcp_flag_returns_none(tmp_path):
    assert load_mcp_servers(_args(no_mcp=True), _workspace(tmp_path)) is None


def test_no_mcp_flag_wins_over_explicit_path(tmp_path):
    explicit = tmp_path / "servers.json"
    explicit.write_text('{"mcpServers":{"fs":{"command":"echo"}}}', encoding="utf-8")

    assert (
        load_mcp_servers(_args(no_mcp=True, mcp_config=explicit), _workspace(tmp_path))
        is None
    )


def test_default_file_is_optional(tmp_path):
    assert load_mcp_servers(_args(), _workspace(tmp_path)) is None


def test_explicit_and_default_paths_parse(tmp_path):
    _write_mcp(tmp_path, '{"mcpServers":{"fs":{"command":"echo"}}}')
    servers = load_mcp_servers(_args(), _workspace(tmp_path))

    assert servers is not None
    assert len(servers) == 1
    assert servers[0].name == "fs"
    assert servers[0].kind is McpServerKind.STDIO
    assert servers[0].command == "echo"


def test_explicit_path_overrides_default(tmp_path):
    _write_mcp(tmp_path, '{"mcpServers":{"default":{"command":"echo"}}}')
    explicit = tmp_path / "other" / "servers.json"
    explicit.parent.mkdir()
    explicit.write_text(
        '{"mcpServers":{"explicit":{"url":"https://example.com/mcp"}}}',
        encoding="utf-8",
    )

    servers = load_mcp_servers(_args(mcp_config=explicit), _workspace(tmp_path))

    assert servers is not None
    assert [server.name for server in servers] == ["explicit"]
    assert servers[0].kind is McpServerKind.HTTP


def test_explicit_missing_file_is_fatal(tmp_path):
    missing = tmp_path / "nope.json"

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        load_mcp_servers(_args(mcp_config=missing), _workspace(tmp_path))

    assert str(missing) in str(excinfo.value)


def test_explicit_malformed_file_is_fatal(tmp_path):
    explicit = tmp_path / "servers.json"
    explicit.write_text("{nope", encoding="utf-8")

    with pytest.raises(MitoCodeMcpConfigError):
        load_mcp_servers(_args(mcp_config=explicit), _workspace(tmp_path))


def test_explicit_path_that_cannot_be_read_is_fatal(tmp_path):
    directory = tmp_path / "servers.json"
    directory.mkdir()

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        load_mcp_servers(_args(mcp_config=directory), _workspace(tmp_path))

    assert str(directory) in str(excinfo.value)


def test_non_utf8_document_is_fatal(tmp_path):
    path = tmp_path / ".mitocode" / "mcp.json"
    path.parent.mkdir()
    path.write_bytes(b'{"mcpServers": {"fs": {"command": "\xff\xfe"}}}')

    with pytest.raises(MitoCodeMcpConfigError):
        load_mcp_servers(_args(), _workspace(tmp_path))


def test_malformed_document_raises(tmp_path):
    _write_mcp(tmp_path, "{}")

    with pytest.raises(MitoCodeMcpConfigError):
        load_mcp_servers(_args(), _workspace(tmp_path))


def test_workspace_without_mcp_directory_returns_none(tmp_path):
    assert not (tmp_path / ".mitocode").exists()
    assert load_mcp_servers(_args(), _workspace(tmp_path)) is None


def test_error_message_names_default_location(tmp_path):
    _write_mcp(tmp_path, "{}")

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        load_mcp_servers(_args(), _workspace(tmp_path))

    message = str(excinfo.value)
    assert "mcp.json" in message
    assert "default" in message.lower()


def test_error_message_does_not_leak_header_values(tmp_path):
    _write_mcp(
        tmp_path,
        '{"mcpServers":{"remote":{"url":"https://example.com/mcp",'
        '"headers":{"Authorization":"Bearer super-secret-token"}},"broken":3}}',
    )

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        load_mcp_servers(_args(), _workspace(tmp_path))

    message = str(excinfo.value)
    assert "super-secret-token" not in message
    assert "Authorization" not in message


def test_error_message_does_not_leak_non_string_command_values(tmp_path):
    _write_mcp(
        tmp_path,
        '{"mcpServers":{"fs":{"command":["npx","--token","super-secret-token"]}}}',
    )

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        load_mcp_servers(_args(), _workspace(tmp_path))

    message = str(excinfo.value)
    assert "got list" in message
    assert "super-secret-token" not in message
    assert "--token" not in message


@pytest.mark.asyncio
async def test_loaded_config_matches_resolved_workspace(tmp_path):
    _write_mcp(tmp_path, '{"mcpServers":{"fs":{"command":"echo"}}}')
    _write_config(tmp_path, json.dumps({"version": 2, "name": "x", "description": ""}))
    resolved = await resolve_workspace(tmp_path)

    servers = load_mcp_servers(_args(), resolved.workspace)

    assert servers is not None
    assert [server.name for server in servers] == ["fs"]
