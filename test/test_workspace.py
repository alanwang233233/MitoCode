"""
Unit tests for Workspace class in mitocode.backend.workspace.workspace_manager.

Covers: _convert_to_path, __init__, new_workspace, init_from_path, dump,
        get_map, _validate_within_workspace, open.
"""

import json
import stat
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from mitocode.backend.exception.MitoCodeException import (
    MitoCodePathConvertError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceDumpError,
)
from mitocode.backend.workspace.workspace_manager import (
    DEFAULT_MAP,
    MITOCODE_CONFIG_PATH,
    Workspace,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmp_workspace_root(tmp_path):
    """A temporary directory that will serve as the workspace root."""
    return tmp_path / "my_agent_workspace"


@pytest.fixture
def existing_workspace(tmp_workspace_root):
    """Create a fully-initialised workspace on disk and return the Workspace instance."""
    tmp_workspace_root.mkdir()
    ws = Workspace.new_workspace(tmp_workspace_root, "test-agent", "a test workspace")
    return ws


@pytest.fixture
def empty_dir(tmp_path):
    """An empty directory that is NOT yet a workspace."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def populated_workspace(tmp_path):
    """A workspace whose dir_map directories and file_map files exist on disk.

    Uses custom maps so that paths are relative (no leading '/') and stay
    inside the workspace.  The agent/ directory and config.json file are
    created ahead of time.
    """
    root = tmp_path / "populated_ws"
    root.mkdir()
    ws = Workspace.new_workspace(
        root,
        "populated",
        "workspace with pre-created paths",
        dir_map={"agent": "/agent", "context": "/context"},
        file_map={"config": "/config.json", "mcpServers": "/mcpServers.json"},
    )
    # Create the mapped directories
    for dir_name in ws.config["dir_map"].values():
        (root / dir_name.lstrip("/")).mkdir(parents=True, exist_ok=True)
    # Touch the mapped files so they can be opened
    for file_name in ws.config["file_map"].values():
        (root / file_name.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
        (root / file_name.lstrip("/")).write_text("")
    return ws


# ──────────────────────────────────────────────
# _convert_to_path
# ──────────────────────────────────────────────

class TestConvertToPath:
    """Tests for the static method _convert_to_path."""

    def test_returns_path_from_string(self):
        result = Workspace._convert_to_path("/tmp/foo/bar")
        assert isinstance(result, Path)
        assert str(result) == "/tmp/foo/bar"

    def test_returns_path_unchanged_when_already_path(self):
        p = Path("/some/where")
        result = Workspace._convert_to_path(p)
        assert result is p

    def test_raises_on_none(self):
        with pytest.raises(MitoCodePathConvertError):
            Workspace._convert_to_path(None)

    def test_raises_on_empty_string(self):
        with pytest.raises(MitoCodePathConvertError):
            Workspace._convert_to_path("")

    def test_raises_on_falsy_zero(self):
        """0 is falsy — verify it is rejected."""
        with pytest.raises(MitoCodePathConvertError):
            Workspace._convert_to_path(0)


# ──────────────────────────────────────────────
# __init__
# ──────────────────────────────────────────────

class TestInit:
    """Tests for the Workspace constructor."""

    def test_default_maps_are_used_when_not_provided(self, empty_dir):
        ws = Workspace(empty_dir, "agent-1")
        assert ws.config["name"] == "agent-1"
        assert ws.config["description"] == ""
        assert ws.config["dir_map"] == DEFAULT_MAP["dir_map"]
        assert ws.config["file_map"] == DEFAULT_MAP["file_map"]
        assert ws.config_path == empty_dir / MITOCODE_CONFIG_PATH
        assert ws.config_file == ws.config_path / "config.json"

    def test_custom_maps_override_defaults(self, empty_dir):
        custom_dir = {"custom": "/custom_dir"}
        custom_file = {"secret": "/secret.json"}
        ws = Workspace(empty_dir, "agent-2", "desc", dir_map=custom_dir, file_map=custom_file)
        assert ws.config["description"] == "desc"
        assert ws.config["dir_map"] is custom_dir
        assert ws.config["file_map"] is custom_file

    def test_init_does_not_create_files_on_disk(self, empty_dir):
        """__init__ is a pure data constructor — it must not touch the filesystem."""
        before = set(p.name for p in empty_dir.iterdir()) if empty_dir.exists() else set()
        Workspace(empty_dir, "agent-mem")
        after = set(p.name for p in empty_dir.iterdir()) if empty_dir.exists() else set()
        assert before == after


# ──────────────────────────────────────────────
# new_workspace
# ──────────────────────────────────────────────

class TestNewWorkspace:
    """Tests for the class method new_workspace."""

    def test_creates_workspace_on_disk(self, tmp_workspace_root):
        tmp_workspace_root.mkdir()
        ws = Workspace.new_workspace(tmp_workspace_root, "agent", "test")
        mitocode_dir = tmp_workspace_root / MITOCODE_CONFIG_PATH
        config_file = mitocode_dir / "config.json"

        assert mitocode_dir.is_dir()
        assert config_file.is_file()

        with open(config_file) as f:
            data = json.load(f)
        assert data["name"] == "agent"
        assert data["description"] == "test"

    def test_raises_when_path_does_not_exist(self, tmp_workspace_root):
        with pytest.raises(MitoCodeWorkspaceInitError, match="does not exist"):
            Workspace.new_workspace(tmp_workspace_root, "ghost")

    def test_override_true_replaces_existing_config(self, existing_workspace):
        """When a workspace already exists and override=True, the old .mitocode is removed."""
        ws = Workspace.new_workspace(
            existing_workspace.path, "replaced", "new desc", override=True
        )
        assert ws.config["name"] == "replaced"
        assert ws.config["description"] == "new desc"

        with open(existing_workspace.config_file) as f:
            data = json.load(f)
        assert data["name"] == "replaced"

    def test_override_false_returns_existing(self, existing_workspace):
        """override=False with an existing workspace returns the loaded instance."""
        ws = Workspace.new_workspace(
            existing_workspace.path, "ignored-name", override=False
        )
        # The original name is preserved because the existing config was loaded
        assert ws.config["name"] == "test-agent"


# ──────────────────────────────────────────────
# init_from_path
# ──────────────────────────────────────────────

class TestInitFromPath:
    """Tests for the class method init_from_path."""

    def test_loads_existing_workspace(self, existing_workspace):
        ws = Workspace.init_from_path(existing_workspace.path)
        assert ws.config["name"] == "test-agent"
        assert ws.config["description"] == "a test workspace"

    def test_raises_when_mitocode_dir_missing(self, empty_dir):
        with pytest.raises(MitoCodeWorkspaceInitError, match="not found"):
            Workspace.init_from_path(empty_dir)

    def test_raises_when_config_json_missing(self, tmp_workspace_root):
        tmp_workspace_root.mkdir()
        mitocode = tmp_workspace_root / MITOCODE_CONFIG_PATH
        mitocode.mkdir()
        # config.json is absent
        with pytest.raises(MitoCodeWorkspaceInitError, match="not found"):
            Workspace.init_from_path(tmp_workspace_root)

    def test_raises_when_config_is_empty_dict(self, tmp_workspace_root):
        tmp_workspace_root.mkdir()
        mitocode = tmp_workspace_root / MITOCODE_CONFIG_PATH
        mitocode.mkdir()
        (mitocode / "config.json").write_text("{}")
        with pytest.raises(MitoCodeWorkspaceInitError, match="empty"):
            Workspace.init_from_path(tmp_workspace_root)

    def test_raises_when_config_missing_required_fields(self, tmp_workspace_root):
        tmp_workspace_root.mkdir()
        mitocode = tmp_workspace_root / MITOCODE_CONFIG_PATH
        mitocode.mkdir()
        (mitocode / "config.json").write_text(json.dumps({"name": "no-maps"}))
        with pytest.raises(MitoCodeWorkspaceInitError, match="invalid"):
            Workspace.init_from_path(tmp_workspace_root)

    def test_raises_on_malformed_json(self, tmp_workspace_root):
        tmp_workspace_root.mkdir()
        mitocode = tmp_workspace_root / MITOCODE_CONFIG_PATH
        mitocode.mkdir()
        (mitocode / "config.json").write_text("not json {{{")
        with pytest.raises(MitoCodeWorkspaceInitError):
            Workspace.init_from_path(tmp_workspace_root)

    def test_raises_on_unreadable_config(self, tmp_workspace_root):
        """Simulate an OSError by removing permissions."""
        tmp_workspace_root.mkdir()
        mitocode = tmp_workspace_root / MITOCODE_CONFIG_PATH
        mitocode.mkdir()
        config_file = mitocode / "config.json"
        config_file.write_text(json.dumps({"name": "x", "dir_map": {}, "file_map": {}}))
        # Remove read permission
        config_file.chmod(0o000)
        try:
            with pytest.raises(MitoCodeWorkspaceInitError):
                Workspace.init_from_path(tmp_workspace_root)
        finally:
            config_file.chmod(0o644)


# ──────────────────────────────────────────────
# dump
# ──────────────────────────────────────────────

class TestDump:
    """Tests for the dump method."""

    def test_dump_writes_config_to_disk(self, existing_workspace):
        existing_workspace.config["description"] = "updated description"
        result = existing_workspace.dump()
        assert result is True

        with open(existing_workspace.config_file) as f:
            data = json.load(f)
        assert data["description"] == "updated description"

    def test_dump_returns_false_when_config_path_not_dir(self, empty_dir):
        """If .mitocode doesn't exist, dump returns False."""
        ws = Workspace(empty_dir, "agent-nodir")
        assert ws.dump() is False

    def test_dump_raises_on_write_error(self, existing_workspace):
        """When writing fails after an OSError, MitoCodeWorkspaceDumpError is raised."""
        m = mock_open()
        m.side_effect = OSError("disk full")
        with patch("builtins.open", m):
            with pytest.raises(MitoCodeWorkspaceDumpError):
                existing_workspace.dump()


# ──────────────────────────────────────────────
# _validate_within_workspace
# ──────────────────────────────────────────────

class TestValidateWithinWorkspace:
    """Tests for the security barrier _validate_within_workspace."""

    def test_normal_path_passes(self, existing_workspace):
        resolved = existing_workspace._validate_within_workspace(Path("agent/output.txt"))
        assert resolved.is_relative_to(existing_workspace.path.resolve())

    def test_traversal_with_dotdot_is_blocked(self, existing_workspace):
        with pytest.raises(ValueError, match="Path traversal detected"):
            existing_workspace._validate_within_workspace(Path("../../../etc/passwd"))

    def test_traversal_via_symlink_is_blocked(self, existing_workspace):
        """A symlink pointing outside the workspace must be rejected."""
        outside = existing_workspace.path.parent / "outside.txt"
        outside.write_text("secret")
        try:
            link = existing_workspace.path / "agent" / "escape_link"
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(outside.resolve())
            with pytest.raises(ValueError, match="Path traversal detected"):
                existing_workspace._validate_within_workspace(Path("agent/escape_link"))
        finally:
            if outside.exists():
                outside.unlink()
            link_target = existing_workspace.path / "agent" / "escape_link"
            if link_target.exists() or link_target.is_symlink():
                link_target.unlink()


# ──────────────────────────────────────────────
# get_map
# ──────────────────────────────────────────────

class TestGetMap:
    """Tests for get_map — logical path resolution."""

    def test_dir_map_resolves_known_key(self, existing_workspace):
        result = existing_workspace.get_map("/agent/tasks/output.txt")
        expected = (existing_workspace.path / "agent" / "tasks" / "output.txt").resolve()
        assert result == expected

    def test_dir_map_with_unknown_key_uses_literal(self, existing_workspace):
        """Unknown dir keys are used as the literal directory name."""
        result = existing_workspace.get_map("/custom_area/sub")
        expected = (existing_workspace.path / "custom_area" / "sub").resolve()
        assert result == expected

    def test_dir_map_empty_path_returns_root(self, existing_workspace):
        result = existing_workspace.get_map("/")
        assert result == existing_workspace.path.resolve()

    def test_dir_map_only_slash_returns_root(self, existing_workspace):
        """Just '/' with nothing after it."""
        result = existing_workspace.get_map("/")
        assert result == existing_workspace.path.resolve()

    def test_file_map_resolves_known_key(self, existing_workspace):
        """Known keys in DEFAULT_MAP use paths starting with '/' (e.g. '/config.json').

        The leading '/' is stripped by _validate_within_workspace, so '/config.json'
        resolves to '<workspace_root>/config.json'."""
        result = existing_workspace.get_map("config")
        expected = (existing_workspace.path / "config.json").resolve()
        assert result == expected

    def test_file_map_with_unknown_key_uses_literal(self, existing_workspace):
        result = existing_workspace.get_map("unknown_file.txt")
        expected = (existing_workspace.path / "unknown_file.txt").resolve()
        assert result == expected

    def test_dir_map_traversal_is_blocked(self, existing_workspace):
        with pytest.raises(ValueError, match="Path traversal detected"):
            existing_workspace.get_map("/agent/../../../etc/passwd")

    def test_file_map_traversal_is_blocked(self, existing_workspace):
        with pytest.raises(ValueError, match="Path traversal detected"):
            existing_workspace.get_map("../outside")


# ──────────────────────────────────────────────
# open
# ──────────────────────────────────────────────

class TestOpen:
    """Tests for the open method — the main file-operation entry point."""

    def test_open_via_file_map_writes_and_reads(self, populated_workspace):
        content = '{"key": "value"}'
        with populated_workspace.open("config", "w") as f:
            f.write(content)

        with populated_workspace.open("config", "r") as f:
            assert f.read() == content

    def test_open_via_dir_map_writes_and_reads(self, populated_workspace):
        content = "log line 1\nlog line 2\n"
        with populated_workspace.open("/agent/log.txt", "w") as f:
            f.write(content)

        with populated_workspace.open("/agent/log.txt", "r") as f:
            assert f.read() == content

    def test_open_passes_kwargs_correctly(self, populated_workspace):
        """encoding kwargs are forwarded to builtins.open."""
        with populated_workspace.open("/agent/utf8.txt", "w", encoding="utf-8") as f:
            f.write("café")

        with populated_workspace.open("/agent/utf8.txt", "r", encoding="utf-8") as f:
            assert f.read() == "café"

    def test_open_creates_parent_directories_if_needed_same_as_builtin_open(self, existing_workspace):
        """open() itself does not create dirs — this is a behaviour-documenting test."""
        # Workspace.open delegates to builtins.open which will NOT auto-create dirs.
        # The caller (Agent) is expected to ensure parent dirs exist.
        with pytest.raises(FileNotFoundError):
            existing_workspace.open("/agent/deep/nested/file.txt", "w")

    def test_open_traversal_is_blocked(self, existing_workspace):
        with pytest.raises(ValueError, match="Path traversal detected"):
            existing_workspace.open("/agent/../../../etc/passwd", "r")