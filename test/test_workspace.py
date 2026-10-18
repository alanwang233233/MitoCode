"""
Unit tests for Workspace class in mitocode.core.workspace.workspace_manager.

Covers: _convert_to_path, __init__, new_workspace, init_from_path, dump, _validate_within_workspace.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from mitocode.core.exception.mitocode_exception import (
    MitoCodePathConvertError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceDumpError,
)
from mitocode.core.workspace.workspace import (
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
    root = tmp_path / "populated_ws"
    root.mkdir()
    ws = Workspace.new_workspace(
        root,
        "populated",
        "workspace with pre-created paths",
    )
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

    def test_workspace_properly(self, empty_dir):
        ws = Workspace(empty_dir, "agent-1", 1)
        assert ws.config["name"] == "agent-1"
        assert ws.config["description"] == ""
        assert ws.config_path == empty_dir / MITOCODE_CONFIG_PATH
        assert ws.config_file == ws.config_path / "config.json"

    def test_init_does_not_create_files_on_disk(self, empty_dir):
        """__init__ is a pure data constructor — it must not touch the filesystem."""
        before = (
            set(p.name for p in empty_dir.iterdir()) if empty_dir.exists() else set()
        )
        Workspace(empty_dir, "agent-mem", 2)
        after = (
            set(p.name for p in empty_dir.iterdir()) if empty_dir.exists() else set()
        )
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
        config_file.write_text(json.dumps({"name": "x"}))
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

    def test_dump_raise_when_config_path_not_dir(self, empty_dir):
        """If .mitocode doesn't exist, dump returns False."""
        ws = Workspace(empty_dir, "agent-nodir", 2)
        with pytest.raises(MitoCodeWorkspaceDumpError, match="not found"):
            ws.dump()

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
        resolved = existing_workspace._validate_within_workspace(
            Path("agent/output.txt")
        )
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
# validate
# ──────────────────────────────────────────────


class TestValidate:
    """Tests for the validate method."""

    def test_valid_workspace_returns_true(self, existing_workspace):
        """A properly initialized workspace should validate successfully."""
        assert existing_workspace.validate() is True

    def test_validate_fails_when_path_missing(self, existing_workspace):
        """Validation should fail if the workspace path no longer exists."""
        import shutil

        try:
            shutil.rmtree(existing_workspace.path)
            with pytest.raises(MitoCodeWorkspaceInitError, match="does not exist"):
                existing_workspace.validate()
        finally:
            # Restore the workspace for other tests
            existing_workspace.path.mkdir(exist_ok=True)
            existing_workspace.config_path.mkdir(exist_ok=True)
            existing_workspace.dump()

    def test_validate_fails_when_config_dir_missing(self, existing_workspace):
        """Validation should fail if .mitocode directory is missing."""
        import shutil

        try:
            shutil.rmtree(existing_workspace.config_path)
            with pytest.raises(MitoCodeWorkspaceInitError, match="does not exist"):
                existing_workspace.validate()
        finally:
            # Restore for other tests
            existing_workspace.config_path.mkdir(exist_ok=True)
            existing_workspace.dump()

    def test_validate_fails_when_config_file_missing(self, existing_workspace):
        """Validation should fail if config.json is missing."""
        try:
            existing_workspace.config_file.unlink()
            with pytest.raises(MitoCodeWorkspaceInitError, match="does not exist"):
                existing_workspace.validate()
        finally:
            # Restore for other tests
            existing_workspace.dump()

    def test_validate_fails_when_config_empty(self, existing_workspace):
        """Validation should fail if config.json is empty or invalid."""
        try:
            existing_workspace.config_file.write_text("{}")
            with pytest.raises(MitoCodeWorkspaceInitError, match="empty or invalid"):
                existing_workspace.validate()
        finally:
            # Restore for other tests
            existing_workspace.dump()

    def test_validate_fails_when_name_missing(self, existing_workspace):
        """Validation should fail if name field is missing."""
        try:
            import json

            with open(existing_workspace.config_file, "w") as f:
                json.dump({"version": 1}, f)
            with pytest.raises(
                MitoCodeWorkspaceInitError, match="missing required 'name'"
            ):
                existing_workspace.validate()
        finally:
            # Restore for other tests
            existing_workspace.dump()

    def test_validate_fails_when_version_missing(self, existing_workspace):
        """Validation should fail if version field is missing."""
        try:
            import json

            with open(existing_workspace.config_file, "w") as f:
                json.dump({"name": "test"}, f)
            with pytest.raises(
                MitoCodeWorkspaceInitError, match="missing required 'version'"
            ):
                existing_workspace.validate()
        finally:
            # Restore for other tests
            existing_workspace.dump()


# ──────────────────────────────────────────────
# reload
# ──────────────────────────────────────────────


class TestReload:
    """Tests for the reload method."""

    def test_reload_updates_config(self, existing_workspace):
        """Reload should update in-memory config from disk."""
        # First, modify the config on disk
        with open(existing_workspace.config_file, "w") as f:
            json.dump(
                {
                    "name": "reloaded-name",
                    "description": "reloaded description",
                    "version": 1,
                },
                f,
            )
        # Reload
        assert existing_workspace.reload() is True
        # Verify in-memory config was updated
        assert existing_workspace.config["name"] == "reloaded-name"
        assert existing_workspace.config["description"] == "reloaded description"

    def test_reload_fails_when_config_file_missing(self, existing_workspace):
        """Reload should fail if config.json is missing."""
        try:
            existing_workspace.config_file.unlink()
            with pytest.raises(MitoCodeWorkspaceInitError, match="does not exist"):
                existing_workspace.reload()
        finally:
            # Restore for other tests
            existing_workspace.dump()

    def test_reload_fails_when_config_invalid(self, existing_workspace):
        """Reload should fail if config.json is invalid."""
        try:
            existing_workspace.config_file.write_text("not json")
            with pytest.raises(MitoCodeWorkspaceInitError):
                existing_workspace.reload()
        finally:
            # Restore for other tests
            existing_workspace.dump()

    def test_reload_fails_when_config_empty(self, existing_workspace):
        """Reload should fail if config.json is empty."""
        try:
            existing_workspace.config_file.write_text("{}")
            with pytest.raises(MitoCodeWorkspaceInitError, match="empty or invalid"):
                existing_workspace.reload()
        finally:
            # Restore for other tests
            existing_workspace.dump()


# ──────────────────────────────────────────────
# is_outdated
# ──────────────────────────────────────────────


class TestIsOutdated:
    """Tests for the is_outdated method."""

    def test_not_outdated_when_same(self, existing_workspace):
        """Should return False when in-memory and disk configs match."""
        assert existing_workspace.is_outdated() is False

    def test_outdated_when_different(self, existing_workspace):
        """Should return True when in-memory and disk configs differ."""
        # Modify disk config
        with open(existing_workspace.config_file, "w") as f:
            json.dump(
                {
                    "name": "different-name",
                    "description": "different description",
                    "version": 1,
                },
                f,
            )
        assert existing_workspace.is_outdated() is True

    def test_outdated_after_memory_change(self, existing_workspace):
        """Should return True after changing in-memory config without dumping."""
        existing_workspace.config["name"] = "changed-in-memory"
        assert existing_workspace.is_outdated() is True

    def test_is_outdated_fails_when_config_file_missing(self, existing_workspace):
        """Should raise error if config.json is missing."""
        try:
            existing_workspace.config_file.unlink()
            with pytest.raises(MitoCodeWorkspaceInitError, match="does not exist"):
                existing_workspace.is_outdated()
        finally:
            # Restore for other tests
            existing_workspace.dump()
