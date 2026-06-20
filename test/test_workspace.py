import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.workspace.workspace_manager import Workspace, DEFAULT_MAP, MITOCODE_CONFIG_PATH
from mitocode.backend.exception.MitoCodeException import (
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceDumpError,
    MitoCodePathConvertError,
)


# ===========================================================================
# Workspace._convert_to_path tests
# ===========================================================================

class TestConvertToPath:
    def test_convert_string_to_path(self):
        """Test converting a string path to Path object."""
        result = Workspace._convert_to_path("/tmp/test")
        from pathlib import Path
        assert isinstance(result, Path)
        assert str(result) == "/tmp/test"

    def test_convert_path_to_path(self):
        """Test that Path objects are returned as-is."""
        from pathlib import Path
        input_path = Path("/tmp/test")
        result = Workspace._convert_to_path(input_path)
        assert result is input_path

    def test_convert_none_raises_error(self):
        """Test that None input raises MitoCodePathConvertError."""
        with pytest.raises(MitoCodePathConvertError):
            Workspace._convert_to_path(None)

    def test_convert_empty_string(self):
        """Test converting empty string to Path raises error."""
        # Empty string is falsy, so it raises MitoCodePathConvertError
        with pytest.raises(MitoCodePathConvertError):
            Workspace._convert_to_path("")


# ===========================================================================
# Workspace initialization tests
# ===========================================================================

class TestWorkspaceInit:
    def test_init_with_minimal_args(self, tmp_path):
        """Test initializing Workspace with minimal arguments."""
        ws = Workspace(tmp_path, "test_workspace")
        assert ws.path == tmp_path
        assert ws.config_path == tmp_path / MITOCODE_CONFIG_PATH
        assert ws.config["name"] == "test_workspace"
        assert ws.config["description"] == ""
        assert ws.config["dir_map"] == DEFAULT_MAP["dir_map"]
        assert ws.config["file_map"] == DEFAULT_MAP["file_map"]

    def test_init_with_description(self, tmp_path):
        """Test initializing Workspace with description."""
        ws = Workspace(tmp_path, "test_workspace", "A test workspace")
        assert ws.config["description"] == "A test workspace"

    def test_init_with_custom_dir_map(self, tmp_path):
        """Test initializing Workspace with custom dir_map."""
        custom_dir_map = {"custom": "/custom_dir"}
        ws = Workspace(tmp_path, "test", dir_map=custom_dir_map)
        assert ws.config["dir_map"] == custom_dir_map

    def test_init_with_custom_file_map(self, tmp_path):
        """Test initializing Workspace with custom file_map."""
        custom_file_map = {"custom_file": "/custom.json"}
        ws = Workspace(tmp_path, "test", file_map=custom_file_map)
        assert ws.config["file_map"] == custom_file_map


# ===========================================================================
# Workspace.new_workspace tests
# ===========================================================================

class TestNewWorkspace:
    def test_new_workspace_creates_config(self, tmp_path):
        """Test that new_workspace creates config directory and file."""
        ws = Workspace.new_workspace(tmp_path, "new_ws")
        assert ws is not None
        assert (tmp_path / MITOCODE_CONFIG_PATH).exists()
        assert (tmp_path / MITOCODE_CONFIG_PATH / "config.json").exists()

    def test_new_workspace_nonexistent_path_raises(self, tmp_path):
        """Test that new_workspace raises error for non-existent path."""
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(MitoCodeWorkspaceInitError):
            Workspace.new_workspace(nonexistent, "test")

    def test_new_workspace_override_false_existing_config(self, tmp_path):
        """Test that new_workspace with override=False returns existing workspace."""
        # First create a workspace
        Workspace.new_workspace(tmp_path, "first")
        # Try to create another without override
        ws = Workspace.new_workspace(tmp_path, "second", override=False)
        assert ws is not None
        assert ws.config["name"] == "first"  # Should keep original name

    def test_new_workspace_override_true_replaces_config(self, tmp_path):
        """Test that new_workspace with override=True replaces config."""
        # First create a workspace
        Workspace.new_workspace(tmp_path, "first")
        # Create another with override
        ws = Workspace.new_workspace(tmp_path, "second", override=True)
        assert ws is not None
        assert ws.config["name"] == "second"

    def test_new_workspace_with_custom_params(self, tmp_path):
        """Test new_workspace with custom parameters."""
        custom_dir_map = {"agent": "/my_agent"}
        custom_file_map = {"config": "/my_config.json"}
        ws = Workspace.new_workspace(
            tmp_path, "custom_ws", "Custom description",
            dir_map=custom_dir_map, file_map=custom_file_map
        )
        assert ws.config["dir_map"] == custom_dir_map
        assert ws.config["file_map"] == custom_file_map
        assert ws.config["description"] == "Custom description"


# ===========================================================================
# Workspace.init_from_path tests
# ===========================================================================

class TestInitFromPath:
    def test_init_from_valid_path(self, tmp_path):
        """Test initializing from an existing workspace path."""
        Workspace.new_workspace(tmp_path, "existing")
        ws = Workspace.init_from_path(tmp_path)
        assert ws is not None
        assert ws.config["name"] == "existing"

    def test_init_from_nonexistent_path_returns_none(self, tmp_path):
        """Test that init_from_path returns None for non-existent path."""
        nonexistent = tmp_path / "nonexistent"
        ws = Workspace.init_from_path(nonexistent)
        assert ws is None

    def test_init_from_path_without_config_returns_none(self, tmp_path):
        """Test that init_from_path returns None if config doesn't exist."""
        # Create directory but no config
        ws = Workspace.init_from_path(tmp_path)
        assert ws is None

    def test_init_from_path_with_invalid_config_returns_none(self, tmp_path):
        """Test that init_from_path returns None if config is invalid."""
        config_dir = tmp_path / MITOCODE_CONFIG_PATH
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        # Write invalid config (missing required fields)
        config_file.write_text("{}")
        ws = Workspace.init_from_path(tmp_path)
        assert ws is None

    def test_init_from_path_missing_name_returns_none(self, tmp_path):
        """Test that init_from_path returns None if name is missing."""
        config_dir = tmp_path / MITOCODE_CONFIG_PATH
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        # Write config without name (also missing dir_map and file_map)
        json.dump({"description": "no name"}, config_file)
        ws = Workspace.init_from_path(tmp_path)
        assert ws is None


# ===========================================================================
# Workspace.dump tests
# ===========================================================================

class TestDump:
    def test_dump_writes_config(self, tmp_path):
        """Test that dump writes config to file."""
        # Need to create the config directory first
        ws = Workspace(tmp_path, "dump_test", "Test desc")
        ws.config_path.mkdir(parents=True, exist_ok=True)
        result = ws.dump()
        assert result is True
        config_file = tmp_path / MITOCODE_CONFIG_PATH / "config.json"
        assert config_file.exists()
        with open(config_file) as f:
            saved_config = json.load(f)
        assert saved_config["name"] == "dump_test"
        assert saved_config["description"] == "Test desc"

    def test_dump_returns_false_if_paths_dont_exist(self, tmp_path):
        """Test that dump returns False if paths don't exist."""
        ws = Workspace(tmp_path, "test")
        # Don't create the directories
        result = ws.dump()
        assert result is False

    def test_dump_with_invalid_path_raises(self, tmp_path):
        """Test that dump raises error for invalid path."""
        ws = Workspace(tmp_path, "test")
        # Create config_path as a file instead of directory
        ws.config_path.parent.mkdir(parents=True, exist_ok=True)
        ws.config_path.touch()
        with pytest.raises(MitoCodeWorkspaceDumpError):
            ws.dump()


# ===========================================================================
# Workspace.get_map tests
# ===========================================================================

class TestGetMap:
    def test_get_map_root_directory(self, tmp_path):
        """Test get_map returns root path for empty string."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("")
        assert result == tmp_path

    def test_get_map_dir_map_agent(self, tmp_path):
        """Test get_map resolves agent directory."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("/agent")
        expected = tmp_path / DEFAULT_MAP["dir_map"]["agent"].lstrip("/")
        assert result == expected

    def test_get_map_dir_map_context(self, tmp_path):
        """Test get_map resolves context directory."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("/context")
        expected = tmp_path / DEFAULT_MAP["dir_map"]["context"].lstrip("/")
        assert result == expected

    def test_get_map_dir_map_nested_path(self, tmp_path):
        """Test get_map resolves nested paths in dir_map."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("/agent/subdir/file.txt")
        expected = tmp_path / DEFAULT_MAP["dir_map"]["agent"].lstrip("/") / "subdir" / "file.txt"
        assert result == expected

    def test_get_map_file_map_mcpServers(self, tmp_path):
        """Test get_map resolves mcpServers file."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("mcpServers")
        expected = tmp_path / DEFAULT_MAP["file_map"]["mcpServers"].lstrip("/")
        # The file_map value is "/mcpServers.json", lstrip("/") gives "mcpServers.json"
        assert result == expected

    def test_get_map_file_map_config(self, tmp_path):
        """Test get_map resolves config file."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("config")
        expected = tmp_path / DEFAULT_MAP["file_map"]["config"].lstrip("/")
        # The file_map value is "/config.json", lstrip("/") gives "config.json"
        assert result == expected

    def test_get_map_unknown_dir_key(self, tmp_path):
        """Test get_map uses key as-is for unknown dir_map keys."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("/unknown_dir")
        expected = tmp_path / "unknown_dir"
        assert result == expected

    def test_get_map_unknown_file_key(self, tmp_path):
        """Test get_map uses key as-is for unknown file_map keys."""
        ws = Workspace(tmp_path, "test")
        result = ws.get_map("unknown_file")
        expected = tmp_path / "unknown_file"
        assert result == expected

    def test_get_map_custom_dir_map(self, tmp_path):
        """Test get_map with custom dir_map."""
        custom_dir_map = {"my_dir": "/custom/path"}
        ws = Workspace(tmp_path, "test", dir_map=custom_dir_map)
        result = ws.get_map("/my_dir")
        expected = tmp_path / "custom" / "path"
        assert result == expected

    def test_get_map_custom_file_map(self, tmp_path):
        """Test get_map with custom file_map."""
        custom_file_map = {"my_file": "/custom_file.json"}
        ws = Workspace(tmp_path, "test", file_map=custom_file_map)
        result = ws.get_map("my_file")
        expected = tmp_path / "custom_file.json"
        assert result == expected


# ===========================================================================
# Workspace.open tests
# ===========================================================================

class TestOpen:
    def test_open_file_in_dir_map(self, tmp_path):
        """Test opening a file in a mapped directory."""
        ws = Workspace.new_workspace(tmp_path, "test")
        # Create the agent directory
        agent_dir = tmp_path / DEFAULT_MAP["dir_map"]["agent"].lstrip("/")
        agent_dir.mkdir(parents=True, exist_ok=True)

        test_file = agent_dir / "test.txt"
        test_file.write_text("hello")

        with ws.open("/agent/test.txt", "r") as f:
            content = f.read()
        assert content == "hello"

    def test_open_file_in_file_map(self, tmp_path):
        """Test opening a file from file_map."""
        ws = Workspace.new_workspace(tmp_path, "test")

        config_file = tmp_path / DEFAULT_MAP["file_map"]["config"].lstrip("/")
        config_file.write_text('{"key": "value"}')

        with ws.open("config", "r") as f:
            content = f.read()
        assert content == '{"key": "value"}'

    def test_open_write_mode(self, tmp_path):
        """Test opening a file in write mode."""
        ws = Workspace.new_workspace(tmp_path, "test")
        agent_dir = tmp_path / DEFAULT_MAP["dir_map"]["agent"].lstrip("/")
        agent_dir.mkdir(parents=True, exist_ok=True)

        with ws.open("/agent/new_file.txt", "w") as f:
            f.write("new content")

        test_file = agent_dir / "new_file.txt"
        assert test_file.exists()
        assert test_file.read_text() == "new content"


# ===========================================================================
# Integration tests
# ===========================================================================

class TestWorkspaceIntegration:
    def test_full_workflow(self, tmp_path):
        """Test complete workspace lifecycle."""
        # Create new workspace
        ws = Workspace.new_workspace(tmp_path, "integration_test", "Integration test workspace")

        # Verify config was created
        assert ws.config["name"] == "integration_test"

        # Dump changes
        ws.config["description"] = "Updated description"
        assert ws.dump() is True

        # Re-init from path
        ws2 = Workspace.init_from_path(tmp_path)
        assert ws2 is not None
        assert ws2.config["name"] == "integration_test"
        assert ws2.config["description"] == "Updated description"

        # Test file operations
        agent_dir = tmp_path / DEFAULT_MAP["dir_map"]["agent"].lstrip("/")
        agent_dir.mkdir(parents=True, exist_ok=True)

        with ws2.open("/agent/data.json", "w") as f:
            json.dump({"data": "test"}, f)

        with ws2.open("/agent/data.json", "r") as f:
            loaded = json.load(f)
        assert loaded == {"data": "test"}

    def test_workspace_isolation(self, tmp_path):
        """Test that multiple workspaces are isolated."""
        ws1_path = tmp_path / "ws1"
        ws2_path = tmp_path / "ws2"
        ws1_path.mkdir()
        ws2_path.mkdir()

        ws1 = Workspace.new_workspace(ws1_path, "workspace1")
        ws2 = Workspace.new_workspace(ws2_path, "workspace2")

        # Modify ws1
        ws1.config["custom"] = "value1"
        ws1.dump()

        # Verify ws2 is unaffected
        assert "custom" not in ws2.config

        # Re-load and verify
        ws1_reload = Workspace.init_from_path(ws1_path)
        ws2_reload = Workspace.init_from_path(ws2_path)

        assert ws1_reload.config["custom"] == "value1"
        assert "custom" not in ws2_reload.config