"""Tests for the project/global configuration file locator.

The manager answers one question only: for a given file name, is the project
copy or the global copy the one to open?  It never parses the file and never
merges values, so the tests here stay at the path level: which copy wins, what
happens when neither exists, and which names are refused.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.config import ConfigManager, ConfigScope, default_global_dir
from mitocode.core.exception.mitocode_exception import (
    MitoCodeConfigError,
    MitoCodeConfigNotFoundError,
    MitoCodeConfigPathError,
)


def test_missing_config_error_is_a_file_not_found_error():
    """A missing configuration can be caught as a plain FileNotFoundError."""
    error = MitoCodeConfigNotFoundError("missing")

    assert isinstance(error, MitoCodeConfigError)
    assert isinstance(error, FileNotFoundError)


# ---------------------------------------------------------------------------
# Platform directories
# ---------------------------------------------------------------------------


def test_windows_global_dir_is_home_dot_mitocode(tmp_path):
    """Windows keeps the global configuration directly under the home folder."""
    assert (
        default_global_dir(platform="win32", home=tmp_path, env={})
        == tmp_path / ".mitocode"
    )


def test_windows_ignores_xdg_config_home(tmp_path):
    """XDG_CONFIG_HOME is a Unix convention and must not move the Windows root."""
    environment = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}

    assert (
        default_global_dir(platform="win32", home=tmp_path, env=environment)
        == tmp_path / ".mitocode"
    )


def test_macos_global_dir_is_under_dot_config(tmp_path):
    """macOS follows the same ``~/.config`` layout as Linux."""
    assert (
        default_global_dir(platform="darwin", home=tmp_path, env={})
        == tmp_path / ".config" / ".mitocode"
    )


def test_linux_global_dir_honours_absolute_xdg(tmp_path):
    """An absolute XDG_CONFIG_HOME replaces ``~/.config``."""
    environment = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}

    assert (
        default_global_dir(platform="linux", home=tmp_path, env=environment)
        == tmp_path / "xdg" / ".mitocode"
    )


def test_linux_global_dir_ignores_relative_xdg(tmp_path):
    """The XDG specification says a relative root is invalid, so it is ignored."""
    environment = {"XDG_CONFIG_HOME": "relative/path"}

    assert (
        default_global_dir(platform="linux", home=tmp_path, env=environment)
        == tmp_path / ".config" / ".mitocode"
    )


def test_linux_global_dir_falls_back_when_xdg_is_unset(tmp_path):
    """Without XDG_CONFIG_HOME the Linux root is ``~/.config``."""
    assert (
        default_global_dir(platform="linux", home=tmp_path, env={})
        == tmp_path / ".config" / ".mitocode"
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _manager(tmp_path):
    """Build a manager with both roots under *tmp_path*."""
    return ConfigManager.for_workspace(
        tmp_path / "repo", global_dir=tmp_path / "global"
    )


def test_manager_exposes_both_directories(tmp_path):
    """Both roots are reachable without touching a file."""
    manager = _manager(tmp_path)

    assert manager.project_dir == tmp_path / "repo" / ".mitocode"
    assert manager.global_dir == tmp_path / "global"


def test_project_copy_wins_over_global(tmp_path):
    """The project copy is authoritative when both exist."""
    manager = _manager(tmp_path)
    project_file = manager.project_dir / "provider.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("{}", encoding="utf-8")
    global_file = manager.global_dir / "provider.json"
    global_file.parent.mkdir(parents=True)
    global_file.write_text("{}", encoding="utf-8")

    assert manager.path("provider.json") == project_file


def test_global_copy_is_used_when_project_has_none(tmp_path):
    """A project without its own copy falls back to the global one."""
    manager = _manager(tmp_path)
    global_file = manager.global_dir / "provider.json"
    global_file.parent.mkdir(parents=True)
    global_file.write_text("{}", encoding="utf-8")

    assert manager.path("provider.json") == global_file


def test_missing_file_raises_not_found(tmp_path):
    """Neither copy present is an explicit error, not an empty configuration."""
    manager = _manager(tmp_path)

    with pytest.raises(MitoCodeConfigNotFoundError) as info:
        manager.path("provider.json")

    assert "provider.json" in str(info.value)


def test_forced_project_scope_does_not_check_existence(tmp_path):
    """PROJECT is a pure path computation so a new file can be addressed."""
    manager = _manager(tmp_path)

    assert manager.path("new.json", scope=ConfigScope.PROJECT) == (
        manager.project_dir / "new.json"
    )
    assert not manager.project_dir.exists()


def test_forced_global_scope_returns_global_path(tmp_path):
    """GLOBAL ignores an existing project copy."""
    manager = _manager(tmp_path)
    project_file = manager.project_dir / "provider.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("{}", encoding="utf-8")

    assert manager.path("provider.json", scope=ConfigScope.GLOBAL) == (
        manager.global_dir / "provider.json"
    )


def test_exists_reports_either_copy(tmp_path):
    """``exists`` answers for the pair, not for one copy."""
    manager = _manager(tmp_path)

    assert manager.exists("provider.json") is False
    global_file = manager.global_dir / "provider.json"
    global_file.parent.mkdir(parents=True)
    global_file.write_text("{}", encoding="utf-8")

    assert manager.exists("provider.json") is True


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_open_reads_the_selected_copy(tmp_path):
    """Reading through the manager hides which copy was chosen."""
    manager = _manager(tmp_path)
    project_file = manager.project_dir / "mcp.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(json.dumps({"source": "project"}), encoding="utf-8")

    with manager.open("mcp.json") as handle:
        assert json.load(handle) == {"source": "project"}


def test_open_write_creates_the_project_copy(tmp_path):
    """Writing with AUTO lands in the project layer and creates its directory."""
    manager = _manager(tmp_path)

    with manager.open("mcp.json", "w") as handle:
        json.dump({"source": "project"}, handle)

    project_file = manager.project_dir / "mcp.json"
    assert project_file.is_file()
    assert json.loads(project_file.read_text(encoding="utf-8")) == {"source": "project"}


def test_open_write_can_target_global_explicitly(tmp_path):
    """Only an explicit GLOBAL writes the shared copy."""
    manager = _manager(tmp_path)

    with manager.open("settings.json", "w", scope=ConfigScope.GLOBAL) as handle:
        json.dump({"theme": "dark"}, handle)

    assert (manager.global_dir / "settings.json").is_file()
    assert not (manager.project_dir / "settings.json").exists()


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


def test_absolute_name_is_refused(tmp_path):
    """An absolute name would escape the configuration directory."""
    manager = _manager(tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        manager.path(str(tmp_path / "outside.json"))


def test_parent_traversal_is_refused(tmp_path):
    """A ``..`` segment would escape the configuration directory."""
    manager = _manager(tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        manager.path(f"..{os.sep}outside.json")


def test_empty_name_is_refused(tmp_path):
    """An empty name addresses the directory, not a file."""
    manager = _manager(tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        manager.path("   ")


def test_project_path_requires_a_workspace(tmp_path):
    """Without a workspace there is no project layer to address."""
    manager = ConfigManager(global_dir=tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        _ = manager.project_dir
    with pytest.raises(MitoCodeConfigPathError):
        manager.path("provider.json", scope=ConfigScope.PROJECT)


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows-only")
def test_windows_drive_letter_is_refused(tmp_path):
    """``C:provider.json`` is drive-relative, not a safe relative name."""
    manager = ConfigManager.for_workspace(tmp_path / "repo", global_dir=tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        manager.path("C:provider.json")


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths are Windows-only")
def test_windows_unc_path_is_refused(tmp_path):
    """A UNC path points at another machine, never at this configuration."""
    manager = ConfigManager.for_workspace(tmp_path / "repo", global_dir=tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        manager.path(r"\\server\share\provider.json")


@pytest.mark.skipif(sys.platform != "win32", reason="backslashes are Windows-only")
def test_windows_backslash_traversal_is_refused(tmp_path):
    """Backslash separators must not smuggle a ``..`` past the check."""
    manager = ConfigManager.for_workspace(tmp_path / "repo", global_dir=tmp_path)

    with pytest.raises(MitoCodeConfigPathError):
        manager.path(r"secret\..\provider.json")
