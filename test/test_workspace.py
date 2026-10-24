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
    MitoCodeWorkspaceError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceDumpError,
    MitoCodeWorkspacePathError,
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


# ──────────────────────────────────────────────
# resolve_within / is_within (public path API)
# ──────────────────────────────────────────────


def test_resolve_within_accepts_relative_path_inside(existing_workspace):
    resolved = existing_workspace.resolve_within("agent/inside.txt")
    assert resolved == (existing_workspace.path.resolve() / "agent/inside.txt")


def test_resolve_within_accepts_absolute_path_inside(existing_workspace):
    absolute = existing_workspace.path.resolve() / "agent/inside.txt"
    assert existing_workspace.resolve_within(absolute) == absolute


def test_resolve_within_rejects_traversal(existing_workspace):
    with pytest.raises(ValueError, match="Path traversal detected"):
        existing_workspace.resolve_within(Path("../../../etc/passwd"))


def test_resolve_within_rejects_absolute_outside(existing_workspace):
    with pytest.raises(MitoCodeWorkspacePathError):
        existing_workspace.resolve_within(Path("/etc/passwd"))


def test_resolve_within_error_is_a_value_error(existing_workspace):
    """The new type must stay a ValueError so existing raises(ValueError) calls hold."""
    assert issubclass(MitoCodeWorkspacePathError, ValueError)
    assert issubclass(MitoCodeWorkspacePathError, MitoCodeWorkspaceError)


def test_is_within_does_not_raise(existing_workspace):
    assert existing_workspace.is_within("agent/inside.txt") is True
    assert existing_workspace.is_within("/etc/passwd") is False


def test_private_alias_delegates_and_keeps_message(existing_workspace):
    with pytest.raises(ValueError, match="Path traversal detected"):
        existing_workspace._validate_within_workspace(Path("../../../etc/passwd"))


# ──────────────────────────────────────────────
# resolve_within / is_within — uncovered branches
# ──────────────────────────────────────────────


class _FspathRaisingOSError:
    """An os.PathLike whose __fspath__ fails, to reach the OSError arm of is_within."""

    def __fspath__(self):
        """Fail the way a broken path provider would.

        Returns:
            Never returns; always raises.

        Raises:
            OSError: Always, to exercise is_within's OSError handling.
        """
        raise OSError("path provider unavailable")


def test_resolve_within_normalises_dot_and_dotdot_segments(existing_workspace):
    """Relative and absolute spellings of a/./b/../c must both come back as a/c."""
    root = existing_workspace.path.resolve()

    assert (
        existing_workspace.resolve_within("a/./b/../c") == root / "a/c"
    ), "relative '.' and '..' segments must be collapsed before the check"
    spelled_absolute = root / "a" / "b" / ".." / "c"
    assert (
        existing_workspace.resolve_within(spelled_absolute) == root / "a/c"
    ), "an absolute target with an in-bound '..' must be normalised too"


def test_resolve_within_accepts_the_workspace_root_itself(existing_workspace):
    """Containment is inclusive: the root, however spelled, resolves to the root."""
    root = existing_workspace.path.resolve()

    assert existing_workspace.resolve_within(".") == root, "'.' must name the root"
    assert (
        existing_workspace.resolve_within(str(existing_workspace.path)) == root
    ), "an absolute path to the root is inside the root"


def test_resolve_within_rejects_sibling_directory_sharing_root_prefix(
    existing_workspace,
):
    """A sibling whose name extends the root's name is outside — the check is per-component."""
    root = existing_workspace.path
    sibling = root.parent / (root.name + "_evil")
    sibling.mkdir()
    outside_file = sibling / "loot.txt"
    outside_file.write_text("secret")
    assert str(outside_file).startswith(str(root)), (
        "precondition: the two paths share a string prefix, so a str.startswith check "
        "would wrongly accept this"
    )

    with pytest.raises(MitoCodeWorkspacePathError, match="Path traversal detected"):
        existing_workspace.resolve_within(outside_file)


def test_resolve_within_rejects_symlink_inside_pointing_outside(existing_workspace):
    """A symlink in the workspace pointing at an outside file is refused by both entry points."""
    outside_file = existing_workspace.path.parent / "outside_target.txt"
    outside_file.write_text("secret")
    (existing_workspace.path / "escape_link").symlink_to(outside_file)

    with pytest.raises(MitoCodeWorkspacePathError, match="Path traversal detected"):
        existing_workspace.resolve_within("escape_link")
    assert (
        existing_workspace.is_within("escape_link") is False
    ), "the predicate must agree with the raiser on an escaping symlink"


def test_resolve_within_accepts_symlink_inside_pointing_inside(existing_workspace):
    """A symlink to another in-workspace location is legitimate and resolves to its target."""
    agent_dir = existing_workspace.path / "agent"
    agent_dir.mkdir()
    real_file = agent_dir / "real.txt"
    real_file.write_text("hello")
    (existing_workspace.path / "inside_link").symlink_to(real_file)

    assert (
        existing_workspace.resolve_within("inside_link") == real_file.resolve()
    ), "an in-workspace symlink must resolve to its target, not be rejected"


def test_resolve_within_rejects_dangling_symlink_pointing_outside(existing_workspace):
    """A link whose target does not exist but sits outside is still an escape attempt."""
    outside_dir = existing_workspace.path.parent / "secrets"
    outside_dir.mkdir()
    missing = outside_dir / "never_created.txt"
    (existing_workspace.path / "dangling_escape").symlink_to(missing)
    assert not missing.exists(), "precondition: the link target must not exist on disk"

    with pytest.raises(MitoCodeWorkspacePathError, match="Path traversal detected"):
        existing_workspace.resolve_within("dangling_escape")


def test_resolve_within_accepts_absolute_target_under_symlinked_root_alias(
    tmp_path, existing_workspace
):
    """A workspace rooted at a symlink still accepts the real path, because both sides resolve."""
    alias = tmp_path / "ws_alias"
    alias.symlink_to(existing_workspace.path)
    assert alias.is_symlink(), "precondition: the alias must be a symlink"
    assert str(alias) != str(alias.resolve()), (
        "precondition: the unresolved alias string must differ from the resolved root, "
        "otherwise this does not exercise resolve-both-sides"
    )
    aliased_workspace = Workspace(alias, "alias-rooted", 1)
    target = existing_workspace.path.resolve() / "agent/inside.txt"

    assert (
        aliased_workspace.resolve_within(target) == target
    ), "the real absolute path must be recognised as inside a symlink-rooted workspace"
    assert (
        aliased_workspace.resolve_within(alias / "agent/inside.txt") == target
    ), "the alias-rooted spelling must resolve to the same file"
    assert aliased_workspace.is_within(target) is True


@pytest.mark.parametrize("target", [None, 123, b"agent/inside.txt"])
def test_is_within_returns_false_for_non_path_argument(existing_workspace, target):
    """A non-path argument counts as not-within instead of escaping as a TypeError."""
    assert (
        existing_workspace.is_within(target) is False
    ), f"is_within({target!r}) must answer False, never raise"


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_is_within_returns_false_for_embedded_nul_byte(existing_workspace, kind):
    """An embedded NUL makes the path unresolvable; both spellings must answer False."""
    target = (
        "agent/in\0side.txt"
        if kind == "relative"
        else str(existing_workspace.path / "agent/in\0side.txt")
    )

    assert (
        existing_workspace.is_within(target) is False
    ), f"a NUL byte in a {kind} path must not escape as a ValueError"


def test_is_within_returns_false_when_fspath_raises_oserror(existing_workspace):
    """The OSError arm of is_within's handler is reachable and returns False."""
    assert (
        existing_workspace.is_within(_FspathRaisingOSError()) is False
    ), "an OSError while building the path must count as not-within"


def test_is_within_treats_empty_target_as_the_workspace_root(existing_workspace):
    """Path('') collapses to Path('.'), so an empty target names the root and is within."""
    assert (
        existing_workspace.resolve_within("") == existing_workspace.path.resolve()
    ), "an empty target must resolve to the root, matching Path('') == Path('.')"
    assert (
        existing_workspace.is_within("") is True
    ), "consistent with resolve_within: an empty target names the root, which is in"


def test_is_within_returns_false_for_symlink_loop(existing_workspace):
    """DEFECT (d2a8815): a symlink loop breaks both clauses is_within documents.

    is_within promises it "never" raises and that anything which "cannot be
    resolved into a usable path" counts as not-within.  On Python <= 3.12 the
    first clause breaks: Path.resolve() raises RuntimeError, which is not in the
    (ValueError, TypeError, OSError) tuple, so RuntimeError escapes to the
    caller.  On Python >= 3.13 the second clause breaks instead: resolve() no
    longer raises on a loop and leaves the component unresolved, so the result
    still sits under the root and the predicate answers True.
    """
    first = existing_workspace.path / "loop_a"
    second = existing_workspace.path / "loop_b"
    first.symlink_to(second)
    second.symlink_to(first)

    try:
        verdict = existing_workspace.is_within("loop_a")
    except Exception as exc:  # any escape at all is the defect, not just this type
        pytest.fail(
            "is_within documents 'never raising', but it propagated "
            f"{type(exc).__name__}: {exc}"
        )
    assert verdict is False, (
        "a symlink loop cannot be resolved into a usable path, so it must not count "
        f"as within; got {verdict!r}"
    )


def test_validate_within_workspace_accepts_str_argument(existing_workspace):
    """The private alias takes a str despite its Path annotation, and resolves it."""
    resolved = existing_workspace._validate_within_workspace("agent/inside.txt")

    assert isinstance(resolved, Path), "the alias must return a Path, not a str"
    assert (
        resolved == existing_workspace.path.resolve() / "agent/inside.txt"
    ), "a str target must be resolved exactly as resolve_within resolves it"
