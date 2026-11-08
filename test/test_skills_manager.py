"""Tests for mitocode.core.skills.skills_manager (skill discovery/management)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mitocode.core.exception.mitocode_exception import (
    MitoCodeSkillNotFoundError,
    MitoCodeSkillReadError,
)
from mitocode.core.skills.skill import Skill, SkillIssue, SkillRoot
from mitocode.core.skills.skills_manager import MAX_SKILL_FILE_BYTES, SkillsManager


def _skill_md(name, description="A test skill.", body="Step one.\n\nStep two."):
    """Build minimal valid SKILL.md text for discovery tests."""
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def _write_skill(root: Path, name: str, text: str) -> Path:
    """Create one skill directory holding the given SKILL.md text."""
    entry = root / name
    entry.mkdir(parents=True)
    (entry / "SKILL.md").write_text(text, encoding="utf-8")
    return entry


def test_skills_manager_init_normalises_bare_path_and_str_roots_to_empty_label(
    tmp_path,
):
    """Bare Path and str roots become SkillRoot("", ...) and the order is preserved."""
    path_root = tmp_path / "a"
    str_root = str(tmp_path / "b")
    labeled = SkillRoot("workspace", tmp_path / "c")
    manager = SkillsManager([path_root, str_root, labeled])
    assert manager.roots() == (
        SkillRoot("", path_root),
        SkillRoot("", tmp_path / "b"),
        SkillRoot("workspace", tmp_path / "c"),
    )


def test_skills_manager_rejects_bare_string_roots():
    """A bare string passed as roots raises instead of being scanned per character."""
    with pytest.raises(TypeError, match="sequence"):
        SkillsManager("/some/skills/dir")


@pytest.mark.asyncio
async def test_skills_manager_discover_finds_skills_under_two_roots_with_source_labels(
    tmp_path,
):
    """Skills from two roots carry each root's label and exact path fields."""
    ws = tmp_path / "workspace"
    user = tmp_path / "user"
    _write_skill(ws, "alpha", _skill_md("alpha"))
    _write_skill(user, "beta", _skill_md("beta"))
    manager = SkillsManager([SkillRoot("workspace", ws), SkillRoot("user", user)])
    skills = await manager.discover()
    assert [s.name for s in skills] == ["alpha", "beta"]
    assert all(isinstance(s, Skill) for s in skills)
    assert skills[0].source == "workspace"
    assert skills[1].source == "user"
    assert skills[0].path == ws / "alpha"
    assert skills[0].skill_md_path == ws / "alpha" / "SKILL.md"
    assert skills[1].description == "A test skill."
    assert skills[0].license == ""
    assert skills[0].allowed_tools == ()
    assert dict(skills[0].metadata) == {}


@pytest.mark.asyncio
async def test_skills_manager_discover_no_or_missing_roots_yield_empty_catalogue(
    tmp_path,
):
    """No roots, or roots that do not exist, give an empty catalogue and no issues."""
    empty = SkillsManager([])
    assert await empty.discover() == ()
    missing = SkillsManager([tmp_path / "does-not-exist"])
    assert await missing.discover() == ()
    assert missing.issues() == ()
    assert missing.list_skills() == ()


@pytest.mark.asyncio
async def test_skills_manager_discover_root_that_is_a_file_yields_issue_others_scanned(
    tmp_path,
):
    """A root that exists but is a file becomes one issue; other roots still scan."""
    file_root = tmp_path / "not-a-dir"
    file_root.write_text("I am a file", encoding="utf-8")
    good = tmp_path / "good"
    _write_skill(good, "alpha", _skill_md("alpha"))
    manager = SkillsManager([file_root, good])
    skills = await manager.discover()
    assert [s.name for s in skills] == ["alpha"]
    issues = manager.issues()
    assert len(issues) == 1
    assert isinstance(issues[0], SkillIssue)
    assert issues[0].path == file_root
    assert issues[0].detail != ""


@pytest.mark.asyncio
async def test_skills_manager_discover_directory_without_skill_md_yields_issue(
    tmp_path,
):
    """A scanned directory lacking SKILL.md is reported as an issue, not a skill."""
    root = tmp_path / "root"
    hollow = root / "hollow"
    hollow.mkdir(parents=True)
    manager = SkillsManager([root])
    assert await manager.discover() == ()
    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == hollow
    assert issues[0].detail != ""


@pytest.mark.asyncio
async def test_skills_manager_discover_lowercase_skill_md_case_sensitive_yields_issue(
    tmp_path,
):
    """A directory holding only lowercase skill.md is an issue; nothing is discovered."""
    root = tmp_path / "root"
    lowercase = root / "lowercase"
    lowercase.mkdir(parents=True)
    (lowercase / "skill.md").write_text(_skill_md("lowercase"), encoding="utf-8")
    manager = SkillsManager([root])
    assert await manager.discover() == ()
    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == lowercase


@pytest.mark.asyncio
async def test_skills_manager_discover_invalid_skill_md_records_issue_rest_discovered(
    tmp_path,
):
    """One invalid SKILL.md costs only its own entry; valid siblings still load."""
    root = tmp_path / "root"
    _write_skill(root, "broken", "---\nname: Bad_Name\ndescription: d\n---\nbody\n")
    _write_skill(root, "good", _skill_md("good"))
    manager = SkillsManager([root])
    skills = await manager.discover()
    assert [s.name for s in skills] == ["good"]
    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == root / "broken" / "SKILL.md"
    assert issues[0].detail != ""


@pytest.mark.asyncio
async def test_skills_manager_discover_unreadable_skill_md_yields_issue(tmp_path):
    """A SKILL.md unreadable due to file mode becomes an issue, not an exception."""
    if os.name == "nt":
        pytest.skip("Windows has no mode bits, so the unreadable case cannot be staged")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores mode bits, so the unreadable case cannot be staged")
    root = tmp_path / "root"
    locked = root / "locked"
    skill_md = _write_skill(root, "locked", _skill_md("locked")) / "SKILL.md"
    skill_md.chmod(0o000)
    try:
        manager = SkillsManager([root])
        assert await manager.discover() == ()
        issues = manager.issues()
        assert len(issues) == 1
        assert issues[0].path == locked / "SKILL.md"
        assert issues[0].detail != ""
    finally:
        skill_md.chmod(0o644)


@pytest.mark.asyncio
async def test_skills_manager_discover_duplicate_name_first_root_wins_other_issue(
    tmp_path,
):
    """The first root offering a name wins; the shadowed duplicate becomes an issue."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "dup", _skill_md("dup"))
    _write_skill(second, "dup", _skill_md("dup"))
    manager = SkillsManager([SkillRoot("workspace", first), SkillRoot("user", second)])
    skills = await manager.discover()
    assert [s.name for s in skills] == ["dup"]
    assert skills[0].source == "workspace"
    assert skills[0].skill_md_path == first / "dup" / "SKILL.md"
    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == second / "dup"


@pytest.mark.asyncio
async def test_skills_manager_discover_rescan_rebuilds_catalogue_and_issues(tmp_path):
    """Each discover() fully rebuilds the catalogue and clears fixed issues."""
    root = tmp_path / "root"
    _write_skill(root, "alpha", _skill_md("alpha", description="Old description."))
    _write_skill(root, "beta", "not a skill at all")
    manager = SkillsManager([root])
    await manager.discover()
    assert [s.name for s in manager.list_skills()] == ["alpha"]
    assert len(manager.issues()) == 1
    (root / "alpha" / "SKILL.md").write_text(
        _skill_md("alpha", description="New description."), encoding="utf-8"
    )
    (root / "beta" / "SKILL.md").write_text(_skill_md("beta"), encoding="utf-8")
    skills = await manager.discover()
    assert [s.name for s in skills] == ["alpha", "beta"]
    by_name = {s.name: s for s in skills}
    assert by_name["alpha"].description == "New description."
    assert manager.issues() == ()


@pytest.mark.asyncio
async def test_skills_manager_discover_catalogue_sorted_by_name(tmp_path):
    """The discovered catalogue comes back sorted by skill name."""
    root = tmp_path / "root"
    for name in ("zeta", "alpha", "mid"):
        _write_skill(root, name, _skill_md(name))
    manager = SkillsManager([root])
    skills = await manager.discover()
    assert [s.name for s in skills] == ["alpha", "mid", "zeta"]


def test_skills_manager_before_first_discover_catalogue_get_and_issues_empty():
    """Before any discover(): list_skills and issues are empty and get() is None."""
    manager = SkillsManager([])
    assert manager.list_skills() == ()
    assert manager.get("anything") is None
    assert manager.issues() == ()


@pytest.mark.asyncio
async def test_skills_manager_get_and_read_skill_unknown_name_none_and_raises(tmp_path):
    """An unknown name gives get() -> None and read_skill -> MitoCodeSkillNotFoundError."""
    root = tmp_path / "root"
    _write_skill(root, "alpha", _skill_md("alpha"))
    manager = SkillsManager([root])
    await manager.discover()
    assert manager.get("no-such-skill") is None
    with pytest.raises(MitoCodeSkillNotFoundError):
        await manager.read_skill("no-such-skill")


@pytest.mark.asyncio
async def test_skills_manager_read_skill_returns_body_with_frontmatter_stripped(
    tmp_path,
):
    """read_skill re-reads the file and returns the body below the frontmatter."""
    root = tmp_path / "root"
    _write_skill(
        root,
        "alpha",
        "---\nname: alpha\ndescription: d\n---\nStep one.\n\nStep two.",
    )
    manager = SkillsManager([root])
    await manager.discover()
    assert await manager.read_skill("alpha") == "Step one.\n\nStep two."


@pytest.mark.asyncio
async def test_skills_manager_read_skill_file_deleted_after_discover_raises_read_error(
    tmp_path,
):
    """Deleting a catalogued SKILL.md turns read_skill into MitoCodeSkillReadError."""
    root = tmp_path / "root"
    entry = _write_skill(root, "alpha", _skill_md("alpha"))
    manager = SkillsManager([root])
    await manager.discover()
    (entry / "SKILL.md").unlink()
    with pytest.raises(MitoCodeSkillReadError):
        await manager.read_skill("alpha")


@pytest.mark.asyncio
async def test_skills_manager_discover_plain_file_and_hidden_dir_skipped_silently(
    tmp_path,
):
    """Plain files and dot-prefixed directories are ignored without issues."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "notes.txt").write_text("plain file", encoding="utf-8")
    hidden = root / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text(_skill_md("hidden"), encoding="utf-8")
    manager = SkillsManager([root])
    assert await manager.discover() == ()
    assert manager.issues() == ()


@pytest.mark.asyncio
async def test_skills_manager_discover_unlistable_root_yields_issue(tmp_path):
    """A root whose listing is permission-denied yields one issue, nothing raised."""
    if os.name == "nt":
        pytest.skip("Windows has no mode bits, so the unlistable case cannot be staged")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores mode bits, so the unlistable case cannot be staged")
    locked_root = tmp_path / "locked-root"
    locked_root.mkdir()
    good = tmp_path / "good"
    _write_skill(good, "alpha", _skill_md("alpha"))
    locked_root.chmod(0o000)
    try:
        manager = SkillsManager([locked_root, good])
        skills = await manager.discover()
        assert [s.name for s in skills] == ["alpha"]
        issues = manager.issues()
        assert len(issues) == 1
        assert issues[0].path == locked_root
        assert "cannot list the skill root" in issues[0].detail
    finally:
        locked_root.chmod(0o755)


@pytest.mark.asyncio
async def test_skills_manager_discover_unlistable_skill_dir_yields_issue(tmp_path):
    """A skill directory whose listing is permission-denied yields one issue."""
    if os.name == "nt":
        pytest.skip("Windows has no mode bits, so the unlistable case cannot be staged")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores mode bits, so the unlistable case cannot be staged")
    root = tmp_path / "root"
    locked = root / "locked"
    locked.mkdir(parents=True)
    locked.chmod(0o000)
    try:
        manager = SkillsManager([root])
        assert await manager.discover() == ()
        issues = manager.issues()
        assert len(issues) == 1
        assert issues[0].path == locked
        assert "cannot list the skill directory" in issues[0].detail
    finally:
        locked.chmod(0o755)


@pytest.mark.asyncio
async def test_discover_refuses_fifo_skill_md_instead_of_blocking(tmp_path):
    """A FIFO standing in for SKILL.md becomes an issue; it is never opened."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("staging the named-pipe case needs POSIX FIFOs")
    root = tmp_path / "root"
    entry = root / "locked"
    entry.mkdir(parents=True)
    os.mkfifo(entry / "SKILL.md")
    manager = SkillsManager([root])

    await asyncio.wait_for(manager.discover(), timeout=10)

    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == entry / "SKILL.md"
    assert manager.list_skills() == ()


@pytest.mark.asyncio
async def test_read_skill_refuses_fifo_swapped_in_after_discovery(tmp_path):
    """A SKILL.md swapped to a FIFO after discovery is a read error, not a hang."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("staging the named-pipe case needs POSIX FIFOs")
    root = tmp_path / "root"
    entry = _write_skill(root, "alpha", _skill_md("alpha"))
    manager = SkillsManager([root])
    await manager.discover()
    (entry / "SKILL.md").unlink()
    os.mkfifo(entry / "SKILL.md")

    with pytest.raises(MitoCodeSkillReadError):
        await asyncio.wait_for(manager.read_skill("alpha"), timeout=10)


@pytest.mark.asyncio
async def test_discover_refuses_oversized_skill_md_from_its_size_alone(tmp_path):
    """A SKILL.md above the read cap is refused from st_size, never loaded."""
    root = tmp_path / "root"
    entry = root / "big"
    entry.mkdir(parents=True)
    (entry / "SKILL.md").write_text(
        "---\nname: big\ndescription: d\n---\n" + "x" * (MAX_SKILL_FILE_BYTES + 1),
        encoding="utf-8",
    )
    manager = SkillsManager([root])

    await manager.discover()

    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == entry / "SKILL.md"
    assert manager.list_skills() == ()


@pytest.mark.asyncio
async def test_read_skill_refuses_a_file_grown_past_the_cap(tmp_path):
    """A SKILL.md that grows past the cap after discovery is a read error."""
    root = tmp_path / "root"
    entry = _write_skill(root, "alpha", _skill_md("alpha"))
    manager = SkillsManager([root])
    await manager.discover()
    (entry / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: d\n---\n" + "x" * (MAX_SKILL_FILE_BYTES + 1),
        encoding="utf-8",
    )

    with pytest.raises(MitoCodeSkillReadError):
        await manager.read_skill("alpha")


@pytest.mark.asyncio
async def test_read_skill_accepts_a_file_at_exactly_the_cap(tmp_path):
    """The size boundary is inclusive: a SKILL.md of exactly the cap reads."""
    root = tmp_path / "root"
    entry = root / "alpha"
    entry.mkdir(parents=True)
    header = f"---\nname: alpha\ndescription: d\n---\n"
    (entry / "SKILL.md").write_bytes(
        (header + "x" * (MAX_SKILL_FILE_BYTES - len(header))).encode("utf-8")
    )
    manager = SkillsManager([root])
    await manager.discover()

    assert len(await manager.read_skill("alpha")) == MAX_SKILL_FILE_BYTES - len(header)


@pytest.mark.asyncio
async def test_discover_refuses_symlinked_skill_directory(tmp_path):
    """A skill directory that is a symlink is an issue; it is never followed."""
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "SKILL.md").write_text(_skill_md("link"), encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(real, target_is_directory=True)
    manager = SkillsManager([root])

    await manager.discover()

    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == root / "link"
    assert manager.list_skills() == ()


@pytest.mark.asyncio
async def test_discover_refuses_symlinked_skill_md(tmp_path):
    """A symlinked SKILL.md inside an ordinary directory is an issue."""
    target = tmp_path / "real-skill.md"
    target.write_text(_skill_md("alpha"), encoding="utf-8")
    root = tmp_path / "root"
    entry = root / "alpha"
    entry.mkdir(parents=True)
    (entry / "SKILL.md").symlink_to(target)
    manager = SkillsManager([root])

    await manager.discover()

    issues = manager.issues()
    assert len(issues) == 1
    assert issues[0].path == entry / "SKILL.md"
    assert manager.list_skills() == ()


@pytest.mark.asyncio
async def test_read_skill_refuses_a_file_that_left_the_roots_after_discovery(tmp_path):
    """A SKILL.md re-pointed outside the roots after discovery is refused."""
    root = tmp_path / "root"
    entry = _write_skill(root, "alpha", _skill_md("alpha"))
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("do not read", encoding="utf-8")
    manager = SkillsManager([root])
    await manager.discover()
    (entry / "SKILL.md").unlink()
    (entry / "SKILL.md").symlink_to(secret)

    with pytest.raises(MitoCodeSkillReadError):
        await manager.read_skill("alpha")


@pytest.mark.asyncio
async def test_read_skill_still_serves_a_file_swapped_within_the_roots(tmp_path):
    """A regular in-roots swap keeps read_skill working; revalidation is not a freeze."""
    root = tmp_path / "root"
    entry = _write_skill(root, "alpha", _skill_md("alpha"))
    manager = SkillsManager([root])
    await manager.discover()
    (entry / "SKILL.md").write_text(
        _skill_md("alpha", body="Fresh body."), encoding="utf-8"
    )

    assert await manager.read_skill("alpha") == "Fresh body."
