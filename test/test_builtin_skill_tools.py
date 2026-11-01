"""Tests for the built-in ``Skill`` tool (``use_skill``).

The tool is the progressive-disclosure trigger: the model calls it with a
skill name it saw in the system prompt's skills section, and receives the
skill's instruction body.  Every failure is prose beginning ``"Error: "``
rather than an exception, matching the package's other tools, because
``AgentRuntime`` feeds a tool's return value straight back to the model.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mitocode.core.skills.skill import SkillRoot
from mitocode.core.skills.skills_manager import SkillsManager
from mitocode.core.tool.builtin.context import ToolContext, set_current
from mitocode.core.tool.builtin.skill_tools import use_skill


def _skill_md(name: str, body: str = "Step one.\n\nStep two.") -> str:
    """Build minimal valid SKILL.md text with the given body."""
    return f"---\nname: {name}\ndescription: A test skill.\n---\n{body}\n"


def _write_skill(root: Path, name: str, text: str) -> None:
    """Create one skill directory holding the given SKILL.md text."""
    entry = root / name
    entry.mkdir(parents=True)
    (entry / "SKILL.md").write_text(text, encoding="utf-8")


async def _manager_with(root: Path) -> SkillsManager:
    """Build a manager over *root* and discover its skills."""
    manager = SkillsManager([SkillRoot("workspace", root)])
    await manager.discover()
    return manager


@pytest.mark.asyncio
async def test_use_skill_returns_body_for_known_skill(tmp_path):
    """A known name returns the instruction body below the frontmatter."""
    _write_skill(tmp_path, "alpha", _skill_md("alpha"))
    set_current(ToolContext(skills_manager=await _manager_with(tmp_path)))
    try:
        assert await use_skill("alpha") == "Step one.\n\nStep two."
    finally:
        set_current(ToolContext())


@pytest.mark.asyncio
async def test_use_skill_unknown_name_lists_available_skills(tmp_path):
    """An unknown name is prose naming the skills that do exist."""
    _write_skill(tmp_path, "alpha", _skill_md("alpha"))
    _write_skill(tmp_path, "beta", _skill_md("beta"))
    set_current(ToolContext(skills_manager=await _manager_with(tmp_path)))
    try:
        reply = await use_skill("nope")
        assert reply.startswith("Error: unknown skill 'nope'")
        assert "alpha" in reply and "beta" in reply
    finally:
        set_current(ToolContext())


@pytest.mark.asyncio
async def test_use_skill_unknown_name_with_empty_catalogue_says_none_installed(
    tmp_path,
):
    """With no skills discovered at all the reply says so explicitly."""
    set_current(ToolContext(skills_manager=await _manager_with(tmp_path / "absent")))
    try:
        assert await use_skill("nope") == (
            "Error: unknown skill 'nope'. No skills are installed."
        )
    finally:
        set_current(ToolContext())


@pytest.mark.asyncio
async def test_use_skill_without_manager_reports_no_skills():
    """A context wired without a skills manager answers in prose, not by raising."""
    set_current(ToolContext())
    try:
        assert await use_skill("alpha") == "Error: no skills are installed."
    finally:
        set_current(ToolContext())


@pytest.mark.asyncio
async def test_use_skill_read_failure_returns_error_prose(tmp_path):
    """A skill deleted after discovery is an explicit error, never an exception."""
    entry = tmp_path / "alpha"
    entry.mkdir()
    (entry / "SKILL.md").write_text(_skill_md("alpha"), encoding="utf-8")
    manager = await _manager_with(tmp_path)
    (entry / "SKILL.md").unlink()
    set_current(ToolContext(skills_manager=manager))
    try:
        reply = await use_skill("alpha")
        assert reply.startswith("Error:")
        assert "alpha" in reply
    finally:
        set_current(ToolContext())


@pytest.mark.asyncio
async def test_use_skill_clamps_oversized_body(tmp_path):
    """A body above the output budget comes back clamped, not whole."""
    _write_skill(tmp_path, "alpha", _skill_md("alpha", body="x" * 5000))
    set_current(
        ToolContext(skills_manager=await _manager_with(tmp_path), max_output_chars=200)
    )
    try:
        reply = await use_skill("alpha")
        assert "[truncated:" in reply
        assert len(reply) < 500
    finally:
        set_current(ToolContext())
