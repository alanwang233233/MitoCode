"""Data types describing discovered Agent Skills and their discovery roots.

A skill follows the open Agent Skills format: one directory holding a
``SKILL.md`` whose YAML frontmatter declares the metadata and whose body
carries the instructions.  The types here are the discovery result shared by
:mod:`mitocode.core.skills.skill_markdown` (which validates one file) and
:mod:`mitocode.core.skills.skills_manager` (which scans directories).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple


@dataclass(frozen=True)
class SkillRoot:
    """One directory scanned for skills, with a short display label.

    Attributes:
        label: Short name for the root's origin, e.g. ``"workspace"`` or
            ``"user"``.  A root built from a bare path carries ``""``.
        path: Directory to scan.  May not exist; a missing root simply
            contributes no skills and no issues.
    """

    label: str
    path: Path


@dataclass(frozen=True)
class Skill:  # pylint: disable=too-many-instance-attributes
    """A valid skill's metadata (progressive-disclosure level 1).

    This is what a caller keeps in memory for every skill: enough for the
    model to decide when a skill applies, without reading any file.  The
    instructions themselves are level 2 and are read on demand through
    :meth:`mitocode.core.skills.skills_manager.SkillsManager.read_skill`.

    Attributes:
        name: Frontmatter ``name``, always equal to the skill directory's
            name.
        description: Frontmatter ``description``; the model uses it to decide
            when the skill applies.
        path: The skill's own directory.
        skill_md_path: Full path of the ``SKILL.md`` file.
        source: Label of the root the skill was discovered under; ``""`` when
            that root was given as a bare path.
        license: Optional ``license`` frontmatter value; ``""`` when absent.
        allowed_tools: Optional ``allowed-tools`` frontmatter value split on
            commas; empty when absent.
        metadata: Optional ``metadata`` frontmatter mapping; empty when
            absent.
    """

    name: str
    description: str
    path: Path
    skill_md_path: Path
    source: str
    license: str = ""
    allowed_tools: Tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillIssue:
    """One problem found while scanning a candidate skill entry.

    Issues are recorded, never raised: one broken entry must not cost the
    skills that work, mirroring the MCP manager's philosophy.

    Attributes:
        path: The offending directory or file.  A scanned directory with no
            ``SKILL.md`` reports the directory; an unreadable or invalid
            ``SKILL.md`` reports the file; a root that is not a directory
            reports the root; a shadowed duplicate reports the shadowed
            skill's directory.
        detail: Human-readable reason.  Names identifiers (directory names,
            field names) but never quotes file contents.
    """

    path: Path
    detail: str
