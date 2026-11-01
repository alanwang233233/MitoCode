"""Discovery and management of Agent Skills across ordered roots.

:class:`SkillsManager` scans an ordered list of root directories for skill
entries — one directory holding a ``SKILL.md`` each — and serves the results:
a catalogue of :class:`~mitocode.core.skills.skill.Skill` metadata for
progressive-disclosure level 1, and on-demand instruction bodies for level 2
via :meth:`SkillsManager.read_skill`.

Two rules shape the module.  Roots are scanned in the order given and the
first root offering a name wins, so a workspace can override a user-level
skill of the same name; the shadowed duplicate stays visible through
:meth:`SkillsManager.issues`.  And scanning is strict per entry but forgiving
overall: one broken skill directory costs only itself and is recorded as a
:class:`~mitocode.core.skills.skill.SkillIssue`, never raised, so the skills
that work are always returned — the same philosophy the MCP manager applies
to configured servers.

The module knows nothing about *where* roots come from: choosing the
workspace's ``.mitocode/skills`` or a user-level directory is the caller's
(wiring layer's) decision, and no default path is hardcoded here.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger as loguru_logger

from mitocode.core.exception.mitocode_exception import (
    MitoCodeSkillDefinitionError,
    MitoCodeSkillNotFoundError,
    MitoCodeSkillReadError,
)
from mitocode.core.skills.skill import Skill, SkillIssue, SkillRoot
from mitocode.core.skills.skill_markdown import parse_skill_markdown, split_frontmatter

logger = loguru_logger.bind(name=__name__)

#: The one file that turns a directory into a skill entry.  Matched
#: case-sensitively by name, never by ``is_file()``, which on a
#: case-insensitive filesystem would also accept ``skill.md``.
_SKILL_FILE_NAME = "SKILL.md"


class SkillsManager:
    """Discovers skills under ordered roots and serves their contents.

    The manager holds no filesystem state at construction time: roots are
    normalised immediately, but the catalogue and issue list are only built
    by :meth:`discover` and fully replaced by every subsequent call.  The
    read accessors (:meth:`list_skills`, :meth:`get`, :meth:`issues`) report
    that last scan, which keeps a scan's results coherent with one another.

    Args:
        roots: Directories to scan, in precedence order.  Each entry is a
            :class:`SkillRoot`, or a bare :class:`~pathlib.Path`/``str``,
            which becomes a root labelled ``""``.
    """

    def __init__(self, roots: Sequence[Union[str, Path, SkillRoot]]) -> None:
        """Normalise the roots; no disk access happens here.

        Args:
            roots: Directories to scan, in precedence order, as accepted by
                the class docstring.
        """
        normalised: List[SkillRoot] = []
        for root in roots:
            if isinstance(root, SkillRoot):
                normalised.append(root)
            else:
                normalised.append(SkillRoot(label="", path=Path(root)))
        self._roots: Tuple[SkillRoot, ...] = tuple(normalised)
        self._catalogue: Tuple[Skill, ...] = ()
        self._issues: Tuple[SkillIssue, ...] = ()

    def roots(self) -> Tuple[SkillRoot, ...]:
        """Return the normalised roots in the order they were given.

        Returns:
            One :class:`SkillRoot` per constructor entry, bare paths having
            become roots labelled ``""``.
        """
        return self._roots

    async def discover(self) -> Tuple[Skill, ...]:
        """Rescan every root and rebuild the catalogue and issue list.

        Only one level is scanned: ``<root>/<directory>/SKILL.md``.  Hidden
        entries (names starting with ``.``) and plain files in a root are
        skipped silently; a directory with no case-sensitively named
        ``SKILL.md`` yields an issue.  A root that does not exist contributes
        nothing and no issue; a root that exists but is a file yields one
        issue.  Every problem found is logged at warning level as well as
        recorded.

        Returns:
            One :class:`Skill` per valid skill, sorted by name.  The same
            tuple is cached for :meth:`list_skills` and :meth:`get`, and the
            scan's issues replace whatever earlier scans had recorded.
        """
        catalogue, issues = await asyncio.to_thread(self._scan)
        self._catalogue = catalogue
        self._issues = issues
        for issue in issues:
            logger.warning(f"Skill issue at {issue.path}: {issue.detail}")
        return catalogue

    def list_skills(self) -> Tuple[Skill, ...]:
        """Return the most recent :meth:`discover` catalogue.

        Returns:
            The skills of the last scan, sorted by name; empty before the
            first scan.
        """
        return self._catalogue

    def get(self, name: str) -> Optional[Skill]:
        """Return the named skill from the current catalogue.

        Args:
            name: A skill name as returned by :meth:`discover`.

        Returns:
            The skill, or ``None`` when the name is unknown or no scan has
            run yet.
        """
        for skill in self._catalogue:
            if skill.name == name:
                return skill
        return None

    def issues(self) -> Tuple[SkillIssue, ...]:
        """Return the problems found by the most recent :meth:`discover`.

        Returns:
            The recorded issues, in scan order; empty when the last scan was
            clean or has not run yet.  Every scan clears and rebuilds the
            list, so a problem that goes away stops being reported.
        """
        return self._issues

    async def read_skill(self, name: str) -> str:
        """Read a known skill's instruction body fresh from disk.

        Progressive-disclosure level 2.  The file is re-read on every call
        and its frontmatter block, when structurally present, is stripped;
        field validation is deliberately not repeated, so a file edited after
        the scan still yields its instructions.

        Args:
            name: A skill name as returned by :meth:`discover`.

        Returns:
            The instruction body: the text below the frontmatter block with
            leading and trailing empty lines stripped, or the whole file
            verbatim when it carries no frontmatter block.

        Raises:
            MitoCodeSkillNotFoundError: If *name* is not in the current
                catalogue.
            MitoCodeSkillReadError: If the skill's file cannot be read from
                disk (deleted, permission denied, decoding failure).
        """
        skill = self.get(name)
        if skill is None:
            raise MitoCodeSkillNotFoundError(
                f"unknown skill {name!r}: not in the catalogue; run discover() "
                "and check the name"
            )
        try:
            text = await asyncio.to_thread(
                skill.skill_md_path.read_text, encoding="utf-8"
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise MitoCodeSkillReadError(
                f"cannot read the SKILL.md of skill {name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        _raw, body = split_frontmatter(text)
        return body

    def _scan(self) -> Tuple[Tuple[Skill, ...], Tuple[SkillIssue, ...]]:
        """Walk the roots and collect the catalogue and issues.

        Runs inside a worker thread (via :meth:`discover`), so blocking
        filesystem calls never hold the event loop.  Nothing here raises for
        a problem local to one entry: every failure becomes an issue, which
        is what keeps a broken skill from costing the working ones.

        Returns:
            ``(catalogue, issues)`` — the valid skills sorted by name, and
            the problems in scan order.
        """
        skills: List[Skill] = []
        issues: List[SkillIssue] = []
        claimed: Dict[str, str] = {}

        for root in self._roots:
            if not root.path.exists():
                continue
            if not root.path.is_dir():
                issues.append(
                    SkillIssue(path=root.path, detail="skill root is not a directory")
                )
                continue
            self._scan_root(root, skills, issues, claimed)

        return (tuple(sorted(skills, key=lambda skill: skill.name)), tuple(issues))

    def _scan_root(
        self,
        root: SkillRoot,
        skills: List[Skill],
        issues: List[SkillIssue],
        claimed: Dict[str, str],
    ) -> None:
        """Scan one root's immediate children, appending to the shared lists.

        Args:
            root: The root being scanned.
            skills: Catalogue accumulator shared across roots.
            issues: Issue accumulator shared across roots.
            claimed: Skill names already taken by an earlier root, mapped to
                a string identifying the winning root.
        """
        try:
            entries = sorted(root.path.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            issues.append(
                SkillIssue(
                    path=root.path,
                    detail=f"cannot list the skill root: {type(exc).__name__}: {exc}",
                )
            )
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            self._scan_entry(root, entry, skills, issues, claimed)

    def _scan_entry(
        self,
        root: SkillRoot,
        entry: Path,
        skills: List[Skill],
        issues: List[SkillIssue],
        claimed: Dict[str, str],
    ) -> None:
        """Judge one candidate skill directory.

        An entry becomes a skill only when it holds a case-sensitively named
        ``SKILL.md`` whose contents validate; every other outcome appends an
        issue.  A name an earlier root already claimed is shadowed: recorded
        as an issue and skipped, the earlier root's skill standing.

        Args:
            root: The root the entry was found under.
            entry: The candidate skill directory.
            skills: Catalogue accumulator shared across roots.
            issues: Issue accumulator shared across roots.
            claimed: Skill names already taken by an earlier root, mapped to
                a string identifying the winning root.
        """
        try:
            children = {child.name for child in entry.iterdir()}
        except OSError as exc:
            issues.append(
                SkillIssue(
                    path=entry,
                    detail=f"cannot list the skill directory: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return
        if _SKILL_FILE_NAME not in children:
            issues.append(
                SkillIssue(path=entry, detail="directory contains no SKILL.md")
            )
            return

        skill_md = entry / _SKILL_FILE_NAME
        try:
            text = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(
                SkillIssue(
                    path=skill_md,
                    detail=f"cannot read SKILL.md: {type(exc).__name__}: {exc}",
                )
            )
            return

        try:
            parsed = parse_skill_markdown(text, expected_name=entry.name)
        except MitoCodeSkillDefinitionError as exc:
            issues.append(SkillIssue(path=skill_md, detail=str(exc)))
            return

        if parsed.name in claimed:
            issues.append(
                SkillIssue(
                    path=entry,
                    detail=f"skill {parsed.name!r} is shadowed by the same name "
                    f"in root {claimed[parsed.name]!r}, which takes precedence",
                )
            )
            return
        claimed[parsed.name] = root.label if root.label else str(root.path)

        skills.append(
            Skill(
                name=parsed.name,
                description=parsed.description,
                path=entry,
                skill_md_path=skill_md,
                source=root.label,
                license=parsed.license,
                allowed_tools=parsed.allowed_tools,
                metadata=dict(parsed.metadata),
            )
        )
