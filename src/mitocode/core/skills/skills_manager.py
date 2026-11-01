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
import stat
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger as loguru_logger

from mitocode.core.exception.mitocode_exception import (
    MitoCodeSkillDefinitionError,
    MitoCodeSkillNotFoundError,
    MitoCodeSkillReadError,
)
from mitocode.core.skills.skill import Skill, SkillIssue, SkillRoot
from mitocode.core.skills.skill_markdown import (
    SkillFrontmatter,
    parse_skill_markdown,
    split_frontmatter,
)

logger = loguru_logger.bind(name=__name__)

#: The one file that turns a directory into a skill entry.  Matched
#: case-sensitively by name, never by ``is_file()``, which on a
#: case-insensitive filesystem would also accept ``skill.md``.
_SKILL_FILE_NAME = "SKILL.md"

#: Largest accepted ``SKILL.md``, checked from ``st_size`` before anything is
#: opened — the same guard the file tools apply, and for the same reason: the
#: body is loaded whole before any consumer clamps it, so the output budget
#: does not bound the read itself.
MAX_SKILL_FILE_BYTES = 10 * 1024 * 1024


def _read_skill_text(path: Path) -> str:
    """Read one SKILL.md with the file tools' admission checks applied.

    A named pipe would block the reading thread forever, a directory or a
    symlink is not the text file a skill entry promises, and an oversized
    regular file would be loaded whole only to be clamped afterwards — so
    all three are refused from the ``lstat`` result before anything is
    opened, mirroring ``fs_tools``' regular-file and 10 MiB-size checks.
    ``lstat`` rather than ``stat`` is deliberate: a symlinked SKILL.md is
    refused outright, in step with discovery's symlink rejection.

    Args:
        path: The SKILL.md file to read.

    Returns:
        The file's decoded text.

    Raises:
        MitoCodeSkillReadError: If the path is not a regular file, is above
            :data:`MAX_SKILL_FILE_BYTES`, or cannot be read or decoded.
    """
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise MitoCodeSkillReadError(f"SKILL.md at {path} is not a regular file")
        if info.st_size > MAX_SKILL_FILE_BYTES:
            raise MitoCodeSkillReadError(
                f"SKILL.md at {path} is {info.st_size} bytes, above the "
                f"{MAX_SKILL_FILE_BYTES}-byte read cap"
            )
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MitoCodeSkillReadError(
            f"SKILL.md at {path} cannot be read: {type(exc).__name__}: {exc}"
        ) from exc


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

        Raises:
            TypeError: If *roots* is a bare string (or bytes).  A string is
                technically a sequence of strings, so accepting it here would
                silently scan one character per root; the caller almost
                certainly meant a one-element list.
        """
        if isinstance(roots, (str, bytes)):
            raise TypeError(
                "roots must be a sequence of str, Path or SkillRoot entries, "
                "not a single string: a bare string would be scanned one "
                "character at a time"
            )
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
        the scan still yields its instructions.  Two things *are* re-checked
        on every read, because a scan-time verdict can go stale: the file
        must still resolve inside the scanned roots, and it must pass the
        regular-file and size admission checks — a SKILL.md swapped for a
        symlink, a named pipe or an oversized file after discovery is
        refused, never read.

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
                disk (deleted, permission denied, decoding failure), is not
                a regular file or an admitted size, or no longer resolves
                inside the scanned roots.
        """
        skill = self.get(name)
        if skill is None:
            raise MitoCodeSkillNotFoundError(
                f"unknown skill {name!r}: not in the catalogue; run discover() "
                "and check the name"
            )
        if not self._resolved_within_roots(skill.skill_md_path):
            raise MitoCodeSkillReadError(
                f"cannot read the SKILL.md of skill {name!r}: it no longer "
                "resolves inside the scanned skill roots"
            )
        try:
            text = await asyncio.to_thread(_read_skill_text, skill.skill_md_path)
        except MitoCodeSkillReadError as exc:
            raise MitoCodeSkillReadError(
                f"cannot read the SKILL.md of skill {name!r}: {exc}"
            ) from exc
        _raw, body = split_frontmatter(text)
        return body

    def _resolved_within_roots(self, path: Path) -> bool:
        """Report whether *path*'s full resolution stays inside the scan roots.

        The containment re-read :meth:`read_skill` needs: discovery admitted
        the file, but the tree can change underneath it, and a symlink
        swapped in afterwards would otherwise serve whatever it points at.
        Resolving and comparing against the resolved roots covers a symlinked
        file, a symlinked ancestor directory and any ``..`` climb in one
        check; a symlink loop, which cannot be resolved, counts as outside.

        Args:
            path: The skill file to re-check.

        Returns:
            True only when the fully resolved path lies under at least one
            scanned root's own full resolution.
        """
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            # RuntimeError is a symlink loop on 3.10-3.12; either way the
            # path cannot be served.
            return False
        return any(resolved.is_relative_to(root.path.resolve()) for root in self._roots)

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

        An entry becomes a skill only when it holds a case-sensitively named,
        non-symlink ``SKILL.md`` whose contents validate; every other outcome
        appends an issue.  Symlinked entries — the directory itself or the
        file — are refused rather than followed: a skill's body is served to
        the model later by name, and a link planted where the user saw an
        innocent directory must not turn into an arbitrary-file read.  A
        name an earlier root already claimed is shadowed: recorded as an
        issue and skipped, the earlier root's skill standing.

        Args:
            root: The root the entry was found under.
            entry: The candidate skill directory.
            skills: Catalogue accumulator shared across roots.
            issues: Issue accumulator shared across roots.
            claimed: Skill names already taken by an earlier root, mapped to
                a string identifying the winning root.
        """
        skill_md = self._locate_skill_file(entry, issues)
        if skill_md is None:
            return
        parsed = self._load_skill_markdown(entry, skill_md, issues)
        if parsed is None:
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

    def _locate_skill_file(
        self, entry: Path, issues: List[SkillIssue]
    ) -> Optional[Path]:
        """Reject an inadmissible directory and return its SKILL.md path.

        The structural half of admission: symlinked directories, unlistable
        directories, directories without the skill file, and symlinked skill
        files all fail here, before any content is read.

        Args:
            entry: The candidate skill directory.
            issues: Issue accumulator; an entry rejected here appends its
                issue and the caller skips it.

        Returns:
            The SKILL.md path to load, or ``None`` when the directory is not
            admissible.
        """
        if entry.is_symlink():
            issues.append(
                SkillIssue(
                    path=entry,
                    detail="skill directory is a symlink; symlinks are not admitted",
                )
            )
            return None
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
            return None
        if _SKILL_FILE_NAME not in children:
            issues.append(
                SkillIssue(path=entry, detail="directory contains no SKILL.md")
            )
            return None

        skill_md = entry / _SKILL_FILE_NAME
        if skill_md.is_symlink():
            issues.append(
                SkillIssue(
                    path=skill_md,
                    detail="SKILL.md is a symlink; symlinks are not admitted",
                )
            )
            return None
        return skill_md

    def _load_skill_markdown(
        self, entry: Path, skill_md: Path, issues: List[SkillIssue]
    ) -> Optional[SkillFrontmatter]:
        """Read and validate one located SKILL.md.

        The content half of admission: the admission-checked read, then full
        frontmatter validation against the directory name.

        Args:
            entry: The candidate skill directory, supplying the expected name.
            skill_md: The located SKILL.md path from
                :meth:`_locate_skill_file`.
            issues: Issue accumulator; an entry rejected here appends its
                issue and the caller skips it.

        Returns:
            The validated frontmatter, or ``None`` when the file is not
            admissible.
        """
        try:
            text = _read_skill_text(skill_md)
        except MitoCodeSkillReadError as exc:
            issues.append(SkillIssue(path=skill_md, detail=str(exc)))
            return None

        try:
            return parse_skill_markdown(text, expected_name=entry.name)
        except MitoCodeSkillDefinitionError as exc:
            issues.append(SkillIssue(path=skill_md, detail=str(exc)))
            return None
