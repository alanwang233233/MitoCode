"""Discovery and management of agent templates across ordered roots.

:class:`AgentTemplateManager` scans an ordered list of root directories for
template entries — one directory holding a ``template.json`` each — and serves
the resulting catalogue: name, description and the validated message blocks a
session is seeded with.

Two rules shape the module, and they are the ones the skills manager applies.

Roots are scanned in the order given and the first root offering a name wins, so
a workspace can override a template that ships with the package; the shadowed
duplicate stays visible through :meth:`AgentTemplateManager.issues`.  Scanning
is strict per entry but forgiving overall: one broken template directory costs
only itself and is recorded as a :class:`~mitocode.core.agent.template.template.TemplateIssue`,
never raised, so the templates that work are always returned.

The module knows nothing about *where* roots come from: choosing the workspace's
``.mitocode/agent_templates``, the global copy or the packaged directory is the
caller's (wiring layer's) decision, and no default path is hardcoded here.
"""

import asyncio
import stat
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger as loguru_logger

from mitocode.core.agent.template.template import (
    AgentTemplate,
    TemplateDocument,
    TemplateIssue,
    TemplateRoot,
)
from mitocode.core.agent.template.template_json import (
    MAX_TEMPLATE_FILE_BYTES,
    parse_template_json,
)
from mitocode.core.exception.mitocode_exception import (
    MitoCodeTemplateDefinitionError,
    MitoCodeTemplateReadError,
)

logger = loguru_logger.bind(name=__name__)

#: The one file that turns a directory into a template entry.  Matched
#: case-sensitively by name, never by ``is_file()``, which on a
#: case-insensitive filesystem would also accept ``Template.json``.
_TEMPLATE_FILE_NAME = "template.json"


def _read_template_text(path: Path) -> str:
    """Read one template.json with the file tools' admission checks applied.

    A named pipe would block the reading thread forever, a directory or a
    symlink is not the text file a template entry promises, and an oversized
    regular file would be loaded whole only to be replayed into every session it
    creates.  All three are refused from the ``lstat`` result before anything is
    opened, mirroring the skills manager's read guard.

    Args:
        path: The template.json file to read.

    Returns:
        The file's decoded text.

    Raises:
        MitoCodeTemplateReadError: If the path is not a regular file, is above
            :data:`~mitocode.core.agent.template.template_json.MAX_TEMPLATE_FILE_BYTES`,
            or cannot be read or decoded.
    """
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise MitoCodeTemplateReadError(
                f"template.json at {path} is not a regular file"
            )
        if info.st_size > MAX_TEMPLATE_FILE_BYTES:
            raise MitoCodeTemplateReadError(
                f"template.json at {path} is {info.st_size} bytes, above the "
                f"{MAX_TEMPLATE_FILE_BYTES}-byte read cap"
            )
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MitoCodeTemplateReadError(
            f"template.json at {path} cannot be read: {type(exc).__name__}: {exc}"
        ) from exc


class AgentTemplateManager:
    """Discovers templates under ordered roots and serves their contents.

    The manager holds no filesystem state at construction time: roots are
    normalised immediately, but the catalogue and issue list are only built by
    :meth:`discover` and fully replaced by every subsequent call.  The read
    accessors (:meth:`list_templates`, :meth:`get`, :meth:`issues`) report that
    last scan, which keeps a scan's results coherent with one another.

    Args:
        roots: Directories to scan, in precedence order.  Each entry is a
            :class:`~mitocode.core.agent.template.template.TemplateRoot`, or a
            bare :class:`~pathlib.Path`/``str``, which becomes a root labelled
            ``""``.
    """

    def __init__(self, roots: Sequence[Union[str, Path, TemplateRoot]]) -> None:
        """Normalise the roots; no disk access happens here.

        Args:
            roots: Directories to scan, in precedence order.

        Raises:
            TypeError: If *roots* is a bare string (or bytes).  A string is
                technically a sequence of strings, so accepting it would
                silently scan one character per root.
        """
        if isinstance(roots, (str, bytes)):
            raise TypeError(
                "roots must be a sequence of str, Path or TemplateRoot entries, "
                "not a single string: a bare string would be scanned one "
                "character at a time"
            )
        normalised: List[TemplateRoot] = []
        for root in roots:
            if isinstance(root, TemplateRoot):
                normalised.append(root)
            else:
                normalised.append(TemplateRoot(label="", path=Path(root)))
        self._roots: Tuple[TemplateRoot, ...] = tuple(normalised)
        self._catalogue: Tuple[AgentTemplate, ...] = ()
        self._issues: Tuple[TemplateIssue, ...] = ()

    def roots(self) -> Tuple[TemplateRoot, ...]:
        """Return the normalised roots in the order they were given.

        Returns:
            One :class:`TemplateRoot` per constructor entry, bare paths having
            become roots labelled ``""``.
        """
        return self._roots

    async def discover(self) -> Tuple[AgentTemplate, ...]:
        """Rescan every root and rebuild the catalogue and issue list.

        Only one level is scanned: ``<root>/<directory>/template.json``.  Hidden
        entries and plain files in a root are skipped silently; a directory with
        no case-sensitively named ``template.json`` yields an issue.  A root
        that does not exist contributes nothing and no issue; a root that exists
        but is a file yields one issue.  Every problem found is logged at
        warning level as well as recorded.

        Returns:
            One :class:`AgentTemplate` per valid template, sorted by name.  The
            same tuple is cached for :meth:`list_templates` and :meth:`get`, and
            the scan's issues replace whatever earlier scans had recorded.
        """
        catalogue, issues = await asyncio.to_thread(self._scan)
        self._catalogue = catalogue
        self._issues = issues
        for issue in issues:
            logger.warning(f"Template issue at {issue.path}: {issue.detail}")
        return catalogue

    def list_templates(self) -> Tuple[AgentTemplate, ...]:
        """Return the most recent :meth:`discover` catalogue.

        Returns:
            The templates of the last scan, sorted by name; empty before the
            first scan.
        """
        return self._catalogue

    def get(self, name: str) -> Optional[AgentTemplate]:
        """Return the named template from the current catalogue.

        Args:
            name: A template name as returned by :meth:`discover`.

        Returns:
            The template, or ``None`` when the name is unknown or no scan has
            run yet.
        """
        for template in self._catalogue:
            if template.name == name:
                return template
        return None

    def issues(self) -> Tuple[TemplateIssue, ...]:
        """Return the problems found by the most recent :meth:`discover`.

        Returns:
            The recorded issues, in scan order; empty when the last scan was
            clean or has not run yet.
        """
        return self._issues

    def _scan(self) -> Tuple[Tuple[AgentTemplate, ...], Tuple[TemplateIssue, ...]]:
        """Walk the roots and collect the catalogue and issues.

        Runs inside a worker thread (via :meth:`discover`), so blocking
        filesystem calls never hold the event loop.  Nothing here raises for a
        problem local to one entry: every failure becomes an issue, which is
        what keeps one broken template from costing the working ones.

        Returns:
            ``(catalogue, issues)`` — the valid templates sorted by name, and
            the problems in scan order.
        """
        templates: List[AgentTemplate] = []
        issues: List[TemplateIssue] = []
        claimed: Dict[str, str] = {}

        for root in self._roots:
            if not root.path.exists():
                continue
            if not root.path.is_dir():
                issues.append(
                    TemplateIssue(
                        path=root.path, detail="template root is not a directory"
                    )
                )
                continue
            self._scan_root(root, templates, issues, claimed)

        return (tuple(sorted(templates, key=lambda item: item.name)), tuple(issues))

    def _scan_root(
        self,
        root: TemplateRoot,
        templates: List[AgentTemplate],
        issues: List[TemplateIssue],
        claimed: Dict[str, str],
    ) -> None:
        """Scan one root's immediate children, appending to the shared lists.

        Args:
            root: The root being scanned.
            templates: Catalogue accumulator shared across roots.
            issues: Issue accumulator shared across roots.
            claimed: Template names already taken by an earlier root, mapped to
                a string identifying the winning root.
        """
        try:
            entries = sorted(root.path.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            issues.append(
                TemplateIssue(
                    path=root.path,
                    detail=f"cannot list the template root: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            self._scan_entry(root, entry, templates, issues, claimed)

    def _scan_entry(
        self,
        root: TemplateRoot,
        entry: Path,
        templates: List[AgentTemplate],
        issues: List[TemplateIssue],
        claimed: Dict[str, str],
    ) -> None:
        """Judge one candidate template directory.

        An entry becomes a template only when it holds a case-sensitively named,
        non-symlink ``template.json`` whose contents validate; every other
        outcome appends an issue.  Symlinked entries — the directory itself or
        the file — are refused rather than followed: a template is replayed into
        a session, and a link planted where the user saw an innocent directory
        must not turn into an arbitrary-file read.

        Args:
            root: The root the entry was found under.
            entry: The candidate template directory.
            templates: Catalogue accumulator shared across roots.
            issues: Issue accumulator shared across roots.
            claimed: Template names already taken by an earlier root, mapped to
                a string identifying the winning root.
        """
        template_json = self._locate_template_file(entry, issues)
        if template_json is None:
            return
        document = self._load_document(entry, template_json, issues)
        if document is None:
            return

        if document.name in claimed:
            issues.append(
                TemplateIssue(
                    path=entry,
                    detail=f"template {document.name!r} is shadowed by the same "
                    f"name in root {claimed[document.name]!r}, which takes "
                    f"precedence",
                )
            )
            return
        claimed[document.name] = root.label if root.label else str(root.path)

        templates.append(
            AgentTemplate(
                name=document.name,
                description=document.description,
                path=entry,
                template_path=template_json,
                source=root.label,
                session_autoload=document.session_autoload,
            )
        )

    def _locate_template_file(
        self, entry: Path, issues: List[TemplateIssue]
    ) -> Optional[Path]:
        """Reject an inadmissible directory and return its template.json path.

        The structural half of admission: symlinked directories, unlistable
        directories, directories without the template file, and symlinked
        template files all fail here, before any content is read.

        Args:
            entry: The candidate template directory.
            issues: Issue accumulator; an entry rejected here appends its issue.

        Returns:
            The template.json path to load, or ``None``.
        """
        if entry.is_symlink():
            issues.append(
                TemplateIssue(
                    path=entry,
                    detail="template directory is a symlink; symlinks are not "
                    "admitted",
                )
            )
            return None
        try:
            children = {child.name for child in entry.iterdir()}
        except OSError as exc:
            issues.append(
                TemplateIssue(
                    path=entry,
                    detail=f"cannot list the template directory: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return None
        if _TEMPLATE_FILE_NAME not in children:
            issues.append(
                TemplateIssue(path=entry, detail="directory contains no template.json")
            )
            return None

        template_json = entry / _TEMPLATE_FILE_NAME
        if template_json.is_symlink():
            issues.append(
                TemplateIssue(
                    path=template_json,
                    detail="template.json is a symlink; symlinks are not admitted",
                )
            )
            return None
        return template_json

    def _load_document(
        self, entry: Path, template_json: Path, issues: List[TemplateIssue]
    ) -> Optional[TemplateDocument]:
        """Read and validate one located template.json.

        The content half of admission: the admission-checked read, then full
        validation against the directory name.

        Args:
            entry: The candidate template directory, supplying the expected name.
            template_json: The located template.json path.
            issues: Issue accumulator; an entry rejected here appends its issue.

        Returns:
            The validated document, or ``None`` when the file is not admissible.
        """
        try:
            text = _read_template_text(template_json)
        except MitoCodeTemplateReadError as exc:
            issues.append(TemplateIssue(path=template_json, detail=str(exc)))
            return None

        try:
            return parse_template_json(text, expected_name=entry.name)
        except MitoCodeTemplateDefinitionError as exc:
            issues.append(TemplateIssue(path=template_json, detail=str(exc)))
            return None
