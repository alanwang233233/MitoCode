"""Data types describing discovered agent templates and their discovery roots.

An agent template is a directory holding a ``template.json``: a name, a
description, and an optional ``session_autoload`` list of message blocks that is
replayed into a session at creation time.  The types here are the shared result
of :mod:`mitocode.core.agent.template.template_json` (which validates one file)
and :mod:`mitocode.core.agent.template.template_manager` (which scans roots).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

#: The literal a template body may use to name the workspace it runs in.
#:
#: Replaced with the workspace's absolute path when the template is applied.
#: A single named placeholder rather than general interpolation: a template is
#: prose and code, so ``str.format`` over it would turn every brace into a
#: KeyError waiting to happen.
WORKSPACE_PLACEHOLDER = "{workspace}"


def render_text(text: str, workspace: Optional[Path]) -> str:
    """Substitute :data:`WORKSPACE_PLACEHOLDER` in *text*.

    Args:
        text: Template text, possibly carrying ``{workspace}``.
        workspace: The workspace root to substitute, or ``None`` to leave the
            placeholder untouched (a caller with no workspace has no path to
            put there, and silently deleting it would hide the omission).

    Returns:
        *text* with every ``{workspace}`` replaced.  Every other brace sequence
        is returned unchanged.
    """
    if workspace is None:
        return text
    return text.replace(WORKSPACE_PLACEHOLDER, str(workspace))


@dataclass(frozen=True)
class TemplateRoot:
    """One directory scanned for templates, with a short display label.

    Attributes:
        label: Short name for the root's origin, e.g. ``"workspace"``,
            ``"global"`` or ``"builtin"``.  A root built from a bare path
            carries ``""``.
        path: Directory to scan.  May not exist; a missing root simply
            contributes no templates and no issues.
    """

    label: str
    path: Path


@dataclass(frozen=True)
class TemplateIssue:
    """One problem found while scanning a candidate template entry.

    Issues are recorded, never raised: one broken entry must not cost the
    templates that work, mirroring the skills and MCP managers.

    Attributes:
        path: The offending directory or file.
        detail: Human-readable reason.  Names identifiers (directory names,
            field names) but never quotes file contents.
    """

    path: Path
    detail: str


@dataclass(frozen=True)
class TemplateDocument:
    """A validated ``template.json``, before it is attached to a root.

    Attributes:
        name: ``name`` field; matches ``^[a-z0-9]+(-[a-z0-9]+)*$``, is at most
            64 characters, and equals the template directory's name.
        description: ``description`` field collapsed to one whitespace-normalised
            line; 1..1024 characters after collapsing.
        session_autoload: Message blocks to replay, in order, each a plain
            mapping whose ``kind`` is ``"text"``, ``"tool_call"`` or
            ``"tool_result"``.  Every call carries a resolved ``id`` and every
            result carries the ``tool_call_id`` it answers.
    """

    name: str
    description: str
    session_autoload: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class AgentTemplate:
    """A template discovered under a root: content plus provenance.

    Attributes:
        name: ``name`` field, always equal to the template directory's name.
        description: What the template is for; the model chooses by this.
        path: The template's own directory.
        template_path: Full path of the ``template.json`` file.
        source: Label of the root the template was discovered under.
        session_autoload: Validated message blocks, as in
            :class:`TemplateDocument`.
    """

    name: str
    description: str
    path: Path
    template_path: Path
    source: str
    session_autoload: Tuple[Mapping[str, Any], ...] = ()

    def system_message(self, workspace: Optional[Path] = None) -> str:
        """Return the template's own system prompt, when it declares one.

        Only the **first** block can replace a session's system prompt; a
        system block further down is ordinary history.  That is what makes
        "template carries a persona" and "template carries context" two
        distinguishable shapes rather than a guess.

        Args:
            workspace: Substituted into ``{workspace}`` when given.

        Returns:
            The first block's rendered ``message`` when that block is a
            ``text`` block with ``role="system"``, otherwise ``""``.
        """
        if not self.session_autoload:
            return ""
        first = self.session_autoload[0]
        if first.get("kind") != "text" or first.get("role") != "system":
            return ""
        return render_text(str(first.get("message", "")), workspace)


def builtin_templates_dir() -> Path:
    """Return the packaged root holding the templates shipped with MitoCode.

    Resolved rather than used verbatim so a symlinked install still finds the
    files next to the module they came from.

    Returns:
        ``<package>/agent/template/builtin``, whether or not it exists on disk.
    """
    return Path(__file__).resolve().parent / "builtin"
