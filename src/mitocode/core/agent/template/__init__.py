"""Agent templates: declarative session seeds discovered from three roots.

A template is a directory holding a ``template.json`` whose ``session_autoload``
blocks are replayed into a session at creation time.  This package owns the
format (:mod:`.template_json`), the discovery (:mod:`.template_manager`) and the
types they share (:mod:`.template`); applying a template to a live session lives
in :mod:`.apply`, which is imported explicitly by the orchestrator so that
importing the data types never drags in the agent runtime.
"""

from mitocode.core.agent.template.template import (
    WORKSPACE_PLACEHOLDER,
    AgentTemplate,
    TemplateDocument,
    TemplateIssue,
    TemplateRoot,
    builtin_templates_dir,
    render_text,
)
from mitocode.core.agent.template.template_json import (
    MAX_TEMPLATE_FILE_BYTES,
    parse_template_json,
)
from mitocode.core.agent.template.template_manager import AgentTemplateManager

__all__ = [
    "MAX_TEMPLATE_FILE_BYTES",
    "WORKSPACE_PLACEHOLDER",
    "AgentTemplate",
    "AgentTemplateManager",
    "TemplateDocument",
    "TemplateIssue",
    "TemplateRoot",
    "builtin_templates_dir",
    "parse_template_json",
    "render_text",
]
