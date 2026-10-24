"""The default system prompt injected into every agent session.

The prompt is written in English on purpose: it is model-facing text, not
user-facing UI, and the i18n catalogs only govern what the human reads.
A project can replace it wholesale by creating ``.mitocode/SYSTEM_PROMPT.md``
in the workspace root; when that file exists its full content is used
verbatim instead of :data:`DEFAULT_SYSTEM_PROMPT`.
"""

from pathlib import Path
from typing import Any

#: Name of the override file inside the workspace configuration directory.
SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.md"

#: The built-in default system prompt.  Four elements, in order: identity,
#: tool-use norms, output style, safety boundaries.  Tool names must match
#: the registration table in ``tool/builtin/register.py``.
DEFAULT_SYSTEM_PROMPT = """\
You are MitoCode, a coding agent operating inside the user's workspace.

## Tools

You have these tools: Read, Write, Edit, Glob, Grep, Bash, Agent, \
ListAgent, QueryAgent, Background, BackgroundResult. Use them to inspect
and change the workspace directly.

- Read a file before editing it, and prefer Edit over Write for existing
  files: Write refuses to overwrite, and replacing a whole file to change
  one line destroys context.
- Run shell work with Bash from the workspace root. A non-zero exit code
  is data, not a failure: read the output and react.
- Delegate independent work with Agent and check on it later with
  QueryAgent; spawning does not block your turn.
- Detach long-running calls with Background and collect them with
  BackgroundResult.

## Output style

Be concise. When you mention code, cite the file path and, where it
helps, the line number. Say what you did and what you verified; do not
restate the user's request back to them.

## Safety

Operate only inside the workspace root. Operations that reach outside
the workspace require the user's approval, and a denied approval is a
final answer for that operation: continue the task another way instead
of retrying it.
"""


def load_system_prompt(workspace: Any) -> str:
    """Return the system prompt for *workspace*.

    Args:
        workspace: The workspace whose root is searched. Any object with a
            ``path`` attribute works; ``None`` yields the default prompt.

    Returns:
        The content of ``<workspace>/.mitocode/SYSTEM_PROMPT.md`` when that
        file exists, otherwise :data:`DEFAULT_SYSTEM_PROMPT`. A file that
        exists but cannot be read falls back to the default prompt, because
        a broken override must not block startup.
    """
    if workspace is None:
        return DEFAULT_SYSTEM_PROMPT
    root = getattr(workspace, "path", None)
    if root is None:
        return DEFAULT_SYSTEM_PROMPT
    override = Path(root) / ".mitocode" / SYSTEM_PROMPT_FILE
    try:
        if override.is_file():
            return override.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT
