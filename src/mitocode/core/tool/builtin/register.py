"""One call that publishes the eleven built-in tools onto a ``ToolManager``.

The tool modules in this package each export plain coroutine functions --
``read``, ``bash``, ``spawn_agent`` and so on.  None of them knows what name
it will be published under or what the model will be told about it: that is
registration's job, and this module is the single place it happens.  The
whole model-facing surface is fixed here, in one table, in the order the
tools are published to the registry:

* the registered name (``Read``, ``Glob``, ``Agent`` ...), which is what the
  model calls, what the mini-DSL behind ``Background`` dispatches on, and
  what the TUI labels;
* the tool function from its module;
* a human-facing and an agent-facing description.  The agent doc is written
  for a model that has never seen this workspace: it states the argument
  semantics, the refusal behaviours and the error shapes, because the model
  has no other source for any of that;
* the display action and display parameter the TUI styles a tool call with.

Three behaviours are the point of the function, and each is deliberate.

**Attempt all eleven; return all eleven.**  The returned list holds the
names *attempted*, not the names that landed.  A name skipped because it was
already taken is still reported, so the caller always sees the full intent,
and a skip is visible in the log -- never as a silently shrunken list that
the caller would have to diff against a hard-coded eleven to notice.

**Refuse to clobber.**  ``ToolManager.register`` overwrites by name
silently, and the order between user tools and built-ins is not controlled:
a user tool named ``Read`` registered first must survive, and so must ours
when the order is reversed.  A name that already has a registration is
skipped with a warning and the other ten still land.  Nothing here ever
unregisters or replaces anything.

**Schemas stay annotation-derived.**  ``parameters_schema`` is never passed:
``ToolManager.register`` derives each tool's JSON Schema from the function
signature, and a hand-written copy would drift from that signature the first
time either changed.  The defaults a reader might pin -- ``Read``'s line
window, ``Grep``'s ``output_mode`` enum, ``Bash``'s timeout -- are the
signatures' own defaults, read straight off the functions.  One wrinkle the
registry cannot see past: the agent and background tool modules opt into
``from __future__ import annotations``, so their five tools carry PEP 563
*string* annotations, and the schema builder refuses those.  Registration
therefore resolves the strings to the types they already name -- see
:func:`_resolve_string_annotations` -- so the registry's own derivation path
stays the only path a schema travels.

When *context* is given it is installed with
:func:`~mitocode.core.tool.builtin.context.set_current`, so the module-level
current context is live for tools that read it lazily, and its
``tool_manager`` is filled in when still unset, so ``Background`` can resolve
the mini-DSL against this registry.  With no *context* nothing is installed
here: :func:`~mitocode.core.tool.builtin.context.current` lazily creates and
caches a default at the first tool call, which is the documented fallback
for REPL and test use.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Tuple, get_type_hints

from loguru import logger as loguru_logger

from mitocode.core.tool.builtin.agent_tools import (
    list_agents,
    query_agent,
    spawn_agent,
)
from mitocode.core.tool.builtin.background_tools import (
    background,
    background_result,
)
from mitocode.core.tool.builtin.context import ToolContext, set_current
from mitocode.core.tool.builtin.fs_tools import (
    edit,
    glob_files,
    grep_files,
    read,
    write,
)
from mitocode.core.tool.builtin.shell_tools import bash
from mitocode.core.tool.tool_manager import ToolManager

logger = loguru_logger.bind(name=__name__)

#: The eleven built-in tools in registration order, each as ``(registered
#: name, function, human doc, agent doc, display action, display param)``.
#: The order is the order the registry publishes them in.  The agent docs are
#: written for a model that has never seen the workspace: each states what
#: its tool does, what its arguments mean, and what it returns, assuming
#: nothing about the caller's surroundings.  ``display_param`` is empty for
#: the tools whose invocation has no single argument worth highlighting.
_BUILTIN_TOOLS: Tuple[Tuple[str, Callable[..., Any], str, str, str, str], ...] = (
    (
        "Read",
        read,
        "Read a text file from the workspace and return its lines numbered " "from 1.",
        "Read a text file and return its lines numbered from 1, each "
        'rendered as "{number}\\t{line}" with absolute 1-indexed numbers, so '
        "paging through a large file shows the same line with the same "
        "number every time. offset is the 1-indexed line to start from (0 "
        "and 1 both start at the first line) and limit caps how many lines "
        "come back. A missing file, an empty file, an offset past the end, "
        "a target that is not a regular file, and a file above the 10 MiB "
        'read cap each return a distinct "Error: ..." message instead of '
        "raising. file_path is relative to the workspace root or absolute; "
        "a path whose target leaves the workspace is decided by the "
        "approval policy.",
        "file.read",
        "file_path",
    ),
    (
        "Write",
        write,
        "Create a new file in the workspace; never overwrites an existing one.",
        "Create a file with the exact content given and report how many "
        "characters were written. Parent directories are created as needed, "
        "and empty content creates an empty file. Write never overwrites: "
        "when the file already exists the call is refused with advice to "
        "read the file and use Edit, because replacing a whole file to make "
        "one change destroys the rest without a trace. file_path is "
        "relative to the workspace root or absolute; a path whose target "
        "leaves the workspace is decided by the approval policy.",
        "file.write",
        "file_path",
    ),
    (
        "Edit",
        edit,
        "Replace exact text in an existing file, uniquely unless replace_all is set.",
        "Replace exact occurrences of old_string with new_string in an "
        "existing text file. The match is literal text rather than a "
        "pattern, so copy it exactly including indentation: one occurrence "
        "is replaced; a match found in more than one place is refused with "
        "the count unless replace_all is true; a match found nowhere "
        "returns the most similar existing lines to retry from. An empty "
        "old_string, and old_string identical to new_string, are both "
        "refused. file_path is relative to the workspace root or absolute; "
        "a path whose target leaves the workspace is decided by the "
        "approval policy.",
        "file.edit",
        "file_path",
    ),
    (
        "Glob",
        glob_files,
        "List workspace files matching an fnmatch pattern, sorted, one per line.",
        "List files matching an fnmatch pattern such as '*.py' or "
        "'src/*.py', sorted, one path per line, relative to the workspace "
        "root. A pattern without '/' matches basenames at any depth; a "
        "pattern containing '/' matches the path relative to the searched "
        "directory. The walk skips version-control and cache directories "
        "and dot-directories -- a pattern that itself starts with a dot "
        "unlocks hidden directories -- honours a minimal .gitignore at the "
        "search root, and lists files only, never directories. path is the "
        "directory to search and defaults to the whole workspace; a path "
        "whose target leaves the workspace is decided by the approval "
        "policy. When nothing matches, the reply says so.",
        "file.glob",
        "pattern",
    ),
    (
        "Grep",
        grep_files,
        "Search workspace files for a regular expression, one output line per result.",
        "Search the files under path for a regular expression and return "
        "one output line per result. output_mode selects the shape: "
        "'files_with_matches' (the default) lists each matching path once, "
        "'content' emits 'path:lineno:line' for every matching line, and "
        "'count' emits 'path:n' per matching file. glob restricts which "
        "files are read by basename pattern such as '*.py'; head_limit caps "
        "the number of output lines; case_insensitive ignores case. An "
        "invalid regex falls back to a literal substring match. "
        "Dot-directories are searched only when the pattern itself starts "
        "with a dot; binary files are skipped, noted with a '[binary]' "
        "marker in content mode. When nothing matches the result is an "
        'explicit "Error: no matches" message, never empty output.',
        "file.grep",
        "pattern",
    ),
    (
        "Bash",
        bash,
        "Run a shell command in the workspace root and return its merged output.",
        "Run a shell command under /bin/sh with its working directory at "
        "the workspace root and return its merged stdout and stderr in "
        "order. timeout is the ceiling in milliseconds (default 120000); "
        "when it fires the whole process group is killed, so children the "
        "command spawned die with it. Before anything runs, the command is "
        "scanned for tokens reaching outside the workspace, and a flagged "
        "command is submitted to the approval policy: allowed, it runs with "
        'no second check; denied, it is refused with an "Error: ..." '
        "message. A non-zero exit code is appended as '[exit code: N]' "
        "rather than reported as an error, so the command's own failure is "
        "data for you, not a tool failure.",
        "bash",
        "command",
    ),
    (
        "Agent",
        spawn_agent,
        "Spawn a sub-agent with a task and return its id without waiting.",
        "Create a sub-agent, send it the prompt as its first message, and "
        "return its new id immediately. This call does NOT wait for the "
        "answer: the sub-agent runs in the background on its own, so check "
        "on it later with QueryAgent rather than re-sending the task. The "
        "first line of description becomes the agent's label in the agent "
        "tree, and the whole description is kept for QueryAgent to report. "
        "parent_id defaults to the agent whose turn is making this call. "
        "Nesting is capped at a fixed depth, and a spawn past the cap is "
        "refused with the depth named.",
        "agent.create",
        "description",
    ),
    (
        "ListAgent",
        list_agents,
        "List every agent with its id, name, state, parent and child count.",
        "List every agent that currently exists: one line per agent with "
        "its id, name, state, parent and number of children. Takes no "
        "arguments. Use it to find the ids QueryAgent needs, to see the "
        "agent tree before spawning more agents, or to check whether any "
        "work is still running.",
        "agent.list",
        "",
    ),
    (
        "QueryAgent",
        query_agent,
        "Report one agent's state and last assistant message.",
        "Report one agent's id, name, description, state and last assistant "
        "message -- the usual check on whether a delegated task finished "
        "and what it produced. This call does not wait: a still-running "
        "agent is reported as running, so ask again later. agent_id is the "
        "id returned by Agent or shown by ListAgent. An unknown id returns "
        "an explicit error naming that the agent is gone, never an "
        "exception.",
        "agent.query",
        "agent_id",
    ),
    (
        "Background",
        background,
        "Detach a tool call written in the mini-DSL and return a background handle.",
        'Detach one tool call as a background task and return a "bg_<n>" '
        "handle immediately, without waiting for the call to run. The call "
        'is written in the mini-DSL as ToolName("argument") with literal '
        'arguments, e.g. Bash("python -m pytest test/") or '
        'Read("notes.md", offset=100): the tool name must match a '
        "registered tool exactly. timeout is the ceiling in milliseconds "
        "the detached call is held to -- never a wait this call performs; "
        "state a concrete value such as 300000, since zero or less means "
        "the default ceiling and anything above one hour is clamped down "
        "to it. Collect the output later with BackgroundResult using the "
        "handle.",
        "background",
        "call",
    ),
    (
        "BackgroundResult",
        background_result,
        "Redeem a background handle and report the task's outcome.",
        "Redeem one background handle returned by Background and report "
        "what is known about that task. This call does not wait: a task "
        "still in flight is reported as running, so ask again later. A "
        "finished task returns its output and a failed one returns its "
        "error; timed-out, cancelled, already-collected and unknown "
        "handles each return a distinct message -- 'already collected' "
        "means the output was handed over once and then dropped, while "
        "'unknown' means no task with that handle was ever issued. handle "
        "may be the bare bg token or the whole reply sentence that carried "
        "it.",
        "background.result",
        "handle",
    ),
)


def _resolve_string_annotations(func: Callable[..., Any]) -> None:
    """Evaluate any PEP 563 string annotations on *func*'s underlying tool.

    ``agent_tools`` and ``background_tools`` opt into
    ``from __future__ import annotations``, so the five tools they export
    carry their parameter annotations as *strings* -- ``"Optional[int]"``
    rather than ``Optional[int]``.  The schema builder in ``ToolManager``
    reads raw ``inspect.signature`` annotations and refuses strings, which
    would make registration of those five tools impossible without either
    hand-writing schemas (banned here: they drift) or resolving the strings.

    Resolution is the faithful spelling of what is already there:
    :func:`typing.get_type_hints` evaluates each string in the tool module's
    own namespace -- the same namespace the interpreter would use -- and the
    resulting annotations dictionary is rebound onto the tool.  No parameter
    name, kind, default or order changes, and the tools never read their
    annotations at runtime, so behaviour is untouched; the strings and the
    types are two spellings of one signature.  The wrapper's own copy is
    rebound too, because ``functools.wraps`` shares the dictionary by
    reference and a stale string copy on the wrapper would mislead any
    introspection that does not follow ``__wrapped__``.

    Idempotent: a second call finds no strings and rebinds nothing.

    Args:
        func: The registered callable -- the ``tool_boundary`` wrapper, whose
            ``__wrapped__`` chain leads to the raw tool coroutine.
    """
    raw = inspect.unwrap(func)
    annotations = getattr(raw, "__annotations__", None)
    if not annotations or not any(
        isinstance(value, str) for value in annotations.values()
    ):
        return
    resolved = get_type_hints(raw)
    raw.__annotations__ = resolved
    if func is not raw:
        func.__annotations__ = resolved


def register_builtin_tools(
    tool_manager: ToolManager, context: Optional[ToolContext] = None
) -> list[str]:
    """Register all eleven built-in tools on *tool_manager* in one call.

    Every built-in in :data:`_BUILTIN_TOOLS` is attempted in order.  A name
    that already has a registration on *tool_manager* is skipped with a
    warning instead of being overwritten -- ``ToolManager.register``
    replaces by name silently, and the order between user tools and
    built-ins is not controlled, so a user tool named ``Read`` must survive
    a registration that happens after it.  The other ten are registered
    normally, and no name is ever unregistered or replaced here.

    Each tool's JSON Schema is derived from its function signature by the
    registry; ``parameters_schema`` is deliberately never passed, so the
    schema cannot drift from the signature it describes.  Because two of the
    tool modules opt into ``from __future__ import annotations``, the string
    annotations their five tools carry are first resolved to the types they
    name (see :func:`_resolve_string_annotations`) -- without that, the
    registry's schema builder would refuse them.

    Args:
        tool_manager: The registry to publish the tools onto.  Existing
            registrations under any built-in name are left untouched.
        context: The :class:`ToolContext` the tools should read their world
            from.  When given, it is installed with
            :func:`~mitocode.core.tool.builtin.context.set_current` and its
            ``tool_manager`` is pointed at *tool_manager* when still unset,
            so ``Background`` can resolve the mini-DSL against this
            registry.  When ``None``, nothing is installed here and
            :func:`~mitocode.core.tool.builtin.context.current` lazily
            creates a default at the first tool call.

    Returns:
        The eleven attempted names in registration order -- including any
        skipped because the name was already taken.  The list is therefore
        the caller's record of full intent rather than a report of what
        landed; a skip is visible only in the log.
    """
    if context is not None:
        if context.tool_manager is None:
            context.tool_manager = tool_manager
        set_current(context)

    attempted: list[str] = []
    for entry in _BUILTIN_TOOLS:
        name, func, human_doc, agent_doc, display_action, display_param = entry
        # Resolved for every tool, skipped or not: the annotation fixup is
        # idempotent and global to the function object, and leaving it
        # dependent on the registration outcome would make the schema
        # readability of a tool depend on which manager it met first.
        _resolve_string_annotations(func)
        if tool_manager.get_spec(name) is not None:
            logger.warning(
                "Skipping built-in tool '{}': a tool is already registered "
                "under that name and built-ins never overwrite; the "
                "existing registration is left in place.",
                name,
            )
        else:
            tool_manager.register(
                func,
                name=name,
                human_doc=human_doc,
                agent_doc=agent_doc,
                display_action=display_action,
                display_param=display_param,
            )
        attempted.append(name)
    return attempted
