"""Session persistence for the CLI: ids, records, disk store and selection.

This module owns the whole on-disk session protocol for the interactive CLI:
generating session ids, serializing :class:`AgentSession` message blocks into
JSON records, writing them atomically under
``<workspace>/.mitocode/sessions`` and listing/loading/validating them again.
The serialization protocol deliberately lives here instead of inside the core
session classes, so ``mitocode.core`` stays free of CLI, JSON and terminal
concerns (import direction: this module may import private block classes from
``core.agent.agent_session``; the core never imports this module).
"""

import asyncio
import copy
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from loguru import logger as loguru_logger
from rich.table import Table
from rich.text import Text

# noinspection PyProtectedMember
from mitocode.core.agent.agent_session import (
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)

SCHEMA_VERSION = 1
"""Version of the on-disk session record schema written by this module."""

AUTOSAVE_INTERVAL = 30.0
"""Seconds between automatic background saves of the current session."""

TITLE_TIMEOUT = 30.0
"""Ceiling in seconds for one background title-generation request."""

TITLE_MAX_CHARS = 60
"""Maximum characters of a generated session title."""

FALLBACK_TITLE_MAX_CHARS = 40
"""Maximum characters of a fallback session title (ellipsis included)."""

_TITLE_SYSTEM_PROMPT = (
    "You name chat sessions. Reply with one short title of at most 40 "
    "characters for the conversation start below. Output the title only: "
    "no quotes, no preamble, single line."
)

_TITLE_QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"
"""Quote characters stripped when they wrap the whole generated title."""


class SessionLoadError(Exception):
    """Raised only inside this module to report an unreadable session file.

    Covers a missing file, a malformed id, broken JSON and a record that
    fails schema validation; callers turn it into warnings rather than
    letting it abort startup or the REPL.
    """


@dataclass(frozen=True)
class SessionSummary:
    """Display facts about one saved session, newest-first sorted upstream.

    Attributes:
        session_id: Disk id the record is stored under.
        title: Human-readable session title.
        updated_at: ISO-8601 UTC last-write timestamp.
        created_at: ISO-8601 UTC creation timestamp.
        message_count: Total number of message blocks across all agents.
        model: Provider model name recorded for display only.
    """

    session_id: str
    title: str
    updated_at: str
    created_at: str
    message_count: int
    model: str


_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
"""Only these characters may appear in a session id used as a file stem."""


def new_session_id() -> str:
    """Return a new sortable session id: UTC stamp plus random hex.

    Returns:
        ``YYYYMMDDTHHMMSSZ-<8 hex chars>``, e.g. ``20261107T152000Z-1a2b3c4d``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def is_valid_session_id(value: object) -> bool:
    """Return whether *value* is safe to use as a session file stem.

    Only ``[A-Za-z0-9._-]`` is allowed and the dot-dot path aliases are
    rejected explicitly, which keeps a stored id from escaping the sessions
    directory when it is later used to build a file name.

    Args:
        value: Candidate id.

    Returns:
        ``True`` for a non-empty string of allowed characters that is neither
        ``"."`` nor ``".."``.
    """
    if not isinstance(value, str) or value in (".", ".."):
        return False
    return bool(_SESSION_ID_PATTERN.fullmatch(value))


def redact_secrets_in(text: str, secrets: Sequence[str]) -> str:
    """Replace every literal secret in *text* with a placeholder.

    Args:
        text: Text that may contain a secret.
        secrets: Literal secret values to replace.

    Returns:
        The text with every non-blank secret replaced by ``***``.
    """
    for secret in secrets:
        if secret and secret.strip():
            text = text.replace(secret, "***")
    return text


def _redact_value(value: Any, secrets: Sequence[str]) -> Any:
    """Return *value* with every literal secret replaced.

    Args:
        value: A string, or a structure that may contain strings.
        secrets: Literal secret values to replace.

    Returns:
        The value with the secrets replaced; the type is preserved for the
        containers it can walk.
    """
    if isinstance(value, str):
        return redact_secrets_in(value, secrets)
    if isinstance(value, Mapping):
        return {key: _redact_value(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        rebuilt = [_redact_value(item, secrets) for item in value]
        return type(value)(rebuilt) if isinstance(value, tuple) else rebuilt
    return value


def block_to_dict(block: Any, secrets: Sequence[str] = ()) -> Optional[dict]:
    """Serialize one history block into its JSON record form.

    A tool result carries the ``success`` flag the runtime decided on.  It
    is written out as it is: the outcome cannot be recovered from the
    content, because a failure does not have to start with ``error:`` and a
    successful tool may well print that word.

    Every string in the record is scrubbed of the literal secrets the
    caller knows about.  A tool that read the credential file would
    otherwise put the API key into the record, which is read back and
    echoed at startup.

    Args:
        block: One entry from ``AgentSession.messages``.
        secrets: Literal secret values to replace before writing.

    Returns:
        The record dict, or ``None`` for a block type this schema does not
        know (the caller skips it rather than failing the save).
    """
    if isinstance(block, _PlainTextMessageBlock):
        return {
            "kind": "text",
            "role": block.role,
            "message": redact_secrets_in(block.message, secrets),
            "reasoning_content": redact_secrets_in(block.reasoning_content, secrets),
        }
    if isinstance(block, _ToolCallMessageBlock):
        return {
            "kind": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments_string": redact_secrets_in(block.arguments_string, secrets),
            "parameter": _redact_value(block.parameter, secrets),
            "reasoning_content": redact_secrets_in(block.reasoning_content, secrets),
            "finished": block.finished,
            "parse_error": block.parse_error,
        }
    if isinstance(block, _ToolResultMessageBlock):
        return {
            "kind": "tool_result",
            "tool_call_id": block.tool_call_id,
            "content": redact_secrets_in(str(block.content), secrets),
            "success": bool(getattr(block, "success", True)),
        }
    return None


def _block_from_dict(data: Mapping[str, Any]) -> Optional[Any]:
    """Rebuild one message block from its record form.

    Args:
        data: One entry of a persisted ``messages`` list.

    Returns:
        A fresh block instance, or ``None`` when the ``kind`` is unknown; a
        bad single block must not lose the whole session.
    """
    if not isinstance(data, Mapping):
        return None
    kind = data.get("kind")
    if kind == "text":
        block = _PlainTextMessageBlock(
            str(data.get("role", "user") or "user"),
            str(data.get("message", "") or ""),
        )
        block.reasoning_content = str(data.get("reasoning_content", "") or "")
        return block
    if kind == "tool_call":
        call = _ToolCallMessageBlock(
            str(data.get("name", "") or ""),
            str(data.get("id", "") or ""),
        )
        call.arguments_string = str(data.get("arguments_string", "") or "")
        parameter = data.get("parameter")
        call.parameter = dict(parameter) if isinstance(parameter, Mapping) else {}
        call.reasoning_content = str(data.get("reasoning_content", "") or "")
        parse_error = data.get("parse_error")
        call.parse_error = None if parse_error is None else str(parse_error)
        call.finished = bool(data.get("finished", False))
        return call
    if kind == "tool_result":
        return _ToolResultMessageBlock(
            str(data.get("tool_call_id", "") or ""),
            str(data.get("content", "") or ""),
            success=bool(data.get("success", True)),
        )
    return None


def serialize_tree(
    orchestrator: Any,
    *,
    main_agent_id: int,
    session_id: str,
    title: str,
    created_at: str,
    model: str,
    workspace: str,
    updated_at: str,
    secrets: Sequence[str] = (),
) -> dict:
    """Serialize the whole session tree into one record dict.

    Walks the orchestrator's tree snapshot, serializes every live session's
    message blocks, and assembles the schema document.

    Args:
        orchestrator: Registry providing ``get_tree_snapshot`` and
            ``get_session``.
        main_agent_id: Id of the interactive main session.
        session_id: Disk id this record is stored under.
        title: Current session title.
        created_at: ISO-8601 UTC creation timestamp.
        model: Provider model name for display only.
        workspace: Absolute workspace path, display/diagnostic only.
        updated_at: ISO-8601 UTC last-write timestamp.
        secrets: Literal secret values scrubbed from every string written.

    Returns:
        The complete schema record, ready for ``SessionStore.save``.
    """
    agents: List[dict] = []
    for node in orchestrator.get_tree_snapshot():
        agent_id = node.get("id")
        session = orchestrator.get_session(agent_id)
        if session is None:
            continue
        agents.append(
            {
                "agent_id": agent_id,
                "name": str(node.get("name", "") or ""),
                "description": str(node.get("description", "") or ""),
                "parent_id": node.get("parent_id"),
                "sub_agent_ids": list(node.get("sub_agent") or []),
                "system_prompt": redact_secrets_in(session.system_prompt, secrets),
                "messages": [
                    record
                    for record in (
                        block_to_dict(block, secrets) for block in session.messages
                    )
                    if record is not None
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "workspace": workspace,
        "model": model,
        "main_agent_id": main_agent_id,
        "agents": agents,
    }


def restore_agent_messages(session: Any, agent_record: Mapping[str, Any]) -> None:
    """Fill one restored session with its persisted prompt and blocks.

    Args:
        session: A session built by ``AgentOrchestrator.restore_tree`` (its
            constructor ran with ``system_prompt=""`` so no system block was
            seeded).
        agent_record: The matching ``agents`` entry of the record.

    Raises:
        ValueError: When *agent_record* is not a mapping with a list
            ``messages`` field.
    """
    if not isinstance(agent_record, Mapping):
        raise ValueError("agent record must be a mapping")
    messages = agent_record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("agent record must carry a list messages field")
    session.system_prompt = str(agent_record.get("system_prompt", "") or "")
    session.messages = [
        block
        for block in (_block_from_dict(item) for item in messages)
        if block is not None
    ]


def display_messages_from_record(record: Mapping[str, Any]) -> List[Any]:
    """Rebuild the main agent's message blocks for a read-only replay.

    ``/demo`` shows a saved session without opening it: there is no live
    :class:`AgentSession` to read, so the record's own ``messages`` are
    converted back into block instances.  The blocks are fresh objects; the
    caller may render them freely without any risk of touching the live
    session tree.

    Args:
        record: A loaded session document.

    Returns:
        The main agent's message blocks, unknown kinds skipped, or an empty
        list when the record carries no usable ``messages``.
    """
    main_agent_id = record.get("main_agent_id")
    if not isinstance(main_agent_id, int) or isinstance(main_agent_id, bool):
        return []
    agent_record = _agent_record(record, main_agent_id)
    if agent_record is None:
        return []
    messages = agent_record.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        block
        for block in (_block_from_dict(item) for item in messages)
        if block is not None
    ]


class SessionStore:
    """Read and write session records under ``<workspace>/.mitocode/sessions``.

    Writes are atomic: the JSON body goes to a ``.tmp-`` file in the same
    directory (mode ``0600``, ``fsync``), then ``os.replace`` swaps it into
    place, so a crash can never leave a half-written session and a concurrent
    autosave can never corrupt the current file.
    """

    def __init__(self, workspace: Any) -> None:
        """Store the sessions directory inside *workspace*'s config dir.

        Args:
            workspace: Workspace root; the directory is
                ``<root>/.mitocode/sessions`` and is created lazily on the
                first write.
        """
        self._sessions_dir = Path(workspace) / ".mitocode" / "sessions"

    @property
    def sessions_dir(self) -> Path:
        """Return the directory session records are written to."""
        return self._sessions_dir

    def save(self, record: Mapping[str, Any]) -> Path:
        """Atomically write *record* as ``<session_id>.json``.

        The write is atomic: the document goes to a private sibling temporary
        file and is moved into place, so an interrupted save cannot leave a
        truncated record behind.  Two CLI instances in one workspace are kept
        apart by the workspace lock, which is taken once for the whole run
        rather than per write.

        Args:
            record: A validated schema document.

        Returns:
            The path the record was written to.

        Raises:
            SessionLoadError: The ``session_id`` is missing or unsafe as a
                file name.
            OSError: The directory or file could not be written.
        """
        session_id = record.get("session_id") if isinstance(record, Mapping) else None
        if not is_valid_session_id(session_id):
            raise SessionLoadError("session_id is missing or unsafe as a file name")

        self._sessions_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self._sessions_dir.chmod(0o700)
        except OSError:
            pass

        target = self._sessions_dir / f"{session_id}.json"
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=self._sessions_dir)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(dict(record), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except BaseException:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        return target

    def load(self, session_id: str) -> dict:
        """Return the full validated record stored under *session_id*.

        Args:
            session_id: Disk id previously returned by :meth:`save` or
                :meth:`list_sessions`.

        Returns:
            The complete schema document.

        Raises:
            SessionLoadError: The id is unsafe, the file is missing, the
                JSON is broken or the record fails validation.
        """
        if not is_valid_session_id(session_id):
            raise SessionLoadError(f"invalid session id: {session_id!r}")
        path = self._sessions_dir / f"{session_id}.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SessionLoadError(f"cannot read session {session_id}: {exc}") from exc
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SessionLoadError(f"session {session_id} is not valid JSON") from exc
        reason = validate_record(record)
        if reason is not None:
            raise SessionLoadError(f"session {session_id} is invalid: {reason}")
        return record

    def list_sessions(self) -> Tuple[List[SessionSummary], int]:
        """List valid session summaries newest-first and count bad files.

        ``.tmp-*`` files, hidden files and anything without a ``.json``
        suffix are ignored without counting; every ``.json`` file that
        cannot be read, parsed or validated is skipped and counted, with
        only the file name and reason logged (never its content).

        A directory that cannot be enumerated at all is a different thing
        from an empty one: it means the sessions are there but unreadable,
        so the caller is told instead of being handed a silent "no saved
        sessions".

        Returns:
            A ``(summaries, skipped)`` pair. *summaries* is sorted by
            ``updated_at`` descending; *skipped* is the number of ignored
            ``.json`` files.

        Raises:
            SessionLoadError: When the sessions directory cannot be listed.
        """
        if not self._sessions_dir.is_dir():
            return [], 0
        summaries: List[SessionSummary] = []
        skipped = 0
        try:
            entries = sorted(self._sessions_dir.iterdir())
        except OSError as exc:
            raise SessionLoadError(
                f"the session directory '{self._sessions_dir}' could not be "
                f"read: {exc.strerror or 'I/O error'}"
            ) from exc
        for path in entries:
            name = path.name
            if name.startswith(".") or not name.endswith(".json"):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                reason = validate_record(record)
                if reason is not None:
                    raise SessionLoadError(reason)
                agents = record.get("agents") or []
                summaries.append(
                    SessionSummary(
                        session_id=str(record["session_id"]),
                        title=str(record.get("title", "") or ""),
                        updated_at=str(record.get("updated_at", "") or ""),
                        created_at=str(record.get("created_at", "") or ""),
                        message_count=sum(
                            len(agent.get("messages") or []) for agent in agents
                        ),
                        model=str(record.get("model", "") or ""),
                    )
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                skipped += 1
                loguru_logger.debug("Skipped session file {}: {}", name, exc)
        summaries.sort(key=lambda item: item.updated_at, reverse=True)
        return summaries, skipped


def validate_record(record: object) -> Optional[str]:
    """Return why *record* is not a valid session document, or ``None``.

    The checks mirror the schema contract: top-level object, current
    schema version, a safe session id, string metadata fields, a non-empty
    ``agents`` list whose entries carry an integer ``agent_id`` and a list
    ``messages`` field, a ``main_agent_id`` present among the agents, and
    closed ``parent_id`` references.

    The duplicate-id and parent-cycle checks mirror
    :meth:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator.restore_tree`
    exactly, so a record that lists and loads is a record that restores: the
    two layers cannot disagree about what a usable tree is.

    Args:
        record: Parsed JSON value of a candidate session file.

    Returns:
        ``None`` when the record is valid, otherwise a short reason string
        safe to log (it never contains record content).
    """
    if not isinstance(record, Mapping):
        return "record is not a JSON object"
    if record.get("schema_version") != SCHEMA_VERSION:
        return f"unsupported schema_version: {record.get('schema_version')!r}"
    if not is_valid_session_id(record.get("session_id")):
        return "session_id is missing or unsafe"
    for field in ("title", "created_at", "updated_at", "model", "workspace"):
        if not isinstance(record.get(field), str):
            return f"{field} must be a string"
    agents = record.get("agents")
    if not isinstance(agents, list) or not agents:
        return "agents must be a non-empty list"
    agent_ids = set()
    for agent in agents:
        if not isinstance(agent, Mapping):
            return "every agent must be an object"
        agent_id = agent.get("agent_id")
        if not isinstance(agent_id, int) or isinstance(agent_id, bool):
            return "every agent needs an integer agent_id"
        if not isinstance(agent.get("messages"), list):
            return f"agent {agent_id} needs a list messages field"
        if agent_id in agent_ids:
            return f"duplicate agent_id {agent_id}"
        agent_ids.add(agent_id)
    main_agent_id = record.get("main_agent_id")
    if not isinstance(main_agent_id, int) or isinstance(main_agent_id, bool):
        return "main_agent_id must be an integer"
    if main_agent_id not in agent_ids:
        return "main_agent_id is not among the agents"
    parents: dict = {}
    for agent in agents:
        parent_id = agent.get("parent_id")
        if parent_id is None:
            continue
        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            return f"agent {agent.get('agent_id')} has a non-integer parent_id"
        if parent_id not in agent_ids:
            return (
                f"agent {agent.get('agent_id')} references missing parent {parent_id}"
            )
        parents[agent.get("agent_id")] = parent_id
    # A parent chain that loops back on itself can never be restored, and the
    # loop is invisible to the checks above: every parent id exists, so only
    # walking the chain finds it.
    for agent_id in agent_ids:
        visited = set()
        current: Optional[int] = agent_id
        while current is not None:
            if current in visited:
                return f"cycle detected in parent chain at agent {current}"
            visited.add(current)
            current = parents.get(current)
    return None


_PANEL_TITLE_MAX_CHARS = 40
"""Maximum characters of a session title shown in the selection panel."""


def _truncate_title(title: str, limit: int = _PANEL_TITLE_MAX_CHARS) -> str:
    """Shorten *title* to *limit* characters with an ellipsis.

    Args:
        title: Session title to display.
        limit: Maximum length of the returned string.

    Returns:
        The title unchanged when it fits, otherwise its head followed by
        ``…``.
    """
    if len(title) <= limit:
        return title
    return title[: limit - 1] + "…"


def _format_panel_time(updated_at: str) -> str:
    """Render a record timestamp in the user's local timezone.

    Args:
        updated_at: ISO-8601 UTC timestamp from the record.

    Returns:
        ``YYYY-MM-DD HH:MM`` in local time, or the raw string when it does
        not parse (display must never crash on a hand-edited file).
    """
    try:
        moment = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return updated_at
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


async def select_session(
    ui: Any, summaries: List[SessionSummary], *, allow_new: bool = True
) -> str:
    """Print the saved-session table and return the user's choice.

    With nothing to choose from the answer is ``"new"`` and with exactly
    one session that session is returned without any output; only two or
    more summaries get the interactive table. Every cell is wrapped in
    ``Text`` so session content can never be parsed as Rich markup.

    Args:
        ui: Terminal surface used to print the table (``ui.console.print``)
            and read a line (``await ui.ask``).
        summaries: Valid summaries, already sorted newest-first.
        allow_new: Whether the trailing ``n`` / new-session choice is
            offered.

    Returns:
        ``"new"`` when there is nothing to choose from or the user picks
        the new-session row; otherwise the chosen session id.

    Raises:
        EOFError: The input stream closed; startup aborts.
        KeyboardInterrupt: The user interrupted; startup aborts.
    """
    if not summaries:
        return "new"
    if len(summaries) == 1 and not allow_new:
        # Nothing to choose between and no new-session row to offer, so the
        # single session is the answer.  With ``allow_new`` the panel is
        # shown anyway: the user asked to be able to start a new session.
        return summaries[0].session_id

    table = Table(box=None, show_header=True, header_style="bold")
    table.add_column("#")
    table.add_column("Title")
    table.add_column("Updated")
    table.add_column("Messages")
    table.add_column("Model")
    for index, summary in enumerate(summaries, start=1):
        table.add_row(
            Text(str(index)),
            Text(_truncate_title(summary.title)),
            Text(_format_panel_time(summary.updated_at)),
            Text(str(summary.message_count)),
            Text(summary.model),
        )
    if allow_new:
        table.add_row(Text("n"), Text("New session"), Text(""), Text(""), Text(""))
    ui.console.print(table)

    hint = f"[1-{len(summaries)}, n]" if allow_new else f"[1-{len(summaries)}]"
    while True:
        raw = await ui.ask(f"Select a session {hint}: ")
        choice = raw.strip().lower()
        if allow_new and choice in ("n", "new"):
            return "new"
        if choice.isdigit():
            number = int(choice)
            if 1 <= number <= len(summaries):
                return summaries[number - 1].session_id
        tail = ", or n for a new session" if allow_new else ""
        ui.warn(f"Please enter a number between 1 and {len(summaries)}{tail}.")


# ---------------------------------------------------------------------------
# SessionManager: the current on-disk session identity
# ---------------------------------------------------------------------------

_NEW_SESSION_TITLE = "New Session"
"""Title every fresh session starts with until a real one is generated."""

_SAVE_WARN_MESSAGE = (
    "Saving the session to disk failed; continuing without persistence."
)
"""User-facing warning shown at most once per session on save failures."""


def _format_clock(clock: Any) -> str:
    """Render one clock reading as an ISO-8601 UTC timestamp.

    Args:
        clock: Zero-argument callable returning a ``datetime``.

    Returns:
        ``YYYY-MM-DDTHH:MM:SSZ`` in UTC.
    """
    return clock().strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the change-detection fingerprint of a session record.

    The ``updated_at`` field is blanked first so a timestamp-only change
    never counts as new content; the rest is dumped with sorted keys so the
    comparison is stable across processes.

    Args:
        record: A schema document (or any mapping).

    Returns:
        A canonical JSON string for comparison.
    """
    data = dict(record)
    data["updated_at"] = ""
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _agent_record(
    record: Mapping[str, Any], agent_id: int
) -> Optional[Mapping[str, Any]]:
    """Find the ``agents`` entry of *record* carrying *agent_id*.

    Args:
        record: A schema document.
        agent_id: The agent id to look up.

    Returns:
        The matching mapping, or ``None`` when the id is absent.
    """
    for agent in record.get("agents") or []:
        if isinstance(agent, Mapping) and agent.get("agent_id") == agent_id:
            return agent
    return None


class SessionManager:
    """Own the on-disk identity of the current interactive session.

    One lock serializes saving *and* switching: an autosave that fires
    mid-switch would otherwise read the new tree and write it into the old
    file.  Every save failure is contained — logged and warned at most once
    per session — so the REPL never dies over disk trouble.
    """

    def __init__(
        self,
        store: Any,
        orchestrator: Any,
        ui: Any,
        *,
        state: Any,
        provider: Any,
        workspace: str,
        clock: Any = None,
        redact_secrets: Sequence[str] = (),
    ) -> None:
        """Wire the manager to its collaborators.

        Args:
            store: Disk store for session records.
            orchestrator: Session registry to serialize and restore.
            ui: Surface used for warnings and notes.
            state: CLI state providing ``main_id`` and the counters and
                tree snapshot a switch resets.
            provider: Provider used for background title generation; its
                ``default_model`` is recorded into saved records.
            workspace: Absolute workspace path written into records.
            clock: ``datetime``-like factory for timestamps; tests inject a
                frozen clock, production uses ``datetime.now``.
            redact_secrets: Literal secret values scrubbed from every string
                written to a record.  A tool that read the credential file
                would otherwise put the API key into the record, which is
                read back and echoed at startup.
        """
        self._store = store
        self._orchestrator = orchestrator
        self._ui = ui
        self._state = state
        self._provider = provider
        self._workspace = workspace
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._redact_secrets: tuple[str, ...] = tuple(
            secret for secret in redact_secrets if secret and secret.strip()
        )
        self._lock = asyncio.Lock()
        self._current_session_id: Optional[str] = None
        self._current_title: str = _NEW_SESSION_TITLE
        self._created_at: str = ""
        self._last_fingerprint: Optional[str] = None
        self._save_warned = False
        self._title_task: Optional[asyncio.Task] = None
        self._title_attempted: set = set()

    @property
    def current_session_id(self) -> Optional[str]:
        """Return the disk id of the current session, or ``None``."""
        return self._current_session_id

    @property
    def current_title(self) -> str:
        """Return the current session's display title."""
        return self._current_title

    def _model(self) -> str:
        """Return the provider's default model name for record display."""
        return str(getattr(self._provider, "default_model", "") or "")

    def _snapshot_agents(self) -> List[dict]:
        """Return the live tree as agent records, for a switch rollback.

        The snapshot is taken before the registry is reset, so a restore that
        fails can put the previous conversation back instead of leaving the
        CLI without a main session.

        Returns:
            One record per live agent, in tree-snapshot order.  Empty when
            there is no live tree to preserve.
        """
        agents: List[dict] = []
        for node in self._orchestrator.get_tree_snapshot():
            agent_id = node.get("id")
            session = self._orchestrator.get_session(agent_id)
            if session is None:
                continue
            agents.append(
                {
                    "agent_id": agent_id,
                    "name": str(node.get("name", "") or ""),
                    "description": str(node.get("description", "") or ""),
                    "parent_id": node.get("parent_id"),
                    "sub_agent_ids": list(node.get("sub_agent") or []),
                    "system_prompt": redact_secrets_in(
                        getattr(session, "system_prompt", ""), self._redact_secrets
                    ),
                    "messages": [
                        record
                        for record in (
                            block_to_dict(block, self._redact_secrets)
                            for block in session.messages
                        )
                        if record is not None
                    ],
                }
            )
        return agents

    async def _restore_agents(self, agents: List[dict]) -> bool:
        """Rebuild the tree from *agents*, restoring each session's messages.

        Args:
            agents: Agent records in the shape ``serialize_tree`` produces.

        Returns:
            ``True`` when the tree was rebuilt, ``False`` when it could not
            be restored — in which case the caller must not leave the state
            pointing at a session that does not exist.
        """
        if not agents:
            return False
        try:
            sessions = await self._orchestrator.restore_tree(agents)
            for agent_id, session in sessions.items():
                agent_record = _agent_record({"agents": agents}, agent_id)
                if agent_record is not None:
                    restore_agent_messages(session, agent_record)
        except Exception:  # pylint: disable=broad-exception-caught
            loguru_logger.exception("Session rollback failed")
            return False
        return True

    async def _ensure_main_session(self) -> None:
        """Guarantee that the state points at a session that exists.

        Called after a failed switch: the registry may be empty, in which
        case a fresh main session is created so the next turn still runs and
        saving still reaches the disk.
        """
        if self._orchestrator.get_session(self._state.main_id) is not None:
            return
        try:
            session = await self._orchestrator.create_session(
                name="Main", description="Interactive session"
            )
        except Exception:  # pylint: disable=broad-exception-caught
            loguru_logger.exception("Could not recreate the main session")
            return
        self._state.main_id = session.agent_id

    def _first_user_text(self) -> str:
        """Return the main session's first non-blank user message.

        Returns:
            The stripped message text, or ``""`` when the main session does
            not exist or no user text has been sent yet.
        """
        session = self._orchestrator.get_session(self._state.main_id)
        if session is None:
            return ""
        for block in getattr(session, "messages", None) or []:
            if getattr(block, "role", "") != "user":
                continue
            text = str(getattr(block, "message", "") or "").strip()
            if text:
                return text
        return ""

    async def save_current(
        self, *, force: bool = False, report_failure: bool = False
    ) -> Optional[Path]:
        """Serialize and store the current session tree.

        Args:
            force: Write even when the record is unchanged.
            report_failure: Warn on every failure instead of once.

        Returns:
            The written path, or ``None`` when there is no session, the
            content was unchanged, or the write failed.
        """
        async with self._lock:
            return await self._save_locked(force=force, report_failure=report_failure)

    async def _save_locked(
        self, *, force: bool, report_failure: bool
    ) -> Optional[Path]:
        """Save the current tree; the caller must hold the manager lock.

        Args:
            force: Write even when the record is unchanged.
            report_failure: Warn on every failure instead of once.

        Returns:
            The written path, or ``None`` as for :meth:`save_current`.
        """
        if self._current_session_id is None:
            return None
        if self._orchestrator.get_session(self._state.main_id) is None:
            return None
        record = serialize_tree(
            self._orchestrator,
            main_agent_id=self._state.main_id,
            session_id=self._current_session_id,
            title=self._current_title,
            created_at=self._created_at,
            model=self._model(),
            workspace=self._workspace,
            updated_at="",
            secrets=self._redact_secrets,
        )
        fingerprint = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if not force and fingerprint == self._last_fingerprint:
            return None
        record["updated_at"] = _format_clock(self._clock)
        try:
            path = await asyncio.to_thread(self._store.save, record)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            loguru_logger.exception(
                "Session {} save failed: {}", self._current_session_id, exc
            )
            if report_failure or not self._save_warned:
                self._ui.warn(_SAVE_WARN_MESSAGE)
                self._save_warned = True
            return None
        self._save_warned = False
        self._last_fingerprint = fingerprint
        return path

    def _register_new_locked(self) -> None:
        """Adopt a fresh id, the default title and a blank fingerprint.

        The caller must hold the manager lock.
        """
        self._current_session_id = new_session_id()
        self._current_title = _NEW_SESSION_TITLE
        self._created_at = _format_clock(self._clock)
        self._last_fingerprint = None
        self._save_warned = False

    async def start_new_session(self, main_session: Any) -> None:
        """Register a fresh session id and save an empty record.

        Args:
            main_session: The newly created main session (unused for
                serialization, which reads the whole tree live).
        """
        async with self._lock:
            self._register_new_locked()
            await self._save_locked(force=False, report_failure=False)

    def attach_record(self, record: Mapping[str, Any]) -> None:
        """Adopt an existing record as the current on-disk session.

        Args:
            record: A schema document previously loaded from the store; the
                pre-seeded fingerprint makes the next unchanged save a no-op.
        """
        self._adopt_record_locked(record)

    def maybe_generate_title(self) -> None:
        """Start one background title task for the current session.

        Runs only while the title is still the default, a first user text
        exists, and this session has not been titled before; the task binds
        the session id at start and its result is discarded if the current
        session changed by the time the title is written.
        """
        if self._current_session_id is None:
            return
        if self._current_title != _NEW_SESSION_TITLE:
            return
        if self._current_session_id in self._title_attempted:
            return
        first_user_message = self._first_user_text()
        if not first_user_message:
            return
        session_id = self._current_session_id
        self._title_attempted.add(session_id)
        self._title_task = asyncio.create_task(
            self._title_job(session_id, first_user_message)
        )

    async def _title_job(self, session_id: str, first_user_message: str) -> None:
        """Generate one title off the interactive path and write it back.

        Args:
            session_id: Session the request was started for; results for any
                other current session are dropped.
            first_user_message: The user text the title is generated from.
        """
        try:
            try:
                title = await asyncio.wait_for(
                    generate_session_title(self._provider, first_user_message),
                    TITLE_TIMEOUT,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                title = fallback_title(first_user_message)
            await self._write_title_locked(session_id, title)
        except asyncio.CancelledError:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            loguru_logger.exception("Session title generation failed")

    async def _write_title_locked(self, session_id: str, title: str) -> None:
        """Apply a generated title and force-save, under the manager lock.

        Args:
            session_id: Session the title belongs to; a mismatch with the
                current session means the result is stale and is dropped.
            title: Cleaned or fallback title; blanks keep the old title.
        """
        async with self._lock:
            if self._current_session_id != session_id:
                return
            title = title.strip()
            if not title:
                return
            self._current_title = title
            await self._save_locked(force=True, report_failure=False)

    async def _cancel_title_task_locked(self) -> None:
        """Cancel any pending title task and wait for it to unwind.

        The caller must hold the manager lock.
        """
        task = self._title_task
        self._title_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def shutdown(self) -> None:
        """Cancel title generation and write a final forced save.

        Used at process exit: a title that had not finished is dropped (the
        next start regenerates it) and the last tree state reaches disk.
        Save failures stay contained, so shutdown never raises.
        """
        await self._cancel_title_task_locked()
        await self.save_current(force=True)

    async def _drain_pump(self) -> None:
        """Wait until the event pump has rendered everything queued so far.

        A missing pump (tests, or early startup) is skipped silently.
        """
        pump = getattr(self._state, "pump", None)
        if pump is not None:
            await pump.wait_idle()

    async def _stop_background_locked(self) -> None:
        """Stop every detached tool task so none outlives the switch."""
        tool_context = getattr(self._state, "tool_context", None)
        background = getattr(tool_context, "background", None)
        if background is not None:
            await background.stop_all()

    def _adopt_record_locked(self, record: Mapping[str, Any]) -> None:
        """Register a loaded record as the current session; caller holds lock."""
        self._current_session_id = str(record["session_id"])
        self._current_title = str(record.get("title", "") or _NEW_SESSION_TITLE)
        self._created_at = str(record.get("created_at", "") or "")
        self._last_fingerprint = _record_fingerprint(record)
        self._save_warned = False

    def _reset_counters_locked(self) -> None:
        """Zero the per-process counters and refresh the tree snapshot."""
        self._state.turn_count = 0
        self._state.tool_call_count = 0
        self._state.total_chars = 0
        self._state.agents = self._orchestrator.get_tree_snapshot()
        self._ui.agent_count = len(self._state.agents)

    async def switch_to(self, record: Mapping[str, Any]) -> bool:
        """Switch the live tree to a loaded record, all under the save lock.

        The old tree is saved first, then the registry is reset and the new
        tree restored.  A failure anywhere in the restore puts the previous
        tree back, so the session the user was in stays usable and saving
        keeps working.

        Args:
            record: A validated schema document (typically from
                ``SessionStore.load``).

        Returns:
            ``True`` on success; ``False`` when the restore failed, in which
                case the previous session has been restored.
        """
        async with self._lock:
            await self._cancel_title_task_locked()
            await self._stop_background_locked()
            await self._save_locked(force=False, report_failure=False)
            previous_agents = self._snapshot_agents()
            previous_main_id = self._state.main_id
            # Stop the event source before draining: draining a queue that a
            # still-running turn keeps filling can wait for the whole turn,
            # and messages appended while waiting are lost with the reset.
            await self._orchestrator.reset_sessions()
            await self._drain_pump()
            try:
                sessions = await self._orchestrator.restore_tree(
                    record.get("agents") or []
                )
                for agent_id, session in sessions.items():
                    agent_record = _agent_record(record, agent_id)
                    if agent_record is not None:
                        restore_agent_messages(session, agent_record)
                main_agent_id = record.get("main_agent_id")
                if main_agent_id not in sessions:
                    raise ValueError("main_agent_id is not among the restored agents")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                loguru_logger.exception(
                    "Session switch to {} failed: {}",
                    record.get("session_id"),
                    exc,
                )
                await self._orchestrator.reset_sessions()
                # Put the previous tree back: an empty registry would leave
                # the state pointing at a session that no longer exists, so
                # the next turn could not run and no save would reach disk.
                if not await self._restore_agents(previous_agents):
                    self._state.main_id = previous_main_id
                    await self._ensure_main_session()
                else:
                    self._state.main_id = previous_main_id
                await self._drain_pump()
                return False
            self._state.main_id = main_agent_id
            self._adopt_record_locked(record)
            self._reset_counters_locked()
            await self._drain_pump()
            self.maybe_generate_title()
            return True

    async def switch_new(self) -> None:
        """Destroy the current tree and start a brand-new session, locked.

        The old session is saved first; the new one gets a fresh disk id, the
        default title and zeroed counters, and its empty record is written
        immediately so a crash never leaves the new session unsaved.  A
        failure to create the new session is contained: the previous tree is
        restored and the exception never reaches the REPL.
        """
        async with self._lock:
            await self._cancel_title_task_locked()
            await self._stop_background_locked()
            await self._save_locked(force=False, report_failure=False)
            previous_agents = self._snapshot_agents()
            previous_main_id = self._state.main_id
            # Stop the event source before draining, for the same reason as in
            # switch_to: a live turn would otherwise keep the queue filled.
            await self._orchestrator.reset_sessions()
            await self._drain_pump()
            try:
                main_session = await self._orchestrator.create_session(
                    name="Main", description="Interactive session"
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                loguru_logger.exception("Starting a new session failed: %s", exc)
                await self._orchestrator.reset_sessions()
                if not await self._restore_agents(previous_agents):
                    self._state.main_id = previous_main_id
                    await self._ensure_main_session()
                else:
                    self._state.main_id = previous_main_id
                await self._drain_pump()
                return
            self._state.main_id = main_session.agent_id
            self._register_new_locked()
            self._reset_counters_locked()
            await self._drain_pump()
            await self._save_locked(force=True, report_failure=False)


class SessionAutosaver:
    """Periodically save the current session in the background.

    One asyncio task sleeps for the configured interval, calls
    :meth:`SessionManager.save_current` and repeats until stopped. Saving
    itself is a no-op while no session exists, so the loop does not need to
    know whether one is live; the manager's lock keeps it from interleaving
    with a session switch.
    """

    def __init__(
        self, manager: SessionManager, interval: float = AUTOSAVE_INTERVAL
    ) -> None:
        """Bind the autosaver to a manager.

        Args:
            manager: Manager whose :meth:`~SessionManager.save_current` is
                called once per tick.
            interval: Seconds between saves; tests inject a small value.
        """
        self._manager = manager
        self._interval = interval
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the background loop unless it is already running."""
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def run(self) -> None:
        """Save once per interval until cancelled.

        Raises:
            asyncio.CancelledError: When the task is cancelled by
                :meth:`stop`; the error propagates to the gatherer.
        """
        while True:
            await asyncio.sleep(self._interval)
            await self._manager.save_current()

    async def stop(self) -> None:
        """Cancel the loop and wait until it has fully unwound."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Session title generation
# ---------------------------------------------------------------------------


def _clean_title_text(raw: str) -> str:
    """Clean a model-produced title down to one short line.

    The order follows the design: strip surrounding whitespace, keep the
    first line only, drop a pair of wrapping quotes, collapse internal
    whitespace and cut at :data:`TITLE_MAX_CHARS`.

    Args:
        raw: The raw model output.

    Returns:
        The cleaned title, possibly empty.
    """
    text = raw.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if len(text) >= 2 and text[0] in _TITLE_QUOTE_CHARS and text[-1] == text[0]:
        text = text[1:-1].strip()
    return " ".join(text.split())[:TITLE_MAX_CHARS]


async def generate_session_title(provider: Any, first_user_message: str) -> str:
    """Ask the provider for a one-line session title.

    The provider is shallow-copied and the copy's ``reasoning_effort`` is
    cleared, so the title request neither inherits the interactive tier nor
    races the main turn's attribute.

    Args:
        provider: Chat provider exposing ``chat`` and ``default_model``.
        first_user_message: The session's first user text.

    Returns:
        The cleaned title (never blank).

    Raises:
        Exception: Whatever the provider raises, and ``ValueError`` when the
            response carries no usable content; callers fall back.
    """
    clone = copy.copy(provider)
    clone.reasoning_effort = None
    messages = [
        {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": first_user_message},
    ]
    response = await clone.chat(messages=messages, tools=[], stream=False)
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if not choices:
        raise ValueError("title response has no choices")
    first = choices[0] if isinstance(choices[0], Mapping) else None
    message = first.get("message") if first is not None else None
    content = message.get("content") if isinstance(message, Mapping) else None
    title = _clean_title_text(str(content or ""))
    if not title:
        raise ValueError("title response was empty")
    return title


def fallback_title(first_user_message: str) -> str:
    """Derive a title from the user message when generation fails.

    Args:
        first_user_message: The session's first user text.

    Returns:
        The first line with collapsed whitespace, cut at
        :data:`FALLBACK_TITLE_MAX_CHARS` with an ellipsis; an empty string
        when the message carries no usable text.
    """
    text = str(first_user_message or "").strip()
    if not text:
        return ""
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    text = " ".join(text.split())
    if not text:
        return ""
    return _truncate_title(text, FALLBACK_TITLE_MAX_CHARS)
