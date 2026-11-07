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

import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from loguru import logger as loguru_logger
from rich.table import Table
from rich.text import Text

# noinspection PyProtectedMember
from mitocode.core.agent.agent_session import (
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from mitocode.cli.format import ERROR_PREFIX

SCHEMA_VERSION = 1
"""Version of the on-disk session record schema written by this module."""


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


def block_to_dict(block: Any) -> Optional[dict]:
    """Serialize one history block into its JSON record form.

    Tool results carry no ``success`` field at runtime, so it is inferred
    from the ``error:`` prefix rule shared with :mod:`mitocode.cli.format`
    and :mod:`mitocode.cli.pump`: a result whose stripped content starts
    with ``error:`` (case-insensitive) is saved as a failure.

    Args:
        block: One entry from ``AgentSession.messages``.

    Returns:
        The record dict, or ``None`` for a block type this schema does not
        know (the caller skips it rather than failing the save).
    """
    if isinstance(block, _PlainTextMessageBlock):
        return {
            "kind": "text",
            "role": block.role,
            "message": block.message,
            "reasoning_content": block.reasoning_content,
        }
    if isinstance(block, _ToolCallMessageBlock):
        return {
            "kind": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments_string": block.arguments_string,
            "parameter": block.parameter,
            "reasoning_content": block.reasoning_content,
            "finished": block.finished,
            "parse_error": block.parse_error,
        }
    if isinstance(block, _ToolResultMessageBlock):
        content = str(block.content)
        success = not content.strip().lower().startswith(ERROR_PREFIX)
        return {
            "kind": "tool_result",
            "tool_call_id": block.tool_call_id,
            "content": content,
            "success": success,
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
                "system_prompt": session.system_prompt,
                "messages": [
                    record
                    for record in (block_to_dict(block) for block in session.messages)
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

        Returns:
            A ``(summaries, skipped)`` pair. *summaries* is sorted by
            ``updated_at`` descending; *skipped* is the number of ignored
            ``.json`` files.
        """
        if not self._sessions_dir.is_dir():
            return [], 0
        summaries: List[SessionSummary] = []
        skipped = 0
        for path in sorted(self._sessions_dir.iterdir()):
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
        agent_ids.add(agent_id)
    main_agent_id = record.get("main_agent_id")
    if not isinstance(main_agent_id, int) or isinstance(main_agent_id, bool):
        return "main_agent_id must be an integer"
    if main_agent_id not in agent_ids:
        return "main_agent_id is not among the agents"
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
    if len(summaries) == 1:
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
