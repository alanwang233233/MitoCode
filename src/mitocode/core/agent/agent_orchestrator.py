"""Central coordinator that owns agent sessions, tasks, and event routing."""

import asyncio
import copy
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from loguru import logger as loguru_logger

from .agent_runtime import AgentRuntime
from .agent_session import AgentSession
from .agent_state import AgentState
from .template.apply import apply_template
from .template.template import AgentTemplate
from ..bridge.event_adapter import EventBridge
from ..chat_provider.sample_chat_provider import SampleChatProvider
from ..chat_provider.scenario_provider import ScenarioProvider
from ..tool.tool_manager import ToolManager

logger = loguru_logger.bind(name=__name__)

# Matches the shape of an OpenAI-style secret key. Only the value is replaced;
# surrounding words are preserved so the message stays diagnostically useful.
_API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]+")


class AgentOrchestrator:  # pylint: disable=too-many-instance-attributes
    """Central controller that directly manages multiple agent sessions.

    Responsibilities:

    * Session registry with auto-incrementing ids.
    * Tree-structure maintenance (parent / child relationships), tracked via
      ``parent_id`` / ``sub_agent_ids`` fields on each :class:`AgentSession`.
    * Running sessions as ``asyncio.Task`` instances (tracked in ``_tasks``).
    * Event routing from internal :class:`AgentRuntime` events through
      :class:`EventBridge` to a shared ``asyncio.Queue`` consumed by the TUI.
    * Cascading cancellation.
    """

    def __init__(
        self,
        tool_manager: ToolManager | None = None,
        default_provider: SampleChatProvider | ScenarioProvider | None = None,
        system_prompt: str = "",
        redact_secrets: Sequence[str] = (),
        workspace: Optional[Path] = None,
    ):
        """Create an orchestrator.

        Args:
            tool_manager: Optional registry every new session snapshots.
            default_provider: Provider used by sessions that do not pass one.
            system_prompt: Role instructions seeded into every session this
                orchestrator creates, including child agents. An explicit
                ``system_prompt`` passed to :meth:`create_session` wins.
            redact_secrets: Literal secret values (e.g. the configured API
                key) scrubbed from error messages and error logs alongside the
                ``sk-`` pattern. Blank entries are ignored.
            workspace: Workspace root substituted into a template's
                ``{workspace}`` placeholder. ``None`` leaves the placeholder
                visible: a caller with no workspace has no path to put there.
        """
        self._sessions: Dict[int, AgentSession] = {}
        self._tasks: Dict[int, asyncio.Task | None] = {}
        self._counter: int = 0
        self._tool_manager: ToolManager | None = tool_manager
        self._default_provider: SampleChatProvider | ScenarioProvider | None = (
            default_provider
        )
        self._system_prompt: str = system_prompt or ""
        self._workspace: Optional[Path] = workspace
        self._redact_secrets: tuple[str, ...] = tuple(
            secret for secret in redact_secrets if secret and secret.strip()
        )
        self._bridge = EventBridge(tool_manager)
        self.event_queue: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Error reporting
    # ------------------------------------------------------------------

    @staticmethod
    def redact_api_keys(message: str) -> str:
        """Replace OpenAI-style ``sk-`` secrets in ``message`` with ``sk-***``.

        The replacement is best-effort: it targets the literal ``sk-`` prefix
        followed by key-shaped characters, which covers the API keys this
        project reads from the environment without touching ordinary prose.

        Args:
            message: Raw text, typically an exception message.

        Returns:
            The text with every matched secret replaced by ``sk-***``.
        """
        return _API_KEY_PATTERN.sub("sk-***", message)

    def _redact_message(self, message: str) -> str:
        """Scrub this orchestrator's configured secrets out of *message*.

        Every literal secret passed as ``redact_secrets`` is replaced first, so
        a key the ``sk-`` pattern cannot shape-match (the CLI accepts any
        non-blank key) is still gone; the static :meth:`redact_api_keys` pass
        then runs unchanged on the result.

        Args:
            message: Raw text, typically an exception message or traceback.

        Returns:
            The text with every configured secret replaced by ``***`` and
            every ``sk-``-shaped key replaced by ``sk-***``.
        """
        for secret in self._redact_secrets:
            message = message.replace(secret, "***")
        return self.redact_api_keys(message)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        name: str = "",
        provider: SampleChatProvider | ScenarioProvider | None = None,
        description: str = "",
        parent_id: int | None = None,
        template: AgentTemplate | None = None,
        **session_kwargs,
    ) -> AgentSession:
        """Create a new :class:`AgentSession` and register it.

        This is the single entry point for session construction: it validates
        the provider and parent, allocates an id, wires up the tree structure
        (``parent_id`` / ``sub_agent_ids``) and pushes a tree sync event.

        Args:
            name: Display name for the agent.
            provider: Chat provider instance. Falls back to the orchestrator's
                default_provider if not given.
            description: Optional description shown in the tree.
            parent_id: If set, the new session becomes a child of this node.
            template: Optional agent template whose ``session_autoload`` blocks
                are replayed into the new session before it is registered. Its
                first block replaces the session's system prompt when that
                block is a system message; see
                :func:`~mitocode.core.agent.template.apply.apply_template`.
            **session_kwargs: Forwarded to :class:`AgentSession`. The
                orchestrator's ``system_prompt`` is used unless the caller
                passes one explicitly.

        Returns:
            The newly created, already registered :class:`AgentSession`.

        Raises:
            ValueError: If no provider is available, or parent_id references
                a non-existent session.
        """
        if provider is None:
            provider = self._default_provider
        if provider is None:
            raise ValueError(
                "provider is required — pass it directly or set "
                "default_provider on the orchestrator"
            )
        if parent_id is not None and parent_id not in self._sessions:
            raise ValueError(f"Parent agent {parent_id} does not exist")
        if hasattr(provider, "turn"):
            provider = copy.deepcopy(provider)
            setattr(provider, "turn", 0)

        agent_id = self._counter
        self._counter += 1

        session_kwargs.setdefault("system_prompt", self._system_prompt)

        session = AgentSession(
            agent_id=agent_id,
            name=name,
            description=description,
            parent_id=parent_id,
            provider=provider,
            tool_manager=self._tool_manager,
            **session_kwargs,
        )
        # Before registration: a template is part of the session's history, and
        # nothing must be able to observe the session between the two states.
        if template is not None:
            apply_template(session, template, workspace=self._workspace)
        self._sessions[agent_id] = session
        self._tasks[agent_id] = None
        if parent_id is not None:
            self._sessions[parent_id].sub_agent_ids.append(agent_id)

        await self._push_tree_sync()
        return session

    async def create_agent(
        self,
        name: str,
        provider: SampleChatProvider | ScenarioProvider | None = None,
        description: str = "",
        parent_id: int | None = None,
        **session_kwargs,
    ) -> int:
        """Create a new session and return its id.

        Backward-compatible alias for :meth:`create_session`.

        Args:
            name: Display name for the agent.
            provider: Chat provider instance. Falls back to the orchestrator's
                default_provider if not given.
            description: Optional description shown in the tree.
            parent_id: If set, the new session becomes a child of this node.
            **session_kwargs: Forwarded to :class:`AgentSession`.

        Returns:
            The numeric id of the newly created session.
        """
        session = await self.create_session(
            name=name,
            provider=provider,
            description=description,
            parent_id=parent_id,
            **session_kwargs,
        )
        return session.agent_id

    async def restore_tree(
        self, agents: Sequence[Mapping[str, Any]]
    ) -> Dict[int, AgentSession]:
        """Rebuild sessions from persisted records, keeping their original ids.

        The registry is assumed empty (the caller has just run
        :meth:`reset_sessions`).  Sessions are constructed in two passes:
        first every session object with its original id, then the tree
        wiring derived from ``parent_id``, so a record may list children
        before its parent.  The counter restarts above the largest
        restored id.

        Args:
            agents: Mapping records each carrying ``agent_id`` (int) and
                ``parent_id`` (``None`` or an id present in *agents*), plus
                optional ``name`` and ``description``.  Messages are NOT
                restored here; the caller fills them afterwards.

        Returns:
            Restored sessions keyed by their original agent id.

        Raises:
            ValueError: On an empty list, duplicate ids, a missing parent,
                or a cycle in the parent chain.
        """
        records: List[Mapping[str, Any]] = list(agents)
        if not records:
            raise ValueError("restore_tree requires at least one agent record")

        ids: List[int] = []
        for record in records:
            agent_id = record.get("agent_id")
            if not isinstance(agent_id, int) or isinstance(agent_id, bool):
                raise ValueError(f"agent_id must be an int, got {agent_id!r}")
            ids.append(agent_id)
        duplicates = {agent_id for agent_id in ids if ids.count(agent_id) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate agent ids in restore records: {sorted(duplicates)}"
            )

        known_ids = set(ids)
        parents: Dict[int, Any] = {}
        for record in records:
            agent_id = record["agent_id"]
            parent_id = record.get("parent_id")
            if parent_id is not None and parent_id not in known_ids:
                raise ValueError(f"Parent agent {parent_id} does not exist")
            parents[agent_id] = parent_id

        for agent_id in known_ids:
            visited: set[int] = set()
            current: Any = agent_id
            while current is not None:
                if current in visited:
                    raise ValueError(
                        f"Cycle detected in parent chain at agent {current}"
                    )
                visited.add(current)
                current = parents.get(current)

        if self._default_provider is None:
            raise ValueError(
                "provider is required — set default_provider on the orchestrator"
            )

        sessions: Dict[int, AgentSession] = {}
        for record in records:
            agent_id = record["agent_id"]
            provider = self._default_provider
            if hasattr(provider, "turn"):
                provider = copy.deepcopy(provider)
                setattr(provider, "turn", 0)
            sessions[agent_id] = AgentSession(
                agent_id=agent_id,
                name=str(record.get("name", "") or ""),
                description=str(record.get("description", "") or ""),
                parent_id=parents[agent_id],
                provider=provider,
                tool_manager=self._tool_manager,
                system_prompt="",
            )

        for agent_id in sorted(sessions):
            parent_id = sessions[agent_id].parent_id
            if parent_id is not None:
                sessions[parent_id].sub_agent_ids.append(agent_id)

        self._sessions.update(sessions)
        self._tasks.update({agent_id: None for agent_id in sessions})
        self._counter = max(ids) + 1

        await self._push_tree_sync()
        return sessions

    async def destroy_agent(self, agent_id: int, _sync: bool = True) -> None:
        """Remove a session and all of its descendants.

        Args:
            agent_id: The session to remove.
            _sync: Internal flag.  Only the top-level call triggers a tree
                sync event to the TUI.
        """
        session = self._sessions.get(agent_id)
        if session is None:
            return
        await self.cancel_agent(agent_id, cascade=True)

        parent_id = session.parent_id
        if parent_id is not None and parent_id in self._sessions:
            parent = self._sessions[parent_id]
            if agent_id in parent.sub_agent_ids:
                parent.sub_agent_ids.remove(agent_id)

        for sub_id in list(session.sub_agent_ids):
            await self.destroy_agent(sub_id, _sync=False)

        self._tasks.pop(agent_id, None)
        self._sessions.pop(agent_id, None)
        if _sync:
            await self._push_tree_sync()

    async def cancel_agent(self, agent_id: int, cascade: bool = True) -> None:
        """Request cancellation for a session (and optionally its children).

        Sets the session cancel event and, if an asyncio Task is associated,
        cancels it for immediate interruption.

        Args:
            agent_id: The session to cancel.
            cascade: If True, recursively cancel all descendants first.
        """
        session = self._sessions.get(agent_id)
        if session is None:
            return
        if cascade:
            for sub_id in session.sub_agent_ids:
                await self.cancel_agent(sub_id, cascade=True)
        session.cancel()
        task = self._tasks.get(agent_id)
        if task is not None:
            task.cancel()

    async def send_message(self, agent_id: int, message: str) -> asyncio.Task | None:
        """Send a user message to a session and start its run loop.

        If the session is already running the previous task is cancelled and
        awaited before a new one is created, preventing concurrent execution.

        Args:
            agent_id: The target session.
            message: The user prompt to append.

        Returns:
            The newly created asyncio Task, or None if the session does not exist.
        """
        session = self._sessions.get(agent_id)
        if session is None:
            return None

        old_task = self._tasks.get(agent_id)
        if session.state == AgentState.RUNNING and old_task is not None:
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass

        session.append_user_prompt(message)
        session.state = AgentState.RUNNING
        await self._push_tree_sync()

        task = asyncio.create_task(self._run_session(agent_id))
        self._tasks[agent_id] = task
        return task

    def has_active_tasks(self) -> bool:
        """Return whether any session currently has a live run task.

        A task that has finished but has not been reaped does not count:
        ``_run_session`` clears its own entry in ``finally``, so a ``None``
        entry or a done task both mean "idle" for switching purposes.

        Returns:
            ``True`` when at least one registered task is not done.
        """
        return any(
            task is not None and not task.done() for task in self._tasks.values()
        )

    async def reset_sessions(self) -> None:
        """Cancel and await every run task, then drop the whole registry.

        Sessions are cancelled cooperatively first so the runtime's
        ``finally`` branches can still write tool results and synthetic
        cancel events; the tasks are then awaited with
        ``return_exceptions=True`` because a cancelled task raises. The
        counter restarts at zero so a subsequent :meth:`restore_tree` or
        :meth:`create_session` allocates ids from the beginning again.
        """
        tasks = [task for task in self._tasks.values() if task is not None]
        for session in self._sessions.values():
            session.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._sessions.clear()
        self._tasks.clear()
        self._counter = 0
        await self._push_tree_sync()

    def get_session(self, agent_id: int) -> AgentSession | None:
        """Return the session registered under ``agent_id``, or ``None``."""
        return self._sessions.get(agent_id)

    # ------------------------------------------------------------------
    # Tree snapshot
    # ------------------------------------------------------------------

    def get_tree_snapshot(self) -> List[dict]:
        """Return the current session tree as a list of dicts for TUI rendering."""
        return [
            session.to_agent_node().to_dict() for session in self._sessions.values()
        ]

    async def _push_tree_sync(self) -> None:
        snapshot = self.get_tree_snapshot()
        try:
            await self.event_queue.put((-1, {"type": "sync", "agents": snapshot}))
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # Internal run loop
    # ------------------------------------------------------------------

    async def _run_session(self, agent_id: int) -> None:
        session = self._sessions.get(agent_id)
        if session is None:
            # The session was destroyed before this task got scheduled.
            return
        cancel_event_seen = False
        try:
            async for raw_event in AgentRuntime.run(session):
                if raw_event.get("event") == "system.cancelled":
                    cancel_event_seen = True
                adapted = self._bridge.adapt(raw_event, agent_id)
                if adapted is not None:
                    await self.event_queue.put((agent_id, adapted))

            if session.is_cancelled():
                session.state = AgentState.CANCELLED
            else:
                session.state = AgentState.FINISHED
        except asyncio.CancelledError:
            session.state = AgentState.CANCELLED
            self._emit_synthetic_cancel(agent_id, cancel_event_seen)
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Provider failures stay inside the run loop on purpose: the TUI
            # relies on the task completing quietly and the tree showing the
            # error state. The log keeps the traceback, the event carries only
            # the redacted message so a key can never reach a pane or a log sink.
            logger.error(
                f"Agent {agent_id} run failed\n"
                f"{self._redact_message(traceback.format_exc())}"
            )
            await self.event_queue.put(
                (
                    agent_id,
                    {
                        "type": "error",
                        "agent_id": agent_id,
                        "message": self._redact_message(str(exc)),
                    },
                )
            )
            session.state = AgentState.ERROR
        finally:
            if self._tasks.get(agent_id) is asyncio.current_task():
                self._tasks[agent_id] = None
            await self._push_tree_sync()

    def _emit_synthetic_cancel(self, agent_id: int, cancel_event_seen: bool) -> None:
        """Queue the cancellation event a hard task kill never let the runtime yield.

        ``task.cancel()`` injects ``CancelledError`` straight into whatever the
        turn was awaiting, so the runtime's cooperative checkpoints never run
        and no ``system.cancelled`` event is yielded; the consumer would never
        learn that the turn ended. The event is synthesised here instead,
        exactly once per run: when the cooperative path already yielded one,
        *cancel_event_seen* is ``True`` and nothing is queued, so the consumer
        never sees a duplicate. ``put_nowait`` is used because awaiting inside
        a cancelled task's handler would let a second cancellation abort the
        emit.

        Args:
            agent_id: Id of the agent whose turn was cancelled.
            cancel_event_seen: Whether the runtime already yielded its own
                ``system.cancelled`` event during this run.
        """
        if cancel_event_seen:
            return
        adapted = self._bridge.adapt({"event": "system.cancelled"}, agent_id)
        if adapted is None:
            return
        try:
            self.event_queue.put_nowait((agent_id, adapted))
        except asyncio.QueueFull:
            pass
