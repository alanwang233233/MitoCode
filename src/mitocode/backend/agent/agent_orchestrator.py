import asyncio
import copy
from typing import Dict, List

from .agent_runtime import AgentRuntime
from .agent_session import AgentSession
from .agent_state import AgentState
from ..bridge.event_adapter import EventBridge
from ..chat_provider.sample_chat_provider import SampleChatProvider
from ..chat_provider.scenario_provider import ScenarioProvider
from ..tool.tool_manager import ToolManager


class AgentOrchestrator:
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
    ):
        self._sessions: Dict[int, AgentSession] = {}
        self._tasks: Dict[int, asyncio.Task | None] = {}
        self._counter: int = 0
        self._tool_manager: ToolManager | None = tool_manager
        self._default_provider: SampleChatProvider | ScenarioProvider | None = (
            default_provider
        )
        self._bridge = EventBridge(tool_manager)
        self.event_queue: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        name: str = "",
        provider: SampleChatProvider | ScenarioProvider | None = None,
        description: str = "",
        parent_id: int | None = None,
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
            **session_kwargs: Forwarded to :class:`AgentSession`.

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

        session = AgentSession(
            agent_id=agent_id,
            name=name,
            description=description,
            parent_id=parent_id,
            provider=provider,
            tool_manager=self._tool_manager,
            **session_kwargs,
        )
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
        session = self._sessions[agent_id]
        try:
            async for raw_event in AgentRuntime.run(session):
                adapted = self._bridge.adapt(raw_event, agent_id)
                if adapted is not None:
                    await self.event_queue.put((agent_id, adapted))

            if session.is_cancelled():
                session.state = AgentState.CANCELLED
            else:
                session.state = AgentState.FINISHED
        except asyncio.CancelledError:
            session.state = AgentState.CANCELLED
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            session.state = AgentState.ERROR
        finally:
            if self._tasks.get(agent_id) is asyncio.current_task():
                self._tasks[agent_id] = None
            await self._push_tree_sync()
