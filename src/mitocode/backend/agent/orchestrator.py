import asyncio
from typing import Any, Dict, List

from .agent_runtime import AgentRuntime
from .agent_session import AgentSession
from .agent_node import AgentNode
from .managed_agent import ManagedAgent, AgentState
from ..bridge.event_adapter import EventBridge
from ..tool.tool_manager import ToolManager


class AgentOrchestrator:
    """Central controller that manages the lifecycle of multiple agents.

    Responsibilities:

    * Agent registry with auto-incrementing ids.
    * Tree-structure maintenance (parent / child relationships).
    * Running agents as ``asyncio.Task`` instances.
    * Event routing from internal :class:`AgentRuntime` events through
      :class:`EventBridge` to a shared ``asyncio.Queue`` consumed by the TUI.
    * Cascading cancellation.
    """

    def __init__(
        self,
        tool_manager: ToolManager | None = None,
        default_provider: Any = None,
    ):
        self._agents: Dict[int, ManagedAgent] = {}
        self._counter: int = 0
        self._tool_manager: ToolManager | None = tool_manager
        self._default_provider: Any = default_provider
        self._bridge = EventBridge(tool_manager)
        self.event_queue: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        provider: Any = None,
        description: str = "",
        parent_id: int | None = None,
        **session_kwargs,
    ) -> int:
        """Create and register a new agent.

        Args:
            name: Display name for the agent.
            provider: Chat provider instance. Falls back to the orchestrator's
                default_provider if not given.
            description: Optional description shown in the tree.
            parent_id: If set, the new agent becomes a child of this node.
            **session_kwargs: Forwarded to :class:`AgentSession`.

        Returns:
            The numeric id of the newly created agent.

        Raises:
            ValueError: If no provider is available and parent_id references
                a non-existent agent.
        """
        agent_id = self._counter
        self._counter += 1

        provider = provider or self._default_provider
        if provider is None:
            raise ValueError(
                "provider is required — pass it directly or set default_provider on the orchestrator"
            )

        session = AgentSession(
            provider=provider, tool_manager=self._tool_manager, **session_kwargs
        )
        managed = ManagedAgent(
            agent_id=agent_id,
            name=name,
            session=session,
            description=description,
            parent_id=parent_id,
        )
        if parent_id is not None:
            if parent_id not in self._agents:
                raise ValueError(f"Parent agent {parent_id} does not exist")
            self._agents[parent_id].sub_agent_ids.append(agent_id)

        self._agents[agent_id] = managed

        self._push_tree_sync()
        return agent_id

    def destroy_agent(self, agent_id: int, _sync: bool = True) -> None:
        """Remove an agent and all of its descendants.

        Args:
            agent_id: The agent to remove.
            _sync: Internal flag.  Only the top-level call triggers a tree
                sync event to the TUI.
        """
        managed = self._agents.get(agent_id)
        if managed is None:
            return
        self.cancel_agent(agent_id, cascade=True)

        parent_id = managed.parent_id
        if parent_id is not None and parent_id in self._agents:
            parent = self._agents[parent_id]
            if agent_id in parent.sub_agent_ids:
                parent.sub_agent_ids.remove(agent_id)

        for sub_id in list(managed.sub_agent_ids):
            self.destroy_agent(sub_id, _sync=False)

        self._agents.pop(agent_id, None)
        if _sync:
            self._push_tree_sync()

    def cancel_agent(self, agent_id: int, cascade: bool = True) -> None:
        """Request cancellation for an agent (and optionally its children).

        Sets the session cancel event and, if an asyncio Task is associated,
        cancels it for immediate interruption.

        Args:
            agent_id: The agent to cancel.
            cascade: If True, recursively cancel all descendants first.
        """
        managed = self._agents.get(agent_id)
        if managed is None:
            return
        if cascade:
            for sub_id in managed.sub_agent_ids:
                self.cancel_agent(sub_id, cascade=True)
        managed.session.cancel()
        if managed._task is not None:
            managed._task.cancel()

    async def send_message(
        self, agent_id: int, message: str
    ) -> asyncio.Task | None:
        """Send a user message to an agent and start its run loop.

        If the agent is already running the previous task is cancelled and
        awaited before a new one is created, preventing concurrent execution.

        Args:
            agent_id: The target agent.
            message: The user prompt to append.

        Returns:
            The newly created asyncio Task, or None if the agent does not exist.
        """
        managed = self._agents.get(agent_id)
        if managed is None:
            return None

        if managed.state == AgentState.RUNNING and managed._task is not None:
            old_task = managed._task
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass

        managed.session.append_user_prompt(message)
        managed.state = AgentState.RUNNING
        self._push_tree_sync()

        task = asyncio.create_task(self._run_agent(managed))
        managed._task = task
        return task

    # ------------------------------------------------------------------
    # Tree snapshot
    # ------------------------------------------------------------------

    def get_tree_snapshot(self) -> List[dict]:
        """Return the current agent tree as a list of dicts for TUI rendering."""
        return [AgentNode.from_managed(m).to_dict() for m in self._agents.values()]

    def _push_tree_sync(self) -> None:
        snapshot = self.get_tree_snapshot()
        try:
            self.event_queue.put_nowait((-1, {"type": "sync", "agents": snapshot}))
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # Internal run loop
    # ------------------------------------------------------------------

    async def _run_agent(self, managed: ManagedAgent) -> None:
        try:
            async for raw_event in AgentRuntime.run(managed.session):
                adapted = self._bridge.adapt(raw_event, managed.id)
                if adapted is not None:
                    await self.event_queue.put((managed.id, adapted))

            if managed.session._cancel_event.is_set():
                managed.state = AgentState.CANCELLED
            else:
                managed.state = AgentState.FINISHED
        except asyncio.CancelledError:
            managed.state = AgentState.CANCELLED
            raise
        except Exception:
            managed.state = AgentState.ERROR
        finally:
            if managed._task is asyncio.current_task():
                managed._task = None
            self._push_tree_sync()
