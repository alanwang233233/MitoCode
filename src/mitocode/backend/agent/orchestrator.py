import asyncio
from typing import Any, Callable, Dict, List, Optional

from mitocode.backend.agent.agent_runtime import AgentRuntime
from mitocode.backend.agent.agent_session import AgentSession
from mitocode.backend.agent.agent_node import AgentNode
from mitocode.backend.agent.managed_agent import ManagedAgent, AgentState
from mitocode.backend.bridge.event_adapter import EventBridge
from mitocode.backend.tool.tool_manager import ToolManager


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
        self, tool_manager: Optional[ToolManager] = None, default_provider=None
    ):
        self._agents: Dict[int, ManagedAgent] = {}
        self._nodes: Dict[int, AgentNode] = {}
        self._counter: int = 0
        self._tool_manager = tool_manager
        self._default_provider = default_provider
        self._bridge = EventBridge(tool_manager)
        self.event_queue: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        provider=None,
        description: str = "",
        parent_id: Optional[int] = None,
        **session_kwargs,
    ) -> int:
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
        if parent_id is not None and parent_id in self._agents:
            self._agents[parent_id].sub_agent_ids.append(agent_id)

        self._agents[agent_id] = managed
        self._nodes[agent_id] = AgentNode(
            id=agent_id,
            name=name,
            description=description,
            parent_id=parent_id,
        )
        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].sub_agent_ids.append(agent_id)

        self._push_tree_sync()
        return agent_id

    def destroy_agent(self, agent_id: int) -> None:
        managed = self._agents.get(agent_id)
        if managed is None:
            return
        self.cancel_agent(agent_id, cascade=True)

        parent_id = managed.parent_id
        if parent_id is not None and parent_id in self._agents:
            parent = self._agents[parent_id]
            if agent_id in parent.sub_agent_ids:
                parent.sub_agent_ids.remove(agent_id)
        if parent_id is not None and parent_id in self._nodes:
            parent_node = self._nodes[parent_id]
            if agent_id in parent_node.sub_agent_ids:
                parent_node.sub_agent_ids.remove(agent_id)

        for sub_id in list(managed.sub_agent_ids):
            self.destroy_agent(sub_id)

        self._agents.pop(agent_id, None)
        self._nodes.pop(agent_id, None)
        self._push_tree_sync()

    def cancel_agent(self, agent_id: int, cascade: bool = True) -> None:
        managed = self._agents.get(agent_id)
        if managed is None:
            return
        if cascade:
            for sub_id in managed.sub_agent_ids:
                self.cancel_agent(sub_id, cascade=True)
        managed.session.cancel()
        if managed._task is not None:
            managed._task.cancel()

    async def send_message(self, agent_id: int, message: str) -> Optional[asyncio.Task]:
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
        for agent_id, managed in self._agents.items():
            if agent_id in self._nodes:
                self._nodes[agent_id].state = managed.state.value
        return [node.to_dict() for node in self._nodes.values()]

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
