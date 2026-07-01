import asyncio
from enum import Enum
from typing import List, Optional

from mitocode.backend.agent.agent_session import AgentSession


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"


class ManagedAgent:
    """Wraps an :class:`AgentSession` with lifecycle metadata.

    Each ``ManagedAgent`` is owned by :class:`AgentOrchestrator` and tracked
    by a unique numeric id.  The orchestrator is responsible for mutating
    ``state``, ``parent_id``, and ``sub_agent_ids``.
    """

    def __init__(
        self,
        agent_id: int,
        name: str,
        session: AgentSession,
        description: str = "",
        parent_id: Optional[int] = None,
    ):
        self.id: int = agent_id
        self.name: str = name
        self.description: str = description
        self.parent_id: Optional[int] = parent_id
        self.sub_agent_ids: List[int] = []
        self.session: AgentSession = session
        self.state: AgentState = AgentState.IDLE
        self._task: Optional[asyncio.Task] = None
