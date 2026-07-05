import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .agent_session import AgentSession


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"


class Agent:
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
        parent_id: int | None = None,
    ):
        self.id: int = agent_id
        self.name: str = name
        self.description: str = description
        self.parent_id: int | None = parent_id
        self.sub_agent_ids: List[int] = []
        self.session: AgentSession = session
        self.state: AgentState = AgentState.IDLE
        self.task: asyncio.Task | None = None


@dataclass
class AgentNode:
    """Pure data model for the agent tree, consumed by the TUI.

    New instances are created on demand from :class:`ManagedAgent` via
    :meth:`from_managed` — never stored persistently.
    """

    id: int
    name: str
    description: str = ""
    parent_id: int | None = None
    sub_agent_ids: List[int] = field(default_factory=list)
    state: str = "idle"

    @classmethod
    def from_managed(cls, managed: "Agent") -> "AgentNode":
        return cls(
            id=managed.id,
            name=managed.name,
            description=managed.description,
            parent_id=managed.parent_id,
            sub_agent_ids=list(managed.sub_agent_ids),
            state=managed.state.value,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "sub_agent": list(self.sub_agent_ids),
            "state": self.state,
        }
