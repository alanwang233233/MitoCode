from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .managed_agent import ManagedAgent


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
    def from_managed(cls, managed: "ManagedAgent") -> "AgentNode":
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
