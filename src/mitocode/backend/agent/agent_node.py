"""Pure-data node model used to render the agent tree in the TUI."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentNode:
    """Pure data model for the agent tree, consumed by the TUI.

    New instances are created on demand from :class:`AgentSession` via
    ``AgentSession.to_agent_node()`` — never stored persistently.
    """

    id: int
    name: str
    description: str = ""
    parent_id: int | None = None
    sub_agent_ids: List[int] = field(default_factory=list)
    state: str = "idle"

    def to_dict(self) -> dict:
        """Serialize this node to the dict shape consumed by the TUI tree."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "sub_agent": list(self.sub_agent_ids),
            "state": self.state,
        }
