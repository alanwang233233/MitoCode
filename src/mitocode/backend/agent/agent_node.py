from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentNode:
    """Pure data model representing one agent in the agent tree.

    Owned by :class:`AgentOrchestrator` as the single source of truth.
    The TUI reads a snapshot (``get_tree_snapshot()``) and constructs its
    Textual ``Tree`` widget from these nodes without holding state.
    """

    id: int
    name: str
    description: str = ""
    parent_id: Optional[int] = None
    sub_agent_ids: List[int] = field(default_factory=list)
    state: str = "idle"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "sub_agent": list(self.sub_agent_ids),
            "state": self.state,
        }
