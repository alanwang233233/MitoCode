"""Enumeration of agent lifecycle states."""

from enum import Enum


class AgentState(Enum):
    """Lifecycle states of an :class:`AgentSession`."""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"
    DEFAULT = IDLE
