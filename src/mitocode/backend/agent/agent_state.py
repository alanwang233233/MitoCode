from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"
    DEFAULT = IDLE
