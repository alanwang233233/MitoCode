from typing import Any, Dict

from ..tool.tool_manager import ToolManager


class EventBridge:
    """Translates raw :class:`AgentRuntime` events into TUI-compatible dicts.

    Raw events (yielded by ``AgentRuntime.run()``) use internal event types
    such as ``agent.text`` or ``agent.tool_call.header``.  The TUI expects
    ``token``, ``tool_call``, and ``tool_result`` type keys.  ``EventBridge``
    performs this mapping and optionally enriches tool-call headers with
    display metadata from a :class:`ToolManager`.
    """

    def __init__(self, tool_manager: ToolManager | None = None):
        self._tool_manager = tool_manager

    def adapt(self, event: Dict[str, Any], agent_id: int) -> Dict[str, Any] | None:
        """Map one raw runtime event to a TUI-ready dict, or ``None`` to skip.

        Args:
            event: Raw event dict from :meth:`AgentRuntime.run`.
            agent_id: The numeric id of the agent that produced the event.

        Returns:
            A dict with keys ``type`` and ``agent_id`` suitable for the TUI,
            or ``None`` if the event should not be forwarded.
        """
        event_type = event.get("event")

        if event_type == "agent.text":
            return {
                "type": "token",
                "agent_id": agent_id,
                "content": event.get("delta", ""),
            }

        if event_type == "agent.tool_call.header":
            tool_name = event.get("name", "")
            action = tool_name
            description = ""
            if self._tool_manager:
                spec = self._tool_manager._tools.get(tool_name)
                if spec:
                    action = spec.display_action or tool_name
                    description = spec.human_doc or tool_name
            return {
                "type": "tool_call",
                "agent_id": agent_id,
                "action": action,
                "description": description or tool_name,
                "additional": "",
            }

        if event_type == "agent.tool_call.delta":
            return {
                "type": "tool_call",
                "agent_id": agent_id,
                "content": event.get("delta", ""),
            }

        if event_type == "agent.tool_call.result":
            result = event.get("result", "")
            return {
                "type": "tool_result",
                "agent_id": agent_id,
                "result": result,
            }

        return None
