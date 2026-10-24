"""Adapter translating agent runtime events into TUI-ready dicts."""

from typing import Any, Dict

from ..tool.tool_manager import ToolManager


class EventBridge:
    """Translates raw :class:`AgentRuntime` events into TUI-compatible dicts.

    Raw events (yielded by ``AgentRuntime.run()``) use internal event types
    such as ``agent.text`` or ``agent.tool_call.header``.  The TUI expects
    ``token``, ``tool_call``, and ``tool_result`` type keys; the interactive
    CLI additionally consumes ``tool_execution`` (with the call's parsed
    arguments), ``turn_end`` and ``cancelled`` — the old TUI's if/elif
    chain ignores types it does not know, so the additive mappings are safe
    for both consumers.  ``EventBridge`` performs this mapping and
    optionally enriches tool-call headers with display metadata from a
    :class:`ToolManager`.
    """

    def __init__(self, tool_manager: ToolManager | None = None):
        """Initialise the bridge, optionally enriched with tool metadata.

        Args:
            tool_manager: If provided, tool-call headers are enriched with
                display action and description from registered tool specs.
        """
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
                spec = self._tool_manager.get_spec(tool_name)
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
                "success": event.get("success", False),
            }

        if event_type == "agent.tool_call.execution":
            return {
                "type": "tool_execution",
                "agent_id": agent_id,
                "name": event.get("name", ""),
                "args": event.get("args", {}),
            }

        if event_type == "system.finished.standard":
            return {
                "type": "turn_end",
                "agent_id": agent_id,
                "reason": event.get("reason", ""),
            }

        if event_type == "system.cancelled":
            return {"type": "cancelled", "agent_id": agent_id}

        return None
