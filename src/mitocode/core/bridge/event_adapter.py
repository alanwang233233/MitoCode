"""Adapter translating agent runtime events into TUI-ready dicts."""

from typing import Any, Dict

from ..tool.tool_manager import ToolManager

_TOOL_EVENT_TYPES = frozenset(
    {
        "agent.tool_call.header",
        "agent.tool_call.delta",
        "agent.tool_call.execution",
        "agent.tool_call.result",
    }
)


class EventBridge:
    """Translates raw :class:`AgentRuntime` events into TUI-compatible dicts.

    Raw events (yielded by ``AgentRuntime.run()``) use internal event types
    such as ``agent.text`` or ``agent.tool_call.header``.  The TUI expects
    ``token``, ``tool_call``, ``tool_start``, ``tool_result``, ``turn_end``,
    and ``turn_cancelled`` type keys, each carrying the originating call id
    where one exists.  ``EventBridge`` performs this mapping and optionally
    enriches tool events with display metadata from a :class:`ToolManager`.
    Events it does not know are skipped.
    """

    def __init__(self, tool_manager: ToolManager | None = None):
        """Initialise the bridge, optionally enriched with tool metadata.

        Args:
            tool_manager: If provided, tool events are enriched with display
                action, description, and display parameter from registered
                tool specs.
        """
        self._tool_manager = tool_manager

    def _display_metadata(self, tool_name: str) -> tuple[str, str, str]:
        """Look up the human-facing display metadata of one tool.

        Args:
            tool_name: Registered tool name as reported by the runtime.

        Returns:
            A ``(action, description, display_param)`` tuple.  When the tool is
            unknown (or no :class:`ToolManager` was supplied) the action and
            description fall back to *tool_name* and ``display_param`` is
            empty.
        """
        action = tool_name
        description = tool_name
        display_param = ""
        if self._tool_manager:
            spec = self._tool_manager.get_spec(tool_name)
            if spec:
                action = spec.display_action or tool_name
                description = spec.human_doc or tool_name
                display_param = spec.display_param or ""
        return action, description, display_param

    def _adapt_tool_event(
        self, event_type: str, event: Dict[str, Any], agent_id: int
    ) -> Dict[str, Any] | None:
        """Map one ``agent.tool_call.*`` event to its TUI-ready shape.

        Args:
            event_type: The raw event type, expected to be a tool-call event.
            event: Raw event dict from :meth:`AgentRuntime.run`.
            agent_id: The numeric id of the agent that produced the event.

        Returns:
            A dict describing the tool call's header, streaming delta,
            execution, or result, or ``None`` if *event_type* is not a
            tool-call event this bridge understands.
        """
        tool_name = event.get("name", "")
        call_id = event.get("id", "")

        if event_type == "agent.tool_call.header":
            action, description, _ = self._display_metadata(tool_name)
            return {
                "type": "tool_call",
                "agent_id": agent_id,
                "id": call_id,
                "name": tool_name,
                "action": action,
                "description": description,
                "additional": "",
            }

        if event_type == "agent.tool_call.delta":
            return {
                "type": "tool_call",
                "agent_id": agent_id,
                "id": call_id,
                "content": event.get("delta", ""),
            }

        if event_type == "agent.tool_call.execution":
            action, description, display_param = self._display_metadata(tool_name)
            return {
                "type": "tool_start",
                "agent_id": agent_id,
                "id": call_id,
                "name": tool_name,
                "args": event.get("args", {}),
                "action": action,
                "description": description,
                "display_param": display_param,
            }

        if event_type == "agent.tool_call.result":
            return {
                "type": "tool_result",
                "agent_id": agent_id,
                "id": call_id,
                "result": event.get("result", ""),
                "success": event.get("success", False),
            }

        return None

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

        if event_type == "agent.reasoning":
            return {
                "type": "reasoning",
                "agent_id": agent_id,
                "content": event.get("delta", ""),
            }

        if event_type in _TOOL_EVENT_TYPES:
            return self._adapt_tool_event(event_type, event, agent_id)

        if event_type == "system.finished.standard":
            return {
                "type": "turn_end",
                "agent_id": agent_id,
                "reason": event.get("reason", ""),
            }

        if event_type == "system.cancelled":
            return {"type": "turn_cancelled", "agent_id": agent_id}

        return None
