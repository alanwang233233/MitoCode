import asyncio
import json
from typing import List, Any, Dict, Callable

from loguru import logger as loguru_logger
from mitocode.backend.tool.tool_manager import ToolManager

logger = loguru_logger.bind(name=__name__)


class _MessageBlock:
    """Base class for all message elements in the conversation history."""

    def __init__(self):
        self.finished = False


class _PlainTextMessageBlock(_MessageBlock):
    """Represents standard text messages from the user or the assistant."""

    def __init__(self, role: str, message: str = ""):
        super().__init__()
        self.role = role
        self.message: str = message
        # Text blocks are considered valid for context building immediately
        self.finished = True

    def append(self, delta: str):
        self.message += delta


class _ToolCallMessageBlock(_MessageBlock):
    """Represents an assistant's request to call a tool/function."""

    def __init__(self, name: str, call_id: str):
        super().__init__()
        self.name = name
        self.id = call_id
        self.arguments_string = ""
        self.parameter: Dict[str, Any] = {}

        self.parse_error: str | None = None

    def append_argument_chunk(self, delta: str):
        self.arguments_string += delta

    def finish(self):
        """Finalizes the tool call by safely parsing the argument string."""
        self.finished = True
        if self.arguments_string:
            try:
                self.parameter = json.loads(self.arguments_string)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {self.name}: {e}")
                self.parse_error = f"JSON parsing failed: {e}"
                self.parameter = {}
        else:
            self.parameter = {}

    def to_standard(self) -> Dict[str, Any]:
        """Converts the internal block to the standard OpenAI tool call format."""
        if not self.finished:
            return {}
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_string,
            }
        }


class _ToolResultMessageBlock(_MessageBlock):
    """Stores the execution results returned from local tools."""

    def __init__(self, tool_call_id: str, result: str):
        super().__init__()
        self.role = "tool"
        self.tool_call_id = tool_call_id
        self.content = result
        self.finished = True


class AgentSession:
    """
    Strong-state container for a single agent conversation session.

    Holds all mutable conversation state: the message history, the provider,
    tool bindings, and the cancellation event.  One ``AgentSession`` object
    represents one independent conversation thread.
    """

    def __init__(self, **settings):
        provider = settings.get("provider", None)
        tools_map = settings.get("tools_map", {})
        tool_schemas = settings.get("tool_schemas", [])
        tool_manager = settings.get("tool_manager", None)
        on_end = settings.get("on_end", None)

        if not provider:
            raise ValueError("Provider is required.")
        if tool_manager is not None and not isinstance(tool_manager, ToolManager):
            raise TypeError("tool_manager must be an instance of ToolManager.")

        if tool_manager:
            managed_map = tool_manager.tools_map()
            managed_schemas = tool_manager.tool_schemas("agent")
            tools_map = {**managed_map, **tools_map}

            merged_schemas = {s["function"]["name"]: s for s in managed_schemas if
                              "function" in s and "name" in s["function"]}
            for schema in tool_schemas:
                name = schema.get("function", {}).get("name")
                if name:
                    merged_schemas[name] = schema
                else:
                    merged_schemas[id(schema)] = schema
            tool_schemas = list(merged_schemas.values())

        self.provider = provider
        self.messages: List[_MessageBlock] = []
        self.tools_map: Dict[str, Callable] = tools_map
        self.tool_schemas = tool_schemas
        self.tool_manager: ToolManager | None = tool_manager
        self.on_end = on_end

        # Asyncio event to handle cancellation signals safely
        self._cancel_event = asyncio.Event()

    def get_tool_docs_for_human(self) -> Dict[str, str]:
        if not self.tool_manager:
            return {}
        return self.tool_manager.human_docs()

    def get_tool_docs_for_agent(self) -> Dict[str, str]:
        if not self.tool_manager:
            return {}
        return self.tool_manager.agent_docs()

    def cancel(self):
        """Signals the session to halt its current run stream."""
        self._cancel_event.set()

    def append_user_prompt(self, prompt: str):
        """Adds a user prompt to the context and clears any prior cancellation."""
        self._cancel_event.clear()  # Reset cancellation state before new run
        self.messages.append(_PlainTextMessageBlock("user", prompt))

    def append_assistant_delta(self, delta: str):
        """Appends streaming text chunks to the latest assistant message."""
        if (not self.messages or
                not isinstance(self.messages[-1], _PlainTextMessageBlock) or
                self.messages[-1].role != "assistant"):
            self.messages.append(_PlainTextMessageBlock("assistant", delta))
        else:
            self.messages[-1].append(delta)

    def build_standard_context(self) -> List[Dict[str, Any]]:
        """Compiles internal message blocks into a provider-compatible JSON list."""
        context = []
        tool_calls = []

        for block in self.messages:
            if not block.finished:
                continue

            if isinstance(block, _PlainTextMessageBlock):
                if tool_calls:
                    context.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    })
                    tool_calls = []
                context.append({
                    "role": block.role,
                    "content": block.message
                })
            elif isinstance(block, _ToolCallMessageBlock):
                tool_calls.append(block.to_standard())
            elif isinstance(block, _ToolResultMessageBlock):
                if tool_calls:
                    context.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    })
                    tool_calls = []
                context.append({
                    "role": block.role,
                    "tool_call_id": block.tool_call_id,
                    "content": block.content
                })

        # Catch any trailing tool calls at the end of the history
        if tool_calls:
            context.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            })

        return context
