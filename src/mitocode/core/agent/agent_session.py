"""Conversation state container for a single agent session."""

import asyncio
import json
from typing import List, Any, Dict, Callable

from loguru import logger as loguru_logger

from mitocode.core.tool.tool_manager import ToolManager

from .agent_node import AgentNode
from .agent_state import AgentState

# noinspection PyProtectedMember
from ..chat_provider.sample_chat_provider import _SampleChatProvider

logger = loguru_logger.bind(name=__name__)


class _MessageBlock:
    """Base class for all message elements in the conversation history."""

    def __init__(self):
        self.finished: bool = False


class _PlainTextMessageBlock(_MessageBlock):
    """Represents standard text messages from the user or the assistant."""

    def __init__(self, role: str, message: str = ""):
        super().__init__()
        self.role: str = role
        self.message: str = message
        # Text blocks are considered valid for context building immediately
        self.finished: bool = True

    def append(self, delta: str):
        """Append a streaming text delta to this block's message."""
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
        """Append a streaming chunk of raw JSON tool arguments."""
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
            },
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
    """Stateful conversation container for one agent.

    Holds the message history as a list of internal message blocks, merges
    tools from an optional :class:`ToolManager` with user-supplied tools and
    schemas, tracks tree relationships (``parent_id`` / ``sub_agent_ids``)
    and lifecycle :class:`AgentState`, exposes cooperative cancellation, and
    converts the history into provider-compatible dicts via
    :meth:`build_standard_context`.

    A non-empty ``system_prompt`` is seeded as the first message block, so it
    precedes every user and assistant turn in the context.  Managed tools are
    snapshotted at construction time; call :meth:`refresh_tools` to pick up
    tools registered afterwards (for example by an MCP refresh).

    Attributes:
        provider: The chat provider used by the runtime.
        messages: Ordered internal message blocks forming the context.
        system_prompt: The role instructions seeded as the first message.
        tools_map: Merged name-to-callable mapping available to this agent.
        tool_schemas: OpenAI-compatible tool schemas for ``chat`` requests.
        state: Current lifecycle state (idle/running/finished/...).
    """

    @staticmethod
    def factory(**settings) -> "AgentSession":
        """
        Create and return a new :class:`AgentSession`.

        Keyword arguments override the factory-level defaults set in
        :meth:`__init__`.

        Args:
            **settings: Settings accepted by :class:`AgentSession`, e.g.
                ``provider``, ``tools_map``, ``tool_schemas``,
                ``tool_manager``, ``on_end``, ``system_prompt``.

        Returns:
            AgentSession: A freshly constructed session instance.
        """
        return AgentSession(**settings)

    def __init__(self, **settings):
        provider: _SampleChatProvider = settings.get("provider", None)
        # The user-supplied collections are kept intact so that
        # :meth:`refresh_tools` can rebuild the merged view without losing
        # them: only the managed half of the merge is allowed to change.
        self._user_tools_map: Dict[str, Callable] = dict(
            settings.get("tools_map") or {}
        )
        self._user_tool_schemas: List[Dict[str, Any]] = list(
            settings.get("tool_schemas") or []
        )
        tool_manager = settings.get("tool_manager", None)
        self.agent_id: int = settings.get("agent_id", -1)
        self.name: str = settings.get("name", "")
        self.description: str = settings.get("description", "")
        self.parent_id: int | None = settings.get("parent_id", None)
        self.system_prompt: str = settings.get("system_prompt", "") or ""
        self.sub_agent_ids: List[int] = []
        self.state: AgentState = AgentState.DEFAULT

        if not provider:
            raise ValueError("Provider is required.")
        if not isinstance(provider, _SampleChatProvider):
            raise TypeError("Provider must be an instance of _SampleChatProvider.")
        if tool_manager is not None and not isinstance(tool_manager, ToolManager):
            raise TypeError("tool_manager must be an instance of ToolManager.")

        tools_map, tool_schemas = self._merge_tools(tool_manager)

        self.provider = provider
        self.messages: List[_MessageBlock] = []
        # The prompt goes first and before any user message, so every later
        # turn sees the same role instructions.
        if self.system_prompt:
            self.messages.append(_PlainTextMessageBlock("system", self.system_prompt))
        self.tools_map: Dict[str, Callable] = tools_map
        self.tool_schemas = tool_schemas
        self.tool_manager: ToolManager | None = tool_manager
        self._cancel_event = asyncio.Event()

    def _merge_tools(
        self, tool_manager: ToolManager | None
    ) -> tuple[Dict[str, Callable], List[Dict[str, Any]]]:
        """Merge managed and user-supplied tools and schemas.

        User entries win on a name collision, for both the callable mapping and
        the schema list.  Managed schemas missing a ``function``/``name`` pair
        are skipped; user schemas without a name are kept by object identity so
        a caller that passes one does not silently lose it.  Without a manager
        the user collections are copied as-is.

        Args:
            tool_manager: The registry to snapshot, or ``None`` when the
                session only uses user-supplied tools.

        Returns:
            A ``(tools_map, tool_schemas)`` tuple suitable for assigning to the
            session attributes.
        """
        if not tool_manager:
            return dict(self._user_tools_map), list(self._user_tool_schemas)

        tools_map = {**tool_manager.tools_map(), **self._user_tools_map}

        merged = {
            s["function"]["name"]: s
            for s in tool_manager.tool_schemas("agent")
            if "function" in s and "name" in s["function"]
        }
        for schema in self._user_tool_schemas:
            name = schema.get("function", {}).get("name")
            if name:
                merged[name] = schema
            else:
                merged[id(schema)] = schema
        return tools_map, list(merged.values())

    def refresh_tools(self) -> None:
        """Re-snapshot managed tools while keeping user-supplied ones.

        Rebuilds :attr:`tools_map` and :attr:`tool_schemas` from the session's
        :class:`ToolManager`, which is what makes tools registered after the
        session was constructed (e.g. by an MCP refresh) visible to it.  The
        merged result is identical in shape to the one built at construction
        time, including user tools overriding managed ones of the same name.

        Nothing else changes: the message history, provider, cancellation
        state, lifecycle state and tree fields are left untouched, so a refresh
        is safe in the middle of an idle session.
        """
        self.tools_map, self.tool_schemas = self._merge_tools(self.tool_manager)

    def get_tool_docs_for_human(self) -> Dict[str, str]:
        """Return human-readable documentation for all managed tools."""
        if not self.tool_manager:
            return {}
        return self.tool_manager.human_docs()

    def get_tool_docs_for_agent(self) -> Dict[str, str]:
        """Return agent-facing documentation for all managed tools."""
        if not self.tool_manager:
            return {}
        return self.tool_manager.agent_docs()

    def cancel(self):
        """Signals the session to halt its current run stream."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested for this session."""
        return self._cancel_event.is_set()

    def append_user_prompt(self, prompt: str):
        """Adds a user prompt to the context and clears any prior cancellation."""
        self._cancel_event.clear()  # Reset cancellation state before new run
        self.messages.append(_PlainTextMessageBlock("user", prompt))

    def append_assistant_delta(self, delta: str):
        """Appends streaming text chunks to the latest assistant message."""
        if (
            not self.messages
            or not isinstance(self.messages[-1], _PlainTextMessageBlock)
            or self.messages[-1].role != "assistant"
        ):
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
                    context.append(
                        {"role": "assistant", "content": None, "tool_calls": tool_calls}
                    )
                    tool_calls = []
                context.append({"role": block.role, "content": block.message})
            elif isinstance(block, _ToolCallMessageBlock):
                tool_calls.append(block.to_standard())
            elif isinstance(block, _ToolResultMessageBlock):
                if tool_calls:
                    context.append(
                        {"role": "assistant", "content": None, "tool_calls": tool_calls}
                    )
                    tool_calls = []
                context.append(
                    {
                        "role": block.role,
                        "tool_call_id": block.tool_call_id,
                        "content": block.content,
                    }
                )

        # Catch any trailing tool calls at the end of the history
        if tool_calls:
            context.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )

        return context

    def to_agent_node(self) -> AgentNode:
        """Builds a pure-data :class:`AgentNode` snapshot for TUI rendering."""
        return AgentNode(
            id=self.agent_id,
            name=self.name,
            description=self.description,
            parent_id=self.parent_id,
            sub_agent_ids=list(self.sub_agent_ids),
            state=self.state.value,
        )
