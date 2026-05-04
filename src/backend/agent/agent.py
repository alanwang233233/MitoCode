# Re-export message block primitives so existing importers keep working.
from .agent_session import (
    _MessageBlock,
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
    AgentSession,
)
from .agent_runtime import AgentRuntime
from .session_factory import SessionFactory


class Agent:
    """
    Backward-compatible wrapper that composes :class:`AgentSession` and
    :class:`AgentRuntime`.

    All stateful attributes (``provider``, ``messages``, ``tools_map``,
    ``tool_schemas``, ``on_end``, ``_cancel_event``) are transparently
    forwarded to the underlying :class:`AgentSession` so that existing code
    continues to work without modification.

    The :meth:`run` async-generator delegates execution to the shared
    :class:`AgentRuntime` singleton.
    """

    # Attributes that belong to Agent itself and must NOT be forwarded to the
    # session.
    _OWN_ATTRS = frozenset({"_session", "_runtime"})

    def __init__(self, **settings):
        # Use object.__setattr__ to bypass our custom __setattr__.
        object.__setattr__(self, "_session", AgentSession(**settings))
        object.__setattr__(self, "_runtime", AgentRuntime.get_instance())

    # ------------------------------------------------------------------
    # Transparent attribute forwarding to the session
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails, so _session and
        # _runtime are never routed through here.
        return getattr(object.__getattribute__(self, "_session"), name)

    def __setattr__(self, name: str, value):
        if name in Agent._OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_session"), name, value)

    # ------------------------------------------------------------------
    # Explicit delegation for the public API
    # ------------------------------------------------------------------

    def cancel(self):
        """Signals the agent to halt its current run stream."""
        object.__getattribute__(self, "_session").cancel()

    def append_user_prompt(self, prompt: str):
        """Adds a user prompt to the context and clears any prior cancellation."""
        object.__getattribute__(self, "_session").append_user_prompt(prompt)

    def append_assistant_delta(self, delta: str):
        """Appends streaming text chunks to the latest assistant message."""
        object.__getattribute__(self, "_session").append_assistant_delta(delta)

    def build_standard_context(self):
        """Compiles internal message blocks into a provider-compatible JSON list."""
        return object.__getattribute__(self, "_session").build_standard_context()

    async def run(self):
        """
        Main interaction loop: streams responses, handles tools, and yields events.

        Delegates to :class:`AgentRuntime` while passing the underlying
        :class:`AgentSession` as the state container.  Refer to
        :meth:`AgentRuntime.run` for a full description of yielded events.
        """
        session = object.__getattribute__(self, "_session")
        runtime = object.__getattribute__(self, "_runtime")
        async for event in runtime.run(session):
            yield event
