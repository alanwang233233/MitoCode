"""Backward-compatible agent facade over :class:`AgentSession`."""

# Re-export message block primitives so existing importers keep working.
# pylint: disable=unused-import
from .agent_runtime import AgentRuntime
from .agent_session import (
    AgentSession,
)


class LegacyAgent:
    """
    Backward-compatible wrapper that composes :class:`AgentSession` and
    :class:`AgentRuntime`.

    All stateful attributes (``provider``, ``messages``, ``tools_map``,
    ``tool_schemas``, ``tool_manager``, ``on_end``, and cancellation state are
    transparently forwarded to the underlying :class:`AgentSession` so that
    existing code continues to work without modification.

    The :meth:`run` async-generator delegates execution to the shared
    :class:`AgentRuntime` singleton.
    """

    # Attributes that belong to Agent itself and must NOT be forwarded to the
    # session.
    # _OWN_ATTRS = frozenset({"_session", "_runtime"})
    _OWN_ATTRS = frozenset({"_session"})

    def __init__(self, **settings):
        # Use object.__setattr__ to bypass our custom __setattr__.
        object.__setattr__(self, "_session", AgentSession(**settings))
        # object.__setattr__(self, "_runtime", AgentRuntime.get_instance())

    # ------------------------------------------------------------------
    # Transparent attribute forwarding to the session
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails, so _session and
        # _runtime are never routed through here.
        return getattr(object.__getattribute__(self, "_session"), name)

    def __setattr__(self, name: str, value):
        if name in type(self)._OWN_ATTRS:
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

        The ``on_end`` callback (if any) receives *this* ``Agent`` wrapper as its
        first argument to preserve backward compatibility with callers that relied
        on ``agent.run()`` being available from within the callback.
        """
        session = object.__getattribute__(self, "_session")
        runtime = AgentRuntime

        # Wrap on_end so callers receive the Agent wrapper (which exposes the
        # full public API including run()) rather than the bare AgentSession.
        original_on_end = session.on_end
        if original_on_end is not None:
            agent_self = self

            def _wrap(orig):
                def _agent_on_end(sess, messages):
                    return orig(agent_self, messages)

                return _agent_on_end

            session.on_end = _wrap(original_on_end)

        try:
            async for event in runtime.run(session):
                yield event
        finally:
            session.on_end = original_on_end
