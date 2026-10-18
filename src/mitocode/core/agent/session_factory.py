"""Factory for constructing configured agent sessions."""

from .agent_session import AgentSession


class SessionFactory:
    """
    Factory for creating :class:`AgentSession` instances.

    ``SessionFactory`` centralises session construction so that callers do not
    need to import or instantiate ``AgentSession`` directly.  Default settings
    can be embedded in a factory subclass or instance, and individual
    :meth:`create_session` calls may override any of them.
    """

    def __init__(self, **defaults):
        """
        Args:
            **defaults: Default keyword arguments forwarded to every
                :meth:`create_session` call unless overridden by the caller.
        """
        self._defaults = defaults

    def create_session(self, **settings) -> AgentSession:
        """
        Create and return a new :class:`AgentSession`.

        Keyword arguments override the factory-level defaults set in
        :meth:`__init__`.
        """
        return AgentSession.factory(**{**self._defaults, **settings})
