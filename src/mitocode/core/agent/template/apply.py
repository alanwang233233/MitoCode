"""Replaying an agent template's blocks into a live session.

The orchestrator builds a session and then hands it here, so a template is a
*seed* rather than a constructor argument: :class:`~mitocode.core.agent.agent_session.AgentSession`
keeps its single, tested construction path, and everything a template needs to
change is reachable through the message list it already exposes.

The blocks are rebuilt rather than replayed through the session's streaming
API.  ``append_assistant_delta`` and friends exist to accumulate *streamed*
deltas and carry side effects — a fresh cancellation event, a pending-reasoning
buffer — that a template has no business touching.  Building the three block
types directly is the whole replay and leaves no state behind.

Only the **first** block can replace the session's system prompt.  A template
whose first block is ``text`` with ``role="system"`` is asserting the whole
persona (the shipped ``general`` template is exactly that); one whose system
text appears later is ordinary history, because a persona in the middle of a
conversation is a message, not an identity.

The three message-block types are imported from
:mod:`~mitocode.core.agent.agent_session` under their underscored names on
purpose.  They are the package-level contract -- ``mitocode.core.agent``
re-exports all three -- and the alternative would be a second, parallel block
implementation that converts into the first one at the cost of a copy and a
divergence waiting to happen.
"""

from pathlib import Path
from typing import Any, List, Mapping, Optional

from mitocode.core.agent.agent_session import (
    AgentSession,
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from mitocode.core.agent.template.template import AgentTemplate, render_text


def apply_template(
    session: AgentSession,
    template: AgentTemplate,
    workspace: Optional[Path] = None,
) -> None:
    """Seed *session* with *template*'s blocks.

    Args:
        session: A session that has already been constructed.  Its existing
            system prompt is kept unless the template declares its own as the
            first block, in which case that block replaces it.
        template: The template to replay.  Its blocks were validated at
            discovery time, so a tool call here always has an id and every
            result answers one.
        workspace: Substituted into ``{workspace}`` in text blocks.  ``None``
            leaves the placeholder visible rather than deleting it.

    Returns:
        ``None``.  The session is modified in place.
    """
    blocks: List[Any] = [
        _build_block(raw, workspace) for raw in template.session_autoload
    ]
    if not blocks:
        return

    if _is_system_text(blocks[0]):
        # Replacement, not accumulation: two personas contradict each other,
        # and the template is the one that was chosen for this agent.
        session.messages = [
            block
            for block in session.messages
            if getattr(block, "role", None) != "system"
        ]
        session.system_prompt = blocks[0].message

    session.messages.extend(blocks)


def _is_system_text(block: Any) -> bool:
    """Report whether *block* is a system text message.

    Args:
        block: Any message block.

    Returns:
        ``True`` for a plain text block whose role is ``"system"``.
    """
    return isinstance(block, _PlainTextMessageBlock) and block.role == "system"


def _build_block(raw: Mapping[str, Any], workspace: Optional[Path]) -> Any:
    """Build one internal message block from a validated template block.

    Args:
        raw: A block as normalised by
            :func:`~mitocode.core.agent.template.template_json.parse_template_json`.
        workspace: Substituted into ``{workspace}`` in text blocks.

    Returns:
        The corresponding internal block, ready to append to
        ``session.messages``.

    Raises:
        ValueError: The block declares a kind the parser does not produce.
            Unreachable for a template that came from discovery.
        KeyError: The block is missing a field the parser guarantees.  Only a
            hand-built :class:`AgentTemplate` can reach this: every template
            read from disk was normalised before it was admitted.
    """
    kind = raw.get("kind")
    if kind == "text":
        block = _PlainTextMessageBlock(
            raw["role"], render_text(str(raw["message"]), workspace)
        )
        block.reasoning_content = str(raw.get("reasoning_content", ""))
        return block
    if kind == "tool_call":
        call = _ToolCallMessageBlock(raw["name"], raw["id"])
        # `finish()` parses the argument string, so `parameter` is filled the
        # same way it is for a call that streamed in.
        call.arguments_string = str(raw["arguments_string"])
        call.finish()
        return call
    if kind == "tool_result":
        return _ToolResultMessageBlock(
            raw["tool_call_id"], str(raw["content"]), bool(raw.get("success", True))
        )
    raise ValueError(f"unsupported template block kind: {kind!r}")
