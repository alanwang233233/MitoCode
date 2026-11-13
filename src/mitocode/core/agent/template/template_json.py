"""Parsing and validation of ``template.json`` files.

A template file is a JSON object carrying the template's identity and an
optional ``session_autoload`` list of message blocks.  The blocks are replayed
into a new session before the agent's first turn, so this module is the boundary
between a hand-written file and a provider request: everything it accepts must
be a history the provider will accept back.

Two rules carry most of the weight.

**The pairing is positional, but only when it is unambiguous.**  Hand-written
templates routinely omit a ``tool_call``'s ``id`` while its ``tool_result``
still names one from some earlier run.  A call with no id is therefore given
``call_<position>``, and a result whose id matches no open call is bound to the
single open one.  Two open calls and an unknown id is refused rather than
guessed: inventing a transcript produces a request that fails remotely, with an
error the author cannot act on.

**A broken block rejects the whole template.**  Skills are forgiving per entry
because each one is independent; the blocks of a template are not.  Half a
history is worse than none — a result whose call was dropped, or a user turn
stranded between a call and its answer, is a request the provider rejects on
the first turn, long after the file looked fine.

Error messages name the offending field, the block number and the rule broken,
but never quote the file's content: a value can be arbitrarily long, and field
plus rule is what an author needs to fix it.
"""

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mitocode.core.agent.template.template import TemplateDocument
from mitocode.core.exception.mitocode_exception import (
    MitoCodeTemplateDefinitionError,
)

#: Template names are lowercase alphanumeric words joined by single separators
#: (``-`` or ``_``).  Underscores are admitted because the templates written
#: before this format existed are named ``plan_agent`` and
#: ``deep_research_agent``; hyphens because every other identifier in this
#: repository is hyphenated.
_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")

#: Longest accepted ``name`` value, in characters.
_MAX_NAME_LENGTH = 64

#: Longest accepted ``description`` value, in characters after collapsing.
_MAX_DESCRIPTION_LENGTH = 1024

#: Largest accepted ``template.json``, checked from ``st_size`` before anything
#: is opened.  A template is replayed into every session it creates, so an
#: oversized one is a cost paid per spawn, not once.
MAX_TEMPLATE_FILE_BYTES = 1024 * 1024

#: The roles a ``text`` block may declare.
_ROLES = ("system", "user", "assistant")

#: The block kinds a template may carry.
_KINDS = ("text", "tool_call", "tool_result")


def parse_template_json(
    text: str, expected_name: Optional[str] = None
) -> TemplateDocument:
    """Parse and fully validate one ``template.json`` file's contents.

    Unknown top-level keys are ignored for forward compatibility; a document is
    otherwise all-or-nothing, with the first problem raised.

    Args:
        text: Full contents of a ``template.json`` file.
        expected_name: When given, the ``name`` field must equal it (the
            template's directory name); a mismatch is an error.

    Returns:
        The validated document, with every block normalised: a ``tool_call``
        always carries ``id``, ``arguments_string`` and ``parameter``; a
        ``tool_result`` always carries the ``tool_call_id`` it answers.

    Raises:
        MitoCodeTemplateDefinitionError: If *text* is not a string, is not
            valid JSON, is not a JSON object, lacks a valid ``name`` or
            ``description``, or carries a ``session_autoload`` list that is not
            a self-consistent message history.
    """
    if not isinstance(text, str):
        raise MitoCodeTemplateDefinitionError(
            "template.json contents must be a string, got " + type(text).__name__
        )

    parsed = _load_document(text)
    name = _require_name(parsed, expected_name)
    description = _require_description(parsed)
    blocks = _load_blocks(parsed.get("session_autoload"))
    return TemplateDocument(name=name, description=description, session_autoload=blocks)


def _load_document(text: str) -> Mapping[str, Any]:
    """Decode *text* into a JSON object.

    Args:
        text: Full file contents.

    Returns:
        The decoded mapping.

    Raises:
        MitoCodeTemplateDefinitionError: The text is not valid JSON, or decodes
            to something other than an object.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # The decoder's own message quotes the offending line, and the
        # never-quote promise covers that too; "not valid JSON" plus the file
        # name the caller reports is the actionable part.
        raise MitoCodeTemplateDefinitionError(
            "template.json is not valid JSON"
        ) from None
    if not isinstance(parsed, dict):
        raise MitoCodeTemplateDefinitionError(
            "template.json must be a JSON object, got " + type(parsed).__name__
        )
    return parsed


def _require_name(parsed: Mapping[str, Any], expected_name: Optional[str]) -> str:
    """Extract and validate the required ``name`` field.

    Args:
        parsed: The decoded document.
        expected_name: The directory name the field must equal, when known.

    Returns:
        The validated name.

    Raises:
        MitoCodeTemplateDefinitionError: The field is missing, ill-typed,
            outside 1..64 characters, violates the name pattern, or differs
            from *expected_name*.
    """
    if "name" not in parsed:
        raise MitoCodeTemplateDefinitionError(
            "template.json is missing the required 'name' field"
        )
    name = parsed["name"]
    if not isinstance(name, str):
        raise MitoCodeTemplateDefinitionError(
            "template.json field 'name' must be a string"
        )
    if not name or len(name) > _MAX_NAME_LENGTH or not _NAME_PATTERN.fullmatch(name):
        raise MitoCodeTemplateDefinitionError(
            f"template.json field 'name' must be 1..{_MAX_NAME_LENGTH} characters "
            "of lowercase letters, digits and single hyphens or underscores, "
            "with no leading, trailing or doubled separator"
        )
    if expected_name is not None and name != expected_name:
        raise MitoCodeTemplateDefinitionError(
            "template.json field 'name' must match the template directory name "
            f"{expected_name!r}"
        )
    return name


def _require_description(parsed: Mapping[str, Any]) -> str:
    """Extract and validate the required ``description`` field.

    Args:
        parsed: The decoded document.

    Returns:
        The description, collapsed to one whitespace-normalised line.

    Raises:
        MitoCodeTemplateDefinitionError: The field is missing, ill-typed,
            blank after collapsing, or longer than the cap.
    """
    if "description" not in parsed:
        raise MitoCodeTemplateDefinitionError(
            "template.json is missing the required 'description' field"
        )
    description = parsed["description"]
    if not isinstance(description, str):
        raise MitoCodeTemplateDefinitionError(
            "template.json field 'description' must be a string"
        )
    collapsed = " ".join(description.split())
    if not collapsed:
        raise MitoCodeTemplateDefinitionError(
            "template.json field 'description' must not be empty"
        )
    if len(collapsed) > _MAX_DESCRIPTION_LENGTH:
        raise MitoCodeTemplateDefinitionError(
            "template.json field 'description' must be at most "
            f"{_MAX_DESCRIPTION_LENGTH} characters"
        )
    return collapsed


def _load_blocks(raw: Any) -> Tuple[Mapping[str, Any], ...]:
    """Validate and normalise the ``session_autoload`` list.

    Args:
        raw: The document's ``session_autoload`` value, or ``None`` when absent.

    Returns:
        The normalised blocks in file order.

    Raises:
        MitoCodeTemplateDefinitionError: The value is not a list, a block is
            not an object, a block is invalid, or the blocks do not form a
            history the provider would accept.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise MitoCodeTemplateDefinitionError(
            "template.json field 'session_autoload' must be an array of message "
            "blocks"
        )

    blocks: List[Dict[str, Any]] = []
    open_calls: List[int] = []
    used_ids: set[str] = set()
    for position, block in enumerate(raw, start=1):
        if not isinstance(block, Mapping):
            raise MitoCodeTemplateDefinitionError(
                f"session_autoload block {position} must be a JSON object"
            )
        kind = block.get("kind")
        if kind == "text":
            _refuse_text_before_a_result(position, blocks, open_calls)
            blocks.append(_text_block(block, position))
        elif kind == "tool_call":
            call = _call_block(block, position)
            # A duplicate id is the one inconsistency positional binding cannot
            # repair: two calls would share an identity, and the provider would
            # either refuse the turn or answer the wrong one.
            if call["id"] in used_ids:
                raise MitoCodeTemplateDefinitionError(
                    f"session_autoload block {position}: tool_call id "
                    f"'{call['id']}' is already used by an earlier call; a "
                    "generated id looks like call_<block number>, so give this "
                    "call an id of its own"
                )
            used_ids.add(call["id"])
            blocks.append(call)
            open_calls.append(len(blocks) - 1)
        elif kind == "tool_result":
            blocks.append(_result_block(block, position, blocks, open_calls))
        else:
            raise MitoCodeTemplateDefinitionError(
                f"session_autoload block {position} declares an unsupported "
                "kind; expected one of " + ", ".join(f"'{name}'" for name in _KINDS)
            )

    if open_calls:
        first = open_calls[0]
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {first + 1}: tool_call '{blocks[first]['name']}' "
            "is never answered by a tool_result, so the template's history is "
            "incomplete"
        )
    return tuple(blocks)


def _refuse_text_before_a_result(
    position: int, blocks: List[Dict[str, Any]], open_calls: List[int]
) -> None:
    """Refuse a text block that would strand an open tool call.

    The provider requires a tool call's results to follow it directly, so a
    system/user/assistant message in between produces a request that cannot be
    sent.  Refusing here names both blocks; discovering it at send time names
    neither.

    Args:
        position: 1-based position of the text block being validated.
        blocks: Blocks normalised so far.
        open_calls: Indices into *blocks* of calls still awaiting a result.

    Raises:
        MitoCodeTemplateDefinitionError: A call is still open.
    """
    if not open_calls:
        return
    caller = blocks[open_calls[0]]
    raise MitoCodeTemplateDefinitionError(
        f"session_autoload block {position}: a text block cannot appear before "
        f"the tool_call '{caller['name']}' at block {open_calls[0] + 1} is "
        "answered by a tool_result"
    )


def _text_block(block: Mapping[str, Any], position: int) -> Dict[str, Any]:
    """Validate and normalise one ``text`` block.

    Args:
        block: The raw block.
        position: 1-based position, used in error messages.

    Returns:
        The normalised block.

    Raises:
        MitoCodeTemplateDefinitionError: The role is missing or unknown, or the
            message or reasoning is not a string.
    """
    role = block.get("role")
    if role not in _ROLES:
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: a text block requires 'role' to "
            "be one of 'system', 'user' or 'assistant'"
        )
    message = block.get("message")
    if not isinstance(message, str):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: a text block requires a string "
            "'message'"
        )
    reasoning = block.get("reasoning_content", "")
    if not isinstance(reasoning, str):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'reasoning_content' must be a "
            "string"
        )
    return {
        "kind": "text",
        "role": role,
        "message": message,
        "reasoning_content": reasoning,
    }


def _call_block(block: Mapping[str, Any], position: int) -> Dict[str, Any]:
    """Validate and normalise one ``tool_call`` block.

    Args:
        block: The raw block.
        position: 1-based position; also the id-suffix for a call that names
            no id of its own.

    Returns:
        The normalised block, carrying ``id``, ``arguments_string`` and the
        parsed ``parameter`` mapping.

    Raises:
        MitoCodeTemplateDefinitionError: The name is missing, the arguments are
            absent or not a JSON object, or the id is not a string.
    """
    name = block.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: a tool_call block requires a "
            "non-empty string 'name'"
        )

    call_id = block.get("id")
    if call_id is None:
        call_id = f"call_{position}"
    elif not isinstance(call_id, str) or not call_id:
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'id' must be a non-empty string"
        )

    arguments = _require_arguments(block, position)
    return {
        "kind": "tool_call",
        "name": name,
        "id": call_id,
        "arguments_string": arguments[0],
        "parameter": arguments[1],
    }


def _require_arguments(
    block: Mapping[str, Any], position: int
) -> Tuple[str, Dict[str, Any]]:
    """Return the call's argument string and its parsed mapping.

    ``arguments_string`` wins when both are present: it is what the provider
    receives, so the ``parameter`` copy is derived from it rather than trusted
    separately.  A template that names neither has nothing to send.

    Args:
        block: The raw ``tool_call`` block.
        position: 1-based position, used in error messages.

    Returns:
        ``(arguments_string, parameter)``.

    Raises:
        MitoCodeTemplateDefinitionError: The argument string is not a JSON
            object, or neither form is present.
    """
    raw_string = block.get("arguments_string")
    parameter = block.get("parameter")
    if parameter is not None and not isinstance(parameter, Mapping):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'parameter' must be a JSON object"
        )

    if raw_string is None:
        if parameter is None:
            raise MitoCodeTemplateDefinitionError(
                f"session_autoload block {position}: a tool_call block requires "
                "'arguments_string' or 'parameter'"
            )
        raw_string = json.dumps(dict(parameter), ensure_ascii=False)

    if not isinstance(raw_string, str):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'arguments_string' must be a string"
        )
    try:
        parsed = json.loads(raw_string)
    except json.JSONDecodeError:
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'arguments_string' must be a "
            "JSON object"
        ) from None
    if not isinstance(parsed, dict):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'arguments_string' must be a "
            "JSON object"
        )
    return raw_string, parsed


def _result_block(
    block: Mapping[str, Any],
    position: int,
    blocks: List[Dict[str, Any]],
    open_calls: List[int],
) -> Dict[str, Any]:
    """Validate one ``tool_result`` block and bind it to the call it answers.

    The binding is by id when the id names an open call, and by position when
    exactly one call is open — the hand-written-template case, where the id was
    copied from an earlier run or left out entirely.

    Args:
        block: The raw block.
        position: 1-based position, used in error messages.
        blocks: Blocks normalised so far, read for the open calls' ids.
        open_calls: Indices into *blocks* of calls still awaiting a result.
            The bound call is removed from it.

    Returns:
        The normalised block, carrying the id of the call it answers.

    Raises:
        MitoCodeTemplateDefinitionError: The content or success flag is
            ill-typed, or no unambiguous open call can be bound.
    """
    content = block.get("content")
    if not isinstance(content, str):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: a tool_result block requires a "
            "string 'content'"
        )
    success = block.get("success", True)
    if not isinstance(success, bool):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'success' must be true or false"
        )

    given_id = block.get("tool_call_id")
    if given_id is not None and not isinstance(given_id, str):
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: 'tool_call_id' must be a string"
        )

    matched = _match_open_call(given_id, position, blocks, open_calls)
    open_calls.remove(matched)
    return {
        "kind": "tool_result",
        "tool_call_id": blocks[matched]["id"],
        "content": content,
        "success": success,
    }


def _match_open_call(
    given_id: Optional[str],
    position: int,
    blocks: List[Dict[str, Any]],
    open_calls: List[int],
) -> int:
    """Return the index of the open call a result answers.

    Args:
        given_id: The id the file supplied, or ``None``.
        position: 1-based position, used in error messages.
        blocks: Blocks normalised so far.
        open_calls: Indices into *blocks* of calls still awaiting a result.

    Returns:
        The index of the bound call.

    Raises:
        MitoCodeTemplateDefinitionError: No call is open, or more than one is
            open and *given_id* names none of them.
    """
    if given_id:
        for index in open_calls:
            if blocks[index]["id"] == given_id:
                return index
    if not open_calls:
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: this tool_result does not answer "
            "any open tool_call; every result must follow the call it answers"
        )
    if len(open_calls) > 1:
        raise MitoCodeTemplateDefinitionError(
            f"session_autoload block {position}: this tool_result names no open "
            "tool_call and more than one call is open, so the pairing is "
            "ambiguous; give the result a 'tool_call_id' matching one of them"
        )
    return open_calls[0]
