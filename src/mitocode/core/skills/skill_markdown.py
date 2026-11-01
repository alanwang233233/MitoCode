"""Parsing and validation of SKILL.md files (frontmatter plus body).

A ``SKILL.md`` is a markdown file whose YAML frontmatter declares the skill's
metadata and whose body carries the instructions.  Two entry points serve two
different needs:

* :func:`split_frontmatter` is structural only — it finds the ``---`` fences
  and separates the raw frontmatter text from the body without interpreting
  either, which is what on-demand body reading needs.
* :func:`parse_skill_markdown` additionally validates every documented field
  and is what discovery runs before a skill is admitted to the catalogue.

Validation follows the open Agent Skills convention: ``name`` and
``description`` are required, everything else is optional, and unknown keys
are ignored so that skills written for newer revisions of the format still
load.  Error messages name the offending field and the rule it violated but
never quote the field's value — a value can be arbitrarily long, and the
field plus rule is what an author needs to fix the file.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from mitocode.core.exception.mitocode_exception import MitoCodeSkillDefinitionError

#: Skill names are lowercase alphanumeric words joined by single hyphens.
#: Matched with :func:`re.fullmatch` — an ``$``-anchored :meth:`re.Pattern.match`
#: would admit a single trailing newline, which ``$`` accepts.
_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

#: Longest accepted ``name`` value, in characters.
_MAX_NAME_LENGTH = 64

#: Longest accepted ``description`` value, in characters after stripping.
_MAX_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True)
class SkillFrontmatter:
    """Validated frontmatter of one SKILL.md plus its instruction body.

    Attributes:
        name: ``name`` field; matches ``^[a-z0-9]+(-[a-z0-9]+)*$``, is at most
            64 characters, and equals ``expected_name`` when one was given.
        description: ``description`` field collapsed to one whitespace-normalised
            line; 1..1024 characters after collapsing.
        license: ``license`` field; ``""`` when absent.
        allowed_tools: ``allowed-tools`` field split on commas, tokens
            stripped and empty tokens dropped; empty when absent.
        metadata: ``metadata`` field as a string-to-string mapping; empty when
            absent.
        body: Text below the frontmatter block with leading and trailing
            empty lines stripped.  May be empty.
    """

    name: str
    description: str
    license: str
    allowed_tools: Tuple[str, ...]
    metadata: Mapping[str, str]
    body: str


def split_frontmatter(text: str) -> Tuple[Optional[str], str]:
    """Split raw SKILL.md text structurally, without validating anything.

    The frontmatter block must open on the text's very first line with a line
    that rstrips to ``---`` and close with the next such line; the closing
    delimiter is required, so a block that never closes counts as having no
    frontmatter at all.  CRLF line endings are accepted, and the returned raw
    frontmatter is always joined with ``\\n`` regardless of the input's line
    endings.

    Args:
        text: Full contents of a SKILL.md file.

    Returns:
        ``(raw_frontmatter, body)``.  ``raw_frontmatter`` is the unparsed text
        between the delimiters, or ``None`` when there is no frontmatter block
        (in which case *body* is the whole text, verbatim).  *body* is the
        text after the closing delimiter with leading and trailing empty
        lines stripped.
    """
    # A leading UTF-8 BOM (common from Windows editors) would otherwise make
    # the opening fence invisible and a valid skill read as frontmatter-less.
    lines = text.removeprefix("\ufeff").splitlines()
    if not lines or lines[0].rstrip() != "---":
        return (None, text)

    closing = None
    for index in range(1, len(lines)):
        if lines[index].rstrip() == "---":
            closing = index
            break
    if closing is None:
        return (None, text)

    raw = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1 :]).strip("\n")
    return (raw, body)


def parse_skill_markdown(
    text: str, expected_name: Optional[str] = None
) -> SkillFrontmatter:
    """Parse and fully validate one SKILL.md file's contents.

    Unknown frontmatter keys are ignored for forward compatibility.  An
    optional key written with an empty value — ``license:``, ``allowed-tools:``,
    ``metadata:`` — is treated as absent.  The ``description`` is validated and
    reported as one whitespace-normalised line (surrounding whitespace
    stripped, internal runs collapsed) so consumers can render one line per
    skill; ``name`` is validated as-is, which its pattern already keeps
    whitespace-free.

    Args:
        text: Full contents of a SKILL.md file.
        expected_name: When given, the frontmatter ``name`` must equal it (the
            skill's directory name); a mismatch is an error.

    Returns:
        The validated frontmatter and body.

    Raises:
        MitoCodeSkillDefinitionError: If *text* is not a string, there is no
            frontmatter block, the block is not valid YAML, it does not parse
            to a mapping, a required field (``name``, ``description``) is
            missing or ill-typed, ``name`` is longer than 64 characters or
            violates ``^[a-z0-9]+(-[a-z0-9]+)*$``, ``name`` differs from
            *expected_name*, ``description`` is not 1..1024 characters after
            stripping, or an optional field is present with the wrong type
            (``license`` must be a string; ``allowed-tools`` a comma-separated
            string or a list of strings; ``metadata`` a string-to-string
            mapping).  Messages name the offending field and the rule
            violated; they never quote the field's value.
    """
    if not isinstance(text, str):
        raise MitoCodeSkillDefinitionError(
            "SKILL.md contents must be a string, got " + type(text).__name__
        )

    raw, body = split_frontmatter(text)
    if raw is None:
        raise MitoCodeSkillDefinitionError(
            "no frontmatter block: the file must open with a '---' line and "
            "close with a second '---' line"
        )

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        # The dropped cause's own text quotes raw frontmatter lines, and the
        # never-quote promise covers tracebacks too; the field-plus-rule
        # message carries the diagnosis.
        raise MitoCodeSkillDefinitionError("frontmatter is not valid YAML") from None
    if not isinstance(parsed, dict):
        raise MitoCodeSkillDefinitionError(
            "frontmatter must be a YAML mapping, got " + type(parsed).__name__
        )

    name = _require_name(parsed, expected_name)
    description = _require_description(parsed)

    return SkillFrontmatter(
        name=name,
        description=description,
        license=_optional_license(parsed),
        allowed_tools=_optional_allowed_tools(parsed),
        metadata=_optional_metadata(parsed),
        body=body,
    )


def _require_name(parsed: Mapping[str, Any], expected_name: Optional[str]) -> str:
    """Extract and validate the required ``name`` field.

    Args:
        parsed: The frontmatter mapping.
        expected_name: The directory name the field must equal, when known.

    Returns:
        The validated name.

    Raises:
        MitoCodeSkillDefinitionError: If the field is missing, ill-typed,
            outside 1..64 characters, violates the name pattern, or differs
            from *expected_name*.
    """
    if "name" not in parsed:
        raise MitoCodeSkillDefinitionError(
            "frontmatter is missing the required 'name' field"
        )
    name = parsed["name"]
    if not isinstance(name, str):
        raise MitoCodeSkillDefinitionError("frontmatter field 'name' must be a string")
    if not name or len(name) > _MAX_NAME_LENGTH or not _NAME_PATTERN.fullmatch(name):
        raise MitoCodeSkillDefinitionError(
            f"frontmatter field 'name' must be 1..{_MAX_NAME_LENGTH} characters "
            "of lowercase letters, digits and single hyphens, with no leading, "
            "trailing or doubled hyphen"
        )
    if expected_name is not None and name != expected_name:
        raise MitoCodeSkillDefinitionError(
            "frontmatter field 'name' must match the skill directory name "
            f"{expected_name!r}"
        )
    return name


def _require_description(parsed: Mapping[str, Any]) -> str:
    """Extract and validate the required ``description`` field.

    Args:
        parsed: The frontmatter mapping.

    Returns:
        The description, collapsed to one whitespace-normalised line.

    Raises:
        MitoCodeSkillDefinitionError: If the field is missing, ill-typed,
        blank after stripping, or longer than the cap.
    """
    if "description" not in parsed:
        raise MitoCodeSkillDefinitionError(
            "frontmatter is missing the required 'description' field"
        )
    description = parsed["description"]
    if not isinstance(description, str):
        raise MitoCodeSkillDefinitionError(
            "frontmatter field 'description' must be a string"
        )
    # Collapse internal whitespace runs (a YAML ``|`` block is ordinary
    # frontmatter): a description is one catalogue line everywhere it is
    # rendered, most critically in the system prompt's skills section.
    collapsed = " ".join(description.split())
    if not collapsed:
        raise MitoCodeSkillDefinitionError(
            "frontmatter field 'description' must not be empty"
        )
    if len(collapsed) > _MAX_DESCRIPTION_LENGTH:
        raise MitoCodeSkillDefinitionError(
            "frontmatter field 'description' must be at most "
            f"{_MAX_DESCRIPTION_LENGTH} characters"
        )
    return collapsed


def _optional_license(parsed: Mapping[str, Any]) -> str:
    """Extract the optional ``license`` field, treating an empty value as absent.

    Args:
        parsed: The frontmatter mapping.

    Returns:
        The license, or ``""`` when absent.

    Raises:
        MitoCodeSkillDefinitionError: If the field is present with a
            non-string value.
    """
    raw = parsed.get("license")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise MitoCodeSkillDefinitionError(
            "frontmatter field 'license' must be a string"
        )
    return raw


def _optional_allowed_tools(parsed: Mapping[str, Any]) -> Tuple[str, ...]:
    """Extract the optional ``allowed-tools`` field.

    Args:
        parsed: The frontmatter mapping.

    Returns:
        One token per tool, with surrounding whitespace stripped and empty
        tokens dropped; empty when the field is absent.

    Raises:
        MitoCodeSkillDefinitionError: If the field is neither a
            comma-separated string nor a list of strings.
    """
    raw = parsed.get("allowed-tools")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(token.strip() for token in raw.split(",") if token.strip())
    if isinstance(raw, list):
        if not all(isinstance(item, str) for item in raw):
            raise MitoCodeSkillDefinitionError(
                "frontmatter field 'allowed-tools' must be a comma-separated "
                "string or a list of strings"
            )
        return tuple(item.strip() for item in raw if item.strip())
    raise MitoCodeSkillDefinitionError(
        "frontmatter field 'allowed-tools' must be a comma-separated string "
        "or a list of strings"
    )


def _optional_metadata(parsed: Mapping[str, Any]) -> Dict[str, str]:
    """Extract the optional ``metadata`` field as an owned string-to-string dict.

    Args:
        parsed: The frontmatter mapping.

    Returns:
        A fresh mapping of the metadata entries; empty when the field is
        absent.

    Raises:
        MitoCodeSkillDefinitionError: If the field is present but is not a
            mapping, or holds a key or value that is not a string.
    """
    raw = parsed.get("metadata")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MitoCodeSkillDefinitionError(
            "frontmatter field 'metadata' must be a mapping of string to string"
        )
    metadata: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise MitoCodeSkillDefinitionError(
                "frontmatter field 'metadata' must map strings to strings"
            )
        metadata[key] = value
    return metadata
