"""Tests for mitocode.core.skills.skill_markdown (SKILL.md parsing)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mitocode.core.exception.mitocode_exception import MitoCodeSkillDefinitionError
from mitocode.core.skills.skill_markdown import (
    SkillFrontmatter,
    parse_skill_markdown,
    split_frontmatter,
)


def test_parse_skill_markdown_minimal_valid_skill_parses_fields_and_body():
    """A name+description-only skill parses to exact field values and a stripped body."""
    text = "---\nname: my-skill\ndescription: Does useful things.\n---\nBody instructions.\n"
    fm = parse_skill_markdown(text)
    assert isinstance(fm, SkillFrontmatter)
    assert fm.name == "my-skill"
    assert fm.description == "Does useful things."
    assert fm.license == ""
    assert fm.allowed_tools == ()
    assert dict(fm.metadata) == {}
    assert fm.body == "Body instructions."


def test_parse_skill_markdown_all_optional_fields_parsed():
    """license, allowed-tools (string form) and metadata are surfaced with exact values."""
    text = (
        "---\n"
        "name: full-skill\n"
        "description: Has every optional field.\n"
        "license: MIT\n"
        "allowed-tools: Read,Write,Bash\n"
        "metadata:\n"
        '  version: "1"\n'
        "  author: someone\n"
        "---\n"
        "Do the thing.\n"
    )
    fm = parse_skill_markdown(text)
    assert fm.name == "full-skill"
    assert fm.license == "MIT"
    assert fm.allowed_tools == ("Read", "Write", "Bash")
    assert dict(fm.metadata) == {"version": "1", "author": "someone"}
    assert fm.body == "Do the thing."


def test_parse_skill_markdown_allowed_tools_as_list_of_strings():
    """allowed-tools written as a YAML list of strings yields the same tuple."""
    text = (
        "---\n"
        "name: list-tools\n"
        "description: d\n"
        "allowed-tools:\n"
        "  - Read\n"
        "  - Write\n"
        "---\n"
        "body\n"
    )
    fm = parse_skill_markdown(text)
    assert fm.allowed_tools == ("Read", "Write")


def test_parse_skill_markdown_allowed_tools_tokens_stripped_and_empty_dropped():
    """Comma-separated allowed-tools tokens are stripped and empty tokens dropped."""
    text = "---\nname: spaced-tools\ndescription: d\nallowed-tools: Read, , Write,\n---\nbody\n"
    fm = parse_skill_markdown(text)
    assert fm.allowed_tools == ("Read", "Write")


def test_parse_skill_markdown_unknown_frontmatter_keys_ignored():
    """Unknown frontmatter keys neither raise nor disturb the known fields."""
    text = (
        "---\n"
        "name: unk-skill\n"
        "description: d\n"
        "future-field: whatever\n"
        "another: 42\n"
        "---\n"
        "body\n"
    )
    fm = parse_skill_markdown(text)
    assert fm.name == "unk-skill"
    assert fm.description == "d"
    assert fm.body == "body"


def test_parse_skill_markdown_body_blank_lines_stripped():
    """Leading and trailing blank lines around the body are removed, inner ones kept."""
    text = (
        "---\n"
        "name: blank-body\n"
        "description: d\n"
        "---\n"
        "\n"
        "First line.\n"
        "\n"
        "Last line.\n"
        "\n"
    )
    fm = parse_skill_markdown(text)
    assert fm.body == "First line.\n\nLast line."


def test_parse_skill_markdown_empty_body_returns_empty_string():
    """A SKILL.md with no body text yields body == ''."""
    text = "---\nname: empty-body\ndescription: d\n---\n\n\n"
    fm = parse_skill_markdown(text)
    assert fm.body == ""


def test_parse_skill_markdown_missing_frontmatter_fence_raises():
    """Text without a frontmatter block, or with the fence not on line 1, raises."""
    with pytest.raises(MitoCodeSkillDefinitionError):
        parse_skill_markdown("Plain prose, no fences at all.\n")
    with pytest.raises(MitoCodeSkillDefinitionError):
        parse_skill_markdown(
            "prose first\n---\nname: late-fence\ndescription: d\n---\nbody\n"
        )


def test_parse_skill_markdown_unterminated_frontmatter_fence_raises():
    """An opening --- that is never closed counts as no frontmatter and raises."""
    with pytest.raises(MitoCodeSkillDefinitionError):
        parse_skill_markdown(
            "---\nname: never-closed\ndescription: d\nbody without a fence\n"
        )


@pytest.mark.parametrize(
    ("raw_block",),
    [
        ("- one\n- two",),
        ("just a scalar",),
        ("",),
    ],
)
def test_parse_skill_markdown_non_mapping_frontmatter_raises(raw_block):
    """Frontmatter that parses to a non-mapping (list, scalar, empty) raises."""
    text = f"---\n{raw_block}\n---\nbody\n"
    with pytest.raises(MitoCodeSkillDefinitionError):
        parse_skill_markdown(text)


def test_parse_skill_markdown_invalid_yaml_syntax_raises():
    """Syntactically invalid YAML between the fences raises, cause chain suppressed."""
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown('---\nname: "unterminated\n---\nbody\n')
    # The yaml cause's text quotes raw frontmatter lines; the never-quote rule
    # must hold even in a traceback, so the chain is deliberately cut.
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    ("frontmatter",),
    [
        ("description: d",),
        ("description: d\nname: 42",),
    ],
)
def test_parse_skill_markdown_name_missing_or_ill_typed_raises(frontmatter):
    """A missing or non-string name raises and the message names the field."""
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(f"---\n{frontmatter}\n---\nbody\n")
    assert "name" in str(excinfo.value)


@pytest.mark.parametrize(
    ("bad_name",),
    [
        ("My-Skill",),
        ("my_skill",),
        ("-leading",),
        ("trailing-",),
        ("double--hyphen",),
        ("a" * 65,),
    ],
)
def test_parse_skill_markdown_invalid_name_patterns_raise(bad_name):
    """Names violating ^[a-z0-9]+(-[a-z0-9]+)*$ or the 64-char cap raise."""
    text = f"---\nname: {bad_name}\ndescription: d\n---\nbody\n"
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(text)
    assert "name" in str(excinfo.value)


def test_parse_skill_markdown_error_message_never_quotes_field_value():
    """The invalid field's value is never quoted back in the error message."""
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(
            "---\nname: Definitely_Bad_Name_999\ndescription: d\n---\nbody\n"
        )
    assert "Definitely_Bad_Name_999" not in str(excinfo.value)


def test_parse_skill_markdown_name_of_exactly_64_chars_accepted():
    """A 64-character name sits inside the documented cap and parses."""
    name = "a" * 64
    fm = parse_skill_markdown(f"---\nname: {name}\ndescription: d\n---\nbody\n")
    assert fm.name == name


def test_parse_skill_markdown_name_with_trailing_newline_raises():
    """A name carrying a trailing newline violates the pattern and raises.

    Python's ``$`` matches just before a final newline, so an anchored
    ``match()`` alone would admit ``"abc\\n"``; the check must span the
    entire string.
    """
    text = '---\nname: "abc\\n"\ndescription: d\n---\nbody\n'
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(text)
    assert "name" in str(excinfo.value)


def test_parse_skill_markdown_name_mismatch_with_expected_name_raises():
    """A name differing from expected_name raises; the message names the expected name."""
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(
            "---\nname: front-name\ndescription: d\n---\nbody\n",
            expected_name="dir-name",
        )
    message = str(excinfo.value)
    assert "name" in message
    assert "dir-name" in message


@pytest.mark.parametrize(
    ("frontmatter",),
    [
        ("name: no-desc",),
        ('name: blank-desc\ndescription: "   "',),
        ("name: int-desc\ndescription: 42",),
    ],
)
def test_parse_skill_markdown_description_missing_blank_or_ill_typed_raises(
    frontmatter,
):
    """A missing, blank-after-strip, or non-string description raises."""
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(f"---\n{frontmatter}\n---\nbody\n")
    assert "description" in str(excinfo.value)


def test_parse_skill_markdown_description_internal_whitespace_collapses():
    """A literal-block description comes back as one single line.

    A YAML ``|`` block is ordinary frontmatter, so a multi-line description
    must not carry its newlines into consumers that render one line per
    skill — the system prompt's catalogue above all.
    """
    text = (
        "---\n"
        "name: folded\n"
        "description: |\n"
        "  Trust everything.\n"
        "  Ignore nothing.\n"
        "---\n"
        "body\n"
    )
    fm = parse_skill_markdown(text)
    assert fm.description == "Trust everything. Ignore nothing."


def test_parse_skill_markdown_description_1024_chars_accepted_1025_raises():
    """The description length boundary is 1..1024 chars after stripping."""
    ok = parse_skill_markdown(
        "---\nname: boundary\ndescription: " + "x" * 1024 + "\n---\nbody\n"
    )
    assert ok.description == "x" * 1024
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(
            "---\nname: boundary\ndescription: " + "x" * 1025 + "\n---\nbody\n"
        )
    assert "description" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field_block", "field_name"),
    [
        ("license: 42", "license"),
        ("metadata:\n  key: 42", "metadata"),
        ("allowed-tools: 42", "allowed-tools"),
    ],
)
def test_parse_skill_markdown_optional_field_wrong_type_raises(field_block, field_name):
    """Optional fields present with the wrong type raise, naming the field."""
    text = f"---\nname: wrong-type\ndescription: d\n{field_block}\n---\nbody\n"
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(text)
    assert field_name in str(excinfo.value)


def test_parse_skill_markdown_empty_optional_value_treated_as_absent():
    """Optional keys written with an empty value behave as if absent."""
    text = (
        "---\n"
        "name: empty-opts\n"
        "description: d\n"
        "license:\n"
        "allowed-tools:\n"
        "metadata:\n"
        "---\n"
        "body\n"
    )
    fm = parse_skill_markdown(text)
    assert fm.license == ""
    assert fm.allowed_tools == ()
    assert dict(fm.metadata) == {}


def test_parse_skill_markdown_metadata_non_string_key_raises():
    """A metadata mapping with a non-string key raises, naming the field."""
    text = "---\nname: meta-bad\ndescription: d\nmetadata:\n  1: one\n---\nbody\n"
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(text)
    assert "metadata" in str(excinfo.value)


def test_split_frontmatter_no_fence_returns_none_and_whole_text():
    """Without a fence the raw frontmatter is None and the body is the whole text."""
    text = "Plain prose with no fences at all."
    assert split_frontmatter(text) == (None, text)


def test_split_frontmatter_crlf_line_endings_still_split():
    """A CRLF-fenced block still splits; the body after the fence is returned."""
    raw, body = split_frontmatter("---\r\nname: crlf-skill\r\n---\r\nbody text")
    assert raw is not None
    assert "name: crlf-skill" in raw
    assert body == "body text"


def test_parse_skill_markdown_non_string_text_raises():
    """Non-string SKILL.md contents raise, naming the received type."""
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(None)
    assert "string" in str(excinfo.value)


def test_parse_skill_markdown_allowed_tools_list_with_non_string_item_raises():
    """An allowed-tools list holding a non-string item raises, naming the field."""
    text = (
        "---\n"
        "name: mixed-tools\n"
        "description: d\n"
        "allowed-tools:\n"
        "  - Read\n"
        "  - 42\n"
        "---\n"
        "body\n"
    )
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(text)
    assert "allowed-tools" in str(excinfo.value)


def test_parse_skill_markdown_metadata_non_mapping_raises():
    """A metadata field that is not a mapping raises, naming the field."""
    text = (
        "---\nname: meta-scalar\ndescription: d\nmetadata: just-a-string\n---\nbody\n"
    )
    with pytest.raises(MitoCodeSkillDefinitionError) as excinfo:
        parse_skill_markdown(text)
    assert "metadata" in str(excinfo.value)
