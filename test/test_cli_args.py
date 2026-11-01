"""Tests for the CLI argument parsing and help rendering."""

import argparse
import dataclasses
import importlib.metadata
import os
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.args import (
    CliArgs,
    build_parser,
    parse_args,
    render_help,
    version_string,
)
from mitocode.cli.errors import MitoCodeCliConfigError, MitoCodeCliError
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError


def test_defaults():
    args = parse_args([])
    assert args.workspace == Path.cwd()
    assert args.permission_mode == "ask"
    assert args.initial_prompt is None


def test_repr_hides_api_key():
    args = parse_args(["--api-key", "super-secret"])
    assert "super-secret" not in repr(args)


def test_default_optional_fields_are_unset():
    args = parse_args([])
    assert args.model is None
    assert args.api_key is None
    assert args.base_url is None
    assert args.timeout is None
    assert args.mcp_config is None
    assert args.no_mcp is False
    assert args.verbose is False
    assert args.debug is False
    assert args.show_help is False
    assert args.show_version is False


def test_flags_override_and_dangerous_skip_is_allow(tmp_path):
    args = parse_args(
        [
            "-w",
            str(tmp_path),
            "-m",
            "m1",
            "--api-key",
            "k",
            "--base-url",
            "http://x",
            "--timeout",
            "12",
            "--dangerously-skip-permissions",
            "hello",
        ]
    )
    assert args.workspace == tmp_path
    assert (args.model, args.api_key, args.base_url, args.timeout) == (
        "m1",
        "k",
        "http://x",
        12.0,
    )
    assert args.permission_mode == "allow"
    assert args.initial_prompt == "hello"


@pytest.mark.parametrize("mode", ["ask", "allow", "deny"])
def test_permission_mode_accepts_documented_values(mode):
    assert parse_args(["--permission-mode", mode]).permission_mode == mode


@pytest.mark.parametrize("mode", ["deny", "ask"])
def test_permission_mode_conflicts_with_skip_flag(mode):
    with pytest.raises(SystemExit):
        parse_args(["--permission-mode", mode, "--dangerously-skip-permissions"])


def test_unknown_permission_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--permission-mode", "banana"])


def test_help_and_version_are_flags_not_exits():
    assert parse_args(["--help"]).show_help is True
    assert parse_args(["--version"]).show_version is True
    assert parse_args(["-h"]).show_help is True


def test_mcp_and_logging_flags(tmp_path):
    config = tmp_path / "mcp.json"
    args = parse_args(["--mcp-config", str(config), "--no-mcp", "--verbose", "--debug"])
    assert args.mcp_config == config
    assert args.no_mcp is True
    assert args.verbose is True
    assert args.debug is True


def test_cli_args_is_frozen():
    args = parse_args([])
    assert isinstance(args, CliArgs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.permission_mode = "deny"


def test_parser_disables_builtin_help_and_exit():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--unknown-flag"])


def test_render_help_lists_key_options():
    buffer = StringIO()
    render_help(Console(file=buffer, width=100, force_terminal=False))
    text = buffer.getvalue()
    for needle in ("--workspace", "--model", "--permission-mode", "--mcp-config"):
        assert needle in text


def test_render_help_accepts_custom_parser():
    buffer = StringIO()
    render_help(
        Console(file=buffer, width=100, force_terminal=False),
        build_parser(),
    )
    assert "Usage" in buffer.getvalue() or "usage" in buffer.getvalue()


def test_render_help_skips_suppressed_options():
    parser = build_parser()
    parser.add_argument(
        "--hidden-flag",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    buffer = StringIO()
    render_help(Console(file=buffer, width=100, force_terminal=False), parser)
    assert "--hidden-flag" not in buffer.getvalue()


def test_version_string_uses_installed_distribution(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "9.9.9" if name == "mitocode" else "?",
    )
    assert version_string() == "mitocode 9.9.9"


def test_version_string_falls_back_when_not_installed(monkeypatch):
    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    assert version_string() == "mitocode 0.1.0"


def test_cli_error_hierarchy():
    assert issubclass(MitoCodeCliError, MitoCodeBaseError)
    assert issubclass(MitoCodeCliConfigError, MitoCodeCliError)
    error = MitoCodeCliConfigError("bad key")
    assert error.message == "bad key"


def test_cli_package_exports_only_main():
    import mitocode.cli as cli

    assert cli.__all__ == ["main"]
    assert callable(cli.main)


def test_reasoning_effort_flag_parses_and_defaults_to_none():
    args = parse_args(["--reasoning-effort", "HIGH"])
    assert args.reasoning_effort == "HIGH"
    assert parse_args([]).reasoning_effort is None
