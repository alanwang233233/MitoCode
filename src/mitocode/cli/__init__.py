"""Interactive command-line interface for MitoCode.

The package implements the ``mitocode cli`` command: a Claude Code-style
REPL with a fixed input area and streaming Rich output, slash commands for
provider configuration and MCP management, a default system prompt for
every agent session, and an interactive approver for out-of-workspace
operations.  The Typer entry point in :mod:`mitocode.__main__` is a thin
shim; every piece of logic lives here so it stays under coverage.
"""
