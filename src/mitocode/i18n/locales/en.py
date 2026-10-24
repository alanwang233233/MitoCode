"""English message catalog for MitoCode's user-facing strings.

Keys are dot-separated and grouped by the surface that renders them
(``cli.*`` for the command line).  Every key defined here must be defined in
every other catalog; :func:`mitocode.i18n.translator.t` falls back to this
catalog when another locale is missing a key.
"""

CATALOG = {
    "cli.app_help": "MitoCode command line interface.",
    "cli.cli.help": "Start the interactive CLI chat mode.",
    "cli.webui.help": "Placeholder for the web UI.",
    "cli.not_implemented": "'{name}' is not implemented yet.",
    "cli.tui.help": (
        "Launch the terminal UI. The TUI is deprecated and no longer maintained."
    ),
    "cli.tui.deprecated_warning": (
        "Warning: the built-in TUI is deprecated and no longer maintained."
    ),
    "cli.tui.launching_in": "Launching in {seconds} seconds...",
    "cli.tui.loaded_env_file": "Loaded provider settings from {path}",
    "cli.api_key_missing": (
        "No API key found. Set MITOCODE_API_KEY (or OPENAI_API_KEY) in your "
        "environment, or put it in a .env file in the current directory."
    ),
    "cli.tui.invalid_timeout": (
        "Invalid MITOCODE_TIMEOUT value '{value}'; using the default of "
        "{default} seconds."
    ),
    "cli.tui.launch_error": "Failed to launch MitoCode: {error}",
    "cli.tui.workspace_error": ("Failed to prepare the workspace at {path}: {error}"),
    # Interactive CLI (mitocode cli)
    "cli.cli.welcome": "MitoCode CLI — type /help for available commands.",
    "cli.cli.no_provider_hint": (
        "No provider configured yet. Run /connect (or /login) to set one up; "
        "/help lists commands."
    ),
    "cli.cli.exit_hint": "Press Ctrl+D or type /exit to quit.",
    "cli.cli.interrupted": "Interrupted.",
    "cli.cli.queued": "Queued: {text}",
    "cli.cli.bye": "Bye.",
    "cli.cli.agent_label": "#{id} {name}",
    "cli.cli.launch_error": "Failed to launch the CLI: {error}",
    "cli.cli.provider_swapped": "Provider switched to {model} ({base_url}).",
    "cli.cli.provider_busy": (
        "The agent is still running; try again after this turn finishes."
    ),
    "cli.cli.corrupt_secret": (
        "Ignoring unreadable provider config at {path}: {error}"
    ),
    # git bootstrap
    "cli.git.missing": (
        "git was not found on PATH; install it so MitoCode can track "
        "workspace changes."
    ),
    "cli.git.initialized": "Initialized a new git repository at {path}.",
    "cli.git.init_failed": "Failed to run 'git init' at {path}: {error}",
    # slash commands
    "cli.cmd.help.help": "Show the available commands.",
    "cli.cmd.clear.help": "Clear the current conversation context.",
    "cli.cmd.exit.help": "Exit the CLI.",
    "cli.cmd.connect.help": (
        "Configure the chat provider and save it to .mitocode/secret/provider.json."
    ),
    "cli.cmd.login.help": "Alias for /connect.",
    "cli.cmd.model.help": "Show or switch the active model.",
    "cli.cmd.mcp.help": "Manage MCP servers: /mcp list, /mcp add, /mcp remove.",
    "cli.cmd.unknown": "Unknown command: {name} — type /help for the list.",
    "cli.cmd.cleared": "Context cleared.",
    "cli.cmd.connect.ask_api_key": "API key (empty to keep current/default): ",
    "cli.cmd.connect.ask_base_url": "Base URL (empty for {default}): ",
    "cli.cmd.connect.ask_model": "Model (empty for {default}): ",
    "cli.cmd.connect.ask_timeout": "Timeout in seconds (empty for {default}): ",
    "cli.cmd.connect.saved": "Provider configuration saved to {path}.",
    "cli.cmd.connect.gitignored": (
        "Added {path} to .gitignore so secrets stay out of git."
    ),
    "cli.cmd.model.current": "Current model: {model}",
    "cli.cmd.model.list_header": "Available models:",
    "cli.cmd.model.pick": "Model name or number (empty to cancel): ",
    "cli.cmd.model.switched": "Model switched to {model}.",
    "cli.cmd.model.not_available": "Model '{model}' is not in the available model list.",
    "cli.cmd.model.session_only": (
        "No provider.json exists yet, so this switch is session-only; run "
        "/connect to persist it."
    ),
    "cli.cmd.model.invalid_number": "'{value}' is not a valid number.",
    "cli.cmd.model.fetch_failed": "Could not fetch the model list: {error}",
    "cli.cmd.model.empty_list": "The provider returned an empty model list.",
    # MCP commands
    "cli.cmd.mcp.none": "No MCP servers configured ({path}).",
    "cli.cmd.mcp.list_header": "MCP servers:",
    "cli.cmd.mcp.ask_name": "Server name: ",
    "cli.cmd.mcp.ask_kind": "Kind (stdio or http): ",
    "cli.cmd.mcp.ask_command": "Command to run (stdio): ",
    "cli.cmd.mcp.ask_url": "Server URL (http): ",
    "cli.cmd.mcp.invalid_kind": "Kind must be 'stdio' or 'http'.",
    "cli.cmd.mcp.added": "Added MCP server '{name}' to {path}.",
    "cli.cmd.mcp.removed": "Removed MCP server '{name}'.",
    "cli.cmd.mcp.not_found": "No MCP server named '{name}'.",
    "cli.cmd.mcp.ask_remove": "Server name to remove: ",
    "cli.cmd.mcp.status": "{name}: {status}",
    "cli.cmd.mcp.started": (
        "MCP: {connected} connected, {failed} failed, {disabled} disabled."
    ),
    "cli.cmd.mcp.load_error": "Could not load MCP config: {error}",
    # approval prompts
    "cli.approval.prompt": "{tool} wants to: {reason}",
    "cli.approval.choices": "[1] Allow once   [2] Always allow this   [3] Deny",
    "cli.approval.pick": "Choice [1-3]: ",
    "cli.approval.invalid": "Please enter 1, 2 or 3.",
    "cli.approval.confirm_rule": "Remember '{rule}' for future requests? [y/N]: ",
    "cli.approval.rule_saved": "Rule saved: {rule}",
    "cli.approval.denied": "Denied.",
}
