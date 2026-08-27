"""CLI entrypoint for running the MitoCode TUI."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from mitocode import APP_VERSION, MITOCODE_CONFIG_PATH
from mitocode.backend.agent.agent_orchestrator import AgentOrchestrator
from mitocode.backend.chat_provider.chat_provider_openai import ChatProviderOpenAI
from mitocode.backend.chat_provider.scenario_provider import ScenarioProvider
from mitocode.backend.chat_provider.sample_chat_provider import SampleChatProvider
from mitocode.backend.tool.tool_manager import ToolManager
from mitocode.backend.workspace.workspace import Workspace
from mitocode.tui.ui_main import VibeFrontend


def _register_tools(tool_manager: ToolManager, workspace: Workspace) -> None:
    """Register workspace-backed tools (bash, read_file, write_file)."""

    @tool_manager.tool(
        human_doc="Run a shell command inside workspace",
        agent_doc="Execute a shell command in the workspace directory and return output",
        display_action="bash",
    )
    async def bash(command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        output = stdout if stdout else "(no stdout)"
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        return (
            f"$ {command}\n[cwd] {workspace.path}\n[exit_code] {proc.returncode}\n{output}"
        )

    @tool_manager.tool(
        human_doc="Read file content",
        agent_doc="Read UTF-8 text file content from a workspace-relative path",
        display_action="file.read",
        display_param="file_path",
    )
    async def read_file(file_path: str) -> str:
        safe_path = workspace._validate_within_workspace(Path(file_path))
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {safe_path}")
        if not safe_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {safe_path}")
        return await asyncio.to_thread(safe_path.read_text, encoding="utf-8")

    @tool_manager.tool(
        human_doc="Write content to a file",
        agent_doc="Write UTF-8 content into a workspace-relative file path",
        display_action="file.write",
        display_param="file_path",
    )
    async def write_file(file_path: str, content: str) -> str:
        safe_path = workspace._validate_within_workspace(Path(file_path))
        await asyncio.to_thread(safe_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(safe_path.write_text, content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {safe_path}"


def _parse_provider_settings(raw: str) -> Dict[str, Any]:
    """Parse user-provided provider settings JSON string."""
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--provider-settings must be a JSON object")
    return parsed


def _default_provider_kind() -> str:
    return (
        "openai"
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        else "scenario"
    )


def _build_provider(args: argparse.Namespace) -> SampleChatProvider:
    """Build and initialize the provider from CLI arguments."""
    if args.provider == "scenario":
        settings = {"delay": args.speed}
        settings.update(_parse_provider_settings(args.provider_settings))
        return asyncio.run(
            ScenarioProvider.factory(
                default_model=args.model or "scenario-model",
                default_model_list=["scenario-model"],
                default_messages=None,
                default_tools=None,
                stream=True,
                settings=settings,
                delay=settings.get("delay", args.speed),
            )
        )

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenAI provider requires --api-key or OPENAI_API_KEY/OPENROUTER_API_KEY")

    settings: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout,
    }
    if args.base_url:
        settings["base_url"] = args.base_url
    settings.update(_parse_provider_settings(args.provider_settings))

    default_model = args.model or "gpt-4o-mini"
    return asyncio.run(
        ChatProviderOpenAI.factory(
            default_model=default_model,
            default_model_list=[default_model],
            default_messages=None,
            default_tools=None,
            stream=True,
            settings=settings,
        )
    )


def _init_workspace(path_arg: str, name: str) -> Workspace:
    """Initialize or create workspace at the given path."""
    workspace_root = Path(path_arg).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    config_file = workspace_root / MITOCODE_CONFIG_PATH / "config.json"
    if config_file.exists():
        return Workspace.init_from_path(workspace_root)

    return Workspace.new_workspace(
        workspace_path=workspace_root,
        name=name,
        description=f"MitoCode workspace (app v{APP_VERSION})",
        override=False,
    )


def main() -> None:
    """Parse CLI arguments and launch the MitoCode TUI application."""
    parser = argparse.ArgumentParser(description="MitoCode")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.03,
        help="Delay between scenario provider chunks in seconds (default: 0.03)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace root path (default: current directory)",
    )
    parser.add_argument(
        "--workspace-name",
        type=str,
        default="mitocode-workspace",
        help="Workspace name for first-time initialization",
    )
    parser.add_argument(
        "--provider",
        choices=["scenario", "openai"],
        default=_default_provider_kind(),
        help="LLM provider backend",
    )
    parser.add_argument("--model", type=str, default="", help="Provider model name")
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="OpenAI/OpenRouter API key (or use OPENAI_API_KEY / OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible API base URL override",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Provider request timeout in seconds",
    )
    parser.add_argument(
        "--provider-settings",
        type=str,
        default="",
        help="Extra provider settings as JSON object",
    )
    args = parser.parse_args()

    workspace = _init_workspace(args.workspace, args.workspace_name)
    tool_manager = ToolManager()
    _register_tools(tool_manager, workspace)
    provider = _build_provider(args)

    orchestrator = AgentOrchestrator(
        tool_manager=tool_manager, default_provider=provider
    )
    asyncio.run(
        orchestrator.create_agent(
            name="Main",
            description=f"Workspace: {workspace.path}",
        )
    )

    app = VibeFrontend(orchestrator=orchestrator)
    app.run()


if __name__ == "__main__":
    main()
