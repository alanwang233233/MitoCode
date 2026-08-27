"""Command line interface for MitoCode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Coroutine, TypeVar, cast

from mitocode import APP_VERSION
from mitocode.backend.agent.agent_orchestrator import AgentOrchestrator
from mitocode.backend.chat_provider.chat_provider_openai import ChatProviderOpenAI
from mitocode.backend.chat_provider.scenario_provider import ScenarioProvider
from mitocode.backend.tool.tool_manager import ToolManager
from mitocode.backend.workspace.migration import BasicMigration
from mitocode.backend.workspace.workspace import Workspace
from mitocode.tui.ui_main import VibeFrontend

T = TypeVar("T")


def _resolve_awaitable(value: T | Awaitable[T]) -> T:
    if asyncio.iscoroutine(value):
        return asyncio.run(cast(Coroutine[Any, Any, T], value))
    return cast(T, value)


def _ensure_workspace(path: Path) -> Workspace:
    try:
        workspace = Workspace.init_from_path(path)
    except Exception:
        workspace = Workspace.new_workspace(path, path.name, override=False)
    return _resolve_awaitable(workspace)


def _register_tools(tool_manager: ToolManager, workspace: Workspace) -> None:
    @tool_manager.tool(
        human_doc="Run a shell command",
        agent_doc="Execute a bash command in the workspace and return stdout/stderr",
        display_action="bash",
    )
    async def bash(command: str) -> str:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode("utf-8", errors="replace")
        err_output = stderr.decode("utf-8", errors="replace")
        combined = (
            output if process.returncode == 0 else f"{output}{err_output}".strip()
        )
        if process.returncode != 0:
            return f"[exit_code={process.returncode}]\n{combined}".strip()
        return combined

    @tool_manager.tool(
        human_doc="Read file content",
        agent_doc="Read UTF-8 text from a file in the workspace",
        display_action="file.read",
        display_param="file_path",
    )
    async def read_file(file_path: str) -> str:
        target = workspace._validate_within_workspace(Path(file_path))
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return target.read_text(encoding="utf-8")

    @tool_manager.tool(
        human_doc="Write content to a file",
        agent_doc="Write UTF-8 text to a file in the workspace",
        display_action="file.write",
        display_param="file_path",
    )
    async def write_file(file_path: str, content: str) -> str:
        target = workspace._validate_within_workspace(Path(file_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {file_path}"


def _build_provider(args: argparse.Namespace):
    if args.provider == "scenario":
        return ScenarioProvider(delay=args.speed)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    settings = {
        "api_key": api_key,
        "base_url": args.base_url,
        "timeout": args.timeout,
    }
    provider = ChatProviderOpenAI(
        default_model=args.model,
        default_model_list=[args.model],
        settings=settings,
    )
    asyncio.run(provider.init())
    return provider


def run_tui(args: argparse.Namespace) -> int:
    workspace = _ensure_workspace(Path(args.workspace).resolve())
    tool_manager = ToolManager()
    _register_tools(tool_manager, workspace)

    provider = _build_provider(args)
    orchestrator = AgentOrchestrator(
        tool_manager=tool_manager, default_provider=provider
    )
    asyncio.run(
        orchestrator.create_agent(
            name=args.agent_name,
            description=args.agent_description,
        )
    )

    app = VibeFrontend(orchestrator=orchestrator)
    app.run()
    return 0


def _workspace_init(args: argparse.Namespace) -> int:
    ws = _resolve_awaitable(
        Workspace.new_workspace(
            workspace_path=Path(args.path).resolve(),
            name=args.name,
            description=args.description,
            override=args.override,
        )
    )
    print(f"Workspace created: {ws.path}")
    return 0


def _workspace_info(args: argparse.Namespace) -> int:
    ws = _resolve_awaitable(Workspace.init_from_path(Path(args.path).resolve()))
    print(json.dumps({"path": str(ws.path), **ws.config}, ensure_ascii=False, indent=2))
    return 0


def _workspace_validate(args: argparse.Namespace) -> int:
    ws = _resolve_awaitable(Workspace.init_from_path(Path(args.path).resolve()))
    valid = _resolve_awaitable(ws.validate())
    print("valid" if valid else "invalid")
    return 0 if valid else 1


def _workspace_reload(args: argparse.Namespace) -> int:
    ws = _resolve_awaitable(Workspace.init_from_path(Path(args.path).resolve()))
    ok = _resolve_awaitable(ws.reload())
    print("reloaded" if ok else "reload failed")
    return 0 if ok else 1


def _workspace_outdated(args: argparse.Namespace) -> int:
    ws = _resolve_awaitable(Workspace.init_from_path(Path(args.path).resolve()))
    outdated = _resolve_awaitable(ws.is_outdated())
    print("true" if outdated else "false")
    return 0


def _workspace_migrate(args: argparse.Namespace) -> int:
    result = _resolve_awaitable(BasicMigration.migrate(Path(args.path).resolve()))
    print("migrated" if result else "migration failed")
    return 0 if result else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MitoCode CLI")
    parser.add_argument(
        "--version", action="version", version=f"mitocode {APP_VERSION}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    tui = subparsers.add_parser("tui", help="Launch the MitoCode TUI")
    tui.add_argument("--provider", choices=["scenario", "openai"], default="scenario")
    tui.add_argument("--workspace", default=".", help="Workspace root path")
    tui.add_argument("--speed", type=float, default=0.03)
    tui.add_argument("--api-key", default="")
    tui.add_argument("--base-url", default="")
    tui.add_argument("--timeout", type=float, default=60)
    tui.add_argument("--model", default="gpt-4o-mini")
    tui.add_argument("--agent-name", default="Main")
    tui.add_argument("--agent-description", default="Chief orchestrator")
    tui.set_defaults(func=run_tui)

    workspace = subparsers.add_parser("workspace", help="Workspace operations")
    ws_subparsers = workspace.add_subparsers(dest="workspace_command", required=True)

    ws_init = ws_subparsers.add_parser("init", help="Create workspace metadata")
    ws_init.add_argument("--path", default=".")
    ws_init.add_argument("--name", default="workspace")
    ws_init.add_argument("--description", default="")
    ws_init.add_argument(
        "--override", action=argparse.BooleanOptionalAction, default=True
    )
    ws_init.set_defaults(func=_workspace_init)

    ws_info = ws_subparsers.add_parser("info", help="Show workspace configuration")
    ws_info.add_argument("--path", default=".")
    ws_info.set_defaults(func=_workspace_info)

    ws_validate = ws_subparsers.add_parser("validate", help="Validate workspace")
    ws_validate.add_argument("--path", default=".")
    ws_validate.set_defaults(func=_workspace_validate)

    ws_reload = ws_subparsers.add_parser(
        "reload", help="Reload workspace configuration"
    )
    ws_reload.add_argument("--path", default=".")
    ws_reload.set_defaults(func=_workspace_reload)

    ws_outdated = ws_subparsers.add_parser(
        "is-outdated", help="Check if in-memory config differs from disk"
    )
    ws_outdated.add_argument("--path", default=".")
    ws_outdated.set_defaults(func=_workspace_outdated)

    ws_migrate = ws_subparsers.add_parser("migrate", help="Migrate workspace version")
    ws_migrate.add_argument("--path", default=".")
    ws_migrate.set_defaults(func=_workspace_migrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
