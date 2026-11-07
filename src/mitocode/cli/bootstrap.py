"""Startup helpers for the MitoCode command line interface.

The first half of this module holds the startup logic that has no dependency on
the terminal UI: resolving the workspace and reading the MCP server
configuration.  Both are pure functions over ``pathlib``/JSON input, so they can
be exercised without a console, a provider or a live MCP connection.  The second
half, :func:`bootstrap`, assembles the runtime in the one order the design
permits, from the workspace down to the first agent session.

Two rules shape every error raised here:

* a message may name a file or directory, but never a configuration value
  (credentials, headers, environment entries, workspace contents);
* the caller decides whether an error is fatal.  A malformed *default* MCP file
  only disables MCP, while a malformed *explicit* one is the user's own
  pointer and therefore a startup failure.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from loguru import logger as loguru_logger

from mitocode import APP_VERSION, MITOCODE_CONFIG_PATH
from mitocode.cli.approval import CliApprover
from mitocode.cli.args import CliArgs
from mitocode.cli.config import (
    KNOWN_REASONING_EFFORTS,
    ProviderConfig,
    resolve_provider_config,
)
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.cli.ui import CliUI
from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.chat_provider.chat_provider_openai import ChatProviderOpenAI
from mitocode.core.exception.mitocode_exception import (
    MitoCodeAppTooOldError,
    MitoCodeMcpConfigError,
    MitoCodeMigrateError,
    MitoCodeWorkspaceDumpError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceTooOldError,
)
from mitocode.core.mcp.mcp_config import McpServerConfig
from mitocode.core.mcp.mcp_manager import McpManager
from mitocode.core.skills.skill import Skill, SkillRoot
from mitocode.core.skills.skills_manager import SkillsManager
from mitocode.core.tool.builtin import ToolContext, register_builtin_tools
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.migration import BasicMigration
from mitocode.core.workspace.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from mitocode.core.agent.agent_session import AgentSession

from mitocode.cli.sessions import _agent_record, restore_agent_messages

CONFIG_FILENAME = "config.json"
"""Workspace configuration file name inside the ``.mitocode`` directory."""

DEFAULT_MCP_CONFIG_RELATIVE_PATH = Path(MITOCODE_CONFIG_PATH) / "mcp.json"
"""Default MCP configuration file, relative to the workspace root."""

SKILLS_DIR_NAME = "skills"
"""Skill directories' name under the workspace root:
``<workspace>/.mitocode/skills``."""

DEFAULT_PROVIDER_BASE_URL = "https://api.openai.com/v1"
"""Endpoint passed when the credential file names none.

The OpenAI SDK falls back to ``OPENAI_BASE_URL`` when its ``base_url`` argument
is ``None``; the CLI promises that provider settings come only from
``provider.json``, so an explicit default keeps the environment out of it.
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are MitoCode, an interactive coding agent running in the user's terminal.
You help with engineering tasks in the workspace at {workspace}.

- Read files before changing them. Prefer Edit for targeted changes and Write
  for new files; never rewrite a whole file to make one small change.
- Use tools for facts: do not invent file contents, command output, or APIs.
- Run tests or commands when they materially verify your work.
- Keep answers concise and concrete; use Markdown.
- Stay inside the workspace. Operations outside it need user approval.
- If a task is ambiguous or destructive, ask before acting.
"""
"""Role instructions seeded into every CLI session, with a ``{workspace}`` slot."""

SKILLS_PROMPT_SECTION = """

## Skills

Installed skills extend you with packaged, task-specific instruction sets.
When the user's request matches a skill's description below, call the
Skill tool with that skill's name to load its full instructions, then
follow them before acting:

{skills}
"""
"""Appended to the system prompt when at least one skill was discovered, with
the ``- name: description`` catalogue in the ``{skills}`` slot."""

_ResultT = TypeVar("_ResultT")


async def _awaited(value: Union[_ResultT, Awaitable[_ResultT]]) -> _ResultT:
    """Await a sync-or-async return value.

    ``Workspace`` and ``BasicMigration`` return a coroutine when an event loop is
    already running and a plain value otherwise. The CLI always calls them from
    inside its event loop (an ``async`` function cannot be reached without one),
    yet spelling that out as a helper keeps the union honest for type checkers
    that cannot narrow it from the call site.

    Args:
        value: A plain result, or an awaitable resolving to one.

    Returns:
        The resolved value.
    """
    if asyncio.iscoroutine(value):
        return await cast(Awaitable[_ResultT], value)
    return cast(_ResultT, value)


@dataclass(frozen=True)
class WorkspaceResolution:
    """Outcome of resolving the workspace at startup.

    Attributes:
        workspace: The loaded workspace; always present, so every
            ``ToolContext`` gets a root directory boundary.
        migrated_from: Configuration version before an automatic migration, or
            ``None`` when no migration happened.
    """

    workspace: Workspace
    migrated_from: Optional[int] = None


def ensure_workspace(
    workspace_path: Union[str, Path], *, confirm: Callable[[Path], bool]
) -> bool:
    """Create the workspace configuration when it is missing.

    Called before logging is configured, so that a declined prompt leaves
    nothing on disk. An existing ``config.json`` is left untouched; a missing
    one is created only after *confirm* agrees, so a read-only checkout can
    still decline cleanly.

    Args:
        workspace_path: Workspace root the CLI was pointed at.
        confirm: Receives the workspace root and returns whether to create it.

    Returns:
        ``True`` when a usable workspace configuration exists after the call
        (already there or just created), ``False`` when the user declined.

    Raises:
        MitoCodeCliConfigError: The workspace directory does not exist or the
            configuration cannot be written. Messages name the path and never
            the contents.
    """
    root = Path(workspace_path)
    if not root.exists():
        raise MitoCodeCliConfigError(
            f"The workspace directory '{root}' does not exist."
        )
    if not root.is_dir():
        raise MitoCodeCliConfigError(f"The workspace path '{root}' is not a directory.")
    config_file = root / MITOCODE_CONFIG_PATH / CONFIG_FILENAME
    if config_file.is_file():
        return True
    if not confirm(root):
        return False

    # ``Path(".").name`` is empty, yet ``-w .`` names the current directory;
    # making the path absolute normalises that without resolving symlinks.
    workspace_name = root.absolute().name
    if not workspace_name:
        raise MitoCodeCliConfigError(f"Cannot determine a workspace name for '{root}'.")

    config_dir = root / MITOCODE_CONFIG_PATH
    try:
        # Encrypted or shared content never lives here, but the directory also
        # holds logs, history and credentials, so a newly created one starts
        # private. An existing one keeps whatever mode the user chose.
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        workspace = Workspace(root, workspace_name, APP_VERSION)
        result = workspace.dump()
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except (OSError, MitoCodeWorkspaceDumpError) as exc:
        raise MitoCodeCliConfigError(
            f"Cannot create the workspace configuration at '{config_dir}': "
            f"{getattr(exc, 'strerror', None) or 'I/O error'}."
        ) from exc
    if not result:
        raise MitoCodeCliConfigError(
            f"Cannot create the workspace configuration at '{config_dir}'."
        )
    return True


async def resolve_workspace(path: Path) -> WorkspaceResolution:
    """Load the workspace at *path*, migrating an outdated configuration.

    :func:`ensure_workspace` has already guaranteed that
    ``.mitocode/config.json`` exists when the CLI calls this, so a missing file
    is a startup error rather than an in-memory workspace.

    Args:
        path: Workspace root directory.

    Returns:
        The resolution describing the workspace and which version it was
        migrated from.

    Raises:
        MitoCodeCliConfigError: The configuration is missing, unreadable,
            invalid, written by a newer MitoCode, or could not be migrated.
            The message names the configuration file but never its contents.
    """
    workspace_path = Path(path)
    config_file = workspace_path / MITOCODE_CONFIG_PATH / CONFIG_FILENAME

    if not config_file.is_file():
        raise _config_error(
            config_file,
            "it does not exist; create a workspace and run mitocode again",
        )

    try:
        workspace = await _awaited(Workspace.init_from_path(workspace_path))
    except MitoCodeWorkspaceTooOldError:
        return await _migrate_workspace(workspace_path, config_file)
    except MitoCodeAppTooOldError as exc:
        raise _config_error(
            config_file,
            "it was created by a newer version of MitoCode; upgrade MitoCode "
            "to open it",
        ) from exc
    except (MitoCodeWorkspaceInitError, UnicodeDecodeError) as exc:
        raise _config_error(
            config_file,
            "it is unreadable, not valid JSON, or missing required fields",
        ) from exc

    return WorkspaceResolution(workspace=workspace, migrated_from=None)


async def _migrate_workspace(
    workspace_path: Path, config_file: Path
) -> WorkspaceResolution:
    """Migrate an outdated workspace and reload it.

    Args:
        workspace_path: Workspace root directory.
        config_file: Configuration file that triggered the migration, used in
            error messages.

    Returns:
        The resolution for the migrated workspace, with ``migrated_from`` set to
        the version found before the migration.

    Raises:
        MitoCodeCliConfigError: The migration failed or the migrated
            configuration could not be reloaded.
    """
    previous_version = _config_version(config_file)
    try:
        await _awaited(BasicMigration.migrate(workspace_path))
        workspace = await _awaited(Workspace.init_from_path(workspace_path))
    except (
        MitoCodeMigrateError,
        MitoCodeWorkspaceInitError,
        MitoCodeAppTooOldError,
        UnicodeDecodeError,
    ) as exc:
        raise _config_error(config_file, "automatic version migration failed") from exc

    loguru_logger.debug(
        "Migrated workspace at {} from version {} to {}",
        workspace_path,
        previous_version,
        APP_VERSION,
    )
    return WorkspaceResolution(workspace=workspace, migrated_from=previous_version)


def _config_version(config_file: Path) -> Optional[int]:
    """Read the ``version`` field of a workspace configuration file.

    Only used after :class:`MitoCodeWorkspaceTooOldError` reported a valid but
    outdated version, so a failure here means the file changed underneath the
    caller; the version is then simply reported as unknown.

    Args:
        config_file: Configuration file to read.

    Returns:
        The configuration version, or ``None`` when it cannot be determined.
    """
    try:
        document = json.loads(config_file.read_text(encoding="utf-8"))
        return int(document["version"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        loguru_logger.debug(
            "Could not determine the previous workspace version: {}", type(exc).__name__
        )
        return None


def _config_error(config_file: Path, reason: str) -> MitoCodeCliConfigError:
    """Build a value-free workspace configuration error.

    Args:
        config_file: Configuration file the error is about.
        reason: Short explanation of what is wrong with it.

    Returns:
        The error to raise, naming the file and what the user can do about it.
    """
    return MitoCodeCliConfigError(
        f"Cannot open the workspace configuration at '{config_file}': {reason}.\n"
        "Fix or remove that file and run mitocode again."
    )


def load_mcp_servers(
    args: CliArgs, workspace: Workspace
) -> Optional[List[McpServerConfig]]:
    """Read the MCP server configuration for one CLI run.

    ``--no-mcp`` skips reading entirely. Otherwise an explicit ``--mcp-config``
    path wins over the default ``<workspace>/.mitocode/mcp.json``. A missing
    default file means "no MCP servers" and returns ``None``; a missing explicit
    file is the user's own pointer and raises.

    Args:
        args: Parsed command line arguments.
        workspace: Resolved workspace, used to locate the default file.

    Returns:
        The validated server configurations in document order, or ``None`` when
        MCP is disabled or no default file exists.

    Raises:
        MitoCodeMcpConfigError: The explicit file is missing or unreadable, or
            either file is not valid JSON or not a valid ``mcpServers`` document.
            The message names the file and whether it is the explicit or default
            one, but never a header, environment or credential value.
    """
    if args.no_mcp:
        return None

    explicit_path = args.mcp_config
    if explicit_path is not None:
        path = Path(explicit_path)
        label = "explicit MCP configuration file (--mcp-config)"
        is_required = True
    else:
        path = workspace.path / DEFAULT_MCP_CONFIG_RELATIVE_PATH
        label = "default MCP configuration file"
        is_required = False

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if not is_required:
            return None
        raise MitoCodeMcpConfigError(f"The {label} '{path}' does not exist.") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise MitoCodeMcpConfigError(
            f"The {label} '{path}' could not be read as UTF-8 text."
        ) from exc

    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise MitoCodeMcpConfigError(
            f"The {label} '{path}' is not valid JSON."
        ) from exc

    try:
        return McpServerConfig.parse_config(document)
    except MitoCodeMcpConfigError as exc:
        raise MitoCodeMcpConfigError(f"The {label} '{path}' is invalid: {exc}") from exc


@dataclass(frozen=True)
class Bootstrapped:  # pylint: disable=too-many-instance-attributes
    """Everything a started CLI run needs, assembled in the design's order.

    Attributes:
        args: The parsed command line arguments the run started with.
        resolution: Workspace resolution and any automatic version migration.
        provider_config: Validated provider settings, secret value excluded from
            ``repr``.
        tool_manager: Registry holding the built-in and MCP tools.
        tool_context: World the built-in tools read, with the orchestrator
            attached.
        provider: The initialized chat provider.
        orchestrator: Session registry and event router.
        main_session: The interactive ``Main`` session.
        mcp_manager: Live MCP manager, or ``None`` when MCP is disabled or could
            not be configured.
        mcp_error: Message describing why MCP is off, or ``""``.
        skills_manager: Skills manager over the workspace root, with its
            startup discovery already run.  Always present; its catalogue may
            be empty.
        restore_succeeded: Whether a *restore* document was applied to the
            orchestrator; ``False`` when no record was passed or the restore
            fell back to a fresh session.
        restore_error: Exception text of a failed restore, or ``""``.
    """

    args: CliArgs
    resolution: WorkspaceResolution
    provider_config: ProviderConfig
    tool_manager: ToolManager
    tool_context: ToolContext
    provider: ChatProviderOpenAI
    orchestrator: AgentOrchestrator
    main_session: "AgentSession"
    mcp_manager: Optional[McpManager]
    skills_manager: SkillsManager
    mcp_error: str = ""
    restore_succeeded: bool = False
    restore_error: str = ""


def build_system_prompt(workspace: Path, skills: Sequence[Skill] = ()) -> str:
    """Render the default system prompt for one workspace.

    Args:
        workspace: Workspace root. It is resolved first so the model always
            receives an absolute path, whatever the user typed.
        skills: The discovered skill catalogue.  Empty — the default — leaves
            the prompt byte-identical to the base template's render; a
            non-empty catalogue appends the skills section naming each one.

    Returns:
        :data:`SYSTEM_PROMPT_TEMPLATE` with the absolute workspace path filled
        into its ``{workspace}`` slot, plus :data:`SKILLS_PROMPT_SECTION` when
        *skills* is non-empty.
    """
    prompt = SYSTEM_PROMPT_TEMPLATE.format(workspace=str(Path(workspace).resolve()))
    if not skills:
        return prompt
    # One line per skill, whatever a Skill value built outside the parser
    # carries: a description with newlines would otherwise turn the catalogue
    # into uncontrolled multi-line prompt text.
    catalogue = "\n".join(
        f"- {skill.name}: {' '.join(skill.description.split())}" for skill in skills
    )
    return prompt + SKILLS_PROMPT_SECTION.format(skills=catalogue)


def skills_roots(workspace: Workspace) -> Tuple[SkillRoot, ...]:
    """Return the skill roots one CLI run scans.

    Only the workspace's own ``.mitocode/skills`` is scanned. A missing
    directory contributes nothing.

    Args:
        workspace: Resolved workspace whose root the scan derives from.

    Returns:
        The labelled roots in precedence order.
    """
    return (
        SkillRoot(
            label="workspace",
            path=workspace.path / MITOCODE_CONFIG_PATH / SKILLS_DIR_NAME,
        ),
    )


def build_skills_manager(workspace: Workspace) -> SkillsManager:
    """Build the skills manager over :func:`skills_roots`.

    Construction performs no disk access; the caller runs
    :meth:`~mitocode.core.skills.skills_manager.SkillsManager.discover`.

    Args:
        workspace: Resolved workspace the first root derives from.

    Returns:
        The manager, roots in precedence order.
    """
    return SkillsManager(list(skills_roots(workspace)))


async def bootstrap(
    args: CliArgs, ui: CliUI, approver: CliApprover, restore: Optional[dict] = None
) -> Bootstrapped:
    """Assemble a CLI run from arguments to an interactive main session.

    The order is a constraint, not a preference: the workspace has to exist
    before provider settings can be read from it, the tool registry has to exist
    before the orchestrator that owns it, MCP has to finish starting before
    any :class:`~mitocode.core.agent.agent_session.AgentSession` is created
    because a session snapshots the registry in its constructor, and skills
    have to finish being discovered before the orchestrator is built because
    the system prompt carries their catalogue.

    A malformed *default* ``<workspace>/.mitocode/mcp.json`` only disables MCP
    and is reported through ``ui.warn``; a malformed explicit ``--mcp-config``
    file is the user's own pointer and aborts startup.

    When *restore* carries a saved session document, the main session is not
    created but rebuilt from it: the registry is reset, the tree is restored
    with its original ids and the message blocks are filled in.  Any failure
    is logged, the registry is cleaned again and a fresh ``Main`` session is
    created instead; the failure never aborts startup.

    Args:
        args: Parsed command line arguments.
        ui: Console surface used for startup warnings.
        approver: Approval policy installed in the tool context.
        restore: A validated session document from ``SessionStore.load``, or
            ``None`` to start a brand-new session.

    Returns:
        The started components, including the main session, and whether the
        restore succeeded.

    Raises:
        MitoCodeCliConfigError: The workspace or provider configuration is
            invalid, the provider cannot be initialized, or the explicit MCP
            configuration file is unusable. Messages never contain the API key.
    """
    resolution = await resolve_workspace(args.workspace)
    loguru_logger.debug(
        "Workspace resolved at {} (migrated_from={})",
        resolution.workspace.path,
        resolution.migrated_from,
    )

    provider_config = resolve_provider_config(resolution.workspace.path)

    tool_manager = ToolManager()
    tool_context = ToolContext(workspace=resolution.workspace, approver=approver)
    register_builtin_tools(tool_manager, tool_context)

    provider = await _create_provider(provider_config, ui)
    loguru_logger.debug("Provider initialized with model {}", provider_config.model)

    mcp_manager, mcp_error = await _start_mcp(
        args, resolution.workspace, tool_manager, ui
    )

    try:
        # Skill discovery sits inside the same guard as the orchestrator
        # build: it is the step that makes the system prompt, so any failure
        # here — however unreachable the per-entry isolation makes it — must
        # not leave a live MCP manager orphaned behind a raising startup.
        skills_manager = build_skills_manager(resolution.workspace)
        await skills_manager.discover()
        tool_context.skills_manager = skills_manager

        orchestrator = AgentOrchestrator(
            tool_manager=tool_manager,
            default_provider=provider,
            system_prompt=build_system_prompt(
                resolution.workspace.path, skills=skills_manager.list_skills()
            ),
            # resolve_provider_config validated the key as non-blank, so the
            # orchestrator can scrub the literal secret out of every provider
            # failure's log line and error event.
            redact_secrets=(provider_config.api_key,),
        )
        tool_context.attach_orchestrator(orchestrator)

        main_session: Optional[AgentSession] = None
        restore_succeeded = False
        restore_error = ""
        if restore is not None:
            try:
                await orchestrator.reset_sessions()
                sessions = await orchestrator.restore_tree(restore.get("agents") or [])
                for agent_id, session in sessions.items():
                    agent_record = _agent_record(restore, agent_id)
                    if agent_record is not None:
                        restore_agent_messages(session, agent_record)
                main_session = sessions[restore["main_agent_id"]]
                restore_succeeded = True
            except Exception as exc:  # pylint: disable=broad-exception-caught
                restore_error = str(exc) or type(exc).__name__
                loguru_logger.exception("Session restore failed")
                await orchestrator.reset_sessions()
        if main_session is None:
            main_session = await orchestrator.create_session(
                name="Main", description="Interactive session"
            )
    except BaseException:
        # MCP is live at this point and its manager would otherwise be lost
        # with the exception, orphaning every server subprocess it started.
        await _stop_mcp_after_failure(mcp_manager)
        raise
    loguru_logger.debug("Main session created with id {}", main_session.agent_id)

    return Bootstrapped(
        args=args,
        resolution=resolution,
        provider_config=provider_config,
        tool_manager=tool_manager,
        tool_context=tool_context,
        provider=provider,
        orchestrator=orchestrator,
        main_session=main_session,
        mcp_manager=mcp_manager,
        skills_manager=skills_manager,
        mcp_error=mcp_error,
        restore_succeeded=restore_succeeded,
        restore_error=restore_error,
    )


async def _create_provider(config: ProviderConfig, ui: CliUI) -> ChatProviderOpenAI:
    """Initialize the chat provider and keep the configured model usable.

    ``init()`` replaces the seeded model list with the endpoint's live list. The
    configured model may be missing from that list, so it is appended after the
    factory returns; otherwise the first request would be rejected for a model
    the user explicitly named.

    Args:
        config: Validated provider settings.
        ui: Console surface used for the "not advertised" warning.

    Returns:
        The initialized provider.

    Raises:
        MitoCodeCliConfigError: The factory rejected the settings or could not
            reach the endpoint. The API key is scrubbed from the message.
    """
    settings = {
        "api_key": config.api_key,
        # An explicit endpoint keeps the provider SDK from reading
        # OPENAI_BASE_URL when provider.json omitted base_url.
        "base_url": config.base_url or DEFAULT_PROVIDER_BASE_URL,
        "timeout": config.timeout,
        "reasoning_effort": config.reasoning_effort,
    }
    try:
        provider = cast(
            ChatProviderOpenAI,
            await ChatProviderOpenAI.factory(
                default_model=config.model,
                default_model_list=[config.model],
                stream=True,
                settings=settings,
            ),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise MitoCodeCliConfigError(_scrub_secret(str(exc), config.api_key)) from exc

    model_list = list(provider.model_list or [])
    if config.model not in model_list:
        model_list.append(config.model)
        ui.warn(
            f"model '{config.model}' is not advertised by the endpoint; "
            "using it anyway"
        )
    provider.model_list = model_list
    if (
        config.reasoning_effort is not None
        and config.reasoning_effort not in KNOWN_REASONING_EFFORTS
    ):
        ui.warn(
            "reasoning_effort is not a known tier "
            "(none, minimal, low, medium, high, xhigh, max); "
            "passing it through anyway"
        )
    return provider


async def _stop_mcp_after_failure(manager: Optional[McpManager]) -> None:
    """Stop an MCP manager whose startup owner is about to raise.

    The stop coroutine is shielded because it releases subprocesses, and it is
    awaited while the caller still holds the exception. A failure here is
    logged only: the original exception must reach the caller unchanged.

    Args:
        manager: Live manager started earlier in :func:`bootstrap`, or ``None``
            when MCP is disabled.
    """
    if manager is None:
        return
    try:
        await asyncio.shield(manager.stop())
    except BaseException:  # pylint: disable=broad-exception-caught
        loguru_logger.exception("MCP shutdown failed after bootstrap error")


async def _start_mcp(
    args: CliArgs, workspace: Workspace, tool_manager: ToolManager, ui: CliUI
) -> "tuple[Optional[McpManager], str]":
    """Read the MCP configuration and start every configured server.

    A manager whose ``start()`` raises -- including a cancellation that lands
    while servers are still connecting -- is stopped before the exception
    propagates, so no stdio subprocess outlives the failed startup.

    Args:
        args: Parsed command line arguments, carrying ``--no-mcp`` and
            ``--mcp-config``.
        workspace: Resolved workspace, used to find the default file.
        tool_manager: Registry the servers' tools are published into.
        ui: Console surface used when the default file is unusable.

    Returns:
        ``(manager, error)``: the live manager or ``None``, plus the message
        explaining why MCP is off (``""`` when it is on or simply absent).

    Raises:
        MitoCodeCliConfigError: The explicit ``--mcp-config`` file is missing,
            unreadable or invalid. The message names the file and never a
            configuration value.
    """
    try:
        servers = load_mcp_servers(args, workspace)
        if servers is None:
            return None, ""
        manager = McpManager(servers, tool_manager=tool_manager)
    except MitoCodeMcpConfigError as exc:
        if args.mcp_config is not None:
            raise MitoCodeCliConfigError(str(exc)) from exc
        ui.warn(f"MCP is disabled: {exc}")
        return None, str(exc)

    try:
        await manager.start()
    except BaseException:
        # Some servers may already be live when the exception or cancellation
        # arrives; without this teardown the local manager reference would be
        # lost and every subprocess it started would be orphaned.
        await _stop_mcp_after_failure(manager)
        raise

    return manager, ""


def _scrub_secret(message: str, secret: str) -> str:
    """Replace the literal API key in *message* with ``***``.

    Provider libraries usually quote only the endpoint, but a validation error
    can echo the whole settings mapping. Blank secrets are ignored so an empty
    value cannot turn every string into a match.

    Args:
        message: Text from an exception, possibly containing the key.
        secret: The configured API key.

    Returns:
        The message with every occurrence of *secret* replaced.
    """
    if secret:
        return message.replace(secret, "***")
    return message


__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_MCP_CONFIG_RELATIVE_PATH",
    "DEFAULT_PROVIDER_BASE_URL",
    "SKILLS_DIR_NAME",
    "SKILLS_PROMPT_SECTION",
    "SYSTEM_PROMPT_TEMPLATE",
    "Bootstrapped",
    "WorkspaceResolution",
    "bootstrap",
    "build_skills_manager",
    "build_system_prompt",
    "ensure_workspace",
    "load_mcp_servers",
    "resolve_workspace",
    "skills_roots",
]
