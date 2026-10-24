"""Provider configuration for the interactive CLI.

Resolution precedence, per field: ``.mitocode/secret/provider.json`` wins
over the environment, the environment wins over built-in defaults. The
``MITOCODE_*`` names take priority over the ``OPENAI_*`` spellings, exactly
as the ``tui`` entry point does.

Writing goes through :func:`save_provider_config`, which updates
``.gitignore`` **before** the file lands on disk, so an API key never
exists at a trackable path for even a moment. The file itself is created
with ``0600`` permissions.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from rich.console import Console

from mitocode.i18n import t

#: Fallback provider configuration, mirroring ``__main__``.
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 60

#: Environment variables read per field, in priority order.
API_KEY_ENV_VARS = ("MITOCODE_API_KEY", "OPENAI_API_KEY")
BASE_URL_ENV_VARS = ("MITOCODE_BASE_URL", "OPENAI_BASE_URL")
MODEL_ENV_VARS = ("MITOCODE_MODEL",)
TIMEOUT_ENV_VARS = ("MITOCODE_TIMEOUT",)

#: Location of the persisted provider configuration, workspace-relative.
PROVIDER_FILE_RELATIVE = Path(".mitocode") / "secret" / "provider.json"

#: Lines in a .gitignore that already cover the secret directory.
GITIGNORE_COVERS = {".mitocode", ".mitocode/", ".mitocode/secret", ".mitocode/secret/"}

#: Type of the injected input port: renders the prompt text, returns input.
InputReader = Callable[[str], Awaitable[str]]


@dataclass
class ProviderConfig:
    """A fully resolved provider configuration.

    Attributes:
        api_key: Provider API key.
        base_url: OpenAI-compatible base URL.
        model: Default model id.
        timeout: Request timeout in seconds.
    """

    api_key: str
    base_url: str
    model: str
    timeout: int


def _first_env(names: Tuple[str, ...]) -> Optional[str]:
    """Return the first non-blank value among the named environment variables.

    Args:
        names: Variable names in priority order.

    Returns:
        The trimmed value of the first set, non-blank variable, or None.
    """
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _coerce_timeout(raw: Any, console: Optional[Console]) -> Optional[int]:
    """Coerce a raw timeout value to a positive int, warning when invalid.

    Args:
        raw: The raw value (from file, environment or user input).
        console: Console for the localized warning; None stays silent.

    Returns:
        The timeout in seconds, or None when *raw* is absent or invalid
        (the caller then applies the default).
    """
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = -1
    if value <= 0:
        if console is not None:
            console.print(
                t(
                    "cli.tui.invalid_timeout",
                    value=raw,
                    default=DEFAULT_TIMEOUT_SECONDS,
                ),
                style="yellow",
            )
        return None
    return value


def _read_provider_file(root: Path, console: Optional[Console]) -> Dict[str, Any]:
    """Read the persisted provider file, degrading to an empty dict.

    Args:
        root: Workspace root directory.
        console: Console for the unreadable-file warning; None stays silent.

    Returns:
        The parsed mapping, or ``{}`` when the file is missing, unreadable
        or not a JSON object.
    """
    path = root / PROVIDER_FILE_RELATIVE
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if console is not None:
            console.print(
                t("cli.cli.corrupt_secret", path=path, error=exc), style="yellow"
            )
        return {}
    return data if isinstance(data, dict) else {}


def resolve_provider_config(
    root: Path, console: Optional[Console] = None
) -> Optional[ProviderConfig]:
    """Resolve the provider configuration for the workspace at *root*.

    Args:
        root: Workspace root directory.
        console: Console for degradation warnings; None stays silent.

    Returns:
        The resolved configuration, or None when no API key is available
        from either the file or the environment.
    """
    stored = _read_provider_file(root, console)

    api_key = stored.get("api_key") or _first_env(API_KEY_ENV_VARS)
    if not api_key:
        return None

    base_url = stored.get("base_url") or _first_env(BASE_URL_ENV_VARS)
    model = stored.get("model") or _first_env(MODEL_ENV_VARS)
    timeout = _coerce_timeout(stored.get("timeout"), console)
    if timeout is None:
        timeout = _coerce_timeout(_first_env(TIMEOUT_ENV_VARS), console)
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_SECONDS

    return ProviderConfig(
        api_key=str(api_key),
        base_url=str(base_url or DEFAULT_BASE_URL),
        model=str(model or DEFAULT_MODEL),
        timeout=timeout,
    )


def ensure_secret_gitignored(root: Path) -> bool:
    """Make sure ``.mitocode/secret/`` is ignored by git at *root*.

    Args:
        root: Workspace root directory; its ``.gitignore`` is created or
            appended to when needed.

    Returns:
        True when a line had to be appended, False when coverage already
        existed. Never raises: an unwritable ``.gitignore`` surfaces as an
        ``OSError`` to the caller, which decides how loud to be.
    """
    gitignore = root / ".gitignore"
    try:
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    except OSError:
        return False
    if any(line.strip() in GITIGNORE_COVERS for line in existing.splitlines()):
        return False
    with gitignore.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(".mitocode/secret/\n")
    return True


def save_provider_config(root: Path, config: ProviderConfig, console: Console) -> Path:
    """Persist *config* to the workspace secret file, gitignore first.

    Args:
        root: Workspace root directory.
        config: The configuration to store.
        console: Console the gitignore notice is printed to.

    Returns:
        The path of the written file.

    Raises:
        OSError: When the secret directory or file cannot be written.
    """
    if ensure_secret_gitignored(root):
        console.print(
            t("cli.cmd.connect.gitignored", path=str(root / ".gitignore")),
            style="dim",
        )
    path = root / PROVIDER_FILE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "api_key": config.api_key,
                "base_url": config.base_url,
                "model": config.model,
                "timeout": config.timeout,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


async def connect_wizard(
    root: Path, input_reader: InputReader, console: Console
) -> Optional[ProviderConfig]:
    """Run the interactive ``/connect`` wizard and persist the result.

    Each field is asked once; an empty answer keeps the currently resolved
    value (or the built-in default when nothing is configured yet).

    Args:
        root: Workspace root directory.
        input_reader: Injected input port rendering the prompts.
        console: Console for feedback messages.

    Returns:
        The saved configuration, or None when no API key was given and
        none was previously configured.
    """
    current = resolve_provider_config(root, console)

    api_key = (await input_reader(t("cli.cmd.connect.ask_api_key"))).strip()
    if not api_key:
        api_key = current.api_key if current else ""
    if not api_key:
        console.print(t("cli.cli.no_provider_hint"), style="yellow")
        return None

    base_url = (
        await input_reader(
            t(
                "cli.cmd.connect.ask_base_url",
                default=(current.base_url if current else DEFAULT_BASE_URL),
            )
        )
    ).strip()
    model = (
        await input_reader(
            t(
                "cli.cmd.connect.ask_model",
                default=(current.model if current else DEFAULT_MODEL),
            )
        )
    ).strip()
    raw_timeout = (
        await input_reader(
            t(
                "cli.cmd.connect.ask_timeout",
                default=(current.timeout if current else DEFAULT_TIMEOUT_SECONDS),
            )
        )
    ).strip()

    timeout = _coerce_timeout(raw_timeout, console)
    if timeout is None:
        timeout = current.timeout if current else DEFAULT_TIMEOUT_SECONDS

    config = ProviderConfig(
        api_key=api_key,
        base_url=base_url or (current.base_url if current else DEFAULT_BASE_URL),
        model=model or (current.model if current else DEFAULT_MODEL),
        timeout=timeout,
    )
    path = save_provider_config(root, config, console)
    console.print(t("cli.cmd.connect.saved", path=path), style="green")
    return config


async def switch_model(
    provider: Any, root: Path, input_reader: InputReader, console: Console
) -> None:
    """Interactively switch the active provider's default model.

    Args:
        provider: The live chat provider; must expose ``default_model``
            and an awaitable ``get_model_list()``.
        root: Workspace root directory (for persistence).
        input_reader: Injected input port for the pick prompt.
        console: Console for the model list and feedback.
    """
    try:
        models = list(await provider.get_model_list())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        console.print(t("cli.cmd.model.fetch_failed", error=exc), style="yellow")
        return
    if not models:
        console.print(t("cli.cmd.model.empty_list"), style="yellow")
        return

    console.print(t("cli.cmd.model.current", model=provider.default_model))
    console.print(t("cli.cmd.model.list_header"))
    for index, name in enumerate(models, start=1):
        console.print(f"  {index}. {name}")

    pick = (await input_reader(t("cli.cmd.model.pick"))).strip()
    if not pick:
        return
    if pick.isdigit():
        number = int(pick)
        if not 1 <= number <= len(models):
            console.print(t("cli.cmd.model.invalid_number", value=pick), style="yellow")
            return
        chosen = models[number - 1]
    else:
        chosen = pick
    if chosen not in models:
        console.print(t("cli.cmd.model.not_available", model=chosen), style="yellow")
        return

    provider.default_model = chosen
    console.print(t("cli.cmd.model.switched", model=chosen), style="green")

    path = root / PROVIDER_FILE_RELATIVE
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                stored["model"] = chosen
                path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            console.print(
                t("cli.cli.corrupt_secret", path=path, error=""), style="yellow"
            )
    else:
        console.print(t("cli.cmd.model.session_only"), style="dim")
