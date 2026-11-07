"""Provider credential loading and interactive setup for the MitoCode CLI.

The workspace's ``.mitocode/secret/provider.json`` is the **only** source of
provider settings: there are no command-line overrides, no environment
variables and no ``.env`` file. That keeps a workspace self-contained and makes
the credential's origin unambiguous.

Two rules shape this module:

* a credential value is never printed, logged or included in an error message;
  the setup prompts read the API key with hidden input and errors name only the
  file, the field and the reason;
* the credential file is created private: its directory is ``0o700`` and the
  file itself ``0o600``.

A file that exists but is unusable (unreadable, malformed, wrong root type or
missing required fields) is a fatal configuration error rather than a silently
ignored layer, because there is no other source left to fall back to.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from loguru import logger as loguru_logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mitocode.cli.errors import MitoCodeCliConfigError

SECRET_RELATIVE_PATH = Path(".mitocode") / "secret" / "provider.json"
"""The provider credential file, relative to the workspace root."""

DEFAULT_TIMEOUT = 60.0
"""Provider request timeout used when the file does not supply one, in seconds."""

KNOWN_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
"""Effort tiers commonly understood by OpenAI-compatible endpoints.

A value outside this set is still accepted and passed through — different
endpoints expose private tiers — but it triggers a warning, because a
misspelled tier would otherwise only fail (or be silently ignored) at the
endpoint.
"""

_DIRECTORY_MODE = 0o700
"""Permission bits applied to the directory holding the credential file."""

_FILE_MODE = 0o600
"""Permission bits applied to the credential file itself."""

Asker = Callable[..., str]
"""Callable that reads one line; ``console.input`` in production, a fake in tests."""


@dataclass(frozen=True)
class ProviderConfig:
    """Validated provider settings read from the workspace credential file.

    Attributes:
        api_key: API key for the provider. Excluded from ``repr`` so that a
            logged or displayed config can never leak it.
        base_url: Provider endpoint, or ``None`` to use the provider default.
        model: Model name to use for chat requests.
        timeout: Request timeout in seconds, always positive.
        reasoning_effort: Lowercase reasoning effort tier, or ``None`` when the
            file did not supply one. Not a secret.
    """

    api_key: str = field(repr=False)
    base_url: Optional[str] = None
    model: str = ""
    timeout: float = DEFAULT_TIMEOUT
    reasoning_effort: Optional[str] = None


def secret_path(workspace: Path) -> Path:
    """Return the credential file path for a workspace.

    Args:
        workspace: Workspace root.

    Returns:
        ``<workspace>/.mitocode/secret/provider.json``.
    """
    return Path(workspace) / SECRET_RELATIVE_PATH


def resolve_provider_config(workspace: Path) -> ProviderConfig:
    """Read and validate the workspace's provider credential file.

    Args:
        workspace: Workspace root containing the credential file.

    Returns:
        The validated provider configuration.

    Raises:
        MitoCodeCliConfigError: The file is missing, unreadable, not valid
            JSON, not a JSON object, or lacks ``api_key``/``model``; the
            timeout is not a positive number. Messages name the file and the
            field but never a value.
    """
    path = secret_path(workspace)
    data = _read_secret_file(path)

    api_key = _text_field(data, "api_key")
    model = _text_field(data, "model")
    base_url = _text_field(data, "base_url")
    timeout = _timeout_field(data, path)
    reasoning_effort = _reasoning_field(data, path)

    missing = [
        name for name, value in (("api_key", api_key), ("model", model)) if not value
    ]
    if missing:
        raise MitoCodeCliConfigError(
            f"Missing required provider configuration in '{path}': "
            f"{', '.join(missing)}.\n"
            "Add the missing field to that file and run mitocode again."
        )

    # The check above guarantees both required fields were supplied; restating
    # it narrows the optional types for static checkers.
    assert api_key is not None and model is not None

    return ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def ask_provider_settings(
    console: Console, *, ask: Optional[Asker] = None
) -> Dict[str, object]:
    """Interactively collect the provider settings for a new credential file.

    The API key is read with hidden input. Required fields are re-asked while
    blank; a blank optional field is simply omitted from the returned mapping.
    ``EOFError``/``KeyboardInterrupt`` are left to the caller, which decides
    whether they end the process.

    Args:
        console: Console the explanatory panel and the prompts are printed to.
        ask: Line reader override for tests. Defaults to
            :meth:`rich.console.Console.input`.

    Returns:
        Mapping with ``api_key`` and ``model`` plus whichever optional settings
        were supplied. Values are stripped of surrounding whitespace.
    """
    asker = ask if ask is not None else _console_input

    console.print(
        Panel(
            Text.assemble(
                ("MitoCode needs provider credentials.\n\n", ""),
                (
                    "They are stored in "
                    f"{SECRET_RELATIVE_PATH.as_posix()} inside this workspace "
                    "and never leave it.",
                    "mito.muted",
                ),
            ),
            title="Provider setup",
            border_style="mito.accent",
            expand=False,
        )
    )

    api_key = _ask_required(asker, console, "API key", password=True)
    model = _ask_required(asker, console, "Model")
    base_url = asker("  Base URL (optional, Enter to skip): ", console=console).strip()
    timeout = _ask_timeout(asker, console)
    effort_raw = asker(
        "  Reasoning effort (optional, Enter to skip): ", console=console
    ).strip()

    values: Dict[str, object] = {"api_key": api_key, "model": model}
    if base_url:
        values["base_url"] = base_url
    if timeout is not None:
        values["timeout"] = timeout
    if effort_raw:
        tier = effort_raw.lower()
        if tier not in KNOWN_REASONING_EFFORTS:
            console.print(
                Text(
                    f"  '{tier}' is not a known effort tier "
                    "(none, minimal, low, medium, high, xhigh, max); "
                    "passing it through anyway",
                    style="mito.warn",
                )
            )
        values["reasoning_effort"] = tier
    return values


def write_secret_file(workspace: Path, values: Mapping[str, object]) -> Path:
    """Write the provider credential file with private permissions.

    The write is atomic: the document is written to a private sibling temporary
    file and moved into place, so an interrupted setup cannot leave a truncated
    ``provider.json`` behind. An existing file is never replaced, even when it
    is malformed: the user may still be able to repair it by hand, and silently
    discarding a credential would be destructive.

    Args:
        workspace: Workspace root that receives ``.mitocode/secret/``.
        values: Settings to persist; blank optional entries are dropped.

    Returns:
        The path that was written.

    Raises:
        MitoCodeCliConfigError: The file already exists, or the directory or
            file cannot be created. The message names the path and never a
            value.
    """
    path = secret_path(workspace)
    if path.exists() or path.is_symlink():
        raise MitoCodeCliConfigError(
            f"The provider configuration at '{path}' already exists; it was "
            "not overwritten. Repair or remove it by hand, then run mitocode "
            "again."
        )
    payload = {
        key: value
        for key, value in values.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }
    temporary_path: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=_DIRECTORY_MODE)
        path.parent.chmod(_DIRECTORY_MODE)
        # ``mkstemp`` creates the file with mode 0o600, so the credential is
        # never briefly readable through the default umask before it is chmoded.
        descriptor, temporary_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.chmod(temporary_path, _FILE_MODE)
        # Re-check immediately before the move: the file may have appeared
        # while the prompts were being answered.
        if path.exists() or path.is_symlink():
            raise MitoCodeCliConfigError(
                f"The provider configuration at '{path}' already exists; it "
                "was not overwritten."
            )
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as exc:
        raise MitoCodeCliConfigError(
            f"Cannot write the provider configuration at '{path}': "
            f"{getattr(exc, 'strerror', None) or 'I/O error'}."
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
    return path


def _console_input(prompt: str, *, console: Console, password: bool = False) -> str:
    """Read one line through Rich, optionally without echoing it.

    Markup is disabled so prompt text containing brackets (for example the
    default ``[60]`` timeout hint) is printed literally.

    Args:
        prompt: Prompt text shown to the user.
        console: Console that owns the input stream.
        password: Whether the typed value must stay hidden.

    Returns:
        The line the user entered.
    """
    return console.input(prompt, password=password, markup=False)


def _ask_required(
    asker: Asker, console: Console, label: str, *, password: bool = False
) -> str:
    """Ask until the answer is non-blank.

    Args:
        asker: Line reader.
        console: Console used for the warning line.
        label: Field name shown in the prompt and the warning.
        password: Whether the answer must stay hidden.

    Returns:
        The stripped, non-blank answer.
    """
    while True:
        answer = asker(f"  {label}: ", console=console, password=password).strip()
        if answer:
            return answer
        console.print(Text(f"  {label} cannot be blank.", style="mito.warn"))


def _ask_timeout(asker: Asker, console: Console) -> Optional[float]:
    """Ask for an optional positive timeout, re-asking when it is invalid.

    Args:
        asker: Line reader.
        console: Console used for the warning line.

    Returns:
        The timeout in seconds, or ``None`` when the user left it blank so the
        default applies.
    """
    while True:
        raw = asker(
            f"  Timeout in seconds [{int(DEFAULT_TIMEOUT)}]: ", console=console
        ).strip()
        if not raw:
            return None
        try:
            timeout = float(raw)
        except ValueError:
            console.print(
                Text("  Timeout must be a number of seconds.", style="mito.warn")
            )
            continue
        if math.isfinite(timeout) and timeout > 0:
            return timeout
        console.print(
            Text(
                "  Timeout must be a finite number of seconds greater than zero.",
                style="mito.warn",
            )
        )


def _read_secret_file(path: Path) -> Dict[str, object]:
    """Read the credential file, raising a value-free error when it is unusable.

    Args:
        path: Credential file to read.

    Returns:
        Scalar fields keyed by name; booleans are dropped so ``true`` cannot
        become a timeout or a text value.

    Raises:
        MitoCodeCliConfigError: The file is missing, unreadable, not valid
            JSON or not a JSON object.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise MitoCodeCliConfigError(
            f"No provider configuration found at '{path}'.\n"
            'Create it with at least "api_key" and "model", then run mitocode again.'
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise MitoCodeCliConfigError(
            f"The provider configuration at '{path}' could not be read as UTF-8 text."
        ) from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise MitoCodeCliConfigError(
            f"The provider configuration at '{path}' is not valid JSON."
        ) from exc
    if not isinstance(data, dict):
        raise MitoCodeCliConfigError(
            f"The provider configuration at '{path}' must be a JSON object."
        )
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }


def _as_text(value: object) -> Optional[str]:
    """Coerce a scalar credential-file value to text for the layer pickers.

    Args:
        value: Raw value from the JSON credential file.

    Returns:
        The value as a string, or ``None`` when it is blank or not a scalar.
        JSON booleans never reach this helper: :func:`_read_secret_file`
        filters them out so ``true`` cannot become a timeout.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _text_field(data: Mapping[str, object], name: str) -> Optional[str]:
    """Return a non-blank text field, or ``None`` when it is absent or unusable.

    Args:
        data: Fields read from the credential file.
        name: Field name.

    Returns:
        The stripped string value, or ``None``.
    """
    value = data.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _timeout_field(data: Mapping[str, object], path: Path) -> float:
    """Resolve and validate the optional timeout field.

    Args:
        data: Fields read from the credential file.
        path: File the fields came from, named in error messages.

    Returns:
        The timeout in seconds, :data:`DEFAULT_TIMEOUT` when the field is absent
        or blank.

    Raises:
        MitoCodeCliConfigError: The value is not a number, not finite or not
            greater than zero. The message names the file, never the value.
    """
    value = data.get("timeout")
    if value is None:
        return DEFAULT_TIMEOUT
    raw = _as_text(value)
    if raw is None:
        # An optional field holding only whitespace is treated as absent, the
        # same way blank base_url and reasoning_effort values are.
        return DEFAULT_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise MitoCodeCliConfigError(
            f"The provider timeout in '{path}' is not a number."
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise MitoCodeCliConfigError(
            f"The provider timeout in '{path}' must be a finite number of "
            "seconds greater than zero."
        )
    return timeout


def _reasoning_field(data: Mapping[str, object], path: Path) -> Optional[str]:
    """Resolve and normalise the optional reasoning effort field.

    Args:
        data: Fields read from the credential file.
        path: File the fields came from, named in the warning.

    Returns:
        The lowercase tier, or ``None`` when the field is absent. An unknown
        tier is still returned but logged as a value-free warning.
    """
    value = _text_field(data, "reasoning_effort")
    if value is None:
        return None
    tier = value.lower()
    if tier not in KNOWN_REASONING_EFFORTS:
        loguru_logger.warning(
            "reasoning_effort in '{}' is not a known tier ({}); "
            "passing it through unchanged",
            path,
            ", ".join(sorted(KNOWN_REASONING_EFFORTS)),
        )
    return tier


__all__ = [
    "DEFAULT_TIMEOUT",
    "KNOWN_REASONING_EFFORTS",
    "ProviderConfig",
    "SECRET_RELATIVE_PATH",
    "ask_provider_settings",
    "resolve_provider_config",
    "secret_path",
    "write_secret_file",
]
