"""Provider configuration resolution for the MitoCode CLI.

Credentials are resolved per key from four layers, highest first:

1. command line flags (``--api-key``, ``--base-url``, ``--model``, ``--timeout``,
   ``--reasoning-effort``)
2. process environment (``MITOCODE_*``, then the ``OPENAI_*`` aliases)
3. a ``.env`` file in the workspace root (parsed without touching ``os.environ``)
4. the read-only local secret file ``<workspace>/.mitocode/secret/provider.json``

The same environment variable names are recognised in the ``.env`` file. Values
that are blank at a higher layer do not shadow a lower layer. The resolved
:class:`ProviderConfig` never prints its API key, and no error raised here
contains a resolved value.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from dotenv import dotenv_values as _parse_dotenv
from loguru import logger as loguru_logger

from mitocode.cli.args import CliArgs
from mitocode.cli.errors import MitoCodeCliConfigError

DOTENV_FILENAME = ".env"
"""Name of the dotenv file read from the workspace root."""

SECRET_RELATIVE_PATH = Path(".mitocode") / "secret" / "provider.json"
"""Read-only local credential file, relative to the workspace root."""

DEFAULT_TIMEOUT = 60.0
"""Provider request timeout used when no layer supplies one, in seconds."""

_API_KEY_NAMES: Tuple[str, ...] = ("MITOCODE_API_KEY", "OPENAI_API_KEY")
_BASE_URL_NAMES: Tuple[str, ...] = ("MITOCODE_BASE_URL", "OPENAI_BASE_URL")
_MODEL_NAMES: Tuple[str, ...] = ("MITOCODE_MODEL",)
_TIMEOUT_NAMES: Tuple[str, ...] = ("MITOCODE_TIMEOUT",)
_REASONING_EFFORT_NAMES: Tuple[str, ...] = ("MITOCODE_REASONING_EFFORT",)
"""Environment (and dotenv) variable names for the reasoning effort tier.

No ``OPENAI_*`` alias exists: unlike the API key and base URL there is no
de-facto standard variable for this setting.
"""

KNOWN_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
"""Effort tiers commonly understood by OpenAI-compatible endpoints.

A value outside this set is still accepted and passed through — different
endpoints expose private tiers — but it triggers a startup warning, because
a misspelled tier would otherwise only fail (or be silently ignored) at the
endpoint.
"""

_SOURCE_FLAGS = "flags"
_SOURCE_ENVIRONMENT = "environment"
_SOURCE_DOTENV = "dotenv"
_SOURCE_SECRET = "secret"
_SOURCE_DEFAULT = "default"

_WORKSPACE_SOURCES = (_SOURCE_DOTENV, _SOURCE_SECRET)
"""Layers backed by files inside the workspace rather than by the user."""


@dataclass(frozen=True)
class ProviderConfig:
    """Validated provider settings resolved from every configuration layer.

    Attributes:
        api_key: API key for the provider. Excluded from ``repr`` so that a
            logged or displayed config can never leak it.
        base_url: Provider endpoint, or ``None`` to use the provider default.
        model: Model name to use for chat requests.
        timeout: Request timeout in seconds, always positive.
        reasoning_effort: Lowercase reasoning effort tier, or ``None`` when
            no layer supplied one. Not a secret.
        source: Layer that supplied ``api_key``: one of ``flags``,
            ``environment``, ``dotenv``, ``secret`` or ``default``.
    """

    api_key: str = field(repr=False)
    base_url: Optional[str] = None
    model: str = ""
    timeout: float = DEFAULT_TIMEOUT
    reasoning_effort: Optional[str] = None
    source: str = _SOURCE_DEFAULT


def dotenv_values(path: Path) -> Dict[str, str]:
    """Read a dotenv file into a plain dictionary.

    The file is only parsed; ``os.environ`` is never modified, so callers can
    use this at startup without leaking values into child processes.

    Args:
        path: Dotenv file to read.

    Returns:
        Mapping of keys to non-empty string values. Missing, unreadable or
        malformed files yield ``{}`` instead of raising.
    """
    try:
        parsed = _parse_dotenv(dotenv_path=Path(path))
    except (OSError, ValueError) as exc:
        loguru_logger.debug(
            "Ignoring unreadable dotenv file {}: {}", DOTENV_FILENAME, exc
        )
        return {}
    return {key: value for key, value in parsed.items() if value is not None}


def resolve_provider_config(
    args: CliArgs,
    workspace: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> ProviderConfig:
    """Resolve and validate provider settings for one CLI run.

    Args:
        args: Parsed command line arguments; non-``None`` flag values win over
            every file and environment layer.
        workspace: Workspace root, used to locate ``.env`` and the local
            secret file.
        environ: Environment to read. Defaults to :data:`os.environ`; tests pass
            an explicit mapping so the process environment stays out of the way.

    Returns:
        The validated provider configuration.

    Raises:
        MitoCodeCliConfigError: ``api_key`` or ``model`` is missing, a timeout
            is not a positive number, ``timeout`` is below one second, or a
            ``base_url`` from a workspace layer is paired with an ``api_key``
            from the flags or environment. The message names keys and file
            locations but never a value.
    """
    environment = os.environ if environ is None else environ
    root = Path(workspace)
    dotenv = dotenv_values(root / DOTENV_FILENAME)
    secret = _read_secret_file(root)

    api_key, source = _first_text(
        (_SOURCE_FLAGS, args.api_key),
        (_SOURCE_ENVIRONMENT, _first_env(environment, _API_KEY_NAMES)),
        (_SOURCE_DOTENV, _first_env(dotenv, _API_KEY_NAMES)),
        (_SOURCE_SECRET, secret.get("api_key")),
    )
    base_url, base_url_source = _first_text(
        (_SOURCE_FLAGS, args.base_url),
        (_SOURCE_ENVIRONMENT, _first_env(environment, _BASE_URL_NAMES)),
        (_SOURCE_DOTENV, _first_env(dotenv, _BASE_URL_NAMES)),
        (_SOURCE_SECRET, secret.get("base_url")),
    )
    model, _ = _first_text(
        (_SOURCE_FLAGS, args.model),
        (_SOURCE_ENVIRONMENT, _first_env(environment, _MODEL_NAMES)),
        (_SOURCE_DOTENV, _first_env(dotenv, _MODEL_NAMES)),
        (_SOURCE_SECRET, secret.get("model")),
    )
    timeout = _resolve_timeout(args.timeout, environment, dotenv, secret)
    reasoning_effort, reasoning_effort_source = _first_text(
        (_SOURCE_FLAGS, args.reasoning_effort),
        (_SOURCE_ENVIRONMENT, _first_env(environment, _REASONING_EFFORT_NAMES)),
        (_SOURCE_DOTENV, _first_env(dotenv, _REASONING_EFFORT_NAMES)),
        (_SOURCE_SECRET, _as_text(secret.get("reasoning_effort"))),
    )
    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.lower()
        if reasoning_effort not in KNOWN_REASONING_EFFORTS:
            loguru_logger.warning(
                "reasoning_effort from {} is not a known tier ({}); "
                "passing it through unchanged",
                _describe_reasoning_layer(reasoning_effort_source),
                ", ".join(sorted(KNOWN_REASONING_EFFORTS)),
            )

    missing = [
        name for name, value in (("api_key", api_key), ("model", model)) if not value
    ]
    if missing:
        raise MitoCodeCliConfigError(_missing_message(missing))

    # The check above guarantees both required fields were supplied; restating
    # it narrows the optional types for static checkers.
    assert api_key is not None and model is not None

    # A workspace-controlled endpoint must never receive a key the user holds
    # outside the workspace: the request would carry that key to whatever the
    # workspace file points at. Only the winning base_url is judged, so a
    # blank workspace value falling through to the environment stays allowed.
    if base_url_source in _WORKSPACE_SOURCES and source not in _WORKSPACE_SOURCES:
        raise MitoCodeCliConfigError(_workspace_base_url_message(base_url_source))

    return ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        source=source or _SOURCE_DEFAULT,
    )


def _first_env(values: Mapping[str, str], names: Sequence[str]) -> Optional[str]:
    """Return the first non-blank value among *names* in *values*.

    Args:
        values: Environment or dotenv mapping.
        names: Candidate variable names, highest priority first.

    Returns:
        The first non-blank string value, or ``None``.
    """
    for name in names:
        candidate = values.get(name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _first_text(
    *entries: Tuple[str, object],
) -> Tuple[Optional[str], Optional[str]]:
    """Pick the first non-blank text value together with its layer name.

    Args:
        *entries: ``(layer, raw value)`` pairs ordered highest priority first.

    Returns:
        ``(value, layer)`` for the winner, or ``(None, None)`` when every layer
        is blank or holds a non-text value.
    """
    for source, value in entries:
        if isinstance(value, str) and value.strip():
            return value.strip(), source
    return None, None


def _as_text(value: object) -> Optional[str]:
    """Coerce a scalar configuration value to text for the layer pickers.

    Args:
        value: Raw value from a flag or from the JSON secret file.

    Returns:
        The value as a string, or ``None`` when it is blank or not a scalar.
        JSON booleans never reach this helper: :func:`_read_secret_file` filters
        them out so ``true`` cannot become a timeout.
    """
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _resolve_timeout(
    flag_value: Optional[float],
    environment: Mapping[str, str],
    dotenv: Mapping[str, str],
    secret: Mapping[str, object],
) -> float:
    """Resolve the request timeout from every layer and validate it.

    Args:
        flag_value: ``--timeout`` value, already parsed by argparse.
        environment: Process environment mapping.
        dotenv: Parsed ``.env`` mapping.
        secret: Local secret file mapping.

    Returns:
        The timeout in seconds, always greater than zero.

    Raises:
        MitoCodeCliConfigError: The winning value is not a number or is not
            greater than zero. The message names the layer, never the value.
    """
    raw, source = _first_text(
        (_SOURCE_FLAGS, _as_text(flag_value)),
        (_SOURCE_ENVIRONMENT, _first_env(environment, _TIMEOUT_NAMES)),
        (_SOURCE_DOTENV, _first_env(dotenv, _TIMEOUT_NAMES)),
        (_SOURCE_SECRET, _as_text(secret.get("timeout"))),
    )
    if raw is None:
        return DEFAULT_TIMEOUT
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise MitoCodeCliConfigError(
            f"provider timeout from {_describe_layer(source)} is not a number"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise MitoCodeCliConfigError(
            f"provider timeout from {_describe_layer(source)} "
            "must be a finite number of seconds greater than zero"
        )
    return timeout


def _describe_reasoning_layer(source: Optional[str]) -> str:
    """Describe the configuration layer a reasoning effort came from.

    Args:
        source: Layer name from :data:`_SOURCE_FLAGS` and friends.

    Returns:
        A short help-oriented description, e.g. ``"--reasoning-effort"`` for
        flags, naming files and variables but never the value itself.
    """
    if source == _SOURCE_FLAGS:
        return "--reasoning-effort"
    if source == _SOURCE_ENVIRONMENT:
        return "the " + "/".join(_REASONING_EFFORT_NAMES) + " environment variable"
    if source == _SOURCE_DOTENV:
        return "the " + DOTENV_FILENAME + " file"
    return "the " + SECRET_RELATIVE_PATH.as_posix() + " file"


def _describe_layer(source: Optional[str]) -> str:
    """Describe a configuration layer for an error message.

    Args:
        source: Layer name from :data:`_SOURCE_FLAGS` and friends.

    Returns:
        A short help-oriented description, e.g. ``"--timeout"`` for flags.
    """
    if source == _SOURCE_FLAGS:
        return "--timeout"
    if source == _SOURCE_ENVIRONMENT:
        return "the " + "/".join(_TIMEOUT_NAMES) + " environment variable"
    if source == _SOURCE_DOTENV:
        return "the " + DOTENV_FILENAME + " file"
    return "the " + SECRET_RELATIVE_PATH.as_posix() + " file"


def _missing_message(missing: Sequence[str]) -> str:
    """Build the error shown when required provider settings are absent.

    The text names the missing keys, the flags, the environment variables and
    both local file locations. It deliberately contains no resolved value.

    Args:
        missing: Field names that no layer supplied, in display order.

    Returns:
        A multi-line, value-free error message.
    """
    names = ", ".join(missing)
    key_names = " / ".join(_API_KEY_NAMES)
    model_names = " / ".join(_MODEL_NAMES)
    return (
        f"Missing required provider configuration: {names}.\n"
        f"Provide them with --api-key/--model, the {key_names} and "
        f"{model_names} environment variables,\n"
        f"a {DOTENV_FILENAME} file in the workspace root, or the "
        f"{SECRET_RELATIVE_PATH.as_posix()} file."
    )


def _workspace_base_url_message(base_url_source: str) -> str:
    """Build the error shown when a workspace endpoint would capture a foreign key.

    A ``base_url`` resolved from a workspace file is only allowed next to an
    ``api_key`` resolved from the same workspace layers, otherwise the request
    would send the user's key to an endpoint the workspace controls. The text
    names the files, flags and variables involved and never any resolved value.

    Args:
        base_url_source: Layer that supplied ``base_url``; one of
            :data:`_SOURCE_DOTENV` or :data:`_SOURCE_SECRET`.

    Returns:
        A multi-line, value-free error message.
    """
    location = (
        DOTENV_FILENAME
        if base_url_source == _SOURCE_DOTENV
        else SECRET_RELATIVE_PATH.as_posix()
    )
    base_url_names = " / ".join(_BASE_URL_NAMES)
    return (
        "Refusing to pair an API key from the command line or the environment "
        "with a\n"
        f"base URL from the workspace {location} file: that endpoint is\n"
        "controlled by the workspace and would receive the key.\n"
        f"Pass --base-url, set {base_url_names} in the environment,\n"
        "or move the API key into the same workspace file."
    )


def _read_secret_file(workspace: Path) -> Dict[str, object]:
    """Read the local secret file without ever logging its contents.

    A missing, unreadable, malformed or non-object file is treated as though
    that layer did not exist: the caller falls back to the layers above it.
    Only a debug log line records the failure, and it never includes values.

    Args:
        workspace: Workspace root containing :data:`SECRET_RELATIVE_PATH`.

    Returns:
        Scalar fields from the file keyed by name; ``{}`` when unusable.
    """
    path = workspace / SECRET_RELATIVE_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        loguru_logger.debug(
            "Ignoring unreadable provider secret file {}: {}",
            SECRET_RELATIVE_PATH.as_posix(),
            exc,
        )
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        loguru_logger.debug(
            "Ignoring malformed provider secret file {}: {}",
            SECRET_RELATIVE_PATH.as_posix(),
            exc,
        )
        return {}
    if not isinstance(data, dict):
        loguru_logger.debug(
            "Ignoring provider secret file {}: root value is not an object",
            SECRET_RELATIVE_PATH.as_posix(),
        )
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }


__all__ = [
    "DEFAULT_TIMEOUT",
    "DOTENV_FILENAME",
    "SECRET_RELATIVE_PATH",
    "KNOWN_REASONING_EFFORTS",
    "ProviderConfig",
    "dotenv_values",
    "resolve_provider_config",
]
