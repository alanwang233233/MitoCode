"""Locale resolution and message lookup for MitoCode's user-facing strings.

The module keeps a fixed set of in-process message catalogs (:mod:`en` and
:mod:`zh`) and resolves every user-visible string through :func:`t` at call
time, so a process picks its language once at startup and every later message
agrees with it.

Locale selection follows the conventional environment chain, in priority
order: ``MITOCODE_LANG`` (MitoCode's own override), then ``LC_ALL``,
``LC_MESSAGES`` and ``LANG``.  Each value is normalized the way locale names
are usually written -- ``zh_CN.UTF-8`` and ``zh-Hans`` both become ``zh`` --
and a value that does not name an available locale is skipped, so an
unsupported language silently falls through to the next variable and
ultimately to :data:`DEFAULT_LOCALE`.

The selected locale is not cached: :func:`get_locale` re-runs detection on
every call unless :func:`set_locale` pinned one explicitly.  Detection is four
environment reads, cheap enough to run per message, and skipping the cache
keeps tests (and long-lived processes whose environment changed) honest.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from mitocode.i18n.locales import en as _en_catalog
from mitocode.i18n.locales import zh as _zh_catalog

#: Message catalogs keyed by locale code.  Every catalog must define the same
#: key set; :func:`t` falls back to :data:`DEFAULT_LOCALE` for a key a locale
#: is missing, so a partial translation degrades to English instead of
#: raising.
CATALOGS: Mapping[str, Mapping[str, str]] = {
    "en": _en_catalog.CATALOG,
    "zh": _zh_catalog.CATALOG,
}

#: Locale used when detection finds nothing usable and no locale was pinned.
DEFAULT_LOCALE: str = "en"

#: Environment variables consulted for locale detection, in priority order.
LOCALE_ENV_VARS = ("MITOCODE_LANG", "LC_ALL", "LC_MESSAGES", "LANG")

# Single-slot override installed by :func:`set_locale`.  ``None`` means no
# override, so every lookup re-runs :func:`detect_locale`.  A one-key mapping
# rather than a ``global`` statement, to keep the module free of rebinding.
_override: dict = {"locale": None}


def _normalize(raw_value: str) -> Optional[str]:
    """Reduce a raw locale string to an available locale code.

    Args:
        raw_value: A locale name as written in the environment, such as
            ``zh_CN.UTF-8``, ``en`` or ``fr_FR``.

    Returns:
        The language code before the first ``_``/``-``/encoding suffix, lower
       cased, when it names an available catalog; ``None`` otherwise.
    """
    language = raw_value.strip().split(".", 1)[0].split("@", 1)[0]
    if not language:
        return None
    code = language.replace("-", "_").split("_", 1)[0].lower()
    return code if code in CATALOGS else None


def detect_locale() -> str:
    """Resolve the locale from the process environment.

    Args:
        None.

    Returns:
        The code of the first environment variable in
        :data:`LOCALE_ENV_VARS` naming an available locale, or
        :data:`DEFAULT_LOCALE` when none does.  Placeholder values such as
        ``C`` and ``POSIX`` name no catalog and are skipped.
    """
    for name in LOCALE_ENV_VARS:
        value = os.environ.get(name)
        if value:
            normalized = _normalize(value)
            if normalized is not None:
                return normalized
    return DEFAULT_LOCALE


def set_locale(locale: Optional[str] = None) -> None:
    """Pin the locale used by subsequent :func:`t` calls.

    Args:
        locale: A catalog key such as ``"en"`` or ``"zh"`` to pin, or
            ``None`` to clear the pin and return to environment detection.

    Raises:
        ValueError: If *locale* is given but names no available catalog.
    """
    if locale is None:
        _override["locale"] = None
        return
    if locale not in CATALOGS:
        raise ValueError(
            f"Unknown locale {locale!r}; available: {', '.join(sorted(CATALOGS))}"
        )
    _override["locale"] = locale


def get_locale() -> str:
    """Return the locale the next :func:`t` call will use.

    Args:
        None.

    Returns:
        The pinned locale when :func:`set_locale` installed one, otherwise
        the result of :func:`detect_locale`.
    """
    pinned = _override["locale"]
    if pinned is not None:
        return str(pinned)
    return detect_locale()


def available_locales() -> tuple:
    """Return the locale codes that have a catalog.

    Args:
        None.

    Returns:
        Available locale codes in definition order (``("en", "zh")``).
    """
    return tuple(CATALOGS)


def t(key: str, **kwargs: object) -> str:
    """Translate *key* in the active locale and format it with *kwargs*.

    Lookup order: the active locale's catalog, then the default locale's
    catalog (so a locale missing a key degrades to English), then *key*
    itself -- a missing translation surfaces as its key rather than an
    exception, because a typo in a message name must never take down the
    command that is trying to report something.

    Args:
        key: Dot-separated catalog key, e.g. ``"cli.tui.deprecated_warning"``.
        **kwargs: Values interpolated into the message's ``{placeholder}``
            fields.  A malformed template or missing placeholder is reported
            as the unformatted template rather than raised.

    Returns:
        The translated, formatted message.
    """
    catalog = CATALOGS.get(get_locale(), CATALOGS[DEFAULT_LOCALE])
    template = catalog.get(key)
    if template is None:
        template = CATALOGS[DEFAULT_LOCALE].get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return template
    return template
