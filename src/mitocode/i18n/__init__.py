"""Internationalization for MitoCode's user-facing strings.

Public API: :func:`~mitocode.i18n.translator.t` for message lookup,
:func:`~mitocode.i18n.translator.set_locale` / :func:`~mitocode.i18n.translator.get_locale`
to pin or inspect the active locale, :func:`~mitocode.i18n.translator.detect_locale`
for environment-based selection, and
:func:`~mitocode.i18n.translator.available_locales` to enumerate catalogs.
"""

from mitocode.i18n.translator import (
    CATALOGS,
    DEFAULT_LOCALE,
    LOCALE_ENV_VARS,
    available_locales,
    detect_locale,
    get_locale,
    set_locale,
    t,
)

__all__ = [
    "CATALOGS",
    "DEFAULT_LOCALE",
    "LOCALE_ENV_VARS",
    "available_locales",
    "detect_locale",
    "get_locale",
    "set_locale",
    "t",
]
