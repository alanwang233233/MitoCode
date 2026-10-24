"""Message catalogs for the supported locales.

Each :mod:`en` / :mod:`zh` submodule exposes a flat ``CATALOG`` mapping of
dot-separated message keys to strings.  Adding a locale means adding a module
here and registering it in :data:`mitocode.i18n.translator.CATALOGS`.
"""
