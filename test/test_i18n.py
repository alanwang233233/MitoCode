"""Tests for the i18n translator: locale detection, pinning and lookup."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mitocode.i18n import (
    available_locales,
    detect_locale,
    get_locale,
    set_locale,
    t,
)
from mitocode.i18n import translator


@pytest.fixture(autouse=True)
def _clean_locale_state():
    """Clear the locale pin around every test so detection starts fresh."""
    set_locale(None)
    yield
    set_locale(None)


def _clear_locale_env(monkeypatch):
    """Remove every environment variable locale detection consults."""
    for name in translator.LOCALE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_detect_locale_reads_mitocode_lang(monkeypatch):
    """MITOCODE_LANG wins over the standard locale variables."""
    _clear_locale_env(monkeypatch)
    monkeypatch.setenv("MITOCODE_LANG", "zh_CN.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert detect_locale() == "zh"


def test_detect_locale_falls_through_to_lc_all(monkeypatch):
    """An unsupported MITOCODE_LANG is skipped and LC_ALL is consulted."""
    _clear_locale_env(monkeypatch)
    monkeypatch.setenv("MITOCODE_LANG", "fr_FR.UTF-8")
    monkeypatch.setenv("LC_ALL", "zh")
    assert detect_locale() == "zh"


def test_detect_locale_parses_hyphen_and_encoding(monkeypatch):
    """Hyphenated names and encoding suffixes normalize to the base code."""
    _clear_locale_env(monkeypatch)
    monkeypatch.setenv("LANG", "zh-Hans.UTF-8")
    assert detect_locale() == "zh"


def test_detect_locale_unknown_language_uses_default(monkeypatch):
    """A language with no catalog falls back to the default locale."""
    _clear_locale_env(monkeypatch)
    monkeypatch.setenv("LANG", "fr_FR.UTF-8")
    assert detect_locale() == "en"


def test_detect_locale_c_posix_and_blank_fall_back(monkeypatch):
    """Placeholder locale values name no catalog and are skipped."""
    _clear_locale_env(monkeypatch)
    monkeypatch.setenv("MITOCODE_LANG", "C")
    monkeypatch.setenv("LC_ALL", "POSIX")
    monkeypatch.setenv("LANG", "   ")
    assert detect_locale() == "en"


def test_detect_locale_unset_environment_uses_default(monkeypatch):
    """With no locale variables at all, the default locale is used."""
    _clear_locale_env(monkeypatch)
    assert detect_locale() == "en"


def test_set_locale_pins_lookup():
    """A pinned locale is returned by get_locale and used by t()."""
    set_locale("zh")
    assert get_locale() == "zh"
    assert t("cli.app_help") == "MitoCode 命令行界面。"


def test_set_locale_none_restores_detection(monkeypatch):
    """Clearing the pin returns to environment-based detection."""
    monkeypatch.setenv("MITOCODE_LANG", "zh")
    set_locale("en")
    assert get_locale() == "en"
    set_locale(None)
    assert get_locale() == "zh"


def test_set_locale_rejects_unknown():
    """An unknown locale code is rejected instead of silently ignored."""
    with pytest.raises(ValueError):
        set_locale("fr")


def test_available_locales():
    """The catalog registry exposes both shipped locales."""
    assert available_locales() == ("en", "zh")


def test_t_formats_parameters():
    """Placeholders are interpolated from the keyword arguments."""
    assert t("cli.not_implemented", name="cli") == "'cli' is not implemented yet."


def test_t_missing_key_returns_key_itself():
    """A key missing everywhere surfaces as the key, never an exception."""
    assert t("no.such.key") == "no.such.key"


def test_t_missing_key_in_locale_falls_back_to_default(monkeypatch):
    """A key a locale lacks is read from the default (English) catalog."""
    monkeypatch.setitem(translator.CATALOGS, "zh", {})
    set_locale("zh")
    assert t("cli.app_help") == "MitoCode command line interface."


def test_t_broken_template_returns_template(monkeypatch):
    """A template naming an unknown placeholder degrades to raw text."""
    monkeypatch.setitem(translator.CATALOGS["en"], "test.broken", "hi {bogus}")
    assert t("test.broken", value=1) == "hi {bogus}"


def test_t_without_kwargs_returns_raw_template(monkeypatch):
    """Without keyword arguments the template is returned unformatted."""
    assert t("cli.tui.launching_in") == "Launching in {seconds} seconds..."
