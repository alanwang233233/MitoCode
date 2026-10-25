"""Tests for the CLI loguru takeover."""

import os
import stat
import sys
from pathlib import Path

import pytest
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.logging_setup import (
    DEFAULT_LOG_SUBDIR,
    LOG_FILENAME,
    LOG_LEVEL,
    RETENTION,
    ROTATION,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _isolated_loguru_state():
    """Restore a plain stderr sink so no test leaks global loguru state."""
    yield
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def _sinks():
    return [handler._sink for handler in logger._core.handlers.values()]


def test_configure_logging_only_writes_file(tmp_path):
    path = configure_logging(log_dir=tmp_path)
    logger.info("hello-from-test")
    logger.complete()
    assert path == tmp_path / LOG_FILENAME
    assert "hello-from-test" in path.read_text(encoding="utf-8")
    assert not any(sink is sys.stderr for sink in _sinks())
    assert not any(sink is sys.stdout for sink in _sinks())
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


def test_configure_logging_removes_previous_sinks(tmp_path):
    stale = tmp_path / "stale.log"
    logger.add(sys.stderr, level="INFO")
    logger.add(str(stale), level="INFO")

    configure_logging(log_dir=tmp_path)
    logger.info("after-takeover")
    logger.complete()

    assert len(logger._core.handlers) == 1
    assert "after-takeover" not in stale.read_text(encoding="utf-8")


def test_default_log_dir_lives_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    path = configure_logging()
    logger.info("default-path-line")
    logger.complete()

    expected_dir = tmp_path / ".mitocode" / "logs"
    assert path == expected_dir / LOG_FILENAME
    assert path.is_file()
    assert "default-path-line" in path.read_text(encoding="utf-8")


def test_creates_missing_log_directory_with_private_mode(tmp_path):
    log_dir = tmp_path / "nested" / "logs"

    configure_logging(log_dir=log_dir)

    assert log_dir.is_dir()
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700


def test_tightens_existing_log_directory_mode(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chmod(log_dir, 0o755)

    configure_logging(log_dir=log_dir)

    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700


def test_file_sink_is_always_debug(tmp_path):
    configure_logging(log_dir=tmp_path)

    handler = next(iter(logger._core.handlers.values()))
    assert handler._levelno == logger.level("DEBUG").no
    assert handler._sink._kwargs["encoding"] == "utf-8"
    assert callable(handler._sink._rotation_function)
    assert callable(handler._sink._retention_function)


def test_debug_flag_keeps_file_at_debug_level(tmp_path):
    path = configure_logging(debug=True, log_dir=tmp_path)

    logger.debug("debug-line")
    logger.complete()

    text = path.read_text(encoding="utf-8")
    assert "DEBUG" in text
    assert "debug-line" in text


def test_existing_log_file_is_appended(tmp_path):
    log_path = tmp_path / LOG_FILENAME
    log_path.write_text("earlier-line\n", encoding="utf-8")

    configure_logging(log_dir=tmp_path)
    logger.info("later-line")
    logger.complete()

    text = log_path.read_text(encoding="utf-8")
    assert "earlier-line" in text
    assert "later-line" in text


def test_exception_traceback_is_written_to_file(tmp_path):
    path = configure_logging(log_dir=tmp_path)

    try:
        raise RuntimeError("boom-in-log")
    except RuntimeError:
        logger.exception("caught")

    logger.complete()

    text = path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: boom-in-log" in text


def test_configure_twice_keeps_one_sink(tmp_path):
    first = configure_logging(log_dir=tmp_path)
    second = configure_logging(log_dir=tmp_path)

    assert first == second
    assert len(logger._core.handlers) == 1


def test_configuration_constants():
    assert LOG_FILENAME == "cli.log"
    assert LOG_LEVEL == "DEBUG"
    assert ROTATION == "10 MB"
    assert RETENTION == 5
    assert DEFAULT_LOG_SUBDIR == Path(".mitocode") / "logs"
