"""Tests for the workspace lock that keeps a single CLI per workspace.

Two MitoCode processes in one workspace write the same session records: each
keeps its own conversation in memory, each autosaves, and the last write wins.
The lock turns that silent conflict into an explicit question.

The tests here cover the lock itself only.  The application-level behaviour --
warning the user, naming the risk and asking whether to force entry -- lives in
``test_cli_app.py``.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.exception.mitocode_exception import MitoCodeWorkspaceLockError
from mitocode.core.workspace.workspace import WorkspaceLock


def test_the_lock_is_taken_when_the_workspace_is_free(tmp_path):
    """A workspace nobody holds can be locked."""
    lock = WorkspaceLock(tmp_path)

    assert lock.acquire() is True

    lock.release()


def test_a_second_lock_on_the_same_workspace_is_refused(tmp_path):
    """A second holder is turned away while the first one keeps the lock."""
    first = WorkspaceLock(tmp_path)
    second = WorkspaceLock(tmp_path)

    assert first.acquire() is True
    try:
        assert (
            second.acquire() is False
        ), "the lock must be refused while another holder keeps it"
    finally:
        first.release()


def test_releasing_the_lock_lets_the_next_holder_in(tmp_path):
    """The lock is handed over when the holder releases it."""
    first = WorkspaceLock(tmp_path)
    assert first.acquire() is True
    first.release()

    second = WorkspaceLock(tmp_path)
    assert second.acquire() is True, "a released lock must be available again"
    second.release()


def test_the_lock_does_not_reach_other_workspaces(tmp_path):
    """The lock is scoped to one workspace, not to the machine."""
    one = WorkspaceLock(tmp_path / "one")
    two = WorkspaceLock(tmp_path / "two")

    assert one.acquire() is True
    try:
        assert (
            two.acquire() is True
        ), "a different workspace must not be blocked by another one"
    finally:
        one.release()
        two.release()


def test_the_lock_file_records_who_holds_it(tmp_path):
    """The lock file names the holder so the user can be told about it."""
    lock = WorkspaceLock(tmp_path)

    assert lock.acquire() is True
    try:
        info = json.loads(lock.path.read_text(encoding="utf-8").strip())
        assert info["pid"] == os.getpid()
        assert info["started_at"]
    finally:
        lock.release()


def test_the_holder_is_described_for_the_user(tmp_path):
    """A refused holder can say who is in the way."""
    holder = WorkspaceLock(tmp_path)
    other = WorkspaceLock(tmp_path)

    assert holder.acquire() is True
    try:
        assert other.acquire() is False
        description = other.holder()
        assert str(os.getpid()) in description
    finally:
        holder.release()


def test_releasing_twice_is_harmless(tmp_path):
    """Release is idempotent, so teardown can call it unconditionally."""
    lock = WorkspaceLock(tmp_path)
    assert lock.acquire() is True

    lock.release()
    lock.release()


def test_a_lock_file_that_cannot_be_created_is_reported(tmp_path):
    """A lock that cannot be taken for a technical reason is an error.

    The caller must be able to tell "someone else holds it" apart from "the
    lock could not be set up at all", because only the first is a conflict the
    user can decide about.
    """
    (tmp_path / ".mitocode").write_text("not a directory", encoding="utf-8")
    lock = WorkspaceLock(tmp_path)

    with pytest.raises(MitoCodeWorkspaceLockError):
        lock.acquire()


def test_the_lock_file_stays_inside_the_workspace(tmp_path):
    """The lock lives in the workspace, so it follows the workspace."""
    lock = WorkspaceLock(tmp_path)

    assert lock.path.parent == tmp_path / ".mitocode"
