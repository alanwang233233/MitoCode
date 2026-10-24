"""The five file tools -- Read, Write, Edit, Glob and Grep.

Every tool here begins the same way: resolve the path through
:func:`~mitocode.core.tool.builtin.paths.resolve`, refuse if that refuses,
then do all of the blocking I/O inside :func:`asyncio.to_thread`.  Failures
come back as prose beginning ``"Error: "`` rather than as exceptions, because
``AgentRuntime`` feeds a tool's return value straight back to the model and a
sentence can be routed around where a traceback cannot.  The single-file
tools -- Read, Write and Edit -- refuse a resolved target that is not a
regular file before opening anything: ``read_text`` on a named pipe would
block the worker thread forever, and no result is worth a leaked thread.

The function names are ``read``, ``write``, ``edit``, ``glob_files`` and
``grep_files`` -- deliberately *not* ``glob`` and ``grep``, because
:func:`grep_files` takes a keyword parameter named ``glob`` that a function of
the same name would shadow.  The registered tool names -- ``Read``, ``Write``,
``Edit``, ``Glob``, ``Grep`` -- are chosen at registration time, which is not
this module's concern.  Schemas are derived from these signatures by
``ToolManager.register``, which is also why this module must not use
``from __future__ import annotations``: the schema builder reads raw
``inspect.signature`` annotations, and stringified ones would break it,
including the ``enum`` that ``Literal`` produces for ``output_mode``.

**Containment is decided in exactly one place.**
:func:`~mitocode.core.tool.builtin.paths.resolve` is the sole authority on
whether a *requested* path may be operated on: an in-root path comes back
without the approver being consulted, and an escape -- an explicit outside
path, or an in-root symlink that resolves out of the workspace -- is
submitted to the approver exactly once, and its verdict is final.  This
module adds no second check on a requested path: a symlink pointing out of
the workspace is an escape like any other, decided by the approver, and an
approver that allows an escape makes these tools operate on the resolved
outside path.  Re-checking containment here would override that verdict,
leaving the approver's decision inert for all five tools -- a future
interactive approver could never grant an out-of-root file operation,
because the tools would refuse it whatever the dialog said.  That claim is
scoped to requested paths, and the walker is the known asymmetry: the same
out-of-workspace symlink that sends ``read("link.txt")`` to the approver is
silently dropped by ``grep_files`` and ``glob_files`` with no approver call
at all, because the walker's containment comparison
(:func:`~mitocode.core.tool.builtin.file_walker._iter_files`)
scopes what the walk *enumerates* -- it can only omit a candidate, never
operate outside one.

**Write refuses to overwrite, and Edit exists because of it.**  A model that
rewrote a 400-line file to change one function cannot tell from a success
message that 399 lines were destroyed, and there is no undo.  So ``write``
creates files and refuses existing ones -- naming ``Edit`` as the tool for
changes -- and ``edit`` replaces exact strings, the only form of address that
survives the file shifting underneath the model.  The replacement is a single
``str.replace`` and never a hand-written scan loop: a loop that re-searches
from position 0 after each replacement runs forever when ``old_string`` is a
substring of ``new_string``, while ``str.replace`` never rescans what it has
produced and terminates by construction.

**The gitignore support is minimal and must never be described as more.**
``Glob`` and ``Grep`` share one walker,
:func:`~mitocode.core.tool.builtin.file_walker._iter_files`, which always
prunes ``.git``, ``.hg``, ``.svn`` and ``__pycache__``, prunes dot-directories
-- unless the caller's pattern itself starts with a dot, which unlocks them
through
:meth:`~mitocode.core.tool.builtin.file_walker._Ignore.with_hidden_directories`
-- never follows a directory
symlink, and honours a single ``.gitignore`` at the walk root with per-line
:func:`fnmatch.fnmatch` matching and ``!`` negation, last match winning so
``!keep.log`` after ``*.log`` negates correctly.  There is no ``pathspec``
dependency and adding one is out of scope: no nested ``.gitignore`` files,
no ``**`` rules, no directory-only rules (``build/`` never matches), no
anchoring.  ``fnmatch`` has no dotfile exception, so a bare ``*`` matches a
dot-file's basename; dot *directories* are what get pruned.

**Binary files are recognised by a NUL byte in the first 8 KiB**
(:func:`~mitocode.core.tool.builtin.file_walker._is_binary`) and are never
fed to the regex: ``Grep`` skips them, noting ``[binary]`` in content mode
where a silent skip would be invisible.
An invalid regex falls back to a substring match, not a crash.

**Every result is clamped** to ``ToolContext.max_output_chars`` via
:func:`~mitocode.core.tool.builtin.context.clamp`, and every tool carries
:func:`~mitocode.core.tool.builtin.errors.tool_boundary`, so even an
unexpected exception reaches the model as one line instead of a lost turn.
All blocking I/O runs in :func:`asyncio.to_thread`: the event loop that runs
these tools also runs every sub-agent, and a synchronous walk would starve
them all.
"""

import asyncio
import difflib
import os
import re
import stat
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from mitocode.core.tool.builtin.context import clamp, current
from mitocode.core.tool.builtin.errors import tool_boundary
from mitocode.core.tool.builtin.file_walker import (
    BINARY_SNIFF_BYTES,
    _is_binary,
    _iter_files,
    _load_ignore,
    _rel_posix,
)
from mitocode.core.tool.builtin.paths import resolve

#: The byte cap on a whole-file Read.  ``read``'s ``limit`` caps *lines* and
#: is applied only after the file is in memory, so on its own a read of a
#: multi-gigabyte file would be loaded whole and then sliced down to a few
#: thousand lines.  A regular file whose size exceeds this cap is refused
#: from its ``st_size`` alone, before anything is read.  Ten mebibytes is
#: comfortably above any legitimate window -- 2000 lines of even 5 KiB each
#: is 10 MiB -- and bounds a single read's worst case to a fraction of the
#: spike the cap exists to prevent.
MAX_READ_BYTES = 10 * 1024 * 1024


def _kind_noun(mode: int) -> str:
    """Name the filesystem object a ``st_mode`` describes, for refusal prose.

    The tools refuse non-regular files with a sentence that names the object
    rather than the abstraction: "is a named pipe (fifo)" tells the model
    what it hit and implies there is nothing to read, where a bare "not a
    regular file" leaves it guessing between a directory (Glob lists that),
    a fifo (nothing can) and a device.

    Args:
        mode: A ``st_mode`` value from :meth:`pathlib.Path.stat` or
            :meth:`pathlib.Path.lstat`.

    Returns:
        One of ``"a directory"``, ``"a named pipe (fifo)"``, ``"a socket"``,
        ``"a character device"``, ``"a block device"``, ``"a symlink"`` or
        ``"a non-regular file"``.
    """
    if stat.S_ISDIR(mode):
        noun = "a directory"
    elif stat.S_ISFIFO(mode):
        noun = "a named pipe (fifo)"
    elif stat.S_ISSOCK(mode):
        noun = "a socket"
    elif stat.S_ISCHR(mode):
        noun = "a character device"
    elif stat.S_ISBLK(mode):
        noun = "a block device"
    elif stat.S_ISLNK(mode):
        # Reachable only from an lstat: a stat follows links, so a mode from
        # stat() is never S_IFLNK.  Kept for completeness of the noun table.
        noun = "a symlink"
    else:
        noun = "a non-regular file"
    return noun


def _read_sync(  # pylint: disable=too-many-return-statements
    path: Path, display: str, offset: int, limit: int
) -> str:
    """Read *path* and return its lines numbered from 1, or an error string.

    Runs inside :func:`asyncio.to_thread`.  Line numbers are absolute,
    1-indexed positions in the file, not positions within the returned
    window, so a model paging through a large file sees the same number for
    the same line however it slices the reads.  ``offset=0`` and
    ``offset=1`` both mean the first line -- models do not consistently
    0-index.  A missing file, an empty file and an offset past the end are
    three different sentences, because collapsing them would hide which of
    the three happened.

    Two guards run before anything is read, because :meth:`read_text` on the
    wrong object either never comes back or comes back enormous.  A ``stat``
    first refuses any target that is not a regular file -- a named pipe
    would block this worker thread forever, the same fate the walker's
    regular-file check saves Grep from -- and then refuses a regular file
    larger than :data:`MAX_READ_BYTES` from its ``st_size`` alone, so a
    multi-gigabyte file is never loaded whole into memory only to be sliced
    down to a window afterwards.

    Args:
        path: The resolved file to read.
        display: The path as the model spelled it, for error messages.
        offset: The 1-indexed line to start from; 0 and 1 both start at the
            first line, and any value below 1 is treated the same way.
        limit: The maximum number of lines to return; must be at least 1.

    Returns:
        The numbered lines joined with newlines and no trailing newline, or
        a prose error beginning ``"Error: "`` -- including the refusals for
        a non-regular-file target and a file above the byte cap.  Never
        raises.
    """
    if limit <= 0:
        return f"Error: read of {display}: limit must be at least 1, got {limit}."
    try:
        info = path.stat()
    except FileNotFoundError:
        return f"Error: read failed: {display} does not exist."
    except OSError as exc:
        return f"Error: read failed on {display}: {exc}"
    if not stat.S_ISREG(info.st_mode):
        return (
            f"Error: read refused: {display} is {_kind_noun(info.st_mode)}, "
            f"not a regular file; only regular files can be read as text."
        )
    if info.st_size > MAX_READ_BYTES:
        return (
            f"Error: read refused: {display} is {info.st_size} bytes, above "
            f"the {MAX_READ_BYTES}-byte read cap; a file this large is never "
            f"loaded whole."
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        # The stat above saw the file; a deletion in between is the same
        # documented race every other check here shares.
        return f"Error: read failed: {display} does not exist."
    except OSError as exc:
        return f"Error: read failed on {display}: {exc}"
    lines = text.splitlines()
    if not lines:
        return f"Error: {display} is empty; there is nothing to read."
    start = max(offset - 1, 0)
    if start >= len(lines):
        plural = "" if len(lines) == 1 else "s"
        return (
            f"Error: offset {offset} is past the end of {display}: it has "
            f"{len(lines)} line{plural}. Offsets count from 1; use offset 1 "
            f"to start from the first line."
        )
    window = lines[start : start + limit]
    numbered = [
        f"{number}\t{line}" for number, line in enumerate(window, start=start + 1)
    ]
    return "\n".join(numbered)


def _write_sync(path: Path, display: str, content: str) -> str:
    """Create *path* with *content*, refusing to overwrite anything.

    Runs inside :func:`asyncio.to_thread`.  The clobber refusal is the
    whole design: a success message for a destructive overwrite would look
    identical to one for a harmless create, and the model would learn
    nothing until the damage was unreadable.  Parents are created as needed.
    A target that exists and is not a regular file is refused before any
    open: the exists check below would refuse it anyway, but with advice
    ("read the file and use Edit") that a fifo, socket or device cannot
    follow, and writing through one would block or misbehave.

    The refusal is not carried by the pre-checks alone.  They ran in an
    earlier instant, and a concurrent writer can create the target after
    they pass; the create itself is therefore exclusive (`open(path, "x")`,
    O_EXCL at the OS level), and a file that appears in between is refused
    with the same already-exists sentence rather than silently truncated.
    Two concurrent writes thus end one-won, one-refused, whatever the
    interleaving.

    Args:
        path: The resolved file to create.
        display: The path as the model spelled it, for messages.
        content: The exact text to write.  Empty content is legitimate and
            produces an empty file.

    Returns:
        A sentence reporting the character count on success, or a prose
        error beginning ``"Error: "`` naming Edit for the clobber case.
        Never raises.
    """
    if path.is_dir():
        return f"Error: write refused: {display} is a directory."
    if path.exists() and not path.is_file():
        # The stat names the object for the message; if it vanishes between
        # the exists() above and here -- the same race every check in this
        # module shares -- the refusal still fires with a generic noun.
        try:
            kind = _kind_noun(path.stat().st_mode)
        except OSError:
            kind = "a non-regular file"
        return (
            f"Error: write refused: {display} is {kind}, not a regular "
            f"file; Write only creates regular files."
        )
    if path.exists() or path.is_symlink():
        # `is_symlink` too: a broken link reports exists() == False, but
        # writing through it would create the link's target elsewhere --
        # exactly the surprise this tool exists to prevent.
        return (
            f"Error: write refused: {display} already exists. Write never "
            f"overwrites: read the file and use Edit to change it, because "
            f"replacing a whole file to make one change destroys the rest "
            f"without a trace."
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        # The pre-checks above passed an instant ago; a concurrent writer
        # created the target in between.  The exclusive create is the real
        # no-overwrite guarantee -- refuse with the same sentence the
        # pre-check would have produced, never a truncated file.
        return (
            f"Error: write refused: {display} already exists. Write never "
            f"overwrites: read the file and use Edit to change it, because "
            f"replacing a whole file to make one change destroys the rest "
            f"without a trace."
        )
    except OSError as exc:
        return f"Error: write failed on {display}: {exc}"
    return f"Wrote {len(content)} characters to {display}."


def _edit_sync(  # pylint: disable=too-many-return-statements
    path: Path, display: str, old_string: str, new_string: str, replace_all: bool
) -> str:
    """Replace exact occurrences of *old_string* in *path* with *new_string*.

    Runs inside :func:`asyncio.to_thread`.  The replacement is a single
    ``str.replace`` -- forward-scanning, single-pass -- and never a
    hand-written loop: a loop that re-searches from the start after each
    replacement runs forever when ``new_string`` contains ``old_string``
    (``a`` -> ``aa``), while ``str.replace`` terminates by construction
    because it never rescans what it has produced.

    A ``stat`` refuses any target that is not a regular file before the
    read, for the same reason Read refuses one: ``read_text`` on a named
    pipe would block this worker thread forever, and Edit would then be
    waiting to rewrite a file no replacement could describe.

    Every outcome is prose.  No match suggests the nearest lines via
    :func:`difflib.get_close_matches`; more than one match reports the count
    and points at ``replace_all``; success reports how many occurrences
    were replaced and how the line count moved.

    Args:
        path: The resolved file to edit.
        display: The path as the model spelled it, for messages.
        old_string: The exact text to find.  Empty strings are refused:
            ``""`` matches between every character and would rewrite the
            whole file.
        new_string: The replacement text.
        replace_all: Replace every occurrence instead of refusing when the
            match is not unique.

    Returns:
        A sentence describing the replacement on success, or a prose error
        beginning ``"Error: "`` -- including the refusal for a
        non-regular-file target.  Never raises.
    """
    if not old_string:
        return f"Error: edit on {display}: old_string must not be empty."
    if old_string == new_string:
        return (
            f"Error: edit on {display}: new_string is identical to "
            f"old_string; nothing would change."
        )
    try:
        info = path.stat()
    except FileNotFoundError:
        return f"Error: edit failed: {display} does not exist."
    except OSError as exc:
        return f"Error: edit failed on {display}: {exc}"
    if not stat.S_ISREG(info.st_mode):
        return (
            f"Error: edit on {display} refused: it is "
            f"{_kind_noun(info.st_mode)}, not a regular file; Edit only "
            f"rewrites regular files."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: edit failed: {display} does not exist."
    except UnicodeDecodeError:
        return (
            f"Error: edit on {display} failed: the file is not valid UTF-8 "
            f"text; Edit refuses to rewrite a file it cannot decode."
        )
    except OSError as exc:
        return f"Error: edit failed on {display}: {exc}"
    count = text.count(old_string)
    if count == 0:
        lines = text.splitlines()
        if not lines:
            return (
                f"Error: edit on {display}: old_string was not found and "
                f"the file is empty."
            )
        # cutoff=0.0 so the closest lines are always suggested rather than
        # the suggestion vanishing whenever the file is dissimilar; three
        # is enough for the model to recognise the neighbourhood.
        closest = difflib.get_close_matches(old_string, lines, n=3, cutoff=0.0)
        suggestions = ", ".join(f"'{line}'" for line in dict.fromkeys(closest))
        return (
            f"Error: edit on {display}: old_string was not found. The most "
            f"similar existing lines are: {suggestions}. Read the file and "
            f"retry with an exact copy of the text to replace."
        )
    if count > 1 and not replace_all:
        return (
            f"Error: edit on {display}: old_string matches {count} places. "
            f"Include more surrounding text to make it unique, or pass "
            f"replace_all to replace every occurrence."
        )
    # The one replacement.  str.replace, never a scan loop -- see the
    # docstring for why the loop is the bug this line prevents.
    new_text = text.replace(old_string, new_string)
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        return f"Error: edit failed on {display}: {exc}"
    plural = "" if count == 1 else "s"
    return (
        f"Replaced {count} occurrence{plural} in {display}: the file went "
        f"from {len(text.splitlines())} to {len(new_text.splitlines())} lines."
    )


def _glob_sync(
    walk_root: Path, display_root: str, pattern: str, base: Path, include_hidden: bool
) -> str:
    """Walk *walk_root* and list every file matching *pattern*, sorted.

    Runs inside :func:`asyncio.to_thread`; matching follows the rules
    documented on :func:`glob_files`.

    Args:
        walk_root: The resolved directory to walk.
        display_root: The root as the model spelled it, for messages.
        pattern: The fnmatch pattern.  Must be non-empty; the caller checks.
        base: The workspace root, for rendering results the way the model
            spelled them.
        include_hidden: Whether dot-directories are walked -- True when the
            pattern itself starts with a dot.

    Returns:
        The matching paths, one per line, sorted, workspace-relative; or a
        sentence beginning "No files matched" when there are none.  Never
        raises.
    """
    if not walk_root.is_dir():
        if not walk_root.exists():
            return (
                f"Error: glob failed: {display_root} does not exist, so "
                f"there is nothing to walk."
            )
        return f"Error: glob on {display_root} failed: it is not a directory."
    ignore = _load_ignore(walk_root)
    if include_hidden:
        ignore = ignore.with_hidden_directories()
    matches: List[str] = []
    for candidate in _iter_files(walk_root, ignore):
        if "/" in pattern:
            hit = fnmatch(_rel_posix(candidate, walk_root), pattern)
        else:
            hit = fnmatch(candidate.name, pattern)
        if hit:
            matches.append(_rel_posix(candidate, base))
    if not matches:
        return f"No files matched '{pattern}' under {display_root}."
    matches.sort()
    return "\n".join(matches)


def _line_matches(
    line: str, compiled: Optional[re.Pattern[str]], needle: str, case_insensitive: bool
) -> bool:
    """Report whether one line matches, by regex or by substring fallback.

    Args:
        line: The line to test, without its terminator.
        compiled: The compiled regex, or None when the model's pattern was
            not a valid regex and the caller fell back to a substring match.
        needle: The literal pattern text, lowercased when the search is
            case-insensitive.
        case_insensitive: Whether matching ignores case; only consulted on
            the substring path, because the regex already carries
            ``re.IGNORECASE``.

    Returns:
        True when the line matches.  An invalid regex never raises here:
        the caller compiled it once, failed, and passed None.
    """
    if compiled is not None:
        return compiled.search(line) is not None
    if case_insensitive:
        return needle in line.lower()
    return needle in line


def _mode_entries(
    shown: str, matched: List[Tuple[int, str]], output_mode: str
) -> List[str]:
    """Render one file's matches in the requested output mode.

    Args:
        shown: The file's path as the model spelled it.
        matched: The ``(lineno, line)`` pairs that matched, in file order.
        output_mode: One of the three validated modes; anything else is
            treated as ``count`` -- the caller has rejected unknown modes.

    Returns:
        The output lines contributed by this file: the path once for
        ``files_with_matches``, one ``path:lineno:line`` per match for
        ``content``, or a single ``path:count`` for ``count``.
    """
    if output_mode == "files_with_matches":
        return [shown]
    if output_mode == "content":
        return [f"{shown}:{lineno}:{line}" for lineno, line in matched]
    return [f"{shown}:{len(matched)}"]


def _grep_one_file(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    candidate: Path,
    base: Path,
    glob_filter: str,
    compiled: Optional[re.Pattern[str]],
    needle: str,
    case_insensitive: bool,
    output_mode: str,
) -> List[str]:
    """Match one file and return the output lines it contributes.

    Reads the first :data:`BINARY_SNIFF_BYTES` bytes and skips the file when
    they contain a NUL byte -- a regex over bytes that decode to replacement
    soup produces garbage matches.  A skipped file still contributes a
    ``[binary]`` marker in content mode, where a silent skip would be
    invisible.  Filtered-out, unreadable and matchless files contribute
    nothing.

    Args:
        candidate: The file to examine.
        base: The workspace root, for rendering the path.
        glob_filter: A basename fnmatch pattern, or empty for no filtering.
        compiled: The compiled regex, or None for substring fallback.
        needle: The literal pattern, lowercased for case-insensitive search.
        case_insensitive: Whether matching ignores case.
        output_mode: One of the three validated modes.

    Returns:
        The output lines this file contributes -- empty when it is skipped
        or has no matches.  Never raises.
    """
    if glob_filter and not fnmatch(candidate.name, glob_filter):
        return []
    shown = _rel_posix(candidate, base)
    try:
        with open(candidate, "rb") as handle:
            head = handle.read(BINARY_SNIFF_BYTES)
            if _is_binary(head):
                if output_mode == "content":
                    return [f"{shown}: [binary]"]
                return []
            # Byte cap: refuse files larger than MAX_READ_BYTES before reading
            # the rest, to avoid memory spikes on huge files.  This mirrors the
            # cap in Read.
            if handle.tell() == BINARY_SNIFF_BYTES:
                # We only read the sniff bytes so far; check file size via stat
                stat_result = os.fstat(handle.fileno())
                if stat_result.st_size > MAX_READ_BYTES:
                    if output_mode == "content":
                        return [f"{shown}: [file too large, skipped]"]
                    return []
            data = head + handle.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    matched = [
        (lineno, line)
        for lineno, line in enumerate(text.splitlines(), start=1)
        if _line_matches(line, compiled, needle, case_insensitive)
    ]
    if not matched:
        return []
    return _mode_entries(shown, matched, output_mode)


def _grep_sync(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    walk_root: Path,
    display_root: str,
    pattern: str,
    glob_filter: str,
    output_mode: str,
    head_limit: int,
    case_insensitive: bool,
    base: Path,
    include_hidden: bool,
) -> str:
    """Search the files under *walk_root* for *pattern*.

    Runs inside :func:`asyncio.to_thread`.  The pattern is compiled once;
    on :class:`re.error` the search falls back to a substring match, because
    an invalid regex is a model mistake the tools should absorb, not a
    crash.  Files arrive from
    :func:`~mitocode.core.tool.builtin.file_walker._iter_files` in sorted
    order, so
    ``head_limit`` truncates a stable prefix; in ``files_with_matches``
    mode the walk itself stops as soon as the budget is full, because one
    output line per matching file means no later file can change the
    result.

    Args:
        walk_root: The resolved directory to search.
        display_root: The root as the model spelled it, for messages.
        pattern: The regex, or the literal text after a fallback.
        glob_filter: A basename fnmatch pattern restricting which files are
            read, or empty for no restriction.
        output_mode: ``files_with_matches``, ``content`` or ``count``; the
            caller has already rejected anything else.
        head_limit: The maximum number of output lines; a non-positive
            value truncates to none.
        case_insensitive: Whether matching ignores case.
        base: The workspace root, for rendering paths.
        include_hidden: Whether dot-directories are searched -- True when
            the pattern itself starts with a dot.

    Returns:
        The output lines joined with newlines, or a prose error beginning
        ``"Error: "`` when nothing matched.  Never raises.
    """
    if not walk_root.is_dir():
        if not walk_root.exists():
            return (
                f"Error: grep failed: {display_root} does not exist, so "
                f"there is nothing to search."
            )
        return f"Error: grep on {display_root} failed: it is not a directory."
    ignore = _load_ignore(walk_root)
    if include_hidden:
        ignore = ignore.with_hidden_directories()
    try:
        compiled: Optional[re.Pattern[str]] = re.compile(
            pattern, re.IGNORECASE if case_insensitive else 0
        )
    except re.error:
        compiled = None
    needle = pattern.lower() if case_insensitive else pattern
    ceiling = max(head_limit, 0)
    entries: List[str] = []
    for candidate in _iter_files(walk_root, ignore):
        entries.extend(
            _grep_one_file(
                candidate,
                base,
                glob_filter,
                compiled,
                needle,
                case_insensitive,
                output_mode,
            )
        )
        if output_mode == "files_with_matches" and len(entries) >= ceiling:
            # One output line per matching file, so once the budget is full
            # no later file can change the output: stop walking instead of
            # scanning the rest of the tree.  In content and count modes the
            # per-file output is unbounded, so those still scan everything.
            break
    entries = entries[:ceiling]
    if not entries:
        return f"Error: no matches for '{pattern}' under {display_root}."
    return "\n".join(entries)


@tool_boundary
async def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a text file and return its lines numbered from 1.

    Each line is returned as ``"{number}\\t{line}"`` with absolute,
    1-indexed numbers, so paging through a large file shows each line with
    the same number every time; ``limit`` caps how many lines come back.

    Two independent caps apply, and the model should know both exist.
    ``limit`` is a *line* cap, applied to the file after it is read.  A
    separate *byte* cap (:data:`MAX_READ_BYTES`, 10 MiB) refuses any regular
    file larger than the cap outright, from its size alone and before
    anything is read, so a multi-gigabyte file is never loaded whole only to
    be sliced down to a window afterwards.  A target that is not a regular
    file -- a named pipe, a directory -- is refused the same way, before any
    open: reading a named pipe would never return at all.

    Args:
        file_path: The file to read, relative to the workspace root or
            absolute.  Resolved through the approval chokepoint; a path
            whose target leaves the workspace -- a symlink out, for
            instance -- is an escape, decided by the approver exactly once.
        offset: The 1-indexed line to start from.  Values below 1 start at
            the first line; an offset past the end of the file is an error
            that says so, because an empty success would be
            indistinguishable from an empty file.
        limit: The maximum number of lines to return.  Must be at least 1.
            Complementary to the byte cap: this bounds the window, the byte
            cap bounds what is ever read into memory.

    Returns:
        The numbered lines joined with newlines, or a prose message
        beginning ``"Error: "`` for a missing file, an empty file, an
        offset past the end, a non-regular-file target, a file above the
        byte cap, or a refused path -- missing and empty produce different
        messages.  Never raises.
    """
    ctx = current()
    path, error = await resolve(ctx, file_path, "read")
    if path is None:
        return (
            error
            if error is not None
            else "Error: read failed: the path could not be resolved."
        )
    text = await asyncio.to_thread(
        _read_sync, path, _rel_posix(path, ctx.root()), offset, limit
    )
    return clamp(text, ctx.max_output_chars)


@tool_boundary
async def write(file_path: str, content: str) -> str:
    """Create a file with *content*, refusing to overwrite an existing one.

    Parent directories are created as needed.  The refusal on existing
    files is the point of the tool's shape: Write creates, Edit changes,
    and nothing silently replaces a whole file -- a success message cannot
    tell the model that anything was lost.

    Args:
        file_path: The file to create, relative to the workspace root or
            absolute.  Resolved through the approval chokepoint; a target
            outside the workspace, including through a symlinked parent, is
            an escape, decided by the approver exactly once.
        content: The exact text to write.  Empty content creates an empty
            file.

    Returns:
        A sentence reporting the character count on success, or a prose
        error beginning ``"Error: "`` when the file already exists (naming
        Edit as the tool to use), when the target exists and is not a
        regular file, when the path is refused, or when the write fails.
        Never raises: anything unexpected is rendered by
        ``tool_boundary``.
    """
    ctx = current()
    path, error = await resolve(ctx, file_path, "write")
    if path is None:
        return (
            error
            if error is not None
            else "Error: write failed: the path could not be resolved."
        )
    text = await asyncio.to_thread(
        _write_sync, path, _rel_posix(path, ctx.root()), content
    )
    return clamp(text, ctx.max_output_chars)


@tool_boundary
async def edit(
    file_path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """Replace exact text in a file, uniquely unless *replace_all* is set.

    Matches exact strings rather than line numbers -- the only form of
    address that still points at the right place after the file has shifted
    under the model.  One occurrence is replaced; more than one is refused
    with the count unless *replace_all* is set; none suggests the most
    similar existing lines instead.

    Args:
        file_path: The file to edit, relative to the workspace root or
            absolute.  Resolved through the approval chokepoint; a target
            outside the workspace is an escape, decided by the approver
            exactly once.
        old_string: The exact text to find, including any indentation.
            Empty strings are refused, because the empty string matches
            between every character.
        new_string: The replacement text.
        replace_all: Replace every occurrence instead of refusing when the
            match is ambiguous.

    Returns:
        A sentence describing the replacement -- how many occurrences and
        how the line count moved -- or a prose error beginning
        ``"Error: "``, including the refusal for a target that is not a
        regular file.  Never raises.
    """
    ctx = current()
    path, error = await resolve(ctx, file_path, "edit")
    if path is None:
        return (
            error
            if error is not None
            else "Error: edit failed: the path could not be resolved."
        )
    text = await asyncio.to_thread(
        _edit_sync,
        path,
        _rel_posix(path, ctx.root()),
        old_string,
        new_string,
        replace_all,
    )
    return clamp(text, ctx.max_output_chars)


@tool_boundary
async def glob_files(pattern: str, path: str = ".") -> str:
    """List files matching an fnmatch pattern, sorted, one per line.

    Walks the directory at *path* recursively, skipping version-control and
    cache directories, dot-directories (unless *pattern* itself starts with
    a dot), and anything the minimal gitignore at the walk root excludes --
    see the module docstring for exactly what that support does and does
    not do.  A pattern without ``/`` matches basenames at any depth; a
    pattern with ``/`` matches the path relative to the walk root.  Only
    files are listed, never directories.

    Args:
        pattern: The fnmatch pattern, such as ``*.py`` or ``src/*.py``.
            Must be non-empty.
        path: The directory to search, relative to the workspace root or
            absolute.  Defaults to the whole workspace.  A path whose
            target leaves the workspace is an escape, decided by the
            approver exactly once.

    Returns:
        The matching paths, sorted and rendered relative to the workspace
        root; or a sentence beginning "No files matched" when there are
        none, or a prose error beginning ``"Error: "`` for a refused path.
        Never raises: anything unexpected is rendered by ``tool_boundary``.
    """
    ctx = current()
    search_root, error = await resolve(ctx, path, "glob")
    if search_root is None:
        return (
            error
            if error is not None
            else "Error: glob failed: the path could not be resolved."
        )
    if not pattern:
        return "Error: glob: pattern must not be empty."
    text = await asyncio.to_thread(
        _glob_sync,
        search_root,
        _rel_posix(search_root, ctx.root()),
        pattern,
        ctx.root(),
        pattern.startswith("."),
    )
    return clamp(text, ctx.max_output_chars)


@tool_boundary
async def grep_files(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    pattern: str,
    path: str = ".",
    glob: str = "",
    output_mode: Literal[
        "files_with_matches", "content", "count"
    ] = "files_with_matches",
    head_limit: int = 100,
    case_insensitive: bool = False,
) -> str:
    """Search files under *path* for a regex, one output line per result.

    Walks the directory like Glob -- the same skips, the same minimal
    gitignore, the same binary and symlink guards -- and matches each line
    of each file against *pattern*.  The dot rule is *pattern's* own, not
    the glob's: when *pattern* itself starts with a dot, dot-directories
    are searched too, so ``grep_files(".env")`` widens the walk into hidden
    directories while ``grep_files("foo")`` never searches ``.config/`` --
    the same rule Glob states, applied to a regex, where it is far more
    surprising.  An invalid regex falls back to a substring match rather
    than failing.  When nothing matches, the result is an error string
    rather than empty output, so "no matches" is never indistinguishable
    from a tool that did not run.

    Two independent caps apply, and the model should know both exist.
    ``head_limit`` is a *line* cap on the number of output lines.  A
    separate *byte* cap (:data:`MAX_READ_BYTES`, 10 MiB) refuses any regular
    file larger than the cap outright, from its size alone and before
    anything is read, so a multi-gigabyte file is never loaded whole only to
    be scanned.  A target that is not a regular file -- a named pipe, a
    directory -- is refused the same way, before any open: reading a named
    pipe would never return at all.

    Args:
        pattern: The regular expression, or the literal text when the regex
            is invalid.  Must be non-empty.
        path: The directory to search, relative to the workspace root or
            absolute.  Defaults to the whole workspace.  A path whose
            target leaves the workspace is an escape, decided by the
            approver exactly once.
        glob: A basename fnmatch pattern restricting which files are read,
            such as ``*.py``.  Empty means no restriction.
        output_mode: ``files_with_matches`` lists each matching path once;
            ``content`` emits ``path:lineno:line`` per matching line;
            ``count`` emits ``path:n`` per matching file.
        head_limit: The maximum number of output lines.  Non-positive
            values truncate to none.  In ``files_with_matches`` mode the
            walk stops early once this many paths are found; the other
            modes scan every file.
        case_insensitive: Match without regard to case.

    Returns:
        The output lines joined with newlines and clamped to the output
        budget, or a prose error beginning ``"Error: "`` when nothing
        matched or the path was refused.  Binary files are skipped, noted
        with a ``[binary]`` marker in ``content`` mode.
    """
    ctx = current()
    if output_mode not in ("files_with_matches", "content", "count"):
        # `Literal` keeps the model from ever sending anything else, but a
        # direct call can, and the defence costs one line.
        return (
            f"Error: grep: unknown output_mode {output_mode!r}; use "
            f"'files_with_matches', 'content' or 'count'."
        )
    search_root, error = await resolve(ctx, path, "grep")
    if search_root is None:
        return (
            error
            if error is not None
            else "Error: grep failed: the path could not be resolved."
        )
    if not pattern:
        return "Error: grep: pattern must not be empty."
    text = await asyncio.to_thread(
        _grep_sync,
        search_root,
        _rel_posix(search_root, ctx.root()),
        pattern,
        glob,
        output_mode,
        head_limit,
        case_insensitive,
        ctx.root(),
        pattern.startswith("."),
    )
    return clamp(text, ctx.max_output_chars)
