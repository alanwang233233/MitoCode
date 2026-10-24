"""The shared file-walking machinery behind the Glob and Grep tools.

Split out of :mod:`~mitocode.core.tool.builtin.fs_tools`, which keeps the
five file tools themselves.  This module holds what their walks share and
nothing else: it registers no tools, and it imports nothing from
``fs_tools`` -- the dependency points one way, from the tools down to the
walker they enumerate through.

Three pieces live here:

* the minimal gitignore matcher, :class:`_Ignore` loaded by
  :func:`_load_ignore` -- minimal by design and documented as such: one
  ``.gitignore`` at the walk root, per-line :func:`fnmatch.fnmatch`
  matching, ``!`` negation with last match winning, and nothing more;
* :func:`_iter_files`, the sorted walker.  Its containment comparison
  scopes what the walk *enumerates* -- it can only omit a candidate, never
  operate outside one -- so it is walk scoping, not a verdict on a
  requested path.  The ruling that
  :func:`~mitocode.core.tool.builtin.paths.resolve` is the sole authority
  on a requested path, and the asymmetry that follows for Glob and Grep,
  is stated in the fs_tools module docstring and in :func:`_iter_files`'s
  own;
* :func:`_is_binary`, the NUL-byte sniff, with :data:`BINARY_SNIFF_BYTES`
  -- the prefix length it and fs_tools' grep share -- plus
  :func:`_rel_posix`, the relative-path renderer the walk uses for
  gitignore matching and fs_tools imports back for rendering paths in
  output and error messages.
"""

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

#: Directories never entered, whatever the pattern or the ``.gitignore`` says.
#: These are version-control internals and bytecode caches: nothing an agent
#: reads there is useful, and walking them is pure cost.
ALWAYS_SKIP_DIRECTORIES = frozenset({".git", ".hg", ".svn", "__pycache__"})

#: How many leading bytes :func:`_is_binary` inspects for a NUL byte.  Eight
#: kilobytes is enough for the magic numbers of every common binary format
#: while keeping the sniff cheap on a large tree.
BINARY_SNIFF_BYTES = 8192

#: One parsed line of a ``.gitignore``: the fnmatch pattern, and whether the
#: line was negated with a leading ``!``.
IgnoreRule = Tuple[str, bool]


class _Ignore:  # pylint: disable=too-few-public-methods
    """The minimal gitignore matcher shared by Glob and Grep.

    Holds ``(pattern, negated)`` pairs loaded from a single ``.gitignore`` at
    a walk root, matched with :func:`fnmatch.fnmatch` against the basename
    for bare patterns and against the relative posix path for patterns
    containing ``/``.  *Not* gitignore semantics: no nested files, no
    ``**``, no directory-only rules, no anchoring.

    Attributes:
        prune_hidden_dirs: Whether dot-directories are pruned during the
            walk.  True by default -- the minimal reading skips them -- and
            cleared when the caller's own pattern starts with a dot, which is
            the one signal that the model is deliberately asking about
            hidden files.
    """

    def __init__(
        self, patterns: Sequence[IgnoreRule] = (), prune_hidden_dirs: bool = True
    ) -> None:
        """Store the rules and the hidden-directory policy.

        Args:
            patterns: The ``(pattern, negated)`` pairs, in file order; later
                pairs override earlier ones.
            prune_hidden_dirs: Whether dot-directories are pruned.  See the
                class docstring.
        """
        self._patterns: List[IgnoreRule] = list(patterns)
        self.prune_hidden_dirs = prune_hidden_dirs

    def with_hidden_directories(self) -> "_Ignore":
        """Return a copy that walks dot-directories instead of pruning them.

        Returns:
            A new :class:`_Ignore` carrying the same rules with
            ``prune_hidden_dirs`` cleared.  A copy, not a mutation, because
            whether hidden directories are walked is a per-request decision
            and must not leak into a matcher another request reuses.
        """
        return _Ignore(self._patterns, prune_hidden_dirs=False)

    def excludes(self, rel_path: str, name: str) -> bool:
        """Decide whether a path is excluded, the last matching rule winning.

        Last-match-wins is what makes negation usable: for ``*.log`` followed
        by ``!keep.log``, a candidate named ``keep.log`` is matched by both
        rules in order and the negated one, being last, decides.  A pattern
        containing ``/`` is matched against the relative path; bare patterns
        match the basename, so ``*.log`` matches at any depth.

        Args:
            rel_path: The candidate's path relative to the walk root, in
                posix form (``src/a.py``).  Only consulted for patterns that
                contain a slash.
            name: The candidate's basename (``a.py``).

        Returns:
            True when the last matching rule excludes the candidate, False
            when no rule matches or the last match negates the exclusion.
        """
        excluded = False
        for pattern, negated in self._patterns:
            if "/" in pattern:
                hit = fnmatch(rel_path, pattern)
            else:
                hit = fnmatch(name, pattern)
            if hit:
                excluded = not negated
        return excluded


def _load_ignore(root: Path) -> _Ignore:
    """Parse the ``.gitignore`` at *root*, or return an empty matcher.

    Only the file at the walk root is read; nested ``.gitignore`` files are
    part of full gitignore semantics and are deliberately not supported.
    Blank lines and ``#`` comments are skipped, a leading ``!`` marks the
    rule as a negation, and any read failure yields the empty matcher: the
    walk proceeds unfiltered rather than failing over an optional hint file.

    Args:
        root: The walk root whose ``.gitignore`` should be loaded -- the
            workspace root itself for the common ``path="."`` requests.

    Returns:
        An :class:`_Ignore` with one rule per meaningful line, in file order.
    """
    rules: List[IgnoreRule] = []
    try:
        raw = (root / ".gitignore").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        # No file, unreadable file, or a path that cannot even be opened:
        # the minimal matcher answers "nothing is excluded" rather than
        # failing the whole walk over an optional hint file.
        return _Ignore(rules)
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        if not line:
            continue
        rules.append((line, negated))
    return _Ignore(rules)


def _is_binary(data: bytes) -> bool:
    """Report whether *data* looks binary: a NUL byte in the first 8 KiB.

    A NUL byte is the cheapest reliable signal: every text encoding a
    workspace file is likely to use never produces one, while virtually
    every binary format contains them early.  Encoding heuristics are
    deliberately not attempted; they misfire, and treating compressed or
    encrypted files as binary is correct anyway.

    Args:
        data: The bytes to sniff.  Callers normally pass the first
            :data:`BINARY_SNIFF_BYTES` bytes of a file, but any prefix works.

    Returns:
        True when ``b"\\x00"`` appears within the sniffed prefix, False
        otherwise.
    """
    return b"\x00" in data[:BINARY_SNIFF_BYTES]


def _rel_posix(path: Path, root: Path) -> str:
    """Return *path* relative to *root* as a posix-style string.

    Used both for gitignore matching inside the walk and for rendering
    paths in output and error messages, so a model always sees the same
    shape it used to ask for the file.

    Args:
        path: The path to render, normally resolved and absolute.
        root: The root to render against -- the walk root for matching, the
            workspace root for display.

    Returns:
        ``src/a.py``-style text when *path* is inside *root*.  Falls back to
        the absolute path when it is not -- possible only if the tree changed
        mid-walk, or when *path* is an approved escape and genuinely lives
        outside the root -- so no caller ever sees a wrong relative form.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _iter_files(root: Path, ignore: _Ignore) -> Iterator[Path]:
    """Yield every regular file under *root* the minimal gitignore allows.

    The shared walker behind Glob and Grep.  Sorted, so both tools present
    deterministic output no matter how the filesystem orders its entries.
    ``.git``, ``.hg``, ``.svn`` and ``__pycache__`` are never entered;
    dot-directories are pruned unless *ignore* says otherwise; a directory
    symlink is never descended into -- ``os.walk`` runs with
    ``followlinks=False`` -- so the walk never leaves the tree it was
    pointed at.

    Every yielded candidate is re-checked for containment after
    :meth:`pathlib.Path.resolve`: a file symlink that points outside the
    walk root resolves outside and is skipped, because the walk enumerates
    the tree rooted at *root* and nothing beyond it.  That scopes what the
    walk reports; it is not a verdict on the requested path, which belongs
    to :func:`~mitocode.core.tool.builtin.paths.resolve` alone.  Candidates
    that are not regular files -- broken symlinks, fifos, devices -- are
    skipped too, which keeps Grep from blocking forever on a named pipe.

    Args:
        root: The directory to walk, normally a resolved absolute path from
            :func:`resolve`.
        ignore: The matcher from :func:`_load_ignore`, possibly adjusted for
            hidden directories by the caller.

    Returns:
        An iterator over the files, sorted by path.  The walk completes
        before the iterator is returned; the tools call this inside
        :func:`asyncio.to_thread`, so the eager walk runs off the loop.

    Raises:
        Nothing is expected: ``os.walk`` with the default error handler
        ignores unreadable directories, and every per-candidate call is
        guarded.
    """
    try:
        boundary = root.resolve()
    except (OSError, RuntimeError, ValueError):
        boundary = root
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        kept: List[str] = []
        for name in dirnames:
            if name in ALWAYS_SKIP_DIRECTORIES:
                continue
            if ignore.prune_hidden_dirs and name.startswith("."):
                continue
            child = here / name
            if ignore.excludes(_rel_posix(child, root), name):
                continue
            kept.append(name)
        # Pruning in place is what stops os.walk from descending; a plain
        # `dirnames = kept` would leave the walk oblivious.
        dirnames[:] = kept
        for name in filenames:
            candidate = here / name
            if ignore.excludes(_rel_posix(candidate, root), name):
                continue
            if not candidate.is_file():
                # Broken symlink, fifo, device: nothing a tool can read.
                continue
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if not resolved.is_relative_to(boundary):
                continue
            found.append(candidate)
    found.sort()
    return iter(found)
