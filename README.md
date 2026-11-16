# MitoCode

MitoCode is an interactive coding agent that runs in your terminal.

## Platform support

MitoCode runs on Linux and macOS, and its test suite runs on Windows too.
The `bash` tool is the one POSIX-only piece: it needs `/bin/sh`, process
groups and `pgrep`, so on Windows it is skipped rather than emulated.
Everything else — the CLI, the session store, the file tools and the
workspace lock — works on all three.

The private-permission promise below (`0o700` directory, `0o600` file) is a
POSIX one. Windows has no such mode bits: `os.chmod` there only toggles the
read-only attribute and access is decided by the directory's ACL. On
Windows, keep the workspace somewhere only your own account can read.

## Interactive CLI

Install the project, then start it from a terminal (the CLI intentionally has
no non-interactive mode):

```bash
pip install -e .
mitocode
```

Provider settings come from `.mitocode/secret/provider.json` in the workspace.
A user-level copy under the global configuration directory (`~/.mitocode` on
Windows, `~/.config/.mitocode` on macOS and Linux) is read when the workspace
has none. An API key and a model are both required.
The first time mitocode sees a directory without a workspace, it asks before
creating `.mitocode/` there (declining exits), and the first time it sees
that file is missing, it collects the credentials interactively and writes
them with private permissions (`0o700` directory, `0o600` file; see the
platform note above). Nothing is read from the environment, and credentials
are never written outside the workspace.

`mitocode` opens a REPL in the current directory; pass `-w/--workspace PATH`
to work elsewhere. Type `/help` for the available slash commands and `/exit`
(or press `Ctrl+D`) to leave. `python -m mitocode` is equivalent, and
`mitocode --version` / `mitocode --help` work without a terminal.

## Sessions

Every conversation is persisted under `<workspace>/.mitocode/sessions/` as one
JSON file per session. The directory is created with `0o700` and each record is
written with `0o600`, so session history stays private to the workspace on
POSIX; on Windows the same files are protected by the workspace's own
directory permissions instead.

- **Startup choice**: with no saved session the CLI starts a new one; with
  exactly one it restores it directly and echoes the previous conversation as
  context; with two or more a panel lists them newest-first and you pick by
  number (`n` starts a new session). Files that cannot be read or validated
  are skipped with one warning.
- **`/save`**: force-saves the current session to disk right away.
- **`/sessions`**: saves the current session, lists the saved ones and switches
  to the chosen record (`n` = new session). If the session is still running —
  an agent task, a running turn or a background tool — MitoCode asks
  `Force stop and switch?` first.
- **`/demo`**: replays a saved session as a speed showcase of the agent's run,
  at `1x` by default (`←`/`→` change the speed, `space` pauses/resumes, `q`,
  `Esc` or `Ctrl+C` quit). The replay never opens the shown session: when it
  ends you press any key to return, and the prompt comes back to the session
  you were in. Playback never calls the model or changes any session history.
- **Titles and autosave**: a fresh session is titled `New Session`; after the
  first turn a title is generated in the background from the first user
  message, and an autosave runs every 30 seconds plus at the end of every
  turn and on exit.
- **Failure fallback**: if a saved session cannot be loaded or restored, the
  CLI warns and starts a new session instead — startup never aborts over a
  broken session file, and a failed restore never overwrites the old file.
