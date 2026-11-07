# MitoCode

MitoCode is an interactive coding agent that runs in your terminal.

## Interactive CLI

Install the project, then start it from a terminal (the CLI intentionally has
no non-interactive mode):

```bash
pip install -e .
mitocode
```

Provider settings come from one place only: the workspace-local
`.mitocode/secret/provider.json`. An API key and a model are both required.
The first time mitocode sees a directory without a workspace, it asks before
creating `.mitocode/` there (declining exits), and the first time it sees
that file is missing, it collects the credentials interactively and writes
them with private permissions (`0o700` directory, `0o600` file). Nothing is
read from the environment, and nothing is written outside the workspace.

`mitocode` opens a REPL in the current directory; pass `-w/--workspace PATH`
to work elsewhere. Type `/help` for the available slash commands and `/exit`
(or press `Ctrl+D`) to leave. `python -m mitocode` is equivalent, and
`mitocode --version` / `mitocode --help` work without a terminal.
