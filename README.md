# MitoCode

MitoCode is an interactive coding agent that runs in your terminal.

## Interactive CLI

Install the project, then start it from a terminal (the CLI intentionally has
no non-interactive mode):

```bash
pip install -e .
mitocode
```

Provider settings are resolved per key, highest priority first: the
`--api-key`, `--base-url`, `--model` and `--timeout` flags, then the
`MITOCODE_*` environment variables (`OPENAI_*` aliases are also accepted), then
a `.env` file in the workspace root, then the read-only
`.mitocode/secret/provider.json`. An API key (`MITOCODE_API_KEY` or
`OPENAI_API_KEY`) is the minimum required value.

`mitocode` opens a REPL in the current directory; pass `-w/--workspace PATH`
to work elsewhere. Type `/help` for the available slash commands and `/exit`
(or press `Ctrl+D`) to leave. `python -m mitocode` is equivalent, and
`mitocode --version` / `mitocode --help` work without a terminal.
