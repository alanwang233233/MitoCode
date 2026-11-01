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
`OPENAI_API_KEY`) and a model (`--model` or `MITOCODE_MODEL`) are both
required. A `base_url` from the workspace `.env` or secret file is only
paired with an API key that also comes from the workspace files (either
one); supply `--base-url` (or `MITOCODE_BASE_URL`) yourself to use it
with a key from your environment.

`mitocode` opens a REPL in the current directory; pass `-w/--workspace PATH`
to work elsewhere. Type `/help` for the available slash commands and `/exit`
(or press `Ctrl+D`) to leave. `python -m mitocode` is equivalent, and
`mitocode --version` / `mitocode --help` work without a terminal.
