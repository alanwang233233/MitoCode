"""Package entrypoint for `python -m mitocode`."""

from __future__ import annotations

import sys

from mitocode.cli import main


if __name__ == "__main__":
    argv = sys.argv[1:] or ["tui"]
    raise SystemExit(main(argv))
