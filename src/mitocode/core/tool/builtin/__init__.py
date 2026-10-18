"""Built-in tools for MitoCode: workspace-confined file, shell, and agent tools.

This package is assembled bottom-up.  ``approval`` is the approval protocol;
``context``, ``paths``, ``errors``, ``call_dsl`` and ``background`` arrive in
later steps, so this module deliberately re-exports nothing yet and keeps no
imports — importing a sibling that does not exist would make the whole package
unimportable mid-build.
"""
