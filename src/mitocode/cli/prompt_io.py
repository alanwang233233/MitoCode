"""Production input port for the REPL: a prompt_toolkit thin shell.

The bindings stay deliberately dumb — Esc asks the REPL to interrupt the
running turn, Ctrl+C prints the localized exit hint, and Ctrl+D falls
through as the ``EOFError`` the REPL loop treats as exit. Every decision
lives in :class:`~mitocode.cli.repl.CliRepl`; this module only binds real
keys to it and wraps the session in ``patch_stdout`` so streaming output
prints above a stable input line.
"""

from typing import Any

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession

from mitocode.i18n import t


async def run_with_prompt_toolkit(repl: Any) -> None:
    """Bind *repl* to a real terminal and run it.

    Args:
        repl: A :class:`~mitocode.cli.repl.CliRepl` whose ``input_reader``
            is replaced with a prompt_toolkit-backed reader. The call
            returns when the REPL's main loop exits.
    """
    bindings = KeyBindings()

    @bindings.add("escape", eager=True)
    def _escape(event: Any) -> None:  # pylint: disable=unused-argument
        """Esc: interrupt the agent's running turn, if any."""
        repl.request_interrupt()

    @bindings.add("c-c")
    def _ctrl_c(event: Any) -> None:  # pylint: disable=unused-argument
        """Ctrl+C: print how to leave; interrupting is what Esc is for."""
        repl.console.print(t("cli.cli.exit_hint"), style="dim")

    session: PromptSession = PromptSession(key_bindings=bindings)

    async def reader(prompt_text: str) -> str:
        return await session.prompt_async(prompt_text)

    repl.input_reader = reader
    with patch_stdout(raw=True):
        await repl.run()
