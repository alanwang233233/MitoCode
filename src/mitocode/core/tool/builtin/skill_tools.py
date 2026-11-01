"""The one built-in tool that loads an Agent Skill's instructions: ``Skill``.

The system prompt's skills section lists only each skill's name and
description (progressive-disclosure level 1); this tool is how the model
reaches level 2.  It serves a skill by name from the catalogue the manager
discovered at startup, which is also the boundary: a name outside the
catalogue reads nothing, whatever path it might correspond to.  That is why
this tool exists alongside ``Read`` rather than the model reading a
``SKILL.md`` itself -- user-level skills live outside the workspace, where
the file tools' containment correctly refuses to go, and the catalogue, not
the filesystem, decides what is reachable.

Every failure is prose beginning ``"Error: "`` -- no manager wired, an
unknown name, a skill that vanished after discovery -- matching the rest of
the package, because ``AgentRuntime`` feeds a tool's return value straight
back to the model.  An unknown name names the skills that *do* exist, so a
model that misread the list can recover on its next call without asking the
user.
"""

from __future__ import annotations

from mitocode.core.exception.mitocode_exception import MitoCodeSkillError
from mitocode.core.tool.builtin.context import clamp, current
from mitocode.core.tool.builtin.errors import tool_boundary


@tool_boundary
async def use_skill(name: str) -> str:
    """Return the instruction body of the named skill.

    Args:
        name: A skill name as listed in the system prompt's skills section
            (or shown by ``/skills``).

    Returns:
        The skill's instruction body, clamped to the context's output
        budget, or an ``"Error: ..."`` message: no skills manager is wired,
        the name is unknown (with the available names listed, or an explicit
        "none installed" when the catalogue is empty), or the skill's file
        could no longer be read.
    """
    manager = current().skills_manager
    if manager is None:
        return "Error: no skills are installed."

    if manager.get(name) is None:
        available = [skill.name for skill in manager.list_skills()]
        if not available:
            return f"Error: unknown skill '{name}'. No skills are installed."
        return (
            f"Error: unknown skill '{name}'. Available skills: "
            f"{', '.join(available)}."
        )

    try:
        body = await manager.read_skill(name)
    except MitoCodeSkillError as exc:
        return f"Error: could not load skill '{name}': {exc}"
    return clamp(body, current().max_output_chars)
