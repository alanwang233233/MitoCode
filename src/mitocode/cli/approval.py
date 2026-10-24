"""Interactive approver with persisted pattern rules.

The two-valued :class:`ApprovalDecision` protocol is untouched — "always
allow" is a UI choice that derives a rule, persists it to
``.mitocode/approval.json`` and returns ``ALLOW``, so the deprecated TUI
path and every existing approval test are unaffected.

Prompting goes through an injected ``ask`` port rather than a terminal:
in the REPL the port is the approval broker on :class:`CliRepl`, which
renders the question above the standing input line and routes the user's
next typed line to this approver while the calling tool stays suspended.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rich.console import Console

from mitocode.core.tool.builtin.approval import (
    ApprovalDecision,
    ApprovalRequest,
)
from mitocode.i18n import t

#: Location of the persisted approval rules, workspace-relative.
APPROVAL_FILE_RELATIVE = Path(".mitocode") / "approval.json"

#: Type of the injected ask port: renders the prompt, returns the answer.
AskPort = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class ApprovalRule:
    """One persisted "always allow" rule.

    Attributes:
        action: The verb the rule applies to (``bash`` / ``read`` / ...);
            a rule never widens a different action.
        pattern: Bash command pattern in the ``npm install:*`` style;
            only meaningful for ``action="bash"``.
        path_prefix: Directory prefix for file operations; only
            meaningful for non-bash actions.
    """

    action: str
    pattern: Optional[str] = None
    path_prefix: Optional[str] = None

    def matches(self, request: ApprovalRequest) -> bool:
        """Return True when *request* is covered by this rule.

        Args:
            request: The approval request being checked.

        Returns:
            True when the request's action and detail fall inside the
            rule; False otherwise.
        """
        if request.action != self.action:
            return False
        if self.pattern is not None and request.action == "bash":
            return _command_matches(request.detail, self.pattern)
        if self.path_prefix is not None and request.action != "bash":
            return _path_matches(request.detail, self.path_prefix)
        return False

    def describe(self) -> str:
        """Return the human-readable rule spelling shown in prompts."""
        if self.pattern is not None:
            return f"{self.action}({self.pattern})"
        return f"{self.action}({self.path_prefix}/**)"


def _command_matches(command: str, pattern: str) -> bool:
    """Match a bash command against a CC-style pattern.

    Args:
        command: The command line from the approval request.
        pattern: Either ``prefix:*`` (matches *command* equal to or
            extending *prefix* by a word boundary) or a literal that must
            match exactly.

    Returns:
        True when the command is covered by the pattern.
    """
    if pattern.endswith(":*"):
        prefix = pattern[:-2]
        return command == prefix or command.startswith(prefix + " ")
    return command == pattern


def _path_matches(path: str, prefix: str) -> bool:
    """Match a resolved path against a directory prefix.

    Args:
        path: The absolute path from the approval request.
        prefix: The directory the rule allows.

    Returns:
        True when *path* is *prefix* itself or lies directly beneath it
        (separator boundary — ``/etc`` does not match ``/etcetera``).
    """
    normalized = prefix.rstrip("/") or "/"
    return path == normalized or path.startswith(normalized + os.sep)


def derive_rule(request: ApprovalRequest) -> ApprovalRule:
    """Derive the tightest sensible "always allow" rule for *request*.

    Args:
        request: The request the user chose to always allow.

    Returns:
        For bash, a ``first two tokens:*`` command pattern (``npm
        install:*``, never a bare ``npm:*`` that would allow
        ``npm publish``); for file operations, the target's parent
        directory as a path prefix.
    """
    if request.action == "bash":
        tokens = request.detail.split()
        prefix = " ".join(tokens[:2]) if len(tokens) >= 2 else " ".join(tokens)
        return ApprovalRule(action="bash", pattern=prefix + ":*")
    return ApprovalRule(
        action=request.action, path_prefix=str(Path(request.detail).parent)
    )


def _rule_to_dict(rule: ApprovalRule) -> Dict[str, Any]:
    """Serialize one rule to its persisted dict form."""
    data: Dict[str, Any] = {"action": rule.action}
    if rule.pattern is not None:
        data["pattern"] = rule.pattern
    if rule.path_prefix is not None:
        data["path_prefix"] = rule.path_prefix
    return data


def _rule_from_dict(raw: Any) -> Optional[ApprovalRule]:
    """Parse one rule dict, returning None when it is not a valid rule.

    Args:
        raw: A loaded JSON value.

    Returns:
        The rule, or None when *raw* is not a dict with an ``action`` and
        at least one of ``pattern`` / ``path_prefix``.
    """
    if not isinstance(raw, dict):
        return None
    action = raw.get("action")
    pattern = raw.get("pattern")
    path_prefix = raw.get("path_prefix")
    if not isinstance(action, str) or not action:
        return None
    if pattern is None and path_prefix is None:
        return None
    return ApprovalRule(action=action, pattern=pattern, path_prefix=path_prefix)


class ApprovalRuleStore:
    """Persistence for approval rules in ``.mitocode/approval.json``."""

    def __init__(self, root: Path) -> None:
        """Load rules for the workspace rooted at *root*.

        Args:
            root: Workspace root directory.
        """
        self._path = root / APPROVAL_FILE_RELATIVE
        self._rules: List[ApprovalRule] = self._load()

    def _load(self) -> List[ApprovalRule]:
        """Read the rules file, degrading to an empty list."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        rules = []
        for raw in data.get("rules", []):
            rule = _rule_from_dict(raw)
            if rule is not None:
                rules.append(rule)
        return rules

    def rules(self) -> List[ApprovalRule]:
        """Return a copy of the loaded rules."""
        return list(self._rules)

    def match(self, request: ApprovalRequest) -> Optional[ApprovalRule]:
        """Return the first rule covering *request*, or None.

        Args:
            request: The approval request being checked.

        Returns:
            The matching rule, or None when no rule covers the request.
        """
        for rule in self._rules:
            if rule.matches(request):
                return rule
        return None

    def add(self, rule: ApprovalRule) -> None:
        """Append *rule* and persist the full list.

        Args:
            rule: The rule to remember.
        """
        self._rules.append(rule)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rules": [_rule_to_dict(item) for item in self._rules]}
        self._path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class InteractiveApprover:
    """Three-choice interactive approver backed by a rule store.

    Choice 1 allows once; choice 2 derives a rule, shows it for
    confirmation and allows; choice 3 denies. A request already covered
    by a persisted rule never prompts.
    """

    def __init__(self, root: Path, ask: AskPort, console: Console) -> None:
        """Wire the approver to its workspace and I/O ports.

        Args:
            root: Workspace root directory.
            ask: Injected ask port; in the REPL this is the approval
                broker that renders the question above the input line.
            console: Console for the question and feedback.
        """
        self._store = ApprovalRuleStore(root)
        self._ask = ask
        self._console = console

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        """Decide *req*, prompting through the ask port when no rule hits.

        Args:
            req: The operation awaiting a verdict.

        Returns:
            ``ALLOW`` or ``DENY``; never raises.
        """
        if self._store.match(req) is not None:
            return ApprovalDecision.ALLOW

        self._console.print(
            t("cli.approval.prompt", tool=req.action, reason=req.reason),
            style="yellow",
        )
        self._console.print(t("cli.approval.choices"))
        while True:
            answer = (await self._ask(t("cli.approval.pick"))).strip()
            if answer == "1":
                return ApprovalDecision.ALLOW
            if answer == "3":
                self._console.print(t("cli.approval.denied"), style="red")
                return ApprovalDecision.DENY
            if answer == "2":
                return await self._always(req)
            self._console.print(t("cli.approval.invalid"), style="yellow")

    async def _always(self, req: ApprovalRequest) -> ApprovalDecision:
        """Handle the "always allow" choice: confirm, persist, allow.

        Args:
            req: The request being permanently allowed.

        Returns:
            Always ``ALLOW`` — confirmation only decides whether the
            rule is persisted.
        """
        rule = derive_rule(req)
        confirm = (
            await self._ask(t("cli.approval.confirm_rule", rule=rule.describe()))
        ).strip()
        if confirm.lower() in ("y", "yes"):
            self._store.add(rule)
            self._console.print(
                t("cli.approval.rule_saved", rule=rule.describe()), style="green"
            )
        return ApprovalDecision.ALLOW
