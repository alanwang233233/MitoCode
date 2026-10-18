"""Unit tests for the built-in approval protocol."""

import dataclasses
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.tool.builtin.approval import (
    AlwaysAllow,
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    DenyAll,
)

REQUEST = ApprovalRequest(action="bash", detail="cat /etc/passwd", reason="outside root")


@pytest.mark.asyncio
async def test_always_allow_allows_everything():
    assert await AlwaysAllow().request(REQUEST) is ApprovalDecision.ALLOW


@pytest.mark.asyncio
async def test_deny_all_denies_everything():
    assert await DenyAll().request(REQUEST) is ApprovalDecision.DENY


def test_request_carries_action_detail_and_reason():
    assert (REQUEST.action, REQUEST.detail) == ("bash", "cat /etc/passwd")
    assert "outside root" in REQUEST.reason


# --- Approver protocol conformance (runtime_checkable) ------------------------


def test_always_allow_satisfies_approver_protocol():
    """The shipped default must pass the wiring-time isinstance sanity check."""
    assert isinstance(AlwaysAllow(), Approver), (
        "AlwaysAllow does not satisfy Approver; the shipped default would be "
        "rejected at wiring time"
    )


def test_deny_all_satisfies_approver_protocol():
    """The refusal double must pass the wiring-time isinstance sanity check."""
    assert isinstance(DenyAll(), Approver), (
        "DenyAll does not satisfy Approver; the safe fallback approver would be "
        "rejected at wiring time"
    )


def test_approver_rejects_object_without_request_method():
    """A missing ``request`` is the one disqualifier the structural check applies."""
    class UnrelatedApprover:
        """Structurally wrong: the decision method is not named ``request``."""

        async def decide(self, req):  # noqa: ARG002
            return ApprovalDecision.ALLOW

    assert not isinstance(object(), Approver), (
        "a bare object was accepted as an Approver"
    )
    assert not isinstance(UnrelatedApprover(), Approver), (
        "an object whose decision method is named `decide` was accepted as an "
        "Approver; the protocol's required surface is exactly `request`"
    )


def test_approver_rejects_request_attribute_set_to_none():
    """``request = None`` counts as absent, so a half-built approver is rejected."""
    class NullRequestApprover:
        """A placeholder approver whose method was never implemented."""

        request = None

    assert not isinstance(NullRequestApprover(), Approver), (
        "an object whose `request` is None was accepted as an Approver"
    )


def test_approver_isinstance_ignores_signature_and_asyncness():
    """Documented limit: the structural check tests presence of ``request`` only.

    ``runtime_checkable`` cannot validate a member's signature or its async-ness,
    so an object with a synchronous or wrongly-shaped ``request`` still passes.
    The protocol docstring only promises "a cheap sanity check", so this pins the
    limit rather than asserting signature validation that never happens.
    """
    class SyncApprover:
        """Wrong: ``request`` is synchronous, so ``await`` would fail at the call site."""

        def request(self, req):  # noqa: ARG002
            return ApprovalDecision.ALLOW

    class WrongParamsApprover:
        """Wrong: ``request`` takes a different parameter list."""

        async def request(self, prompt, timeout):  # noqa: ARG002
            return ApprovalDecision.ALLOW

    class LambdaApprover:
        """Wrong: ``request`` is a lambda returning a value, not a coroutine."""

    LambdaApprover.request = lambda self, req: ApprovalDecision.ALLOW  # noqa: ARG005

    for cls in (SyncApprover, WrongParamsApprover, LambdaApprover):
        assert isinstance(cls(), Approver), (
            f"{cls.__name__} was rejected by isinstance even though it exposes a "
            "`request` attribute; if this now fails, the check has started "
            "validating signatures, which callers may now depend on"
        )


def test_approver_protocol_declares_only_the_request_method():
    """``Approver`` must stay a pure method protocol -- no data members.

    A non-method member would make ``issubclass()`` raise ``TypeError``; keeping
    the protocol method-only is what lets both checks stay usable.
    """
    public_members = [name for name in vars(Approver) if not name.startswith("_")]

    assert public_members == ["request"], (
        f"Approver declares {public_members}; only the method `request` is "
        "allowed, a data member would break issubclass() with TypeError"
    )


@pytest.mark.parametrize(
    "approver_cls", [Approver, AlwaysAllow, DenyAll], ids=lambda cls: cls.__name__
)
def test_request_is_a_coroutine_function(approver_cls):
    """Every ``request`` is async, because every call site awaits it."""
    assert inspect.iscoroutinefunction(approver_cls.request), (
        f"{approver_cls.__name__}.request is not a coroutine function; callers "
        "await it, so it must be declared `async def`"
    )


def test_issubclass_against_approver_does_not_raise():
    """``issubclass`` stays usable: it raises TypeError for data protocols."""
    assert issubclass(AlwaysAllow, Approver), (
        "issubclass(AlwaysAllow, Approver) is False; the structural check no "
        "longer recognises the shipped default"
    )
    assert issubclass(DenyAll, Approver), (
        "issubclass(DenyAll, Approver) is False; the structural check no longer "
        "recognises the shipped approvers"
    )


# --- ApprovalDecision wire contract -------------------------------------------


def test_decision_has_exactly_allow_and_deny_members():
    """The verdict vocabulary is exactly two members, so no policy can widen it."""
    assert set(ApprovalDecision.__members__) == {"ALLOW", "DENY"}, (
        f"unexpected verdict vocabulary: {sorted(ApprovalDecision.__members__)}"
    )


def test_decision_values_are_the_allow_and_deny_wire_strings():
    """``.value`` is the string a refusal message quotes back to the model."""
    assert ApprovalDecision.ALLOW.value == "allow", (
        f"ALLOW.value drifted to {ApprovalDecision.ALLOW.value!r}; refusal "
        "strings built from it are a wire contract"
    )
    assert ApprovalDecision.DENY.value == "deny", (
        f"DENY.value drifted to {ApprovalDecision.DENY.value!r}; refusal "
        "strings built from it are a wire contract"
    )


def test_decision_is_looked_up_from_its_wire_value():
    """A decision reconstructed from a stored string resolves to the same member."""
    assert ApprovalDecision("allow") is ApprovalDecision.ALLOW, (
        'lookup of "allow" did not resolve to the ALLOW member'
    )
    assert ApprovalDecision("deny") is ApprovalDecision.DENY, (
        'lookup of "deny" did not resolve to the DENY member'
    )


def test_decision_rejects_an_unknown_wire_value():
    """An unrecognised wire value is refused rather than silently defaulted."""
    with pytest.raises(ValueError, match="ALLOW"):
        ApprovalDecision("ALLOW")


# --- ApprovalRequest dataclass shape ------------------------------------------


def test_request_is_a_dataclass_with_fields_in_declared_order():
    """The field order is action, detail, reason -- the order a dialog reads."""
    assert dataclasses.is_dataclass(ApprovalRequest), (
        "ApprovalRequest is not a dataclass; a plain class would not compare by value"
    )
    assert tuple(f.name for f in dataclasses.fields(ApprovalRequest)) == (
        "action",
        "detail",
        "reason",
    ), (
        "request fields are not (action, detail, reason); a dialog reading them "
        "positionally would show the wrong values"
    )


def test_request_accepts_keyword_construction():
    """The request is constructible with keywords, as every call site does."""
    built = ApprovalRequest(action="write", detail="/etc/hosts", reason="outside root")

    assert (built.action, built.detail, built.reason) == (
        "write",
        "/etc/hosts",
        "outside root",
    ), f"keyword construction mis-bound the fields: {built!r}"


def test_request_positional_construction_follows_field_order():
    """Positional construction binds action, detail, reason in that order."""
    built = ApprovalRequest("edit", "/tmp/x", "outside root")

    assert (built.action, built.detail, built.reason) == (
        "edit",
        "/tmp/x",
        "outside root",
    ), f"positional construction mis-bound the fields: {built!r}"


def test_request_equality_compares_all_three_fields():
    """Two identically populated requests are equal; a differing field is not."""
    same = ApprovalRequest(action="bash", detail="cat /etc/passwd", reason="outside root")
    other_reason = ApprovalRequest(
        action="bash", detail="cat /etc/passwd", reason="symlink escape"
    )

    assert REQUEST == same, (
        "two requests with identical fields compared unequal; equality is "
        "field-wise by contract"
    )
    assert REQUEST != other_reason, (
        "requests differing only in `reason` compared equal; a dialog would show "
        "the wrong justification"
    )


def test_request_is_mutable_so_a_dialog_can_annotate_it():
    """Not frozen: a dialog normalises the request before showing it."""
    built = ApprovalRequest(action="bash", detail="echo " + "a" * 500, reason="outside root")

    built.detail = built.detail[:80] + " ..."
    built.reason = "command escapes the workspace root"

    assert built.detail.endswith(" ..."), (
        f"normalising .detail did not stick: {built.detail[-20:]!r}"
    )
    assert built.reason == "command escapes the workspace root", (
        f"annotating .reason did not stick: {built.reason!r}"
    )


def test_request_stores_an_unknown_action_verbatim():
    """It is a plain value object: no validation, no coercion, no enum."""
    built = ApprovalRequest(action="delete", detail="/", reason="outside root")

    assert built.action == "delete", (
        f"action was rewritten to {built.action!r}; the request carries the "
        "caller's vocabulary verbatim"
    )


# --- A third, custom Approver --------------------------------------------------


class RecordingApprover:
    """A duck-typed approver that is neither AlwaysAllow nor DenyAll."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.seen = []

    async def request(self, req):
        self.seen.append(req)
        return self.verdict


def test_custom_approver_satisfies_the_protocol_without_inheriting():
    """A third implementation needs no base class to be wired in."""
    custom = RecordingApprover(ApprovalDecision.ALLOW)

    assert isinstance(custom, Approver), (
        "a custom duck-typed approver was rejected by the protocol check"
    )
    assert not isinstance(custom, (AlwaysAllow, DenyAll)), (
        "the custom approver must remain a third distinct implementation"
    )


@pytest.mark.asyncio
async def test_custom_approver_answers_through_the_same_call_shape():
    """Its own verdict reaches the caller through the protocol's call shape."""
    custom = RecordingApprover(ApprovalDecision.ALLOW)

    verdict = await custom.request(REQUEST)

    assert verdict is ApprovalDecision.ALLOW, (
        f"custom approver returned {verdict!r} instead of its configured verdict"
    )


@pytest.mark.asyncio
async def test_custom_approver_receives_the_request_instance():
    """The protocol hands over the very object it was given, by reference."""
    custom = RecordingApprover(ApprovalDecision.ALLOW)

    await custom.request(REQUEST)

    assert custom.seen == [REQUEST], (
        f"approver saw {custom.seen!r}, expected the exact request instance"
    )
