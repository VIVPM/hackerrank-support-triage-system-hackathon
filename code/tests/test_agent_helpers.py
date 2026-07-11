"""Tests for the pure helpers in agent.py — coercion, bug-language
override, rules-gate product-area defaults. No network."""

from __future__ import annotations

from agent import (
    _coerce_product_area,
    _coerce_request_type,
    _coerce_status,
    _maybe_override_to_bug,
    _rules_gate_product_area,
)


def test_coerce_status_normalizes_case():
    assert _coerce_status("replied") == "Replied"
    assert _coerce_status("REPLIED") == "Replied"
    assert _coerce_status("Escalated") == "Escalated"
    assert _coerce_status("") == "Escalated"
    assert _coerce_status(None) == "Escalated"
    assert _coerce_status("hmm") == "Escalated"  # unknown → fail safe


def test_coerce_request_type_falls_back_to_product_issue():
    assert _coerce_request_type("bug") == "bug"
    assert _coerce_request_type("INVALID") == "invalid"
    assert _coerce_request_type("feature_request") == "feature_request"
    assert _coerce_request_type("") == "product_issue"
    assert _coerce_request_type("nonsense") == "product_issue"


def test_coerce_product_area_strict_snake_case():
    assert _coerce_product_area("screen") == "screen"
    assert _coerce_product_area("conversation_management") == "conversation_management"
    assert _coerce_product_area("") == ""
    assert _coerce_product_area("Has Spaces") == ""
    # Lowercases first, then regex-checks — "UPPER" becomes "upper" and passes.
    assert _coerce_product_area("UPPER") == "upper"
    # Leading digit fails the regex (^[a-z]).
    assert _coerce_product_area("123start") == ""
    # Dashes / punctuation fail.
    assert _coerce_product_area("not-snake") == ""


def test_bug_language_override_flips_product_issue_to_bug():
    ticket = {"Issue": "Claude has stopped working completely", "Subject": "Claude not responding"}
    assert _maybe_override_to_bug(ticket, "product_issue") == "bug"


def test_bug_language_override_leaves_other_types_alone():
    ticket = {"Issue": "Claude has stopped working completely", "Subject": ""}
    # Already a bug — no double-flip.
    assert _maybe_override_to_bug(ticket, "bug") == "bug"
    # `invalid` and `feature_request` should never be flipped.
    assert _maybe_override_to_bug(ticket, "invalid") == "invalid"
    assert _maybe_override_to_bug(ticket, "feature_request") == "feature_request"


def test_bug_language_override_does_not_flip_clean_ticket():
    ticket = {"Issue": "How do I dispute a charge", "Subject": "Dispute charge"}
    assert _maybe_override_to_bug(ticket, "product_issue") == "product_issue"


def test_rules_gate_product_area_per_company():
    assert _rules_gate_product_area("fraud", "Visa") == "fraud"
    assert _rules_gate_product_area("fraud", "HackerRank") == "account_management"
    assert _rules_gate_product_area("score", "HackerRank") == "screen"
    assert _rules_gate_product_area("account_access", "Claude") == "account_management"


def test_rules_gate_product_area_blank_when_company_unknown():
    assert _rules_gate_product_area("fraud", None) == ""
    assert _rules_gate_product_area("fraud", "") == ""
    assert _rules_gate_product_area("fraud", "Random") == ""
    assert _rules_gate_product_area("unknown_reason", "Visa") == ""
