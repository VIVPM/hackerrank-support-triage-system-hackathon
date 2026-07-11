"""Confirm the main.py failsafe row matches the locked output schema
(including Justification) and uses the canonical escalation string."""

from __future__ import annotations

from io_csv import OUTPUT_HEADERS
from main import _failsafe_row


def test_failsafe_has_all_output_headers():
    row = _failsafe_row({"Issue": "x", "Subject": "y", "Company": "Z"}, "boom")
    for h in OUTPUT_HEADERS:
        assert h in row, f"Failsafe row missing {h!r}"
    extras = set(row) - set(OUTPUT_HEADERS)
    assert not extras, f"Failsafe row has extra fields: {extras}"


def test_failsafe_uses_canonical_escalation_string():
    row = _failsafe_row({"Issue": "x", "Subject": "y", "Company": "Z"}, "boom")
    # Canonical string per architecture.md §5.2 ("Escalate", not "Escalated").
    assert row["Response"] == "Escalate to a human"
    assert row["Status"] == "Escalated"


def test_failsafe_preserves_input_fields():
    row = _failsafe_row(
        {"Issue": "iss", "Subject": "sub", "Company": "Visa"}, "reason"
    )
    assert row["Issue"] == "iss"
    assert row["Subject"] == "sub"
    assert row["Company"] == "Visa"


def test_failsafe_justification_truncated_to_reasonable_length():
    long_reason = "x" * 1000
    row = _failsafe_row({"Issue": "i", "Subject": "s", "Company": ""}, long_reason)
    # We slice the reason to 120 chars; the whole justification is short.
    assert len(row["Justification"]) < 200
