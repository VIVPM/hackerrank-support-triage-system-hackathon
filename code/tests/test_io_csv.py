"""Round-trip tests for io_csv. No network. Confirms the locked output
schema (now including Justification) is preserved."""

from __future__ import annotations

import csv

from io_csv import INPUT_HEADERS, OUTPUT_HEADERS, read_tickets, write_outputs


def test_output_headers_include_justification():
    assert "Justification" in OUTPUT_HEADERS
    assert OUTPUT_HEADERS[-1] == "Justification", (
        "Justification should be the last column so existing 7-column readers"
        " still see Issue..Request Type in their expected positions."
    )


def test_input_headers_unchanged():
    assert INPUT_HEADERS == ["Issue", "Subject", "Company"]


def test_read_then_write_roundtrip(tmp_path):
    src = tmp_path / "in.csv"
    # Write with explicit binary newlines so platform LF/CRLF differences
    # don't change what the csv reader sees.
    content = (
        "Issue,Subject,Company\n"
        '"Test ticket","Test subject",HackerRank\n'
        '"Multi\nline\nbody","",Visa\n'
    )
    src.write_bytes(content.encode("utf-8"))
    tickets = read_tickets(src)
    assert len(tickets) == 2
    assert tickets[0]["Issue"] == "Test ticket"
    assert tickets[1]["Issue"] == "Multi\nline\nbody"
    assert tickets[1]["Company"] == "Visa"

    out = tmp_path / "out.csv"
    rows = [
        {
            "Issue": t["Issue"],
            "Subject": t["Subject"],
            "Company": t["Company"],
            "Response": "ok",
            "Product Area": "screen",
            "Status": "Replied",
            "Request Type": "product_issue",
            "Justification": "tested",
        }
        for t in tickets
    ]
    n = write_outputs(rows, out)
    assert n == 2

    # Re-read the written file with stdlib csv to confirm headers + counts.
    with open(out, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == OUTPUT_HEADERS
        rows_back = list(reader)
    assert len(rows_back) == 2
    assert rows_back[0]["Justification"] == "tested"
    assert rows_back[1]["Issue"] == "Multi\nline\nbody"


def test_write_rejects_missing_field(tmp_path):
    out = tmp_path / "out.csv"
    bad_row = {
        "Issue": "x", "Subject": "x", "Company": "x",
        "Response": "x", "Product Area": "x",
        "Status": "Replied", "Request Type": "bug",
        # missing Justification
    }
    try:
        write_outputs([bad_row], out)
    except ValueError as e:
        assert "Justification" in str(e)
        return
    raise AssertionError("Expected ValueError for missing Justification field")
