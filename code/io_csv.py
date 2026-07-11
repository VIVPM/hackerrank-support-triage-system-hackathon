"""CSV read/write with the schema locked in architecture.md §5.

Input CSV columns:  Issue, Subject, Company
Output CSV columns: Issue, Subject, Company, Response, Product Area, Status,
                    Request Type, Justification

Justification is required by problem_statement.md / evalutation_criteria.md §3
even though the shipped sample CSV does not include it.

We use the stdlib `csv` module (not pandas) because:
- Many input cells span multiple lines inside quoted fields; csv.DictReader
  handles RFC 4180 quoting correctly without surprises.
- pandas can normalize whitespace / NaN in ways that subtly diverge from the
  sample CSV's expected output.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

INPUT_HEADERS = ["Issue", "Subject", "Company"]
OUTPUT_HEADERS = [
    "Issue",
    "Subject",
    "Company",
    "Response",
    "Product Area",
    "Status",
    "Request Type",
    "Justification",
]


def read_tickets(path: str | Path) -> list[dict]:
    """Read the input CSV and return a list of dicts.

    Each dict has keys exactly equal to INPUT_HEADERS. Values are returned
    as-is (no whitespace stripping, no None-coercion) so downstream stages
    can decide how to normalize.
    """
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != INPUT_HEADERS:
            raise ValueError(
                f"Unexpected input header: {reader.fieldnames!r}; "
                f"expected {INPUT_HEADERS!r}"
            )
        return [dict(row) for row in reader]


def write_outputs(rows: Iterable[dict], path: str | Path) -> int:
    """Write rows to the output CSV with the locked schema.

    Each row dict must contain every key in OUTPUT_HEADERS. Returns the
    number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_HEADERS, quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for r in rows:
            missing = [k for k in OUTPUT_HEADERS if k not in r]
            if missing:
                raise ValueError(
                    f"Row missing fields: {missing}; got {list(r.keys())}"
                )
            writer.writerow({k: r[k] for k in OUTPUT_HEADERS})
            n += 1
    return n
