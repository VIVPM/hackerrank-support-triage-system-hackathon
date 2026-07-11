"""Support triage agent — terminal entry point.

Reads support_tickets/support_tickets.csv, runs the 6-stage pipeline
(architecture.md §3) on each ticket, and writes predictions to
support_tickets/output.csv.

Usage:
  python code/main.py                       # full run on all input rows
  python code/main.py --only-rows 1,5,12    # process only rows 1, 5, 12 (1-indexed)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_csv import read_tickets, write_outputs

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"


def _failsafe_row(ticket: dict, reason: str) -> dict:
    """Output schema-valid escalation row when the pipeline raises an exception
    on a single ticket. Lets the rest of the run complete instead of aborting."""
    return {
        "Issue": ticket.get("Issue", ""),
        "Subject": ticket.get("Subject", ""),
        "Company": ticket.get("Company", ""),
        "Response": "Escalate to a human",
        "Product Area": "",
        "Status": "Escalated",
        "Request Type": "product_issue",
        "Justification": f"Pipeline error; escalated for human review ({reason[:120]}).",
    }


def parse_only_rows(arg: str | None) -> set[int] | None:
    if not arg:
        return None
    try:
        return {int(x.strip()) for x in arg.split(",") if x.strip()}
    except ValueError:
        raise SystemExit(f"--only-rows expects comma-separated integers, got: {arg!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Support-ticket triage agent")
    parser.add_argument(
        "--only-rows",
        dest="only_rows",
        default=None,
        help="Comma-separated 1-indexed row numbers to process (e.g. 1,5,12). "
             "Output CSV will contain only those rows.",
    )
    args = parser.parse_args()
    only = parse_only_rows(args.only_rows)

    # Lazy import: keeps the I/O-only stub path light if user just wants to
    # eyeball the input without paying the index/torch import cost.
    from agent import process_ticket
    from retriever import build_or_load_index

    print(f"Loading retrieval index (one-time, then cached)...")
    idx = build_or_load_index()
    print()

    tickets = read_tickets(INPUT_CSV)
    print(f"Read {len(tickets)} tickets from {INPUT_CSV}")
    if only is not None:
        print(f"Filtered to rows: {sorted(only)}")
    print()

    rows: list[dict] = []
    t_start = time.time()
    for i, ticket in enumerate(tickets, 1):
        if only is not None and i not in only:
            continue
        subject = (ticket.get("Subject") or "").strip().replace("\n", " ")[:60]
        company = ticket.get("Company") or "None"
        print(f"[{i:>3}/{len(tickets)}] company={company!r:<13} subject={subject!r}")
        t0 = time.time()
        try:
            row = process_ticket(ticket, idx)
            rows.append(row)
            print(
                f"          → status={row['Status']:<9} "
                f"request_type={row['Request Type']:<16} "
                f"product_area={row['Product Area']!r:<22} "
                f"({time.time() - t0:.1f}s)"
            )
        except Exception as e:
            print(f"          ! ERROR: {e!r}", file=sys.stderr)
            traceback.print_exc()
            rows.append(_failsafe_row(ticket, repr(e)))

    n = write_outputs(rows, OUTPUT_CSV)
    print()
    print(f"Wrote {n} rows to {OUTPUT_CSV}  (total {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
