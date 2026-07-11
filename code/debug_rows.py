"""Step 12 diagnostic — for each requested row, print retrieval + Stage 5 raw
output + Stage 6 verdict so we can see exactly why a ticket got escalated.

Usage:
  python code/debug_rows.py 12,17,23
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_csv import read_tickets
from triage import apply_rules, classify_intent
from retriever import build_or_load_index, search
from agent import generate_answer, validate_groundedness, _resolve_company

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"


def _short(s: str, n: int = 600) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def debug_row(idx, ticket: dict, row_num: int) -> None:
    print("=" * 80)
    print(f"ROW {row_num}")
    print(f"  Subject: {ticket.get('Subject')!r}")
    print(f"  Company: {ticket.get('Company')!r}")
    print(f"  Issue:   {_short(ticket.get('Issue', ''), 300)}")
    print()

    rr = apply_rules(ticket)
    print(f"  Stage 3a rules: hard_bucket={rr.hard_bucket!r}")
    if rr.hard_bucket:
        print("  → short-circuited by rules; no retrieval/LLM ran.")
        print()
        return

    given_company = (ticket.get("Company") or "").strip()
    intent = classify_intent(rr.cleaned_text, given_company)
    print(f"  Stage 3b intent: bucket={intent.bucket!r} inferred_company={intent.inferred_company!r}")
    if intent.bucket != "on_topic":
        print(f"  → Stage 3b short-circuited as {intent.bucket}.")
        print()
        return

    company = _resolve_company(given_company, intent.inferred_company)
    if company is None:
        print("  → no company resolved.")
        print()
        return

    query = rr.cleaned_text or ((ticket.get("Issue") or "") + " " + (ticket.get("Subject") or "")).strip()
    chunks = search(idx, query, company=company, k=5)
    print(f"\n  Retrieved {len(chunks)} chunks (company={company}):")
    for r, sc in enumerate(chunks, 1):
        snippet = _short(sc.chunk.text, 350)
        print(f"    #{r} title={sc.chunk.title!r} doc={sc.chunk.doc_path.name}")
        print(f"        {snippet}")
    print()

    ans = generate_answer(ticket, chunks, company)
    print("  Stage 5 output:")
    print(f"    status:        {ans.status}")
    print(f"    request_type:  {ans.request_type}")
    print(f"    product_area:  {ans.product_area!r}")
    print(f"    response:      {_short(ans.response, 600)}")
    print(f"    justification: {_short(ans.justification, 400)}")
    print()

    if ans.status != "Replied":
        print("  → Stage 5 already escalated; no Stage 6 call.")
        print()
        return
    if not chunks:
        print("  → no chunks; Stage 6 would force-escalate.")
        print()
        return

    val = validate_groundedness(ticket, chunks, ans)
    print("  Stage 6 verdict:")
    print(f"    grounded: {val.is_grounded}")
    print(f"    reason:   {val.reason}")
    print(f"    raw:      {_short(val.raw_response, 300)}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python code/debug_rows.py 12,17,23")
    rows = sorted({int(x) for x in sys.argv[1].split(",") if x.strip()})

    print("Loading index...")
    idx = build_or_load_index()
    print()

    tickets = read_tickets(INPUT_CSV)
    for r in rows:
        if r < 1 or r > len(tickets):
            print(f"Row {r} out of range (1..{len(tickets)})")
            continue
        debug_row(idx, tickets[r - 1], r)


if __name__ == "__main__":
    main()
