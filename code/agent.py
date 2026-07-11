"""Stage 5 — Llama 3.3 70B reasoning + grounded structured output.

Single LLM call that, given a ticket and retrieved corpus chunks, returns
all five output fields jointly. Validation/groundedness self-check is
Stage 6 (added in the next step of plan.md).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import call_chat, parse_json_lenient, stage3b_model, stage5_model
from prompts import build_system_prompt, build_user_message, format_chunks

VALID_STATUS = {"Replied", "Escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

_PRODUCT_AREA_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


@dataclass
class AnswerResult:
    status: str
    request_type: str
    product_area: str
    response: str
    justification: str
    raw_response: str  # unparsed model output, for diagnostics


@dataclass
class ValidationResult:
    is_grounded: bool
    reason: str
    raw_response: str


def _coerce_status(v) -> str:
    s = str(v or "").strip()
    # Accept Replied/Escalated case-insensitively, normalize to Title-Case.
    sl = s.lower()
    if sl == "replied":
        return "Replied"
    if sl == "escalated":
        return "Escalated"
    # Unknown → fail safe.
    return "Escalated"


def _coerce_request_type(v) -> str:
    s = str(v or "").strip().lower()
    if s in VALID_REQUEST_TYPES:
        return s
    return "product_issue"  # safest fallback when uncertain


def _coerce_product_area(v) -> str:
    s = str(v or "").strip().lower()
    # Allow only a single short snake_case token. Architecture §5.1 +
    # bucket contract §5.2 say empty for escalations; the caller can also
    # zero this out after seeing the (escalated) status.
    if not s:
        return ""
    if _PRODUCT_AREA_RE.match(s):
        return s
    return ""


def _coerce_text(v) -> str:
    return str(v or "").strip()


def generate_answer(ticket: dict, scored_chunks, company: str) -> AnswerResult:
    """Call Llama 3.3 70B once; parse + coerce the five fields."""
    system = build_system_prompt(company)
    user = build_user_message(ticket, scored_chunks)
    raw = call_chat(
        stage5_model(),
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        json_object=True,
        max_tokens=1024,
    )
    obj = parse_json_lenient(raw)
    status = _coerce_status(obj.get("status"))
    # Keep Stage 5's product_area even when Escalated — the sample CSV row 7
    # (Iron Man → Replied/invalid/conversation_management) shows the gold
    # doesn't enforce empty product_area on non-Replied rows. Only
    # `invalid` request_type genuinely has no product_area.
    request_type = _coerce_request_type(obj.get("request_type"))
    if request_type == "invalid":
        product_area = ""
    else:
        product_area = _coerce_product_area(obj.get("product_area"))
    return AnswerResult(
        status=status,
        request_type=request_type,
        product_area=product_area,
        response=_coerce_text(obj.get("response")),
        justification=_coerce_text(obj.get("justification")),
        raw_response=raw,
    )


# --- Stage 6: groundedness validation ---------------------------------------

_GROUNDEDNESS_SYSTEM_PROMPT = """You are a validator for a customer-support agent. You check whether the agent's reply is grounded in the retrieved support documents AND addresses what the user asked.

You will receive:
1. The user's ticket (subject + issue text).
2. The retrieved support documents the agent had access to.
3. The agent's response and justification.

PRINCIPLE — bias toward acceptance, with one hard line.
A reply that is topically aligned with the retrieved documents and uses information from them is grounded, even if it does not exhaustively cover every aspect of the user's question. Partial-but-faithful answers are valuable — the alternative is escalating to a human, which has its own cost. ONLY reject when you can point to a specific, concrete failure below.

The one hard line that overrides this bias: if the user explicitly told you a constraint (something missing / not visible / already tried / broken) and the response ignores it, REJECT. A confidently-sourced answer that doesn't apply to the user's situation is worse than an escalation.

Check the three criteria. Reject ONLY for concrete, named failures:

A. FABRICATION. The response invents a concrete fact (a phone number, URL, price, step, policy, time limit, or eligibility rule) that is NOT present in any of the retrieved documents. Paraphrasing, light summarization, and reasonable elaboration are NOT fabrication. Only flag specific invented details.

B. IGNORED USER CONSTRAINT (strict). If the user's ticket explicitly states a constraint — that something is missing, unavailable, already tried, broken, or not visible to them — the response must respect that constraint. A response that tells the user to perform exactly the action they said wasn't possible is NOT grounded, no matter how well-sourced the steps are.

   SAME-LOCATION corollary: if the user said they looked at a specific menu, page, dropdown, settings area, or UI region and the option they needed was not visible there, the response must NOT direct them back to that same menu/page/region — even to pick a *different* sub-option within it. The user already inspected that location from their account/role and reported nothing usable. Suggesting a different selection in the same dropdown is still asking them to use a UI surface they've established is incomplete for them. Escalate so a human can investigate the missing option (often a permission or plan-tier issue, not a navigation issue). This is the one B-mode that overrides the bias toward acceptance, because it produces actively misleading replies.

C. SUBJECT MISMATCH or CONTRADICTION (lenient). Only flag if the response is about a fundamentally different entity (e.g. user asks about removing an employee, response only covers deleting candidate records) AND there is no useful overlap. Topical drift, partial coverage, or "close but imperfect" answers do NOT qualify — accept those. Same for contradiction: only flag if the response directly contradicts a retrieved document (claims free → paid, swaps steps), not if it merely goes beyond what's covered.

If none of A/B/C is clearly violated, the response is grounded — return true. When in doubt, return true. Only return false if you can name the specific failure in one sentence.

Reply with ONE JSON object. No prose, no markdown code fences.
{"grounded": true|false, "reason": "<one short sentence: which of A/B/C failed and the specific invented/mismatched/contradicted detail, or 'all criteria pass'>"}
"""


def validate_groundedness(ticket: dict, scored_chunks, answer: AnswerResult) -> ValidationResult:
    """Single small LLM call (Stage 3b model) that judges groundedness."""
    chunks_text = format_chunks(scored_chunks)
    user = (
        f"User ticket:\n"
        f"Subject: {(ticket.get('Subject') or '').strip() or '(none)'}\n"
        f"Issue:\n{(ticket.get('Issue') or '').strip()}\n\n"
        f"=== Retrieved Documents ===\n\n{chunks_text}\n\n"
        f"=== Agent Response ===\n{answer.response}\n\n"
        f"=== Agent Justification ===\n{answer.justification}\n"
    )
    raw = call_chat(
        stage5_model(),
        [
            {"role": "system", "content": _GROUNDEDNESS_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        json_object=True,
        max_tokens=320,
    )
    obj = parse_json_lenient(raw)
    grounded = obj.get("grounded")
    # Be conservative: only accept explicit boolean true.
    is_grounded = grounded is True or (isinstance(grounded, str) and grounded.strip().lower() == "true")
    reason = str(obj.get("reason", "") or "").strip() or "no reason given"
    return ValidationResult(is_grounded=is_grounded, reason=reason, raw_response=raw)


def validate_and_check(
    ticket: dict, scored_chunks, answer: AnswerResult
) -> AnswerResult:
    """Stage 6 — schema check is already enforced by generate_answer's
    coercion; this layer adds groundedness. If retrieval was empty or the
    groundedness LLM says no, force-escalate per architecture.md §5.2.

    On force-escalate we preserve Stage 5's product_area when it was set
    (sample row 7 shows the gold doesn't enforce empty on non-Replied).
    """
    if answer.status == "Escalated":
        # Already escalated by Stage 5; nothing to validate.
        return answer

    if not scored_chunks:
        return AnswerResult(
            status="Escalated",
            request_type=answer.request_type or "product_issue",
            product_area=answer.product_area,
            response="Escalate to a human",
            justification="No corpus matches found for this ticket; cannot answer safely.",
            raw_response=answer.raw_response,
        )

    val = validate_groundedness(ticket, scored_chunks, answer)
    if not val.is_grounded:
        return AnswerResult(
            status="Escalated",
            request_type=answer.request_type or "product_issue",
            product_area=answer.product_area,
            response="Escalate to a human",
            justification=f"Groundedness check failed: {val.reason}",
            raw_response=answer.raw_response,
        )
    return answer


# --- end-to-end pipeline -----------------------------------------------------

# Canonical short-circuit responses (per architecture.md §5.2).
_FRIENDLY_ACK = "Happy to help"
_OOS_REPLY = "I am sorry, this is out of scope from my capabilities"
_MALICIOUS_REPLY = "Escalate to a human"
_AMBIGUOUS_REPLY = "Escalate to a human"
_ESCALATE_REPLY = "Escalate to a human"


def _output_row(
    ticket: dict,
    *,
    status: str,
    request_type: str,
    product_area: str,
    response: str,
    justification: str,
) -> dict:
    """Assemble the dict matching the locked output CSV schema."""
    return {
        "Issue": ticket.get("Issue", ""),
        "Subject": ticket.get("Subject", ""),
        "Company": ticket.get("Company", ""),
        "Response": response,
        "Product Area": product_area,
        "Status": status,
        "Request Type": request_type,
        "Justification": justification,
    }


def _resolve_company(given: str | None, inferred: str | None) -> str | None:
    """Return canonical company key in {hackerrank, claude, visa} or None."""
    for cand in (inferred, given):
        if not cand:
            continue
        c = str(cand).strip().lower()
        if c in {"hackerrank", "claude", "visa"}:
            return c
    return None


# Sensible product_area for rules-gate escalations, keyed on (escalation reason,
# resolved company). Falls back to "" when company is unknown. Prevents the
# old behavior of forfeiting product_area on every rules-gate Escalated row.
_RULES_GATE_PRODUCT_AREA = {
    ("fraud", "visa"): "fraud",
    ("fraud", "hackerrank"): "account_management",
    ("fraud", "claude"): "security",
    ("score", "hackerrank"): "screen",
    ("score", "claude"): "",
    ("score", "visa"): "",
    ("account_access", "hackerrank"): "account_management",
    ("account_access", "claude"): "account_management",
    ("account_access", "visa"): "account_management",
}


def _rules_gate_product_area(reason: str, given_company: str | None) -> str:
    co = (given_company or "").strip().lower()
    if co not in {"hackerrank", "claude", "visa"}:
        return ""
    return _RULES_GATE_PRODUCT_AREA.get((reason, co), "")


def _maybe_override_to_bug(ticket: dict, current_type: str) -> str:
    """If Stage 5 picked `product_issue` but the ticket is clearly a bug
    report ("down", "not working", "all requests failing", etc.), flip to
    `bug`. Only fires on `product_issue` — never overrides `invalid` /
    `feature_request` / an already-correct `bug`.
    """
    from triage import has_bug_language

    if current_type != "product_issue":
        return current_type
    text = ((ticket.get("Issue") or "") + " " + (ticket.get("Subject") or "")).strip()
    if has_bug_language(text):
        return "bug"
    return current_type


def process_ticket(ticket: dict, idx) -> dict:
    """Run the full 6-stage pipeline on one ticket. Returns an output-row dict.

    Decision order matches architecture.md §3 + the §5.2 routing table.
    """
    from triage import apply_rules, classify_intent
    from retriever import search

    # Stages 1, 2, 3a — normalize + rule-based gate (deterministic)
    rr = apply_rules(ticket)

    if rr.hard_bucket == "social":
        return _output_row(
            ticket, status="Replied", request_type="invalid",
            product_area="", response=_FRIENDLY_ACK,
            justification="Stage 3a: pure social greeting.",
        )
    if rr.hard_bucket == "malicious":
        return _output_row(
            ticket, status="Escalated", request_type="invalid",
            product_area="", response=_MALICIOUS_REPLY,
            justification="Stage 3a: jailbreak / abuse pattern matched.",
        )
    given_company_raw = (ticket.get("Company") or "").strip()
    if rr.hard_bucket == "escalate_fraud":
        return _output_row(
            ticket, status="Escalated", request_type="product_issue",
            product_area=_rules_gate_product_area("fraud", given_company_raw),
            response=_ESCALATE_REPLY,
            justification="Stage 3a: fraud / identity / lost-or-stolen pattern; requires human review.",
        )
    if rr.hard_bucket == "escalate_score":
        return _output_row(
            ticket, status="Escalated", request_type="product_issue",
            product_area=_rules_gate_product_area("score", given_company_raw),
            response=_ESCALATE_REPLY,
            justification="Stage 3a: assessment-score / hiring-decision dispute; requires human review.",
        )
    if rr.hard_bucket == "escalate_account_access":
        return _output_row(
            ticket, status="Escalated", request_type="product_issue",
            product_area=_rules_gate_product_area("account_access", given_company_raw),
            response=_ESCALATE_REPLY,
            justification="Stage 3a: account-access request without verified ownership; requires human review.",
        )

    # Stage 3b — LLM intent classifier
    given_company = (ticket.get("Company") or "").strip()
    intent = classify_intent(rr.cleaned_text, given_company)

    if intent.bucket == "social":
        return _output_row(
            ticket, status="Replied", request_type="invalid",
            product_area="", response=_FRIENDLY_ACK,
            justification="Stage 3b: pure social message.",
        )
    if intent.bucket == "off_topic":
        return _output_row(
            ticket, status="Replied", request_type="invalid",
            product_area="", response=_OOS_REPLY,
            justification="Stage 3b: question is outside the scope of HackerRank/Claude/Visa support.",
        )
    if intent.bucket == "malicious":
        return _output_row(
            ticket, status="Escalated", request_type="invalid",
            product_area="", response=_MALICIOUS_REPLY,
            justification="Stage 3b: jailbreak / abuse pattern detected.",
        )
    if intent.bucket == "ambiguous_real":
        return _output_row(
            ticket, status="Escalated", request_type="bug",
            product_area="", response=_AMBIGUOUS_REPLY,
            justification="Stage 3b: vague-but-real bug report with insufficient detail.",
        )

    # on_topic — resolve a company for retrieval
    company = _resolve_company(given_company, intent.inferred_company)
    if company is None:
        return _output_row(
            ticket, status="Escalated", request_type="product_issue",
            product_area="", response=_ESCALATE_REPLY,
            justification="Could not determine which product this ticket relates to.",
        )

    # Stage 4 — hybrid retrieval, company-scoped
    query_text = rr.cleaned_text or (
        (ticket.get("Issue") or "") + " " + (ticket.get("Subject") or "")
    ).strip()
    chunks = search(idx, query_text, company=company, k=5)

    # Stage 5 — joint structured-output generation
    answer = generate_answer(ticket, chunks, company)

    # Stage 6 — schema + groundedness self-check
    final = validate_and_check(ticket, chunks, answer)

    # Post-process: bug-language override (sample row 2 anchor — "site is
    # down" → bug, not product_issue). Only flips product_issue → bug.
    final_request_type = _maybe_override_to_bug(ticket, final.request_type)

    return _output_row(
        ticket,
        status=final.status,
        request_type=final_request_type,
        product_area=final.product_area,
        response=final.response,
        justification=final.justification,
    )


# --- self-test ---------------------------------------------------------------


def _selftest() -> None:
    """Step 7 — three on_topic tickets, one per company. Eyeball the output:
    grounded? schema-valid? product_area from preferred vocab?"""
    from retriever import build_or_load_index, search

    print("Loading index...")
    idx = build_or_load_index()
    print()

    cases = [
        (
            "hackerrank",
            {
                "Issue": (
                    "Hello! I am trying to remove an interviewer from the platform. "
                    "I am not seeing this as an option when I select the three dots "
                    "next to their name. Can you let me know how to do this?"
                ),
                "Subject": "How to Remove a User",
                "Company": "HackerRank",
            },
        ),
        (
            "claude",
            {
                "Issue": "I am allowing Claude to use my data to improve the models, how long will the data be used for?",
                "Subject": "Personal Data Use",
                "Company": "Claude",
            },
        ),
        (
            "visa",
            {
                "Issue": "How do I dispute a charge",
                "Subject": "Dispute charge",
                "Company": "Visa",
            },
        ),
    ]

    for co, ticket in cases:
        query = (ticket["Issue"] + " " + (ticket["Subject"] or "")).strip()
        chunks = search(idx, query, company=co, k=5)
        print(f"=== {co.upper()} | {ticket['Subject']!r} ===")
        print(f"Retrieved top-{len(chunks)} chunks:")
        for r, sc in enumerate(chunks, 1):
            print(f"  #{r} {sc.chunk.title}  ({sc.chunk.doc_path.name})")
        print("Calling Llama 3.3...")
        ans = generate_answer(ticket, chunks, co)
        print(f"  status:        {ans.status}")
        print(f"  request_type:  {ans.request_type}")
        print(f"  product_area:  {ans.product_area!r}")
        resp_preview = ans.response if len(ans.response) <= 400 else ans.response[:400] + "…"
        print(f"  response:      {resp_preview}")
        print(f"  justification: {ans.justification}")
        print()


def _selftest_validate() -> None:
    """Step 8 — verify the groundedness check escalates a hallucinated answer
    and lets a clean answer through. Uses the same Visa-dispute case from
    Step 7 for the clean path."""
    from retriever import build_or_load_index, search

    print("Loading index...")
    idx = build_or_load_index()
    print()

    # 1) Clean case: real Stage 5 answer on the Visa dispute ticket
    visa_ticket = {
        "Issue": "How do I dispute a charge",
        "Subject": "Dispute charge",
        "Company": "Visa",
    }
    visa_chunks = search(idx, "How do I dispute a charge Dispute charge", company="visa", k=5)
    print("=== CLEAN CASE: Visa dispute ===")
    real_answer = generate_answer(visa_ticket, visa_chunks, "visa")
    print(f"  Stage 5 status: {real_answer.status}")
    print(f"  Stage 5 response: {real_answer.response[:200]}…")
    final = validate_and_check(visa_ticket, visa_chunks, real_answer)
    print(f"  Stage 6 grounded? {final.status == real_answer.status}")
    print(f"  Final status: {final.status}")
    print(f"  Final response: {final.response[:200]}…")
    print(f"  Final justification: {final.justification}")
    print()

    # 2) Hallucinated case: same chunks, but inject a clearly fabricated answer
    hallucinated = AnswerResult(
        status="Replied",
        request_type="product_issue",
        product_area="dispute",
        response=(
            "To dispute a charge, please call Visa's secret 24-hour fraud "
            "hotline at +1-555-FAKE-NUM and quote authorization code "
            "ALPHA-7G. Refunds are guaranteed within 4 hours and a $250 "
            "goodwill credit will be applied automatically."
        ),
        justification="Per Visa's internal SLA document.",
        raw_response="(test fixture, not real)",
    )
    print("=== HALLUCINATED CASE: fabricated phone number, code, SLA ===")
    print(f"  Injected response: {hallucinated.response[:200]}…")
    final = validate_and_check(visa_ticket, visa_chunks, hallucinated)
    if final.status == "Escalated":
        print(f"  ✅ CAUGHT — escalated")
        print(f"  reason: {final.justification}")
    else:
        print(f"  ❌ MISSED — Stage 6 let the hallucination through")
        print(f"  Final response: {final.response[:200]}…")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        _selftest_validate()
    else:
        _selftest()

