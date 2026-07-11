"""Prompts and message-formatting helpers for Stage 5 (Llama 3.3 70B).

Kept separate from agent.py so the prompt text is easy to read, diff,
and iterate on without touching the orchestration logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import PRODUCT_AREA_VOCAB

# Per-chunk text included in the user message; longer chunks would crowd
# context for marginal benefit when 5 chunks already fit comfortably.
CHUNK_SNIPPET_CHARS = 1500

_COMPANY_LABEL = {
    "hackerrank": "HackerRank (hiring/assessment platform)",
    "claude": "Claude (Anthropic's AI assistant + API)",
    "visa": "Visa (payment cards / Visa India support)",
}


_STAGE5_SYSTEM_TEMPLATE = """You are a customer-support agent for {company_label}.

Your job: read the user's ticket and the retrieved support documents, then return a single JSON object with EXACTLY these 5 fields:

- "status": one of "Replied" or "Escalated".
- "request_type": one of "product_issue", "feature_request", "bug", "invalid".
- "product_area": one short lowercase snake_case token, or an empty string if escalating.
   PREFER picking from this list (gold-aligned): {vocab}.
   If nothing in the list fits, emit a short snake_case token in the same style. Do NOT invent multi-word phrases or freeform descriptions.
- "response": the user-facing answer.
   - If status="Replied": write a concise, helpful answer GROUNDED ONLY in the provided documents. Quote/paraphrase steps faithfully. Do NOT invent URLs, phone numbers, policies, or steps that aren't in the documents.
   - If status="Escalated": use the exact string "Escalate to a human".
- "justification": one sentence explaining your decision. When status="Replied", cite the doc(s) you used by title or filename. When status="Escalated", say WHY (e.g. "no matching doc", "high-risk fraud case", "account-access dispute").

ESCALATION RULES — escalate (status="Escalated") in any of these cases:
- The provided documents do not contain a clear answer to the user's question.
   (Note: if the documents direct users to a specific support channel — e.g. "contact AWS Support", "reach out to your account manager", "call your issuing bank" — that IS a valid answer; reply with that pointer instead of escalating.)
- The ticket involves fraud, identity theft, lost/stolen card, or unauthorized charges.
- The ticket disputes an assessment score, hiring decision, or recruiter rejection.
- The ticket asks to bypass an account owner's permission or restore access without ownership.
- The ticket is a vague-but-real bug report ("it's not working", "site is down") with insufficient detail to act on.

LANGUAGE
- If the ticket is in a non-English language (French, Spanish, etc.), respond in English unless the user clearly prefers another language.

OUTPUT FORMAT
- Output ONLY a single JSON object on one or more lines. No surrounding prose, no markdown code fences, no commentary.
- Do not include any field other than the 5 listed above.
"""


def build_system_prompt(company: str) -> str:
    co = (company or "").strip().lower()
    label = _COMPANY_LABEL.get(co, "this product")
    vocab = PRODUCT_AREA_VOCAB.get(co, [])
    vocab_str = ", ".join(vocab) if vocab else "(no preferred vocabulary for this company)"
    return _STAGE5_SYSTEM_TEMPLATE.format(company_label=label, vocab=vocab_str)


def format_chunks(scored_chunks) -> str:
    """Render retrieved chunks for the user message."""
    if not scored_chunks:
        return "(no documents were retrieved)"
    parts: list[str] = []
    for i, sc in enumerate(scored_chunks, 1):
        c = sc.chunk
        snippet = c.text[:CHUNK_SNIPPET_CHARS]
        if len(c.text) > CHUNK_SNIPPET_CHARS:
            snippet += "\n…(truncated)"
        crumb = " > ".join(c.breadcrumbs) if c.breadcrumbs else "(none)"
        url = c.source_url or "(no source URL)"
        parts.append(
            f"[Doc {i}]\n"
            f"file: {c.doc_path.name}\n"
            f"title: {c.title}\n"
            f"breadcrumbs: {crumb}\n"
            f"source_url: {url}\n"
            f"---\n{snippet}"
        )
    return "\n\n".join(parts)


def build_user_message(ticket: dict, scored_chunks) -> str:
    subj = (ticket.get("Subject") or "").strip() or "(none)"
    issue = (ticket.get("Issue") or "").strip()
    return (
        f"Ticket Subject: {subj}\n"
        f"Ticket Issue:\n{issue}\n\n"
        f"=== Retrieved Support Documents ===\n\n{format_chunks(scored_chunks)}\n"
    )
