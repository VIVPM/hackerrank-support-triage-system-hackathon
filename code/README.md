# Support Triage Agent — `code/`

A defensive, RAG-grounded ticket triage agent for the HackerRank Orchestrate
hackathon. Reads `support_tickets/support_tickets.csv` and writes predictions
to `support_tickets/output.csv`.

Design rationale lives in [`../architecture.md`](../architecture.md);
the incremental build steps in [`../plan.md`](../plan.md).

---

## What it does

For each ticket, the agent decides:

- **Status** — `Replied` or `Escalated`
- **Request Type** — `product_issue` / `feature_request` / `bug` / `invalid`
- **Product Area** — short snake_case token from a soft-closed per-company vocab
- **Response** — grounded answer from the corpus, or a canonical escalation string
- **Justification** — internal one-line reason (logged, not in CSV)

It runs a **6-stage pipeline** with a defensive ordering: cheap/deterministic
checks first, expensive LLM generation only after the ticket is sanitized,
scoped, and confirmed safe.

```
Sanitize → Route(company) → Pre-triage rules + intent → Hybrid retrieve → Generate → Validate
   1            2                   3a / 3b                    4            5         6
```

| Stage | Module | What it does |
|---|---|---|
| 1 | `triage.py: strip_injections` | Strip prompt-injection patterns (English/French/Spanish), normalize whitespace. |
| 2 | `agent.py: _resolve_company` | Resolve canonical company key (`hackerrank` / `claude` / `visa`) from `Company` field + Stage 3b inference. |
| 3a | `triage.py: apply_rules` | Deterministic regex rules for high-risk buckets: fraud, score dispute, account-access dispute, social greetings, malicious patterns. Short-circuits before any LLM call. |
| 3b | `triage.py: classify_intent` | Llama 3.1 8B classifies remaining tickets into 5 buckets: `social`, `off_topic`, `malicious`, `on_topic`, `ambiguous_real`. Only `on_topic` continues to retrieval. |
| 4 | `retriever.py: search` | Hybrid retrieval: dense (bge-base-en-v1.5, 768d) + BM25 fused via Reciprocal Rank Fusion (k=60). Company-scoped, doc-deduplicated, top-5. |
| 5 | `agent.py: generate_answer` | Llama 3.3 70B single structured-output call → all 5 fields jointly. |
| 6 | `agent.py: validate_groundedness` | Llama 3.3 70B groundedness validator (criteria A/B/C). Force-escalates if Stage 5 hallucinated, ignored a user constraint, or answered the wrong entity. |

---

## Models

| Role | Model | Why |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (768d) | Strong English retrieval; available on HF Serverless feature-extraction. |
| Stage 3b classifier | `meta-llama/Llama-3.1-8B-Instruct` | Cheap routing call. |
| Stage 5 generator | `meta-llama/Llama-3.3-70B-Instruct` | Reasoning + structured output. |
| Stage 6 validator | `meta-llama/Llama-3.3-70B-Instruct` | 8B was too lenient on entity-mismatch / ignored-constraint cases; upgraded to 70B for principled groundedness checks. |

All LLM calls go through HuggingFace Inference Providers (OpenAI-compatible
router at `https://router.huggingface.co/v1`). One client, retried with
exponential backoff on transient errors.

---

## Key design decisions

- **Defensive-first ordering.** Rules gate runs before any LLM, so abuse /
  fraud / account-access patterns can never reach generation.
- **Hybrid retrieval, not pure dense.** Help-center jargon (`Chakra`, `LTI key`,
  `Bedrock`, `time accommodation`) is lexical-heavy; pure semantic search
  underweights it. RRF fusion is the standard fix.
- **Doc-level dedup in search.** Without it, the same source doc dominates
  top-K with redundant chunks. With it, top-5 = 5 unique docs.
- **Soft-closed product_area vocab.** A hard enum brittle-fails on held-out
  tickets; soft-closed gets the alignment win on known values while
  degrading gracefully on novel cases.
- **Stage 6 groundedness check.** A schema-valid answer can still hallucinate.
  The validator catches three concrete failure modes: invented details
  (criterion A), entity mismatch / ignored user constraint (B), and
  fabricated facts (C).
- **Throttle.** HF Providers free tier rate-limits; we sleep 30s every 4
  successful LLM calls. Configurable via `LLM_THROTTLE_EVERY` /
  `LLM_THROTTLE_SLEEP`.
- **On-disk caches.** The retrieval index pickles to `.cache/index.pkl` (built
  once, ~770 chunks); query embeddings cache to
  `.cache/query_embeddings/<sha256>.npy` so re-runs skip the network call.

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r code/requirements.txt

# Configure HF token (root-level .env; .env.example is shipped):
#   HF_TOKEN=hf_xxx
# Get a token at https://huggingface.co/settings/tokens (read scope is enough).
# Accept the Llama 3.3 70B and 3.1 8B licenses once on huggingface.co.
```

## Run

```bash
python code/main.py                       # full 29-row run
python code/main.py --only-rows 13,14     # debug specific rows (1-indexed)
python code/debug_rows.py 13,14           # rich per-row trace: rules → retrieval → Stage 5 → Stage 6
```

Output goes to `support_tickets/output.csv`.

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | yes | HuggingFace Inference Providers auth |
| `STAGE5_MODEL` | no | Override Llama 3.3 70B |
| `STAGE3B_MODEL` | no | Override Llama 3.1 8B |
| `HF_PROVIDER` | no | Pin a backend (e.g. `together`, `fireworks-ai`, `cerebras`) |
| `LLM_THROTTLE_EVERY` | no | Sleep after every N successful calls (default 4) |
| `LLM_THROTTLE_SLEEP` | no | Seconds to sleep when throttling (default 30) |

Secrets are read from `.env` only. Never hardcode keys.

---

## Module map

| File | Role |
|---|---|
| `main.py` | Entry point — argparse, per-ticket loop, failsafe error row. |
| `agent.py` | Stage 5 + 6 + the end-to-end `process_ticket` orchestrator. |
| `triage.py` | Stage 1 sanitize + Stage 3a rules + Stage 3b intent classifier. |
| `retriever.py` | Stage 4 hybrid index + RRF search + on-disk caches. |
| `corpus.py` | Markdown corpus loader, chunker, product-area vocab. |
| `prompts.py` | Stage 5 system/user prompt templates and chunk formatting. |
| `llm.py` | One HF client; `call_chat`, `parse_json_lenient`, throttle. |
| `io_csv.py` | Strict CSV read/write with header validation. |
| `debug_rows.py` | Diagnostic script — prints retrieval + Stage 5 raw + Stage 6 verdict. |

---

## Result on 29 input tickets

11 Replied / 18 Escalated. All Replieds are grounded in specific corpus docs
(e.g. Pause Subscription, Certifications FAQ, Visa Core Rules, Claude LTI
in Canvas). All Escalateds have a clear reason: rules-gate (fraud, score,
account access), ambiguous bug reports, malicious patterns, no corpus
answer, or Stage 6 entity-mismatch catches.
