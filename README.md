# Support Triage System

A defensive, RAG-grounded AI agent that triages real-world support tickets across three
product ecosystems — **HackerRank**, **Claude**, and **Visa** — using only a local
support corpus (no live web calls). Built for the HackerRank Orchestrate hackathon
(May 2026).

For each ticket it decides a **status** (`Replied` / `Escalated`), a **request type**
(`product_issue` / `feature_request` / `bug` / `invalid`), a **product area**, a
**grounded response**, and a **justification** — and it escalates rather than guess
whenever a ticket is high-risk, malicious, or unsupported by the corpus.

---

## Pipeline

A six-stage pipeline ordered defensively — cheap, deterministic checks first; expensive
LLM generation only after a ticket is sanitized, scoped, and confirmed safe.

```
Sanitize → Route(company) → Pre-triage rules + intent → Hybrid retrieve → Generate → Validate
   1            2                  3a / 3b                     4              5          6
```

| Stage | What it does |
|---|---|
| 1 Sanitize | Strip prompt-injection (EN/FR/ES), normalize text. |
| 2 Route | Resolve company (`hackerrank` / `claude` / `visa`) to scope retrieval. |
| 3a Rules | Deterministic gate for fraud / score-dispute / account-access / social / malicious — short-circuits before any LLM. |
| 3b Intent | Llama 3.1 8B → 5 buckets; only `on_topic` proceeds to retrieval. |
| 4 Retrieve | Hybrid dense (`bge-base-en-v1.5`) + BM25, fused with Reciprocal Rank Fusion, company-scoped, doc-deduped top-5. |
| 5 Generate | Llama 3.3 70B single structured-output call → all 5 fields jointly, grounded only in retrieved passages. |
| 6 Validate | Llama 3.3 70B groundedness check — force-escalates on hallucination, entity mismatch, or ignored constraint. |

Full rationale in [`architecture.md`](./architecture.md); the step-by-step build log in
[`plan.md`](./plan.md); run details in [`code/README.md`](./code/README.md).

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt

cp .env.example .env               # then set HF_TOKEN=hf_xxx
python code/main.py                # full run → support_tickets/output.csv
python code/main.py --only-rows 13,14   # debug specific rows
```

Needs a HuggingFace Inference Providers token (`HF_TOKEN`) with read scope. Secrets are
read from `.env` only — never hardcoded. See [`code/README.md`](./code/README.md) for the
full env-var table and module map.

---

## Layout

```
.
├── architecture.md         # design + rationale (what & why)
├── plan.md                 # step-by-step build log (execution)
├── code/                   # the agent (see code/README.md)
├── data/                   # local support corpus: hackerrank / claude / visa
└── support_tickets/
    ├── support_tickets.csv # inputs
    └── output.csv          # agent predictions
```

---

## Result

**11 Replied / 18 Escalated** on the 29 input tickets. Every Replied is grounded in a
specific corpus doc; every Escalated has a documented reason — rules-gate, ambiguous bug,
malicious pattern, no corpus answer, or a Stage-6 groundedness catch.

---

*Forked from the [interviewstreet/hackerrank-orchestrate-may26](https://github.com/interviewstreet/hackerrank-orchestrate-may26)
starter repo; the `code/` agent, design docs, and outputs are my own work.*
