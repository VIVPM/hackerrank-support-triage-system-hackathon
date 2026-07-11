# Build Plan — Step by Step

**Principle.** Each step adds one capability, has a concrete verification gate, and produces something runnable. We don't move to the next step until the current one passes its check. This way, when something breaks, we know which step introduced it.

**Current status legend.** ☐ not started · 🔄 in progress · ✅ done

**Steps 0–12 are complete.** Steps 13–14 (repro check + submission package) remain.

**Reference.** Architecture decisions live in `architecture.md`. This file tracks *execution*, not design.

---

## Step 0 — Repo & dependency setup ✅

**Goal.** Have a runnable Python environment with all deps pinned.

**Build.**
- `code/requirements.txt` with: `huggingface_hub`, `openai` (for OpenAI-compatible HF endpoint), `python-dotenv`, `pandas`, `numpy`, `FlagEmbedding` (for bge-m3), `tqdm`.
- `code/README.md` skeleton with install + run instructions.
- Confirm `.env` already has `HF_TOKEN` set (done).

**Test.**
```bash
cd code && pip install -r requirements.txt
python -c "import dotenv, openai, FlagEmbedding, numpy, pandas; print('ok')"
```

**Done when.** `print('ok')` prints with no import errors.

---

## Step 1 — CSV I/O skeleton ✅

**Goal.** Read input CSV, write a valid (but stub) output CSV with the locked schema.

**Build.**
- `code/io_csv.py`: `read_tickets(path) -> list[dict]` and `write_outputs(rows, path)` that emits header `Issue,Subject,Company,Response,Product Area,Status,Request Type` exactly (per §5 of architecture.md).
- `code/main.py`: minimal CLI — read input, for each row write a hardcoded stub (`Status="Escalated"`, `Request Type="invalid"`, empty Product Area, response="stub", justification="stub").

**Test.**
```bash
python code/main.py
```
Then open `support_tickets/output.csv`.

**Done when.** Output CSV has 29 rows, header matches sample CSV exactly, all rows have all 7 columns populated. (Spot-check the row count and that quoted multi-line `Issue` values survive round-trip.)

---

## Step 2 — Corpus loader + chunker ✅

**Goal.** Load the 770 markdown files into in-memory chunks with proper metadata.

**Build.**
- `code/corpus.py`:
  - `load_docs() -> list[Doc]` walks `data/{hackerrank,claude,visa}/**/*.md`.
  - Parses YAML frontmatter (`title`, `breadcrumbs`, `source_url`).
  - `chunk_docs(docs) -> list[Chunk]` splits each body into ~500-token chunks with 50-token overlap. Each chunk carries `company`, `title`, `breadcrumbs`, `source_url`, `text`.
  - `mine_product_area_vocab() -> dict[str, list[str]]` returns the per-company vocab from §5.1 of architecture.md (just hard-code the 26 tokens we agreed on).

**Test.**
```bash
python -c "from code.corpus import load_docs, chunk_docs; d=load_docs(); c=chunk_docs(d); print(f'{len(d)} docs, {len(c)} chunks'); print(c[0])"
```

**Done when.** Prints ~770 docs, ~3-5k chunks, and one chunk shows correctly-parsed title/breadcrumbs/text.

---

## Step 3 — Hybrid index build via HF API + BM25 (cached) ✅

**Goal.** Build the dense + sparse retrieval index once, persist to disk.

> **Note (architectural change).** We originally planned to use local `bge-m3` via `FlagEmbedding`, which produces dense + learned-sparse in one pass. We switched to **Path B**: dense embeddings via the **HuggingFace Serverless Inference API** (no local model weights, no GPU dependency) plus a local `rank_bm25` index for sparse. The current dense model is `BAAI/bge-base-en-v1.5` (768-d, English-strong, reliably available on HF Serverless feature-extraction). Trade-offs are documented in `architecture.md` §4.

**Build.**
- `code/retriever.py`:
  - `Index` dataclass holds `chunks`, `dense` (N×768 float32, L2-normalized), `tokenized` (per-chunk token lists), `bm25` (BM25Okapi instance), `model_name`, `cache_version`.
  - `build_index(chunks) -> Index` calls `huggingface_hub.InferenceClient.feature_extraction` in `EMBED_BATCH_SIZE=32` batches with retries (cold-start 503s + rate limits), then builds `BM25Okapi` over a simple lowercase-alphanumeric tokenizer over the **raw** chunk text (not the prefixed string).
  - Documents use no prefix (BGE convention); queries get `"Represent this sentence for searching relevant passages: "` at search time.
  - `save_index(idx, path)` / `load_index(path)` — pickle to `code/.cache/index.pkl`.
  - Content-based signature in `code/.cache/index.sig` invalidates automatically on corpus, model, or prefix change.

**Test.**
```bash
python code/retriever.py
```
Run it twice — second run should be instant (cache hit).

**Done when.** First run completes (~3-5 min over HF Serverless), saves `.cache/index.pkl` (~10 MB), second run prints `Cache HIT` and loads in <2 sec. Self-test prints dense shape `(3334, 768)` and BM25 stats.

---

## Step 4 — Hybrid search with RRF + company filter ✅

**Goal.** Given a query + company, return top-K chunks fused from dense + BM25 ranks.

**Build.**
- `code/retriever.py`:
  - `search(idx, query: str, company: str | None, k: int = 5, rrf_k: int = 60) -> list[ScoredChunk]`
  - Prepend `QUERY_PREFIX` to the query, embed once via the same HF API, L2-normalize.
  - Score dense via cosine over the company-filtered chunk indices.
  - Score sparse via `BM25Okapi.get_scores()` over the same filtered indices.
  - Fuse the two ranked lists via **reciprocal-rank fusion**: `rrf(d) = Σ 1/(rrf_k + rank_in_list)`. RRF is parameter-free in scale and dodges the dense-cosine vs. BM25-score normalization headache.
  - Return top-K `ScoredChunk` with `chunk`, `rrf_score`, `dense_score`, `bm25_score` fields for diagnostics.

**Test.** Smoke-test 3 queries:
1. `("site is down", "HackerRank")` → expect HR troubleshooting docs
2. `("dispute a charge", "Visa")` → expect Visa dispute-resolution doc
3. `("how to delete conversation", "Claude")` → expect Claude conversation-management doc

**Done when.** Top-1 result for each is a topically-relevant article (eyeball check). Each result shows non-zero contributions from both dense and BM25 (so RRF is actually fusing, not collapsing to one signal).

---

## Step 5 — Stage 3a rule-based pre-triage gate ✅

**Goal.** Cheap regex-based detection of malicious / social / fraud-escalation patterns.

**Build.**
- `code/triage.py`:
  - `apply_rules(ticket: dict) -> RuleResult` returns `{cleaned_text, hard_bucket, flags}` where `hard_bucket` ∈ {`social`, `malicious`, `escalate_fraud`, `None`} and `cleaned_text` has prompt-injection markers stripped.
  - Patterns from architecture.md §3.3a (identity theft, score dispute, account access without ownership, jailbreak, social, prompt-injection markers).

**Test.** Hand-craft 6 mini test cases covering every rule, assert correct classification.

**Done when.** A small `python -m code.triage --selftest` prints "6/6 passed".

---

## Step 6 — Stage 3b LLM intent classifier ✅

**Goal.** First real LLM call. 5-bucket classifier via Llama 3.1 8B on HF Inference Providers.

**Build.**
- `code/llm.py`: `call_chat(model, messages, json_schema=None) -> dict` — thin wrapper over the HF Providers OpenAI-compatible endpoint, reads `HF_TOKEN` from env.
- `code/triage.py` extension: `classify_intent(cleaned_text, given_company) -> IntentResult` calls Llama 3.1 8B with a structured-output schema for `{bucket, inferred_company}`.

**Test.** Run on 5 representative tickets from `support_tickets.csv` (one per bucket: social-equivalent, off_topic-equivalent, malicious, on_topic, ambiguous_real). Verify each gets correctly bucketed.

**Done when.** 5/5 buckets correct on the hand-picked rows. If any wrong, refine the classifier prompt.

---

## Step 7 — Stage 5 reasoning + structured output ✅

**Goal.** Llama 3.3 70B call that produces all 5 fields jointly with grounded response.

**Build.**
- `code/prompts.py`: system prompt encoding the 5-field contract, the per-company `product_area` vocab, escalation rules, "ground only in provided passages" rule. JSON schema for the response.
- `code/agent.py`: `generate_answer(ticket, retrieved_chunks) -> AnswerResult` — single Llama 3.3 call with the system prompt + ticket + chunks, returns the 5 fields.

**Test.** Run on 3 `on_topic` tickets (one HR, one Claude, one Visa) with retrieval already done. Eyeball outputs for: (a) is the response actually grounded in the retrieved chunks, (b) is the JSON schema valid, (c) does `product_area` come from the per-company vocab.

**Done when.** 3/3 outputs are grounded, schema-valid, and use vocab tokens.

---

## Step 8 — Stage 6 validate + groundedness self-check ✅

> **Update.** The validator was originally Llama 3.1 8B. After observing false negatives (rejecting cleanly grounded answers) and non-deterministic verdicts at temperature 0, upgraded to **Llama 3.3 70B**. Additionally, criterion B in the groundedness prompt was expanded to explicitly call out three concrete failure modes — entity mismatch, ignored constraint, and wrong action — to catch hallucinations more reliably. Stage 6 max_tokens raised 160 → 320 so the 70B has room to reason through A/B/C.

**Goal.** Catch hallucinated replies and force-escalate them.

**Build.**
- `code/agent.py` extension: `validate_and_check(answer, retrieved_chunks) -> AnswerResult` — schema check, allowed-enum check, groundedness check (the `justification` must reference at least one retrieved chunk's source path; if not → force-escalate with a "could not ground answer" justification).

**Test.** Hand-craft a deliberately-bad LLM output (made-up policy, no grounding) and confirm it gets downgraded to `Escalated`.

**Done when.** Bad output is caught and rewritten; good output passes through unchanged.

---

## Step 9 — End-to-end pipeline wiring ✅

**Goal.** All stages 1-6 connected in `agent.py` per `architecture.md` §3.

**Build.**
- `code/agent.py`: `process_ticket(ticket, idx) -> dict` runs Stages 1→2→3a→3b→(4→5→6 if `on_topic`, else short-circuit per §5.2 routing table).
- `code/main.py` updated: load index, loop over input rows, write output.

**Test.** Run on the **first 5 rows** of `support_tickets.csv` only (use `--only-rows 1,2,3,4,5`).

**Done when.** 5 rows produce schema-valid output that eyeball-passes (no hallucinated policies, escalations look reasonable).

---

## Step 10 — Full run on 29 tickets ✅

**Goal.** Generate the complete `output.csv`.

**Build.** (Nothing new — just run.)

**Test.**
```bash
python code/main.py
```
Inspect `support_tickets/output.csv`.

**Done when.** All 29 rows present, schema correct. Eyeball ~10 rows for quality. Note any obvious mistakes for the next iteration.

---

## Step 11 — Caching layer (b) + `--only-rows` ✅

**Goal.** Speed up iteration. Re-run just one row in <15 sec.

**Build (as implemented).**
- `code/retriever.py`: per-**query** embedding cache at `code/.cache/query_embeddings/<sha256>.npy`. Keyed on `embedding_model + query_prefix + query_text`. Skips the HF API embedding call entirely on a hit.
- `code/main.py`: `--only-rows N,M,K` flag.
- `code/debug_rows.py`: rich diagnostic that prints retrieval + Stage 5 raw + Stage 6 verdict.

> **Note.** Originally planned a per-ticket retrieval cache (top-K chunk IDs as JSON). Implemented as a per-query embedding cache instead — same effect (re-runs skip the network call), simpler key, and naturally invalidates on embedder/prefix change without manual versioning. The downstream BM25 + RRF + dedup are sub-10ms in numpy, so caching only the embedding is sufficient.

---

## Step 12 — Iteration on weak rows ✅

**Goal.** Identify and fix the 3-5 worst rows from Step 10.

**What we did.**
- Used `debug_rows.py` to inspect rows where Stage 6 was producing wrong verdicts (false negatives on Pause Subscription, false positives on entity-mismatch like "Remove employee → delete candidate profile").
- **Generalizable changes** (no row-specific patches):
  1. Upgraded Stage 6 model from Llama 3.1 8B → Llama 3.3 70B.
  2. Expanded Stage 6 criterion B with three concrete failure-mode patterns (entity mismatch, ignored constraint, wrong action) — describes failure shapes, not specific tickets.
  3. Bumped Stage 6 `max_tokens` 160 → 320.
  4. Standardized all escalation response strings to `"Escalate to a human"` (matching sample CSV row 2 exactly).
  5. Added a Stage 5 prompt note: if the corpus directs users to a specific support channel (e.g. "contact AWS Support"), that IS a valid Replied answer.

**Final result.** 11 Replied / 18 Escalated on the 29 input tickets. All Replieds are grounded in specific corpus docs; all Escalateds have a documented reason (rules-gate, ambiguous bug, malicious, no corpus answer, Stage 6 entity-mismatch).

**Stopping criterion.** Each subsequent prompt tweak shifts the borderline-row decisions in both directions (precision/recall trade-off). With a 29-row sample, further tuning would overfit. Locked the system here.

---

## Step 13 — README + repro test ☐

**Goal.** A grader who clones the repo can install + run.

**Build.**
- `code/README.md` finalized: install, env vars, `python main.py`, `--only-rows`, expected output location, the 5-field schema.
- Confirm `.env.example` is committed and `.env` is not.
- Top-level pass: lint with `ruff` (or just eyeball), nothing imports the absolute path.

**Test.** Wipe `code/.cache/`, run `python code/main.py` from scratch, time it. Verify final `output.csv` is identical to the previous full run (determinism check).

**Done when.** Cold run reproduces the same 29 outputs.

---

## Step 14 — Submission package ☐

**Goal.** The three deliverables required by `README.md` and `AGENTS.md` §6.

**Build.**
- Zip `code/` excluding `.cache/`, `__pycache__/`, `.venv/`.
- Confirm `support_tickets/output.csv` is the latest full run.
- Confirm `~/hackerrank_orchestrate/log.txt` is the chat transcript to upload.

**Done when.** All three uploads ready and the submission link from `README.md` is open.

---

## Notes & guardrails

- **If a step's verification fails, stop and fix it before adding more.** Don't pile changes on top of a broken stage.
- **Keep `temperature=0`** for every LLM call — non-determinism here = invisible regressions.
- **Never commit `.env`.** Run `git status` before each commit; the file is gitignored but be paranoid.
- **Every LLM call goes through `code/llm.py`.** No scattered API calls in other files.
- **Log every meaningful step** to `~/hackerrank_orchestrate/log.txt` (per AGENTS.md §5.2). Each "step done" turn is a log entry.
