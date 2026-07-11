"""Hybrid retrieval index — Option B (HF API for dense + local BM25 for sparse).

Step 3: build dense embeddings via HuggingFace Serverless Inference API
(no local GPU needed) and a BM25 sparse index via rank_bm25. Persist both
to code/.cache/index.pkl.

Step 4 will add `search()` that fuses dense + BM25 ranks via RRF.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import REPO_ROOT, Chunk, chunk_docs, load_docs

CACHE_DIR = REPO_ROOT / "code" / ".cache"
INDEX_CACHE = CACHE_DIR / "index.pkl"
SIGNATURE_FILE = CACHE_DIR / "index.sig"
QUERY_EMBED_CACHE_DIR = CACHE_DIR / "query_embeddings"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
# BGE convention: documents are NOT prefixed; queries get an instruction prefix.
# Skipping the query prefix at search time costs ~2-4% retrieval quality on BEIR.
DOC_PREFIX = ""
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = 32
EMBED_RETRIES = 6
EMBED_RETRY_BASE_DELAY = 5  # exponential backoff seed


@dataclass
class Index:
    chunks: list[Chunk]
    dense: np.ndarray  # shape (N, 768) float32, L2-normalized
    tokenized: list[list[str]]  # per-chunk tokens for BM25
    bm25: Any  # rank_bm25.BM25Okapi instance
    model_name: str
    cache_version: int = 4  # bumped: model swapped to BAAI/bge-base-en-v1.5 (768d)


@dataclass
class ScoredChunk:
    chunk: Chunk
    rrf_score: float
    dense_score: float
    bm25_score: float
    dense_rank: int
    bm25_rank: int


# --- signature / cache key ---------------------------------------------------


def _signature(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    h.update(EMBEDDING_MODEL.encode("utf-8"))
    h.update(f"|dim={EMBED_DIM}|prefix={DOC_PREFIX}|".encode("utf-8"))
    for c in chunks:
        h.update(c.text.encode("utf-8"))
    return h.hexdigest()[:16]


# --- BM25 tokenization -------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric split. Sufficient for BM25 over English
    + light multilingual help-center text. Diacritics get folded out, which
    is fine for our corpus (mostly English with rare French/Spanish accents).
    """
    return _TOKEN_RE.findall(text.lower())


# --- HF embedding call -------------------------------------------------------


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _embed_batch(client, texts: list[str]) -> np.ndarray:
    """Call HF Serverless feature-extraction with retries on cold-start 503s."""
    for attempt in range(EMBED_RETRIES):
        try:
            embs = client.feature_extraction(texts, model=EMBEDDING_MODEL)
            arr = np.asarray(embs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as e:
            msg = str(e).lower()
            transient = (
                "loading" in msg
                or "503" in msg
                or "rate" in msg
                or "timeout" in msg
                or "connection" in msg
            )
            if transient and attempt < EMBED_RETRIES - 1:
                wait = EMBED_RETRY_BASE_DELAY * (2**attempt)
                print(f"  HF transient error — retry in {wait}s ({e!r})")
                time.sleep(wait)
                continue
            raise


# --- index build -------------------------------------------------------------


def build_index(chunks: list[Chunk]) -> Index:
    # Lazy imports to keep the cache-load path light.
    from dotenv import load_dotenv
    from huggingface_hub import InferenceClient
    from rank_bm25 import BM25Okapi

    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Put it in .env at the repo root "
            "(see .env.example)."
        )

    print(f"Embedding {len(chunks)} chunks via HF Inference API ({EMBEDDING_MODEL})...")
    client = InferenceClient(token=token)

    # DOC_PREFIX is "" for BGE (which only prefixes queries) but kept as a
    # constant so swapping back to an E5-family model is a 2-line change.
    texts = [DOC_PREFIX + c.text for c in chunks]
    t0 = time.time()
    dense_parts: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE)):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        dense_parts.append(_embed_batch(client, batch))
    dense = np.vstack(dense_parts).astype(np.float32)
    if dense.shape[1] != EMBED_DIM:
        raise RuntimeError(
            f"Expected {EMBED_DIM}-d embeddings, got {dense.shape[1]} from {EMBEDDING_MODEL}"
        )
    dense = _normalize(dense)
    print(f"  embedded in {time.time() - t0:.1f}s — dense shape {dense.shape}")

    print(f"Building BM25 sparse index...")
    t0 = time.time()
    # Tokenize the raw chunk text (NOT the e5-prefixed string), so "passage"
    # doesn't end up as a corpus token.
    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 built in {time.time() - t0:.1f}s")

    return Index(
        chunks=chunks,
        dense=dense,
        tokenized=tokenized,
        bm25=bm25,
        model_name=EMBEDDING_MODEL,
    )


# --- save / load -------------------------------------------------------------


def save_index(idx: Index, path: Path = INDEX_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)


class _PortableUnpickler(pickle.Unpickler):
    """Maps `__main__.Index/ScoredChunk/Chunk` references in pickled caches
    back to their canonical module homes. Needed because the index can be
    *built* by running `python code/retriever.py` (then those classes live
    in `__main__`) but *loaded* from any other entry point (e.g. agent.py).
    """

    _MAIN_REMAP = {
        "Index": "retriever",
        "ScoredChunk": "retriever",
        "Chunk": "corpus",
        "Doc": "corpus",
    }

    def find_class(self, module, name):
        if module == "__main__" and name in self._MAIN_REMAP:
            module = self._MAIN_REMAP[name]
        return super().find_class(module, name)


def load_index(path: Path = INDEX_CACHE) -> Index:
    with open(path, "rb") as f:
        return _PortableUnpickler(f).load()


def build_or_load_index() -> Index:
    docs = load_docs()
    chunks = chunk_docs(docs)
    sig = _signature(chunks)

    if INDEX_CACHE.exists() and SIGNATURE_FILE.exists():
        cached_sig = SIGNATURE_FILE.read_text(encoding="utf-8").strip()
        if cached_sig == sig:
            print(f"Cache HIT — loading {INDEX_CACHE} (sig={sig})")
            return load_index(INDEX_CACHE)
        print(f"Cache MISS — signature changed ({cached_sig} → {sig})")
    else:
        print("Cache MISS — no prior index")

    idx = build_index(chunks)
    save_index(idx)
    SIGNATURE_FILE.write_text(sig, encoding="utf-8")
    size_mb = INDEX_CACHE.stat().st_size / (1024 * 1024)
    print(f"Saved index ({len(chunks)} chunks, {size_mb:.1f} MB) to {INDEX_CACHE}")
    return idx


# --- search ------------------------------------------------------------------

# Cached HF client (one per process; embedding queries reuse the same connection).
_search_client = None


def _get_client():
    global _search_client
    if _search_client is None:
        from dotenv import load_dotenv
        from huggingface_hub import InferenceClient

        load_dotenv(REPO_ROOT / ".env")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set — see .env")
        _search_client = InferenceClient(token=token)
    return _search_client


def _embed_query(query: str) -> np.ndarray:
    """Embed one query string. Returns L2-normalized (768,) float32.

    On-disk cache keyed by sha256(model + prefix + query). Re-runs on the
    same tickets skip the HF API call entirely.
    """
    key = hashlib.sha256(
        f"{EMBEDDING_MODEL}|{QUERY_PREFIX}|{query}".encode("utf-8")
    ).hexdigest()
    cache_path = QUERY_EMBED_CACHE_DIR / f"{key}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    client = _get_client()
    arr = _embed_batch(client, [QUERY_PREFIX + query])  # (1, 768)
    vec = _normalize(arr)[0].astype(np.float32)

    QUERY_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vec)
    return vec


def _ranks_descending(scores: np.ndarray) -> np.ndarray:
    """Return 1-indexed ranks where rank[i] is the position of scores[i]
    when sorted descending. Stable for ties via argsort."""
    order = np.argsort(-scores, kind="stable")
    rank = np.empty_like(order, dtype=np.int64)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank


def search(
    idx: Index,
    query: str,
    company: str | None = None,
    k: int = 5,
    rrf_k: int = 60,
    dedup_by_doc: bool = True,
) -> list[ScoredChunk]:
    """Hybrid retrieval over the cached index.

    Filters by company first (if given), then scores by both dense cosine
    and BM25, and fuses the two ranked lists via Reciprocal Rank Fusion:

        rrf(d) = 1/(rrf_k + dense_rank(d)) + 1/(rrf_k + bm25_rank(d))

    rrf_k=60 is the standard value from the original RRF paper; it controls
    how aggressively top ranks dominate.

    If `dedup_by_doc=True` (default), at most one chunk per source doc is
    returned — the highest-RRF chunk per doc — so top-K reflects K *unique*
    documents rather than K possibly-redundant chunks.

    Returns top-K ScoredChunks.
    """
    if not query.strip():
        return []

    # 1. Company-scope filter
    if company:
        co_norm = company.strip().lower()
        valid_idx = np.fromiter(
            (i for i, c in enumerate(idx.chunks) if c.company == co_norm),
            dtype=np.int64,
        )
    else:
        valid_idx = np.arange(len(idx.chunks), dtype=np.int64)
    if valid_idx.size == 0:
        return []

    # 2. Dense scores (cosine == dot product because both sides are L2-normalized)
    q_dense = _embed_query(query)
    dense_all = idx.dense @ q_dense  # (N,)
    dense_scores = dense_all[valid_idx]

    # 3. BM25 scores over the *full* corpus, then sliced to filtered indices
    q_tokens = tokenize(query)
    if not q_tokens:
        bm25_scores = np.zeros(valid_idx.size, dtype=np.float32)
    else:
        bm25_all = np.asarray(idx.bm25.get_scores(q_tokens), dtype=np.float32)
        bm25_scores = bm25_all[valid_idx]

    # 4. Per-list ranks (1-indexed, descending)
    dense_rank = _ranks_descending(dense_scores)
    bm25_rank = _ranks_descending(bm25_scores)

    # 5. Reciprocal Rank Fusion
    rrf = 1.0 / (rrf_k + dense_rank) + 1.0 / (rrf_k + bm25_rank)

    # 6. Top-K with optional doc-level dedup
    sorted_local = np.argsort(-rrf, kind="stable")
    results: list[ScoredChunk] = []
    seen_docs: set = set()
    for li in sorted_local:
        gi = int(valid_idx[li])
        chunk = idx.chunks[gi]
        if dedup_by_doc:
            key = chunk.doc_path
            if key in seen_docs:
                continue
            seen_docs.add(key)
        results.append(
            ScoredChunk(
                chunk=chunk,
                rrf_score=float(rrf[li]),
                dense_score=float(dense_scores[li]),
                bm25_score=float(bm25_scores[li]),
                dense_rank=int(dense_rank[li]),
                bm25_rank=int(bm25_rank[li]),
            )
        )
        if len(results) >= k:
            break
    return results


# --- self-check --------------------------------------------------------------


def _selftest() -> None:
    idx = build_or_load_index()
    print()
    print(f"chunks indexed: {len(idx.chunks)}")
    print(f"dense:  shape={idx.dense.shape}  dtype={idx.dense.dtype}")
    print(f"BM25:   docs={len(idx.tokenized)}  avg tokens/chunk={sum(len(t) for t in idx.tokenized) / max(1, len(idx.tokenized)):.1f}")
    norms = np.linalg.norm(idx.dense, axis=1)
    print(f"dense norms: min={norms.min():.3f}  max={norms.max():.3f}  mean={norms.mean():.3f}")


def _selftest_search() -> None:
    idx = build_or_load_index()
    print(f"\nLoaded index with {len(idx.chunks)} chunks. Running smoke-test queries...\n")

    queries = [
        ("site is down", "hackerrank"),
        ("dispute a charge", "visa"),
        ("how to delete conversation", "claude"),
    ]
    for q, co in queries:
        print(f"Q: {q!r}  (company={co})")
        results = search(idx, q, company=co, k=3)
        for rank, sc in enumerate(results, 1):
            print(
                f"  #{rank} rrf={sc.rrf_score:.5f}  "
                f"dense={sc.dense_score:.3f} (rank {sc.dense_rank})  "
                f"bm25={sc.bm25_score:.2f} (rank {sc.bm25_rank})"
            )
            print(f"      title: {sc.chunk.title}")
            print(f"      breadcrumbs: {sc.chunk.breadcrumbs}")
            print(f"      doc: {sc.chunk.doc_path.name}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        _selftest_search()
    else:
        _selftest()
