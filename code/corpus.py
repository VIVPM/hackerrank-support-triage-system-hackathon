"""Markdown corpus loader, chunker, and product_area vocabulary.

Walks data/{hackerrank,claude,visa}/**/*.md, parses YAML frontmatter, and
yields chunks suitable for embedding. The vocabulary is the soft-closed
list locked in architecture.md §5.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
COMPANIES = ("hackerrank", "claude", "visa")

# Chunking is character-based because every model tokenizer differs slightly
# and we don't need exact token counts here. ~2000 chars ≈ 500 tokens for
# English markdown; 200-char overlap preserves cross-boundary context.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


@dataclass
class Doc:
    company: str
    path: Path
    title: str
    source_url: str
    breadcrumbs: list[str]
    body: str


@dataclass
class Chunk:
    company: str
    doc_path: Path
    chunk_idx: int
    title: str
    source_url: str
    breadcrumbs: list[str]
    text: str  # body slice with title+breadcrumb header prefixed


# --- frontmatter parsing -----------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_QUOTED_RE = re.compile(r'^"(.*)"$')


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Empty dict if no frontmatter."""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_text, body = m.group(1), m.group(2)
    fm: dict = {}
    current_list_key: str | None = None
    for line in fm_text.splitlines():
        if not line.strip():
            current_list_key = None
            continue
        # list continuation: `  - "Foo"`
        list_match = re.match(r"\s*-\s*(.*)$", line)
        if list_match and current_list_key is not None:
            val = list_match.group(1).strip()
            qm = _QUOTED_RE.match(val)
            fm[current_list_key].append(qm.group(1) if qm else val)
            continue
        # key/value: `title: "Foo"` or `breadcrumbs:`
        kv = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if not kv:
            current_list_key = None
            continue
        key, val = kv.group(1), kv.group(2).strip()
        if not val:
            # opens a list
            fm[key] = []
            current_list_key = key
        else:
            qm = _QUOTED_RE.match(val)
            fm[key] = qm.group(1) if qm else val
            current_list_key = None
    return fm, body


# --- doc loading -------------------------------------------------------------


def _fallback_title(path: Path, body: str) -> str:
    # First H1 if present, else file stem.
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("#").strip()
    return path.stem.replace("-", " ").replace("_", " ").strip()


def load_docs() -> list[Doc]:
    docs: list[Doc] = []
    for co in COMPANIES:
        base = DATA_DIR / co
        if not base.exists():
            continue
        for md_path in base.rglob("*.md"):
            try:
                raw = md_path.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, body = _parse_frontmatter(raw)
            title = fm.get("title") or _fallback_title(md_path, body)
            source_url = fm.get("source_url", "")
            breadcrumbs = fm.get("breadcrumbs", []) or []
            if not isinstance(breadcrumbs, list):
                breadcrumbs = [str(breadcrumbs)]
            body = body.strip()
            if not body:
                continue
            docs.append(
                Doc(
                    company=co,
                    path=md_path,
                    title=str(title),
                    source_url=str(source_url),
                    breadcrumbs=[str(b) for b in breadcrumbs],
                    body=body,
                )
            )
    return docs


# --- chunking ----------------------------------------------------------------


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_docs(docs: list[Doc]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        crumb_path = " > ".join(d.breadcrumbs) if d.breadcrumbs else ""
        header = f"[{d.company}] {d.title}"
        if crumb_path:
            header += f" ({crumb_path})"
        for i, piece in enumerate(_chunk_text(d.body, CHUNK_SIZE, CHUNK_OVERLAP)):
            text = f"{header}\n\n{piece}"
            chunks.append(
                Chunk(
                    company=d.company,
                    doc_path=d.path,
                    chunk_idx=i,
                    title=d.title,
                    source_url=d.source_url,
                    breadcrumbs=d.breadcrumbs,
                    text=text,
                )
            )
    return chunks


# --- product_area vocabulary -------------------------------------------------

# Hardcoded soft-closed vocab from architecture.md §5.1.
PRODUCT_AREA_VOCAB: dict[str, list[str]] = {
    "hackerrank": [
        "screen",
        "interviews",
        "community",
        "integrations",
        "account_management",
        "billing",
        "certifications",
        "chakra",
        "test_integrity",
        "settings",
    ],
    "claude": [
        "privacy",
        "conversation_management",
        "account_management",
        "billing",
        "api",
        "claude_code",
        "features",
        "troubleshooting",
        "security",
        "education",
    ],
    "visa": [
        "travel_support",
        "general_support",
        "fraud",
        "dispute",
        "card_services",
        "payments",
    ],
}


def vocab_for(company: str | None) -> list[str]:
    if not company:
        return []
    return PRODUCT_AREA_VOCAB.get(company.strip().lower(), [])


# --- self-check --------------------------------------------------------------


def _selftest() -> None:
    docs = load_docs()
    print(f"Loaded {len(docs)} docs")
    by_co: dict[str, int] = {}
    for d in docs:
        by_co[d.company] = by_co.get(d.company, 0) + 1
    for co in COMPANIES:
        print(f"  {co:11s} {by_co.get(co, 0):4d} docs")
    chunks = chunk_docs(docs)
    print(f"\nChunked into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    by_co_c: dict[str, int] = {}
    for c in chunks:
        by_co_c[c.company] = by_co_c.get(c.company, 0) + 1
    for co in COMPANIES:
        print(f"  {co:11s} {by_co_c.get(co, 0):4d} chunks")
    print("\nFirst chunk preview:")
    c = chunks[0]
    print(f"  company:  {c.company}")
    print(f"  title:    {c.title}")
    print(f"  breadcrumbs: {c.breadcrumbs}")
    print(f"  source_url: {c.source_url}")
    print(f"  text head: {c.text[:160]!r}")


if __name__ == "__main__":
    _selftest()
