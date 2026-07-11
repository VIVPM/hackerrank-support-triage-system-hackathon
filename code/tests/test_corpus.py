"""Lightweight tests for corpus loader + chunker. Reads the real data/
directory if present; otherwise the corpus tests are skipped."""

from __future__ import annotations

import pytest

from corpus import (
    COMPANIES,
    DATA_DIR,
    PRODUCT_AREA_VOCAB,
    Doc,
    chunk_docs,
    load_docs,
    vocab_for,
)


def test_vocab_has_all_three_companies():
    assert set(PRODUCT_AREA_VOCAB.keys()) == {"hackerrank", "claude", "visa"}
    for co, terms in PRODUCT_AREA_VOCAB.items():
        assert len(terms) >= 5, f"{co} has only {len(terms)} terms"
        for t in terms:
            assert t == t.lower(), f"vocab term not lowercase: {co} / {t}"
            assert " " not in t, f"vocab term has space: {co} / {t}"


def test_vocab_for_normalizes_case():
    assert vocab_for("HackerRank") == PRODUCT_AREA_VOCAB["hackerrank"]
    assert vocab_for("CLAUDE") == PRODUCT_AREA_VOCAB["claude"]
    assert vocab_for(None) == []
    assert vocab_for("unknown") == []


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/ corpus not present")
def test_load_docs_returns_some_docs():
    docs = load_docs()
    assert len(docs) > 0
    # Each company should contribute at least one doc.
    by_co = {co: [d for d in docs if d.company == co] for co in COMPANIES}
    for co in COMPANIES:
        assert len(by_co[co]) > 0, f"No docs loaded for {co}"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/ corpus not present")
def test_chunks_have_required_fields():
    docs = load_docs()
    chunks = chunk_docs(docs[:5])  # only need a handful for this test
    assert chunks, "chunk_docs returned nothing"
    c = chunks[0]
    assert c.company in COMPANIES
    assert c.title
    assert c.text
    # Each chunk's text should start with the [company] title header.
    assert c.text.startswith(f"[{c.company}]")


def test_chunk_a_synthetic_doc_no_data_required():
    doc = Doc(
        company="hackerrank",
        path=DATA_DIR / "fake.md",
        title="Synthetic",
        source_url="https://example.com",
        breadcrumbs=["Top", "Section"],
        body="A" * 4500,  # forces multiple chunks at CHUNK_SIZE=2000, overlap=200
    )
    chunks = chunk_docs([doc])
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    assert all(c.company == "hackerrank" for c in chunks)
    assert "Top > Section" in chunks[0].text
