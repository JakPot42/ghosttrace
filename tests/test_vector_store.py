"""Tests for vector_store.py — chunking, embedding, and search behavior."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from vector_store import VectorStore, chunk_text, embed
from config import CHUNK_CHARS, EMBED_DIM


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_multiple_chunks(self):
        text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_text(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= CHUNK_CHARS

    def test_chunks_break_on_whitespace(self):
        text = " ".join(f"word{i}" for i in range(1000))
        for c in chunk_text(text):
            # no chunk starts or ends mid-word
            assert not c.startswith(" ")
            assert not c.endswith(" ")

    def test_overlap_preserves_content(self):
        # Every word of the original text must appear in some chunk
        words = [f"unique{i}" for i in range(800)]
        text = " ".join(words)
        joined = " ".join(chunk_text(text))
        for w in words:
            assert w in joined


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_correct_dimension(self):
        assert len(embed("some filing text")) == EMBED_DIM

    def test_deterministic(self):
        assert embed("nominee services") == embed("nominee services")

    def test_normalized(self):
        v = embed("beneficial ownership disclosure")
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_empty_text_zero_vector(self):
        assert all(x == 0.0 for x in embed(""))

    def test_case_insensitive(self):
        assert embed("NOMINEE Services") == embed("nominee services")

    def test_different_texts_differ(self):
        assert embed("cayman islands shell company") != embed("delaware pension fund")


# ---------------------------------------------------------------------------
# VectorStore — uses a real embedded Chroma client
# ---------------------------------------------------------------------------

@pytest.fixture()
def store():
    s = VectorStore()
    s.clear()
    return s


def _add(store, accession, text, form="SC 13D", date="2026-01-01", cik=0, doc="doc.txt"):
    return store.add_filing(
        accession_number=accession, document_name=doc, form=form,
        filing_date=date, cik=cik, text=text,
    )


class TestVectorStore:
    def test_add_and_count(self, store):
        added = _add(store, "ACC-001", "Calloway holds shares as nominee for an undisclosed owner.")
        assert added == 1
        assert store.count() == 1

    def test_search_finds_relevant_chunk_first(self, store):
        _add(store, "ACC-001", "Calloway Nominee Services holds shares of record as nominee. "
                               "The beneficial owner has not been disclosed.")
        _add(store, "ACC-002", "The reporting person is a pension fund organized in New York "
                               "investing on behalf of retired teachers.")
        results = store.search("nominee undisclosed beneficial owner")
        assert results
        assert results[0]["accession_number"] == "ACC-001"

    def test_search_returns_metadata(self, store):
        _add(store, "ACC-003", "Registered office at George Town, Cayman Islands.",
             form="SC 13G", date="2026-02-02", cik=12345, doc="x.htm")
        r = store.search("Cayman registered office")[0]
        assert r["form"] == "SC 13G"
        assert r["filing_date"] == "2026-02-02"
        assert r["cik"] == 12345
        assert r["document_name"] == "x.htm"
        assert 0.0 <= r["score"] <= 1.0

    def test_search_empty_query(self, store):
        _add(store, "ACC-001", "some text")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_empty_index(self, store):
        assert store.search("anything") == []

    def test_readd_same_document_is_idempotent(self, store):
        _add(store, "ACC-001", "identical text")
        _add(store, "ACC-001", "identical text")
        assert store.count() == 1

    def test_clear(self, store):
        _add(store, "ACC-001", "some text")
        store.clear()
        assert store.count() == 0

    def test_k_caps_results(self, store):
        for i in range(12):
            _add(store, f"ACC-{i:03d}", f"filing number {i} about ownership and shares")
        assert len(store.search("ownership shares", k=5)) == 5


# ---------------------------------------------------------------------------
# Demo seed relevance — the queries the demo video will actually use
# ---------------------------------------------------------------------------

class TestSeedSearchRelevance:
    def test_nominee_query_finds_calloway_filing(self, store):
        from seed_data import SEED_FILINGS
        for f in SEED_FILINGS:
            _add(store, f["accession_number"], f["text"],
                 form=f["form"], date=f["filing_date"], doc=f["document_name"])
        results = store.search("nominee undisclosed beneficial owner")
        assert results[0]["accession_number"] == "DEMO-13D-002"

    def test_subsidiary_query_finds_exhibit_21(self, store):
        from seed_data import SEED_FILINGS
        for f in SEED_FILINGS:
            _add(store, f["accession_number"], f["text"],
                 form=f["form"], date=f["filing_date"], doc=f["document_name"])
        results = store.search("subsidiaries of the registrant")
        assert results[0]["accession_number"] == "DEMO-10K-001"
