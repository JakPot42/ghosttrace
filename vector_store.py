"""
vector_store.py — filing-text chunking, embedding, and search over ChromaDB.

ChromaDB runs embedded and in-memory; the index is rebuilt from the Filing
table at startup (reindex_from_db), which makes it consistent with the
SQLite-wiped-on-restart reality of Render's free tier.

Embeddings are computed here and passed to Chroma explicitly — we never use
Chroma's default embedding function. See config.py for the full rationale
(cold-start time and RAM on a 512MB instance). The hashed bag-of-words
embedder below does lexical matching, not true semantics: a query matches
chunks that share vocabulary, weighted and normalized. That is an honest,
documented tradeoff — swap `embed()` for a transformer model when the
deployment envelope allows it.
"""

from __future__ import annotations

import threading
import zlib

from config import CHUNK_CHARS, CHUNK_OVERLAP, EMBED_DIM, SEARCH_RESULTS_K

_COLLECTION_NAME = "filings"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    word: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return out


def embed(text: str) -> list[float]:
    """Hashed bag-of-words embedding: tokens and bigrams hashed into a fixed
    vector, L2-normalized. crc32 (not hash()) because Python salts hash()
    per process — embeddings must be stable across restarts."""
    vec = [0.0] * EMBED_DIM
    tokens = _tokenize(text)
    for tok in tokens:
        vec[zlib.crc32(tok.encode()) % EMBED_DIM] += 1.0
    for a, b in zip(tokens, tokens[1:]):
        vec[zlib.crc32(f"{a} {b}".encode()) % EMBED_DIM] += 0.5
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Overlapping character windows, broken on whitespace where possible so
    a chunk never starts or ends mid-word."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # walk back to the last whitespace inside the window
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper over an embedded Chroma collection. Thread-safe adds —
    FastAPI runs sync handlers in a threadpool."""

    def __init__(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            try:
                self._client.delete_collection(_COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )

    def count(self) -> int:
        return self._collection.count()

    def add_filing(
        self,
        accession_number: str,
        document_name: str,
        form: str,
        filing_date: str,
        cik: int,
        text: str,
    ) -> int:
        """Chunk, embed, and index one filing document. Returns chunks added.
        Re-adding the same document is a no-op (ids are deterministic and
        Chroma upserts)."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{accession_number}:{document_name}:{i}" for i in range(len(chunks))]
        embeddings = [embed(c) for c in chunks]
        metadatas = [
            {
                "accession_number": accession_number,
                "document_name": document_name,
                "form": form,
                "filing_date": filing_date,
                "cik": cik,
            }
            for _ in chunks
        ]
        with self._lock:
            self._collection.upsert(
                ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
            )
        return len(chunks)

    def search(self, query: str, k: int = SEARCH_RESULTS_K) -> list[dict]:
        """Returns [{text, accession_number, document_name, form, filing_date,
        cik, score}] best-first. Score is cosine similarity in [0, 1]."""
        if not query.strip() or self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embed(query)],
            n_results=min(k, self.count()),
        )
        out: list[dict] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({
                "text": doc,
                "accession_number": meta["accession_number"],
                "document_name": meta["document_name"],
                "form": meta["form"],
                "filing_date": meta["filing_date"],
                "cik": meta["cik"],
                "score": round(1.0 - dist, 3),
            })
        return out


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()
        return _store


def reindex_from_db(db) -> int:
    """Rebuild the whole index from the Filing table. Called at startup —
    the index is in-memory, the filings are in SQLite, and on Render both
    start empty except for what seed data provides."""
    from models import Filing

    store = get_store()
    store.clear()
    total = 0
    for f in db.query(Filing).all():
        total += store.add_filing(
            accession_number=f.accession_number,
            document_name=f.document_name,
            form=f.form,
            filing_date=f.filing_date,
            cik=f.cik,
            text=f.text,
        )
    return total
