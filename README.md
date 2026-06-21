# GhostTrace — Hidden Ownership & Shell Company Tracer

AI-powered beneficial ownership investigation tool — type a company name, pull SEC EDGAR filings, extract every owner and subsidiary Claude can find, resolve name variants, build an ownership graph, and score for shell company and sanctions risk.

Built for analysts who need to trace the real humans behind layered corporate structures without paying for a Bloomberg terminal.

**Live demo:** https://ghosttrace-aose.onrender.com

---

## What It Does

Shell companies obscure beneficial ownership across multiple jurisdictions and filing types. GhostTrace automates the first stage of that investigation:

1. **Ingest** — searches SEC EDGAR for SC 13D/13G (large ownership disclosures), 10-K Exhibit 21 (subsidiary lists), and DEF 14A (proxy statement ownership tables)
2. **Extract** — Claude reads each filing and pulls every owner, subsidiary, affiliate, address, and percentage mentioned in unstructured text
3. **Resolve** — rapidfuzz token-sort and token-set fuzzy matching deduplicates name variants ("Apex Holdings LLC" vs "APEX HOLDINGS" vs "Apex Hldgs") with jurisdiction-conflict guard and memoized adjudication
4. **Graph** — NetworkX builds a directed ownership graph; matplotlib renders it as a PNG
5. **Score** — deterministic risk engine weights shell company indicators: offshore jurisdiction, no operational address, circular ownership, single-person directorship
6. **Screen** — on-demand OFAC SDN check (rapidfuzz against sdn.csv + alt.csv, token_sort_ratio ≥ 90, 35-point weight per hit)
7. **Trace** — bounded Deep Trace agentic loop: Claude proposes the next investigation step, calls `investigate_entity` or `search_cached_filings`, max 5 tool calls

---

## Features

| Feature | Description |
|---------|-------------|
| EDGAR filing search | SC 13D/13G, 10-K Exhibit 21, DEF 14A across all public companies |
| Claude extraction | Entity names, ownership %, filing dates, jurisdictions from unstructured text |
| Entity resolution | rapidfuzz token-sort + token-set dedup, jurisdiction-conflict guard |
| Ownership graph | NetworkX DiGraph → matplotlib PNG; nodes sized by ownership % |
| Risk scoring | Deterministic shell company indicators (jurisdiction, address, circularity, directorship) |
| ChromaDB semantic search | `/filings` search over cached filing text via in-memory vector store |
| OFAC SDN screening | On-demand match against Treasury SDN list (sdn.csv + alt.csv) |
| Deep Trace loop | Bounded agentic investigation (max 5 tool calls) at `/trace/{id}/deep-trace` |
| Report | Filed-and-cited ownership report linking back to EDGAR source documents |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python |
| AI | Claude Haiku (extraction, Deep Trace loop) |
| Filings | SEC EDGAR EFTS full-text search + filing index |
| Entity resolution | rapidfuzz 3.x (`process.extract`, not deprecated `extractBests`) |
| Graph | NetworkX + matplotlib |
| Semantic search | ChromaDB (in-memory) + custom hashed bag-of-words embedder |
| Sanctions | OFAC SDN list (sdn.csv + alt.csv, downloaded on first use) |
| Database | SQLite + SQLAlchemy 2.0 |
| Frontend | Jinja2 templates + vanilla CSS |
| Deploy | Render (free tier) |

---

## Quick Start

```bash
git clone https://github.com/JaKPoT-Sudo/ghosttrace.git
cd ghosttrace
cp .env.example .env          # add ANTHROPIC_API_KEY=sk-ant-...
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\uvicorn main:app --reload
```

Open http://localhost:8000

---

## Demo

Load `/seed` to pre-populate **Harborview Capital Partners** — a fictional entity designed to demonstrate every detection pattern:

- Nested offshore subsidiaries (Cayman → BVI → Delaware)
- Nominee director with no disclosed beneficial owner
- Circular ownership between two entities
- One entity name-matching the OFAC SDN list

Use the Deep Trace button to watch the agentic loop propose and execute investigation steps against the cached filing corpus.

---

## Architecture

```
edgar_client.py       EFTS full-text search, filing index fetch, 10-K/SC13D/DEF14A download
claude_extractor.py   Claude Haiku: unstructured filing text → structured ownership JSON
entity_resolver.py    rapidfuzz dedup, token-sort + token-set, jurisdiction-conflict guard
graph_builder.py      NetworkX DiGraph construction, matplotlib PNG render
risk_engine.py        Deterministic shell company scoring (jurisdiction, address, circularity)
vector_store.py       ChromaDB in-memory store, custom hashed bag-of-words embedder
ofac_checker.py       sdn.csv + alt.csv download, rapidfuzz token_sort_ratio ≥ 90 match
deep_trace.py         Bounded agentic loop: max 5 tool calls, investigate_entity + search tools
models.py             SQLAlchemy ORM (Company, Filing, Entity, OwnershipEdge, RiskScore)
seed_data.py          Harborview Capital Partners synthetic demo dataset
main.py               FastAPI routes, Jinja rendering, lifespan seed
```

---

## Key Architecture Decisions

**Why a custom embedder instead of ChromaDB's default ONNX model:**
ChromaDB's bundled ONNX embedder requires ~80 MB download and ~200 MB RAM at runtime — more than Render's 512 MB free-tier container has available. The custom hashed bag-of-words embedder is one function, zero extra dependencies, and works well for the keyword-heavy queries typical of filing searches ("nominee," "undisclosed beneficial owner," "Cayman"). One-function swap path if semantic quality ever needs upgrading.

**Why rapidfuzz over fuzzy-wuzzy or difflib:**
rapidfuzz v3 implements the same Levenshtein algorithms at C speed. The key call is `process.extract` — the legacy `extractBests` was renamed in v3, a common silent breakage.

**Why OFAC screening is on-demand, not automatic:**
Downloads the SDN list on first use (3–5 seconds). Automatic screening on every seed would slow cold starts on Render's free tier. Same explicit-action discipline as CFIUS Screener's OFAC module.

**Why the Deep Trace loop is bounded at 5 tool calls:**
Demonstrates the "GO-1 Human Override" principle from the DoD Responsible AI framework — the system cannot run indefinitely without re-authorization. Five calls is enough to show multi-step investigation without runaway cost.

---

## Honest Limitations

- The custom embedder is lexical, not semantic — synonym queries miss results that semantic vectors would catch. Documented tradeoff, one-function swap path exists.
- EDGAR text quality varies: Exhibit 21 subsidiary lists are often formatted as tables that get mangled in plain-text extraction.
- OFAC match at token_sort_ratio ≥ 90 has false positives on common words in entity names — every match requires human review.
- No live corporate registry data beyond SEC filings — state-level and offshore registrations are not covered.
- Demo data is entirely synthetic — Harborview Capital Partners does not exist.

---

## Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
# 116 passed
```

Covers: EDGAR client, entity resolver (token-sort, token-set, jurisdiction-conflict), risk scoring, OFAC matching, ChromaDB store, Deep Trace loop (mocked), Claude extraction parsing.

---

*DEMONSTRATION ONLY — synthetic demo data — Harborview Capital Partners is entirely fictional. Not for legal or compliance use without independent verification.*
