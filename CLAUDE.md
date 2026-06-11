# GhostTrace — Claude Code context

GhostTrace is a FastAPI web app that traces hidden ownership structures in
SEC-registered companies. It fetches EDGAR filings, uses Claude Haiku to
extract entities and relationships, runs deterministic risk scoring, screens
against the OFAC SDN list, and optionally runs a bounded agentic Deep Trace.

## Tech stack

- Python 3.11.9 (Render) / 3.14 (local)
- FastAPI + Jinja2 templates + vanilla CSS
- SQLAlchemy 2.0 + SQLite (ephemeral on Render free tier — seed runs every cold start)
- Claude Haiku (`claude-haiku-4-5-20251001`) — extraction, adjudication, report, deep trace
- ChromaDB (in-memory EphemeralClient) — semantic filing search
- NetworkX + matplotlib — ownership graph PNG
- rapidfuzz — OFAC SDN fuzzy matching

## Milestone status

- **Milestone 1 ✅** — EDGAR fetch → Claude extraction → entity resolution → risk scoring → graph → report
- **Milestone 2 ✅** — Hardened resolution (token-sort, token-set, jurisdiction conflict guard, adjudicator memo) + ChromaDB semantic search at /filings + SQLite filing cache
- **Milestone 3 ✅** — OFAC SDN screening (`ofac_checker.py`) + bounded Deep Trace agentic loop (`deep_trace.py`)

## Architecture decisions

**Rules score, Claude explains.** `risk_engine.py` scores deterministically from
config weights. Claude writes the narrative. Scores never come from Claude.

**Orchestrated pipeline, not autonomous agent (v1).** The standard trace fetches
up to MAX_FILINGS_PER_TRACE (10) documents, extracts from each, then resolves
entities. Deep Trace is the only agentic feature — and it is opt-in, bounded.

**ChromaDB uses a hashed bag-of-words embedder, not the default ONNX model.**
The default (`all-MiniLM` via `onnxruntime`) downloads ~80 MB on first use and
holds ~200 MB RAM. On a 512 MB Render free-tier instance that cold-starts on
every wake, that means OOM. The custom embedder in `vector_store.py` is fast,
dependency-free, and deterministic; the tradeoff is lexical matching rather than
true semantics. It is a one-function swap if resources improve.

**EDGAR rate limit: 8 req/s.** SEC caps automated access at 10 req/s and IP-bans
violators. 8 leaves headroom for network jitter. The `_RateLimiter` in
`edgar_client.py` is process-global and thread-safe — critical on FastAPI's
async request handlers.

**OFAC download is lazy and in-memory.** `ofac_checker.py` downloads `sdn.csv`
and `alt.csv` on the first `screen_entities()` call and caches the name list
for the process lifetime. There is no TTL mechanism because Render free-tier
processes are ephemeral — every cold start is a fresh process. The download
adds ~3–5 seconds to the first OFAC check per session; subsequent checks are
fast (the list lives in memory).

**OFAC hits are candidates, not confirmed violations.** The token_sort_ratio
threshold (90, configurable) produces name-match candidates. Every hit is
labeled "verification required" in the UI and in findings. They add 35 points
to the risk score per hit to surface them, but the display makes their
provisional nature clear.

**Deep Trace budget: 5 tool calls.** The loop removes tools from the Anthropic
API call once the budget is spent, forcing Claude to respond with text. An
explicit "synthesize now" message is also appended. This guarantees a readable
report even when the budget is exhausted mid-investigation.

**Starlette 1.x TemplateResponse signature:** `TemplateResponse(request, name, ctx)` —
request is the first positional argument (not in ctx dict). This changed in
Starlette 0.36+.

## Module map

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, all routes, trace orchestration |
| `config.py` | All tunable constants — no logic |
| `models.py` | SQLAlchemy ORM: Filing, Trace, Entity, OwnershipLink |
| `database.py` | Engine, Base, get_db, init_db |
| `edgar_client.py` | EDGAR HTTP layer — rate limited, no DB, no Claude |
| `claude_extractor.py` | Claude calls: extraction, adjudication, report |
| `entity_resolver.py` | Deterministic name dedup — no web, no Claude |
| `risk_engine.py` | Deterministic risk scoring — no web, no Claude |
| `ofac_checker.py` | OFAC SDN download + rapidfuzz screening |
| `deep_trace.py` | Bounded Claude tool-use agentic loop |
| `graph_builder.py` | NetworkX + matplotlib ownership graph PNG |
| `vector_store.py` | ChromaDB in-memory filing index + hashed embedder |
| `seed_data.py` | Fictional Harborview Capital Partners demo network |

## Deployment

- Render service ID: `srv-d8l0ml9o3t8c73ah5cvg`
- Live URL: https://ghosttrace-aose.onrender.com
- Auto-deploys on push to `master`
- Required env vars on Render: `ANTHROPIC_API_KEY`, `PYTHON_VERSION=3.11.9`, `DEMO_MODE=True`

## Test suite

116 tests, all passing. Run with `py -m pytest`.
No test ever makes a real HTTP call (EDGAR, Anthropic, OFAC) — all are mocked.

## Known limitations

- `get_company_candidates()` only covers SEC-registered companies with tickers.
  Private 13D filers (hedge funds, family offices) often won't resolve.
- `get_filing_documents()` returns `list[str]` (filenames). The `main.py`
  orchestrator has a pre-existing `d["name"]` call that would fail on live
  traces; this is a known issue in the existing code, not introduced in M3.
  `deep_trace.py` uses the correct `list[str]` API.
- Deep Trace is disabled on demo (synthetic) traces — the fictional entity names
  won't resolve in EDGAR.
