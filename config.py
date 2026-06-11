"""
config.py — ALL tunable parameters for GhostTrace.

Every number that controls the program's judgment lives here, and no logic
lives here at all. The engines (risk_engine.py, entity_resolver.py) apply
these values; they never define their own.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# App identity + demo mode
# ---------------------------------------------------------------------------

APP_TITLE = "GhostTrace — Hidden Ownership & Shell Company Tracer"
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() in ("1", "true", "yes")
DEMO_BANNER = (
    "DEMO MODE — Harborview Capital Partners synthetic ownership network loaded. "
    "All entities fictional. No real intelligence."
)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# SEC EDGAR access
#
# The SEC requires a User-Agent header with real contact info and caps
# automated access at 10 requests/second. Violators get IP-banned — a ban on
# Render's shared IPs would kill the live demo permanently. We run at 8/sec
# so network timing jitter never pushes us over the limit.
# ---------------------------------------------------------------------------

EDGAR_USER_AGENT = "GhostTrace portfolio research (jak.potvin@gmail.com)"
EDGAR_RATE_LIMIT_PER_SEC = 8

# Ticker/name → CIK lookup table (one JSON file covering all registrants)
EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# Filing index for one company; CIK must be zero-padded to 10 digits
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# Individual filing documents
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# The form types that reveal ownership. Everything else is noise for our
# purposes — scope control is a feature, not a limitation.
TARGET_FORM_TYPES = ["SC 13D", "SC 13G", "10-K", "DEF 14A"]

# Cost + speed control: a live trace fetches at most this many documents.
MAX_FILINGS_PER_TRACE = 10

# Filings can run to hundreds of pages. Truncate document text before sending
# to Claude — the ownership tables we need are findable within this window,
# and an unbounded prompt is an unbounded API bill.
MAX_DOC_CHARS = 30_000

# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
EXTRACTION_MAX_TOKENS = 1500   # entity lists from a dense proxy can be long
ADJUDICATION_MAX_TOKENS = 300  # "same entity or not" is a short answer
REPORT_MAX_TOKENS = 2000

# ---------------------------------------------------------------------------
# Entity resolution thresholds (0-100 similarity scale)
#
# Three bands: auto-merge / ask Claude / treat as distinct. These two numbers
# control both API cost (wider middle band = more adjudication calls) and the
# false-merge rate — merging two genuinely different companies poisons the
# whole graph, so the auto-merge bar is deliberately high.
# ---------------------------------------------------------------------------

FUZZY_AUTO_MERGE_THRESHOLD = 92
FUZZY_ADJUDICATE_THRESHOLD = 75

# Corporate suffixes stripped during name normalization, before comparison.
NORMALIZE_SUFFIXES = [
    "inc", "inc.", "incorporated",
    "llc", "l.l.c.",
    "lp", "l.p.", "llp", "l.l.p.",
    "ltd", "ltd.", "limited",
    "corp", "corp.", "corporation",
    "co", "co.", "company",
    "plc", "sa", "s.a.", "ag", "gmbh", "nv", "n.v.", "bv", "b.v.",
    "holdings", "holding", "group",
]

# ---------------------------------------------------------------------------
# Risk scoring
#
# Two jurisdiction lists, deliberately separate, because the story differs:
# a secrecy jurisdiction suggests opacity; an adversary jurisdiction suggests
# exposure. The report should say which. ADVERSARY list and +40 weight match
# FriendShore's HIGH_RISK_COUNTRIES for suite consistency.
# ---------------------------------------------------------------------------

SECRECY_JURISDICTIONS = [
    "Cayman Islands", "British Virgin Islands", "Bermuda", "Panama",
    "Cyprus", "Luxembourg", "Seychelles", "Marshall Islands",
    "Liechtenstein", "Isle of Man", "Jersey", "Guernsey",
    "Malta", "Belize", "Bahamas",
]

ADVERSARY_JURISDICTIONS = [
    "China", "PRC", "Russia", "Iran", "North Korea", "DPRK",
    "Belarus", "Venezuela",
]

RISK_WEIGHT_SECRECY_JURISDICTION = 30
RISK_WEIGHT_ADVERSARY_JURISDICTION = 40
RISK_WEIGHT_CIRCULAR_OWNERSHIP = 25
RISK_WEIGHT_CHAIN_DEPTH = 20      # applied when chain depth >= threshold below
CHAIN_DEPTH_THRESHOLD = 3
RISK_WEIGHT_SHARED_AGENT = 15     # multiple entities, one registered agent
RISK_WEIGHT_UNDISCLOSED_OWNER = 10

# Score → level mapping
RISK_LEVEL_HIGH = 60
RISK_LEVEL_MEDIUM = 30

# ---------------------------------------------------------------------------
# OFAC sanctions check (milestone 3 — declared now so the design accounts
# for it)
#
# FLAG: Treasury reorganized its sanctions-list hosting in recent years.
# Verify this URL against the current OFAC site before building the OFAC
# module — do not assume it still resolves.
# ---------------------------------------------------------------------------

OFAC_SDN_CSV_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
OFAC_MATCH_THRESHOLD = 90  # fuzzy score required to flag an SDN match

# ---------------------------------------------------------------------------
# Semantic search (milestone 2)
#
# ChromaDB runs embedded and in-memory (EphemeralClient). Persistence is
# pointless on Render free tier — the disk is wiped on every restart along
# with SQLite — so the index is rebuilt from the Filing table at startup.
#
# Embeddings are a deliberate departure from Chroma's default. The default
# (all-MiniLM via onnxruntime) downloads an ~80MB model on first use and
# holds ~200MB of RAM — on a 512MB free-tier instance that cold-starts on
# every wake, that means multi-minute cold starts and likely OOM. We use a
# dependency-free hashed bag-of-words embedder instead: deterministic,
# instant, identical locally and deployed. The tradeoff is lexical matching
# rather than true semantics; the embedder is one function and swappable.
# ---------------------------------------------------------------------------

EMBED_DIM = 512            # hashed embedding dimensionality
CHUNK_CHARS = 1200         # filing text chunk size for indexing
CHUNK_OVERLAP = 200        # overlap between adjacent chunks (citation context)
SEARCH_RESULTS_K = 8       # results returned per filing search

# ---------------------------------------------------------------------------
# Output + database
# ---------------------------------------------------------------------------

GRAPH_OUTPUT_DIR = "static/graphs"
DATABASE_URL = "sqlite:///./ghosttrace.db"
