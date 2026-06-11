"""
edgar_client.py — SEC EDGAR HTTP layer.

Pure data fetching: no Claude, no database. Everything here is testable with
mocked HTTP responses alone. Filing caching lives in the orchestration layer
(the Filing table), not here — this module always actually fetches.
"""

from __future__ import annotations

import threading
import time
from html.parser import HTMLParser

import httpx

from config import (
    EDGAR_ARCHIVES_BASE,
    EDGAR_COMPANY_TICKERS_URL,
    EDGAR_RATE_LIMIT_PER_SEC,
    EDGAR_SUBMISSIONS_URL,
    EDGAR_USER_AGENT,
    MAX_DOC_CHARS,
    MAX_FILINGS_PER_TRACE,
    TARGET_FORM_TYPES,
)


class EdgarError(Exception):
    pass


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Process-wide rate limiter shared by every EDGAR call.

    The SEC's limit applies to our IP, not to any single code path — so the
    state (last request time) must be shared across concurrent request
    handlers, guarded by a lock. Sleeps only for the deficit since the last
    request, which is often zero when Claude extraction ran in between.
    """

    def __init__(self, max_per_second: float):
        self._min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        with self._lock:
            # monotonic, not time.time(): wall clock can jump backwards
            elapsed = time.monotonic() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.monotonic()


_limiter = _RateLimiter(EDGAR_RATE_LIMIT_PER_SEC)

_HEADERS = {"User-Agent": EDGAR_USER_AGENT}


def _get(url: str) -> httpx.Response:
    _limiter.wait()
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=15.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise EdgarError(f"EDGAR request failed: {url} — {exc}") from exc
    if resp.status_code != 200:
        raise EdgarError(f"EDGAR returned HTTP {resp.status_code} for {url}")
    return resp


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

# The ticker table is ~1 MB covering every listed registrant; cache it in
# memory and refresh daily. Only covers companies with tickers — private
# 13D filers won't resolve here, which is a documented v1 limitation.
_TICKER_CACHE: dict = {"fetched_at": 0.0, "rows": []}
_TICKER_CACHE_TTL_SECONDS = 24 * 3600


def _ticker_table() -> list[dict]:
    age = time.monotonic() - _TICKER_CACHE["fetched_at"]
    if not _TICKER_CACHE["rows"] or age > _TICKER_CACHE_TTL_SECONDS:
        data = _get(EDGAR_COMPANY_TICKERS_URL).json()
        # EDGAR serves this as {"0": {...}, "1": {...}} keyed by row number
        _TICKER_CACHE["rows"] = list(data.values())
        _TICKER_CACHE["fetched_at"] = time.monotonic()
    return _TICKER_CACHE["rows"]


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == " ").strip()


def get_company_candidates(query: str, limit: int = 8) -> list[dict]:
    """Return ranked CIK candidates for a user-typed company name or ticker.

    Never silently picks one match: a confident report about the wrong
    company is worse than no report. The route shows a disambiguation page
    when more than one candidate comes back.
    """
    q = _norm(query)
    if not q:
        return []
    rows = _ticker_table()
    exact_ticker = [r for r in rows if r["ticker"].lower() == query.strip().lower()]
    name_matches = [r for r in rows if q in _norm(r["title"])]

    seen: set[int] = set()
    out: list[dict] = []
    for r in exact_ticker + name_matches:
        cik = int(r["cik_str"])
        if cik in seen:
            continue
        seen.add(cik)
        out.append({"cik": cik, "ticker": r["ticker"], "name": r["title"]})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Filing index
# ---------------------------------------------------------------------------

def get_target_filings(cik: int) -> list[dict]:
    """Newest ownership-relevant filings for a company, capped per trace.

    EDGAR's submissions JSON stores recent filings as parallel arrays — one
    array of forms, one of accession numbers, one of dates — where the same
    index across arrays describes one filing. zip() reassembles them.

    Keeps every 13D/13G (each names a different owner — all signal) but only
    the latest 10-K and DEF 14A (an old subsidiary list is superseded, not
    additive).
    """
    data = _get(EDGAR_SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data.get("filings", {}).get("recent", {})

    out: list[dict] = []
    seen_single_forms: set[str] = set()
    for form, acc, doc, date in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("filingDate", []),
    ):
        if form not in TARGET_FORM_TYPES:
            continue
        if form in ("10-K", "DEF 14A"):
            if form in seen_single_forms:
                continue
            seen_single_forms.add(form)
        out.append({
            "form": form,
            "accession_number": acc,
            "primary_document": doc,
            "filing_date": date,
        })
        if len(out) >= MAX_FILINGS_PER_TRACE:
            break
    return out


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def get_filing_documents(cik: int, accession_number: str) -> list[str]:
    """List the file names inside one filing's directory.

    Needed because a 10-K's subsidiary list (Exhibit 21) is a separate file
    from the primary document. Archive paths use the unpadded CIK and the
    accession number without dashes.
    """
    acc = accession_number.replace("-", "")
    url = f"{EDGAR_ARCHIVES_BASE}/{cik}/{acc}/index.json"
    data = _get(url).json()
    items = data.get("directory", {}).get("item", [])
    return [i["name"] for i in items]


def find_exhibit_21(document_names: list[str]) -> str | None:
    """Best-effort Exhibit 21 finder. Naming isn't standardized; this
    heuristic will occasionally miss — a documented limitation, surfaced to
    the user as 'subsidiary list not found' rather than hidden."""
    for name in document_names:
        lowered = name.lower()
        if ("ex21" in lowered or "ex-21" in lowered or "exhibit21" in lowered) \
                and lowered.endswith((".htm", ".html", ".txt")):
            return name
    return None


def fetch_document_text(cik: int, accession_number: str, document_name: str) -> str:
    """Fetch one document and return plain text, truncated to MAX_DOC_CHARS."""
    acc = accession_number.replace("-", "")
    url = f"{EDGAR_ARCHIVES_BASE}/{cik}/{acc}/{document_name}"
    raw = _get(url).text
    if document_name.lower().endswith((".htm", ".html")):
        raw = _strip_html(raw)
    return raw[:MAX_DOC_CHARS]


# ---------------------------------------------------------------------------
# HTML → text (stdlib only; no BeautifulSoup dependency for one job)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _strip_html(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    return parser.get_text()
