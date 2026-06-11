"""Tests for edgar_client.py — all HTTP mocked, no real EDGAR calls."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from config import MAX_DOC_CHARS, MAX_FILINGS_PER_TRACE
from edgar_client import (
    EdgarError,
    _norm,
    _RateLimiter,
    _strip_html,
    fetch_document_text,
    find_exhibit_21,
    get_company_candidates,
    get_target_filings,
)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_enforces_minimum_spacing(self):
        rl = _RateLimiter(max_per_second=50)  # 20ms interval
        rl.wait()
        start = time.monotonic()
        rl.wait()
        assert time.monotonic() - start >= 0.018

    def test_no_sleep_when_interval_already_elapsed(self):
        rl = _RateLimiter(max_per_second=50)
        rl.wait()
        time.sleep(0.05)  # naturally waited longer than the interval
        start = time.monotonic()
        rl.wait()
        assert time.monotonic() - start < 0.01

    def test_safe_under_concurrent_callers(self):
        import threading
        rl = _RateLimiter(max_per_second=100)  # 10ms interval
        timestamps: list[float] = []
        lock = threading.Lock()

        def worker():
            rl.wait()
            with lock:
                timestamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        timestamps.sort()
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        assert all(g >= 0.008 for g in gaps), f"requests too close together: {gaps}"


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

class TestNorm:
    def test_strips_punctuation_and_case(self):
        assert _norm("Tesla, Inc.") == "tesla inc"

    def test_empty(self):
        assert _norm("") == ""
        assert _norm("...") == ""


# ---------------------------------------------------------------------------
# CIK candidate lookup
# ---------------------------------------------------------------------------

FAKE_ROWS = [
    {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    {"cik_str": 111, "ticker": "APLE", "title": "Apple Hospitality REIT, Inc."},
    {"cik_str": 222, "ticker": "TSLA", "title": "Tesla, Inc."},
    {"cik_str": 333, "ticker": "GOOG", "title": "Alphabet Inc."},
    {"cik_str": 333, "ticker": "GOOGL", "title": "Alphabet Inc."},
]


@patch("edgar_client._ticker_table", return_value=FAKE_ROWS)
class TestGetCompanyCandidates:
    def test_name_match_returns_all_matches(self, _mock):
        out = get_company_candidates("apple")
        names = [c["name"] for c in out]
        assert "Apple Inc." in names
        assert "Apple Hospitality REIT, Inc." in names

    def test_exact_ticker_ranks_first(self, _mock):
        out = get_company_candidates("AAPL")
        assert out[0]["cik"] == 320193

    def test_dedupes_same_cik(self, _mock):
        out = get_company_candidates("alphabet")
        assert len(out) == 1
        assert out[0]["cik"] == 333

    def test_no_match(self, _mock):
        assert get_company_candidates("zzznotacompany") == []

    def test_empty_query(self, _mock):
        assert get_company_candidates("") == []
        assert get_company_candidates("   ") == []


# ---------------------------------------------------------------------------
# Filing index — parallel array parsing
# ---------------------------------------------------------------------------

def _submissions_payload(rows: list[tuple[str, str, str, str]]) -> dict:
    """rows: (form, accession, primary_doc, date) — builds EDGAR's parallel arrays."""
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "accessionNumber": [r[1] for r in rows],
                "primaryDocument": [r[2] for r in rows],
                "filingDate": [r[3] for r in rows],
            }
        }
    }


class TestGetTargetFilings:
    def _mock_get(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = payload
        return MagicMock(return_value=resp)

    def test_filters_to_target_forms_and_keeps_latest_10k_only(self):
        payload = _submissions_payload([
            ("8-K", "acc-1", "a.htm", "2026-06-01"),
            ("SC 13D", "acc-2", "b.htm", "2026-05-20"),
            ("10-K", "acc-3", "c.htm", "2026-03-01"),
            ("10-K", "acc-4", "d.htm", "2025-03-01"),
            ("DEF 14A", "acc-5", "e.htm", "2026-04-01"),
            ("SC 13G", "acc-6", "f.htm", "2026-02-11"),
        ])
        with patch("edgar_client._get", self._mock_get(payload)):
            out = get_target_filings(320193)
        forms = [f["form"] for f in out]
        assert forms == ["SC 13D", "10-K", "DEF 14A", "SC 13G"]
        # the older 10-K (acc-4) was skipped
        assert all(f["accession_number"] != "acc-4" for f in out)

    def test_keeps_multiple_13ds(self):
        payload = _submissions_payload([
            ("SC 13D", f"acc-{i}", f"{i}.htm", "2026-01-01") for i in range(4)
        ])
        with patch("edgar_client._get", self._mock_get(payload)):
            out = get_target_filings(1)
        assert len(out) == 4

    def test_respects_cap(self):
        payload = _submissions_payload([
            ("SC 13D", f"acc-{i}", f"{i}.htm", "2026-01-01") for i in range(25)
        ])
        with patch("edgar_client._get", self._mock_get(payload)):
            out = get_target_filings(1)
        assert len(out) == MAX_FILINGS_PER_TRACE

    def test_empty_history(self):
        with patch("edgar_client._get", self._mock_get({"filings": {"recent": {}}})):
            assert get_target_filings(1) == []


# ---------------------------------------------------------------------------
# Exhibit 21 heuristic
# ---------------------------------------------------------------------------

class TestFindExhibit21:
    def test_common_names(self):
        assert find_exhibit_21(["aapl-10k.htm", "ex21.htm"]) == "ex21.htm"
        assert find_exhibit_21(["ex-21_1.htm"]) == "ex-21_1.htm"
        assert find_exhibit_21(["exhibit21list.txt"]) == "exhibit21list.txt"

    def test_miss_returns_none(self):
        assert find_exhibit_21(["report.htm", "ex32.htm", "graphic.jpg"]) is None
        assert find_exhibit_21([]) is None

    def test_ignores_non_text_files(self):
        assert find_exhibit_21(["ex21.jpg"]) is None


# ---------------------------------------------------------------------------
# HTML stripping + document fetch
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_extracts_text_skips_script_and_style(self):
        raw = (
            "<html><style>p{color:red}</style><body><p>Hello</p>"
            "<script>var a=1;</script><div>World</div></body></html>"
        )
        assert _strip_html(raw) == "Hello World"

    def test_plain_text_passthrough_content(self):
        assert _strip_html("just text") == "just text"


class TestFetchDocumentText:
    def _mock_get(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        return MagicMock(return_value=resp)

    def test_strips_html_documents(self):
        with patch("edgar_client._get", self._mock_get("<p>Ownership table</p>")):
            out = fetch_document_text(320193, "0001-23-456", "ex21.htm")
        assert out == "Ownership table"

    def test_txt_documents_not_stripped(self):
        with patch("edgar_client._get", self._mock_get("<raw> stays")):
            out = fetch_document_text(320193, "0001-23-456", "ex21.txt")
        assert out == "<raw> stays"

    def test_truncates_to_max_chars(self):
        with patch("edgar_client._get", self._mock_get("x" * (MAX_DOC_CHARS + 5000))):
            out = fetch_document_text(320193, "0001-23-456", "doc.txt")
        assert len(out) == MAX_DOC_CHARS

    def test_url_uses_dashless_accession(self):
        mock = self._mock_get("ok")
        with patch("edgar_client._get", mock):
            fetch_document_text(320193, "0001-23-456", "doc.txt")
        called_url = mock.call_args[0][0]
        assert "0001-23-456" not in called_url
        assert "000123456" in called_url
