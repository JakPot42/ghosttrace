"""Tests for ofac_checker — OFAC SDN fuzzy-match screening."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_sdn_cache():
    """Clear the module-level SDN cache between tests."""
    import ofac_checker
    ofac_checker._sdn_entries = None
    yield
    ofac_checker._sdn_entries = None


# Minimal SDN and alt CSV content matching the real file format
_SDN_CSV = (
    '1,"JOHN SMITH",individual,"SDGT",,,,,,,,\r\n'
    '2,"HARBORVIEW INVESTMENT GROUP",entity,"IRAN",,,,,,,,\r\n'
    '3,"ABC TRADING COMPANY LTD",entity,"RUSSIA",,,,,,,,\r\n'
    '4,"-0-",,,,,,,,,\r\n'           # sentinel row — must be skipped
)

_ALT_CSV = (
    '1,strong,"AKA ","JOHNNY SMITH",""\r\n'
    '2,strong,"AKA ","HARBORVIEW HOLDINGS LLC",""\r\n'
)


def _mock_urlopen_responses(*contents: str):
    """Return a side_effect list that yields one context-manager mock per content string."""
    responses = []
    for content in contents:
        m = MagicMock()
        m.read.return_value = content.encode()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        responses.append(m)
    return responses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScreenEntities:

    def test_empty_sdn_list_returns_no_hits(self):
        import ofac_checker
        ofac_checker._sdn_entries = []
        from ofac_checker import screen_entities
        assert screen_entities(["Acme Corp"]) == []

    def test_exact_name_flagged(self):
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            hits = screen_entities(["Harborview Investment Group"])
        assert len(hits) >= 1
        assert any(h.entity_name == "Harborview Investment Group" for h in hits)
        assert all(h.score >= 90 for h in hits)

    def test_alias_name_flagged(self):
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            hits = screen_entities(["Harborview Holdings LLC"])
        assert len(hits) >= 1
        assert any("HARBORVIEW" in h.sdn_name.upper() for h in hits)

    def test_unrelated_name_no_false_positive(self):
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            hits = screen_entities(["XYZ Aerospace Innovations Ltd"])
        assert hits == []

    def test_sentinel_row_skipped(self):
        """The -0- sentinel in sdn.csv must not produce any entries."""
        import ofac_checker
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            entries = ofac_checker._load()
        names = [e[1] for e in entries]
        assert "-0-" not in names

    def test_cache_reused_on_second_call(self):
        """_ensure_loaded() must not re-download on second call."""
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            screen_entities(["Entity One"])
            screen_entities(["Entity Two"])
        assert mock_open.call_count == 2  # sdn.csv + alt.csv downloaded exactly once

    def test_download_failure_returns_empty(self):
        """A network error must not raise — just return empty hits."""
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            hits = screen_entities(["Harborview Investment Group"])
        assert hits == []

    def test_hit_fields_populated(self):
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            hits = screen_entities(["ABC Trading Company"])
        assert len(hits) >= 1
        hit = hits[0]
        assert isinstance(hit.sdn_program, str)
        assert isinstance(hit.sdn_type, str)
        assert 0 <= hit.score <= 100

    def test_multiple_entities_screened(self):
        from ofac_checker import screen_entities
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = _mock_urlopen_responses(_SDN_CSV, _ALT_CSV)
            hits = screen_entities([
                "Harborview Investment Group",
                "XYZ Aerospace",
                "John Smith",
            ])
        entity_names = {h.entity_name for h in hits}
        assert "Harborview Investment Group" in entity_names
        assert "John Smith" in entity_names
        assert "XYZ Aerospace" not in entity_names
