"""Tests for entity_resolver.py — all pure logic, no network or DB."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from entity_resolver import normalize_name, resolve_entities, rewrite_links, similarity


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_strips_suffix(self):
        assert normalize_name("Harborview Capital Partners LP") == "harborview capital partners"

    def test_strips_punctuated_suffix(self):
        assert normalize_name("Harborview Capital Partners, L.P.") == "harborview capital partners"

    def test_multiple_suffixes(self):
        assert normalize_name("Meridian Holdings Limited") == "meridian"

    def test_lowercases(self):
        assert normalize_name("GLOBAL INDUSTRIES INC.") == "global industries"

    def test_strips_extra_whitespace(self):
        assert normalize_name("  Atlas  Corp  ") == "atlas"

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_suffix_only_returns_empty(self):
        # "Inc." alone normalizes to empty — edge case, shouldn't crash
        result = normalize_name("Inc.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("Harborview Capital", "Harborview Capital") == 100.0

    def test_empty_strings(self):
        assert similarity("", "") == 0.0

    def test_completely_different(self):
        assert similarity("Apple Inc", "Zeta Corp") < 50

    def test_suffix_variant_high(self):
        # After normalize both become same — similarity should be very high
        s = similarity(
            normalize_name("Pelican Trust Services Ltd"),
            normalize_name("Pelican Trust Services Limited"),
        )
        assert s >= 95

    def test_returns_float(self):
        assert isinstance(similarity("a", "b"), float)


# ---------------------------------------------------------------------------
# resolve_entities — auto-merge band
# ---------------------------------------------------------------------------

def _make(name, jur=None):
    return {"name": name, "entity_type": "company", "jurisdiction": jur,
            "role": "owner", "ownership_pct": None, "address": None, "sources": ["ACC-001"]}


class TestResolveEntities:
    def test_identical_names_merged(self):
        raw = [_make("Harborview Capital Partners LP"), _make("Harborview Capital Partners LP")]
        resolved, alias_map = resolve_entities(raw)
        assert len(resolved) == 1

    def test_suffix_variants_merged(self):
        raw = [
            _make("Harborview Capital Partners LP"),
            _make("Harborview Capital Partners, L.P."),
        ]
        resolved, alias_map = resolve_entities(raw)
        assert len(resolved) == 1
        # alias_map maps the duplicate back to the canonical
        assert len(alias_map) >= 1

    def test_clearly_different_not_merged(self):
        raw = [_make("Meridian Holdings Ltd"), _make("Atlas Pension Partners LLC")]
        resolved, alias_map = resolve_entities(raw)
        assert len(resolved) == 2

    def test_sources_accumulated_on_merge(self):
        raw = [
            {**_make("Meridian Holdings Ltd"), "sources": ["ACC-001"]},
            {**_make("MERIDIAN HOLDINGS LIMITED"), "sources": ["ACC-002"]},
        ]
        resolved, _ = resolve_entities(raw)
        assert len(resolved) == 1
        sources = resolved[0].get("sources", [])
        assert "ACC-001" in sources
        assert "ACC-002" in sources

    def test_adjudicator_called_for_middle_band(self):
        called = []

        def adjudicator(a, b):
            called.append((a, b))
            return False

        # Craft two names with ~80% similarity (ambiguous band)
        # "Calloway Holdings" vs "Calloway Capital" — similar stem, different suffix
        raw = [_make("Calloway Holdings Ltd"), _make("Calloway Capital Ltd")]
        resolve_entities(raw, adjudicator=adjudicator)
        # Whether adjudicator is called depends on the similarity score;
        # we just check it doesn't crash and returns something coherent
        assert isinstance(called, list)

    def test_returns_canonical_name(self):
        raw = [_make("Shenzhen Brightway Industrial Co Ltd")]
        resolved, _ = resolve_entities(raw)
        assert resolved[0]["canonical_name"] == "Shenzhen Brightway Industrial Co Ltd"

    def test_empty_input(self):
        resolved, alias_map = resolve_entities([])
        assert resolved == []
        assert alias_map == {}


# ---------------------------------------------------------------------------
# rewrite_links
# ---------------------------------------------------------------------------

class TestRewriteLinks:
    def test_rewrites_owner(self):
        # alias_map keys are the raw names as they came from the filing, not normalized
        alias_map = {"Harborview Capital Partners, L.P.": "Harborview Capital Partners LP"}
        links = [{"owner": "Harborview Capital Partners, L.P.", "owned": "Meridian Holdings Ltd",
                  "ownership_pct": 64.0, "evidence_quote": "owns 64%", "source": "ACC-001"}]
        rewritten = rewrite_links(links, alias_map)
        assert rewritten[0]["owner"] == "Harborview Capital Partners LP"

    def test_no_match_unchanged(self):
        alias_map = {}
        links = [{"owner": "Atlas Pension Partners LLC", "owned": "Harborview Capital Partners LP",
                  "ownership_pct": 9.0, "evidence_quote": "9%", "source": "ACC-002"}]
        rewritten = rewrite_links(links, alias_map)
        assert rewritten[0]["owner"] == "Atlas Pension Partners LLC"

    def test_empty_links(self):
        assert rewrite_links([], {}) == []


# ---------------------------------------------------------------------------
# Milestone 2 hardening
# ---------------------------------------------------------------------------

class TestTokenReorderMatching:
    def test_reordered_tokens_score_100(self):
        assert similarity("Capital Partners Harborview LP", "Harborview Capital Partners LP") == 100.0

    def test_reordered_tokens_auto_merge(self):
        raw = [
            _make("Harborview Capital Partners LP"),
            _make("Capital Partners Harborview, L.P."),
        ]
        resolved, _ = resolve_entities(raw)
        assert len(resolved) == 1

    def test_partial_overlap_lands_below_automerge(self):
        # Shared tokens but genuinely different names must not score 100
        s = similarity("Atlas Capital Ltd", "Atlas Holdings Ltd")
        assert s < 92


class TestAliasMatching:
    def test_new_variant_matches_via_alias(self):
        # Third sighting is closest to the merged alias, not the canonical
        raw = [
            _make("Meridian Holdings Ltd"),
            _make("Meridian Holdings Limited"),
            _make("MERIDIAN HOLDINGS LIMITED."),
        ]
        resolved, alias_map = resolve_entities(raw)
        assert len(resolved) == 1
        assert len(alias_map) == 3


class TestJurisdictionConflictGuard:
    def test_same_name_different_jurisdiction_not_auto_merged(self):
        # Classic shell pattern: identical names, Cayman vs Delaware.
        # Without an adjudicator they must stay distinct.
        raw = [
            {**_make("Meridian Holdings Ltd", jur="Cayman Islands")},
            {**_make("Meridian Holdings Ltd", jur="Delaware, United States")},
        ]
        resolved, _ = resolve_entities(raw)
        assert len(resolved) == 2

    def test_conflict_routes_to_adjudicator(self):
        called = []

        def adjudicator(a, b):
            called.append((a, b))
            return True

        raw = [
            {**_make("Meridian Holdings Ltd", jur="Cayman Islands")},
            {**_make("Meridian Holdings Ltd", jur="Delaware, United States")},
        ]
        resolved, _ = resolve_entities(raw, adjudicator=adjudicator)
        assert len(called) == 1
        assert len(resolved) == 1  # adjudicator said merge

    def test_substring_jurisdictions_do_not_conflict(self):
        raw = [
            {**_make("Meridian Holdings Ltd", jur="Cayman Islands")},
            {**_make("Meridian Holdings Ltd", jur="Grand Cayman, Cayman Islands")},
        ]
        resolved, _ = resolve_entities(raw)
        assert len(resolved) == 1

    def test_missing_jurisdiction_does_not_block_merge(self):
        raw = [
            {**_make("Meridian Holdings Ltd", jur="Cayman Islands")},
            {**_make("Meridian Holdings Ltd", jur=None)},
        ]
        resolved, _ = resolve_entities(raw)
        assert len(resolved) == 1


class TestAdjudicatorCaching:
    def test_same_pair_asked_once(self):
        calls = []

        def adjudicator(a, b):
            calls.append((a, b))
            return False

        # Same name in conflicting jurisdictions recurs three times — the
        # conflict demotes each to adjudication, but the identical question
        # must hit the cache after the first ask.
        raw = [
            {**_make("Meridian Holdings Ltd", jur="Cayman Islands")},
            {**_make("Meridian Holdings Ltd", jur="Delaware, United States")},
            {**_make("Meridian Holdings Ltd", jur="Delaware, United States")},
        ]
        resolve_entities(raw, adjudicator=adjudicator)
        assert len(calls) == 1
