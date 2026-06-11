"""
entity_resolver.py — entity deduplication.

No web, no database, no Claude imports. The adjudicator for the ambiguous
band is injected as a plain callable, so this module tests without mocking
anything but a function.

Three bands (thresholds in config):
  similarity >= FUZZY_AUTO_MERGE_THRESHOLD  → merge automatically
  similarity >= FUZZY_ADJUDICATE_THRESHOLD  → ask the adjudicator
  below                                     → distinct entities

Uses stdlib difflib rather than a fuzzy-matching dependency — adequate for
SEC filing names after normalization; swap for rapidfuzz if real data
proves otherwise.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable

from config import (
    FUZZY_ADJUDICATE_THRESHOLD,
    FUZZY_AUTO_MERGE_THRESHOLD,
    NORMALIZE_SUFFIXES,
)

# Adjudicator signature: (name_a, name_b) -> bool (same entity?)
Adjudicator = Callable[[str, str], bool]

_SUFFIX_SET = {s.replace(".", "") for s in NORMALIZE_SUFFIXES}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop leading 'the' and corporate suffixes."""
    cleaned = "".join(ch if ch.isalnum() or ch == " " else " " for ch in name.lower())
    tokens = cleaned.split()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    # Strip standard single-token suffixes
    while tokens and tokens[-1] in _SUFFIX_SET:
        tokens = tokens[:-1]
    # Also try joining the last 2-3 tokens to catch abbreviations like "l.p." → ["l","p"] → "lp"
    for n in (3, 2):
        if len(tokens) >= n and "".join(tokens[-n:]) in _SUFFIX_SET:
            tokens = tokens[:-n]
            break
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    """0-100 similarity between two normalized names."""
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    return SequenceMatcher(None, na, nb).ratio() * 100


def _merge_into(canonical: dict, raw: dict) -> None:
    """Fold a new sighting into an existing canonical entity. Later sightings
    fill gaps but never overwrite known values."""
    if raw["name"] not in canonical["aliases"] and raw["name"] != canonical["canonical_name"]:
        canonical["aliases"].append(raw["name"])
    for field in ("jurisdiction", "address", "entity_type"):
        if not canonical.get(field) and raw.get(field):
            canonical[field] = raw[field]
    role = raw.get("role")
    if role and role not in canonical["roles"]:
        canonical["roles"].append(role)
    for src in raw.get("sources") or ([raw["source"]] if raw.get("source") else []):
        if src and src not in canonical["sources"]:
            canonical["sources"].append(src)


def _new_canonical(raw: dict) -> dict:
    sources = list(raw.get("sources") or ([raw["source"]] if raw.get("source") else []))
    return {
        "canonical_name": raw["name"],
        "aliases": [],
        "entity_type": raw.get("entity_type"),
        "jurisdiction": raw.get("jurisdiction"),
        "address": raw.get("address"),
        "roles": [raw["role"]] if raw.get("role") else [],
        "sources": sources,
    }


def resolve_entities(
    raw_entities: list[dict],
    adjudicator: Adjudicator | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Collapse name variants into canonical entities.

    Returns (resolved_entities, alias_map) where alias_map maps every raw
    name to its canonical name — used to rewrite relationship endpoints.

    With no adjudicator, the ambiguous band stays unmerged: a missed merge
    is recoverable, a wrong merge poisons the graph.
    """
    resolved: list[dict] = []
    alias_map: dict[str, str] = {}

    for raw in raw_entities:
        name = (raw.get("name") or "").strip()
        if not name:
            continue
        raw = {**raw, "name": name}

        best: dict | None = None
        best_score = 0.0
        for canonical in resolved:
            score = similarity(name, canonical["canonical_name"])
            if score > best_score:
                best_score = score
                best = canonical

        merged = False
        if best is not None:
            if best_score >= FUZZY_AUTO_MERGE_THRESHOLD:
                merged = True
            elif best_score >= FUZZY_ADJUDICATE_THRESHOLD and adjudicator is not None:
                merged = adjudicator(name, best["canonical_name"])

        if merged and best is not None:
            _merge_into(best, raw)
            alias_map[name] = best["canonical_name"]
        else:
            entity = _new_canonical(raw)
            resolved.append(entity)
            alias_map[name] = entity["canonical_name"]

    return resolved, alias_map


def rewrite_links(raw_links: list[dict], alias_map: dict[str, str]) -> list[dict]:
    """Rewrite relationship endpoints to canonical names and dedupe.

    A link whose endpoints collapse to the same entity (self-ownership after
    merging) is dropped — it's a resolution artifact, not a finding.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for link in raw_links:
        owner = alias_map.get((link.get("owner") or "").strip(), link.get("owner"))
        owned = alias_map.get((link.get("owned") or "").strip(), link.get("owned"))
        if not owner or not owned or owner == owned:
            continue
        key = (owner, owned)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "owner": owner,
            "owned": owned,
            "ownership_pct": link.get("ownership_pct"),
            "evidence_quote": link.get("evidence_quote"),
            "source": link.get("source"),
        })
    return out
