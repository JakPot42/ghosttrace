"""ofac_checker.py — OFAC SDN fuzzy-match screening.

Downloads the SDN primary-name list (sdn.csv) and alias list (alt.csv) from
the OFAC website on first call within a process and caches them in memory.
On Render free tier the process is fresh on every cold start, so there is
no stale-cache problem; one download per session is acceptable.

All matches are CANDIDATES — fuzzy name matching cannot confirm a legal
identity. Every hit requires human verification before any compliance action.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from typing import NamedTuple

from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

from config import OFAC_MATCH_THRESHOLD, OFAC_SDN_ALT_URL, OFAC_SDN_CSV_URL
from entity_resolver import normalize_name

logger = logging.getLogger(__name__)

# Module-level cache — populated on first screen_entities() call.
# Each entry: (normalized_name, original_name, sdn_program, sdn_type)
_sdn_entries: list[tuple[str, str, str, str]] | None = None


class OFACHit(NamedTuple):
    entity_name: str   # trace entity that triggered the match
    sdn_name: str      # matching SDN list name
    score: int         # 0-100 fuzzy similarity score
    sdn_program: str   # sanctions program (e.g. SDGT, IRAN, RUSSIA)
    sdn_type: str      # individual | entity | vessel | aircraft | alias


def _fetch_csv_rows(url: str) -> list[list[str]]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GhostTrace portfolio research (jak.potvin@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8", errors="replace")
    return list(csv.reader(io.StringIO(content)))


def _load() -> list[tuple[str, str, str, str]]:
    entries: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    # Primary SDN names from sdn.csv
    # Columns: entry_num, name, type, program(s), title, ...
    try:
        for row in _fetch_csv_rows(OFAC_SDN_CSV_URL):
            if len(row) < 2:
                continue
            name = row[1].strip().strip('"')
            program = row[3].strip().strip('"') if len(row) > 3 else ""
            sdn_type = row[2].strip().strip('"').lower() if len(row) > 2 else "entity"
            if not name or name in ("-0-", "SDN Name", "Name"):
                continue
            norm = normalize_name(name)
            if norm and norm not in seen:
                seen.add(norm)
                entries.append((norm, name, program, sdn_type))
        logger.info("OFAC SDN primary: %d names loaded", len(entries))
    except Exception as exc:
        logger.warning("OFAC SDN primary list unavailable: %s", exc)

    # Alias names from alt.csv
    # Columns: entry_num, strength, aka_type, alternate_name, remarks
    before = len(entries)
    try:
        for row in _fetch_csv_rows(OFAC_SDN_ALT_URL):
            if len(row) < 4:
                continue
            name = row[3].strip().strip('"')
            if not name or name in ("-0-", "Alternate Name", "Alternate name"):
                continue
            norm = normalize_name(name)
            if norm and norm not in seen:
                seen.add(norm)
                entries.append((norm, name, "", "alias"))
        logger.info("OFAC SDN aliases: %d additional names loaded", len(entries) - before)
    except Exception as exc:
        logger.warning("OFAC SDN alias list unavailable: %s", exc)

    return entries


def _ensure_loaded() -> list[tuple[str, str, str, str]]:
    global _sdn_entries
    if _sdn_entries is None:
        _sdn_entries = _load()
    return _sdn_entries


def screen_entities(entity_names: list[str]) -> list[OFACHit]:
    """Fuzzy-match entity names against the OFAC SDN list.

    Returns OFACHit records for every entity/SDN pair whose token_sort_ratio
    meets OFAC_MATCH_THRESHOLD. Results are candidates only — manual
    verification required before any compliance action.
    """
    entries = _ensure_loaded()
    if not entries:
        return []

    # Parallel lists for rapidfuzz: normalized name for matching, index for metadata lookup
    sdn_norm_names = [e[0] for e in entries]

    hits: list[OFACHit] = []
    for entity_name in entity_names:
        norm_entity = normalize_name(entity_name)
        if not norm_entity:
            continue

        results = rfprocess.extract(
            norm_entity,
            sdn_norm_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=OFAC_MATCH_THRESHOLD,
            limit=3,
        )
        for _matched_norm, score, idx in results:
            _norm, original, program, sdn_type = entries[idx]
            hits.append(OFACHit(
                entity_name=entity_name,
                sdn_name=original,
                score=int(score),
                sdn_program=program,
                sdn_type=sdn_type,
            ))

    return hits
