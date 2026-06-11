"""Harborview Capital Partners — synthetic ownership network demo dataset.

Fictional scenario: a US fund whose ownership chain runs through Cayman and
BVI entities to an undisclosed beneficial owner, with a Chinese industrial
minority stake and a shared registered agent. All names, companies,
addresses, and accession numbers are completely fabricated.

The risk score is NOT hardcoded — the seed runs the same risk_engine rules
as a live trace, so the demo stays honest to the scoring logic.
"""

from __future__ import annotations

import os

from sqlalchemy.orm import Session

from config import GRAPH_OUTPUT_DIR
from models import Entity, OwnershipLink, Trace
from risk_engine import assess, jurisdiction_category

_SHARED_AGENT_ADDR = "Suite 400, 12 Quayside Lane, George Town, Cayman Islands"

SEED_ENTITIES = [
    {
        "canonical_name": "Harborview Capital Partners LP",
        "aliases": ["Harborview Capital Partners, L.P."],
        "entity_type": "fund",
        "jurisdiction": "Delaware, United States",
        "address": "1800 Commerce Plaza, Wilmington, DE",
        "is_focal": True,
        "sources": ["DEMO-13D-001", "DEMO-10K-001"],
    },
    {
        "canonical_name": "Meridian Holdings Ltd",
        "aliases": ["Meridian Holdings, Ltd.", "MERIDIAN HOLDINGS LIMITED"],
        "entity_type": "company",
        "jurisdiction": "Cayman Islands",
        "address": _SHARED_AGENT_ADDR,
        "is_focal": False,
        "sources": ["DEMO-13D-001"],
    },
    {
        "canonical_name": "Pelican Trust Services Ltd",
        "aliases": [],
        "entity_type": "trust",
        "jurisdiction": "British Virgin Islands",
        "address": "Wickham Quay II, Road Town, Tortola, BVI",
        "is_focal": False,
        "sources": ["DEMO-13D-001"],
    },
    {
        "canonical_name": "Calloway Nominee Services Ltd",
        "aliases": [],
        "entity_type": "company",
        "jurisdiction": "Cayman Islands",
        "address": _SHARED_AGENT_ADDR,
        "is_focal": False,
        "sources": ["DEMO-13D-002"],
    },
    {
        "canonical_name": "Shenzhen Brightway Industrial Co Ltd",
        "aliases": ["Shenzhen Brightway Industrial Company Limited"],
        "entity_type": "company",
        "jurisdiction": "Shenzhen, China",
        "address": "Tower B, Nanshan Technology Park, Shenzhen",
        "is_focal": False,
        "sources": ["DEMO-13G-001"],
    },
    {
        "canonical_name": "Atlas Pension Partners LLC",
        "aliases": [],
        "entity_type": "fund",
        "jurisdiction": "New York, United States",
        "address": "55 Hudson Yards, New York, NY",
        "is_focal": False,
        "sources": ["DEMO-13G-002"],
    },
]

SEED_LINKS = [
    {
        "owner": "Meridian Holdings Ltd",
        "owned": "Harborview Capital Partners LP",
        "ownership_pct": 64.0,
        "evidence_quote": "Meridian Holdings Ltd beneficially owns 64.0% of the outstanding limited partnership interests.",
        "source": "DEMO-13D-001",
    },
    {
        "owner": "Pelican Trust Services Ltd",
        "owned": "Meridian Holdings Ltd",
        "ownership_pct": 100.0,
        "evidence_quote": "Meridian is a wholly owned subsidiary of Pelican Trust Services Ltd.",
        "source": "DEMO-13D-001",
    },
    {
        "owner": "Calloway Nominee Services Ltd",
        "owned": "Pelican Trust Services Ltd",
        "ownership_pct": None,
        "evidence_quote": "Shares of Pelican are held of record by Calloway Nominee Services Ltd as nominee.",
        "source": "DEMO-13D-002",
    },
    {
        "owner": "Shenzhen Brightway Industrial Co Ltd",
        "owned": "Harborview Capital Partners LP",
        "ownership_pct": 11.0,
        "evidence_quote": "Reporting person beneficially owns 11.0% of outstanding interests.",
        "source": "DEMO-13G-001",
    },
    {
        "owner": "Atlas Pension Partners LLC",
        "owned": "Harborview Capital Partners LP",
        "ownership_pct": 9.0,
        "evidence_quote": "Reporting person beneficially owns 9.0% of outstanding interests.",
        "source": "DEMO-13G-002",
    },
]

SEED_HEADLINE = (
    "Harborview's controlling ownership runs through three layers of offshore "
    "entities to an undisclosed beneficial owner, with an additional Chinese "
    "industrial minority stake."
)

SEED_SUMMARY = (
    "Harborview Capital Partners LP is majority-owned (64%) by Meridian Holdings Ltd, "
    "a Cayman Islands company that is itself wholly owned by a British Virgin Islands "
    "trust, whose shares are held by a Cayman nominee service with no disclosed "
    "beneficial owner. Meridian and the nominee share a registered address, a common "
    "shell-structure pattern. Separately, Shenzhen Brightway Industrial Co Ltd holds "
    "an 11% stake. The structure scored HIGH on every category of rule: secrecy "
    "jurisdictions, adversary jurisdiction exposure, chain depth, shared agents, and "
    "undisclosed ownership."
)

SEED_KEY_FINDINGS = [
    "Controlling 64% stake traces through Cayman Islands and BVI entities to a nominee with no disclosed beneficial owner (DEMO-13D-001, DEMO-13D-002)",
    "Meridian Holdings Ltd and Calloway Nominee Services Ltd share a single registered address in George Town — a classic shell-structure indicator",
    "Ownership chain above Harborview is three layers deep, exceeding the structural-opacity threshold",
    "Shenzhen Brightway Industrial Co Ltd (China) holds an 11% stake (DEMO-13G-001) — adversary-jurisdiction exposure",
    "Atlas Pension Partners LLC's 9% stake presents no risk indicators and illustrates what a benign holder looks like",
]

SEED_FULL_TEXT = """OWNERSHIP RISK ASSESSMENT
REF: GT-2026-001 (DEMONSTRATION)
SUBJECT: Harborview Capital Partners LP — Hidden Ownership Analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OWNERSHIP STRUCTURE

Harborview Capital Partners LP (Delaware) is majority-controlled through a
three-layer offshore chain: Meridian Holdings Ltd (Cayman Islands) holds 64%
of Harborview; Meridian is wholly owned by Pelican Trust Services Ltd
(British Virgin Islands); Pelican's shares are held of record by Calloway
Nominee Services Ltd (Cayman Islands) as nominee, with no natural person
disclosed as beneficial owner at any level. Two minority holders sit outside
the chain: Shenzhen Brightway Industrial Co Ltd (11%) and Atlas Pension
Partners LLC (9%).

RISK INDICATORS

The rule-based engine fired on five categories:
1. Secrecy jurisdictions — Meridian (Cayman), Pelican (BVI), Calloway (Cayman).
2. Adversary jurisdiction — Shenzhen Brightway Industrial Co Ltd (China).
3. Deep ownership chain — three layers of owners above the focal company.
4. Shared registered address — Meridian and Calloway both register at
   Suite 400, 12 Quayside Lane, George Town.
5. Undisclosed ownership stake — Calloway's interest in Pelican carries no
   stated percentage and no named beneficial owner.

ASSESSMENT

The structure is consistent with deliberate beneficial-ownership obscuring.
No single indicator is conclusive — Cayman funds are common and often
legitimate — but the combination of nominee holding, shared registered
agents, and an unbroken offshore chain terminating without a natural person
is the canonical shell pattern. The Chinese minority stake is independently
notable for any holder with US defense-adjacent exposure.

RECOMMENDED FOLLOW-UP

1. Request beneficial ownership disclosure for Calloway Nominee Services Ltd.
2. Screen all six entities against the OFAC SDN list.
3. Examine Shenzhen Brightway's other US holdings for a pattern.
4. Pull historical 13D/A amendments to establish when the offshore chain was
   constructed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISCLAIMER: This assessment is generated from synthetic demonstration data.
All companies, individuals, addresses, and filing references are entirely
fictional, created solely for portfolio demonstration purposes.

GhostTrace — Hidden Ownership & Shell Company Tracer — DEMONSTRATION ONLY"""


def load_seed_data(db: Session) -> dict:
    if db.query(Trace).filter_by(company_name="Harborview Capital Partners LP").first():
        return {"status": "already_loaded"}

    # Score with the real engine so the demo reflects actual rule behavior
    result = assess(SEED_ENTITIES, SEED_LINKS, "Harborview Capital Partners LP")

    trace = Trace(
        company_name="Harborview Capital Partners LP",
        cik=None,
        is_demo=True,
        risk_score=result["score"],
        risk_level=result["level"],
        headline=SEED_HEADLINE,
        summary=SEED_SUMMARY,
        full_text=SEED_FULL_TEXT,
    )
    trace.findings = result["findings"]
    trace.key_findings = SEED_KEY_FINDINGS
    db.add(trace)
    db.flush()

    for e in SEED_ENTITIES:
        ent = Entity(
            trace_id=trace.id,
            canonical_name=e["canonical_name"],
            entity_type=e["entity_type"],
            jurisdiction=e["jurisdiction"],
            jurisdiction_category=jurisdiction_category(e["jurisdiction"]),
            address=e["address"],
            is_focal=e["is_focal"],
        )
        ent.aliases = e["aliases"]
        ent.sources = e["sources"]
        db.add(ent)

    for link in SEED_LINKS:
        db.add(OwnershipLink(
            trace_id=trace.id,
            owner_name=link["owner"],
            owned_name=link["owned"],
            ownership_pct=link["ownership_pct"],
            evidence_quote=link["evidence_quote"],
            source_accession=link["source"],
        ))

    # Graph PNG — pure pip dependencies (networkx + matplotlib), no graphviz,
    # so unlike FriendShore this works identically on Render. Guarded anyway:
    # a graph failure must never block seeding.
    try:
        from graph_builder import build_graph_png
        path = os.path.join(GRAPH_OUTPUT_DIR, f"trace_{trace.id}.png")
        build_graph_png(SEED_ENTITIES, SEED_LINKS, "Harborview Capital Partners LP", path)
        trace.graph_image_path = "/" + path.replace(os.sep, "/")
    except Exception:
        pass

    db.commit()
    return {"status": "loaded", "entities": len(SEED_ENTITIES), "links": len(SEED_LINKS)}
