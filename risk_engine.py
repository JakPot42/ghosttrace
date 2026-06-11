"""
risk_engine.py — deterministic risk scoring over a resolved ownership network.

No web, no database, no Claude. Every weight and threshold comes from
config.py. The LLM never touches scores: rules score, Claude explains.
"""

from __future__ import annotations

from collections import defaultdict

from config import (
    ADVERSARY_JURISDICTIONS,
    CHAIN_DEPTH_THRESHOLD,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_MEDIUM,
    RISK_WEIGHT_ADVERSARY_JURISDICTION,
    RISK_WEIGHT_CHAIN_DEPTH,
    RISK_WEIGHT_CIRCULAR_OWNERSHIP,
    RISK_WEIGHT_SECRECY_JURISDICTION,
    RISK_WEIGHT_SHARED_AGENT,
    RISK_WEIGHT_UNDISCLOSED_OWNER,
    SECRECY_JURISDICTIONS,
)


def jurisdiction_category(jurisdiction: str | None) -> str | None:
    """'secrecy', 'adversary', or None. Substring match so 'Grand Cayman,
    Cayman Islands' still hits."""
    if not jurisdiction:
        return None
    j = jurisdiction.lower()
    for adv in ADVERSARY_JURISDICTIONS:
        if adv.lower() in j:
            return "adversary"
    for sec in SECRECY_JURISDICTIONS:
        if sec.lower() in j:
            return "secrecy"
    return None


def _max_chain_depth(links: list[dict], focal_name: str) -> int:
    """Longest ownership chain ending at the focal company, counted in edges.
    Cycles are guarded by the path-stack check, not silently followed."""
    owners_of: dict[str, list[str]] = defaultdict(list)
    for link in links:
        owners_of[link["owned"]].append(link["owner"])

    def depth(node: str, path: frozenset[str]) -> int:
        if node in path:
            return 0
        best = 0
        for owner in owners_of.get(node, []):
            best = max(best, 1 + depth(owner, path | {node}))
        return best

    return depth(focal_name, frozenset())


def _find_cycle(links: list[dict]) -> list[str] | None:
    """Return one ownership cycle as a list of names, or None."""
    edges: dict[str, list[str]] = defaultdict(list)
    for link in links:
        edges[link["owner"]].append(link["owned"])

    visited: set[str] = set()

    def dfs(node: str, stack: list[str]) -> list[str] | None:
        if node in stack:
            return stack[stack.index(node):] + [node]
        if node in visited:
            return None
        visited.add(node)
        for nxt in edges.get(node, []):
            found = dfs(nxt, stack + [node])
            if found:
                return found
        return None

    for start in list(edges):
        found = dfs(start, [])
        if found:
            return found
    return None


def assess(entities: list[dict], links: list[dict], focal_name: str) -> dict:
    """Score the network. Returns {"score", "level", "findings"} where each
    finding is {"rule", "detail", "weight"}."""
    findings: list[dict] = []

    # Jurisdiction exposure — one finding per flagged entity
    for e in entities:
        cat = jurisdiction_category(e.get("jurisdiction"))
        if cat == "adversary":
            findings.append({
                "rule": "adversary_jurisdiction",
                "detail": f"{e['canonical_name']} — {e['jurisdiction']}",
                "weight": RISK_WEIGHT_ADVERSARY_JURISDICTION,
            })
        elif cat == "secrecy":
            findings.append({
                "rule": "secrecy_jurisdiction",
                "detail": f"{e['canonical_name']} — {e['jurisdiction']}",
                "weight": RISK_WEIGHT_SECRECY_JURISDICTION,
            })

    # Circular ownership
    cycle = _find_cycle(links)
    if cycle:
        findings.append({
            "rule": "circular_ownership",
            "detail": " → ".join(cycle),
            "weight": RISK_WEIGHT_CIRCULAR_OWNERSHIP,
        })

    # Deep ownership chain above the focal company
    chain = _max_chain_depth(links, focal_name)
    if chain >= CHAIN_DEPTH_THRESHOLD:
        findings.append({
            "rule": "chain_depth",
            "detail": f"{chain} layers of owners above {focal_name}",
            "weight": RISK_WEIGHT_CHAIN_DEPTH,
        })

    # Shared registered address across distinct entities
    by_address: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        addr = (e.get("address") or "").strip().lower()
        if addr:
            by_address[addr].append(e["canonical_name"])
    for addr, names in by_address.items():
        if len(names) >= 2:
            findings.append({
                "rule": "shared_registered_agent",
                "detail": f"{', '.join(names)} share one address",
                "weight": RISK_WEIGHT_SHARED_AGENT,
            })

    # Undisclosed ownership percentages
    for link in links:
        if link.get("ownership_pct") is None:
            findings.append({
                "rule": "undisclosed_ownership",
                "detail": f"{link['owner']} owns an undisclosed share of {link['owned']}",
                "weight": RISK_WEIGHT_UNDISCLOSED_OWNER,
            })

    score = min(100, sum(f["weight"] for f in findings))
    if score >= RISK_LEVEL_HIGH:
        level = "HIGH"
    elif score >= RISK_LEVEL_MEDIUM:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"score": score, "level": level, "findings": findings}
