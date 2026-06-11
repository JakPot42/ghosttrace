"""
claude_extractor.py — Claude Haiku integration.

Three jobs: extract entities/relationships from filing text, adjudicate
ambiguous entity matches, and write the final report narrative. Risk scores
never come from here — rules score, Claude explains.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import anthropic

from config import (
    ADJUDICATION_MAX_TOKENS,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    EXTRACTION_MAX_TOKENS,
    REPORT_MAX_TOKENS,
)


class ExtractorError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(raw: str, context: str) -> dict:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractorError(f"{context}: invalid JSON — {exc}") from exc


def _require(data: dict, fields: list[str], context: str) -> None:
    for f in fields:
        if f not in data:
            raise ExtractorError(f"{context}: missing field '{f}'")


def _call(prompt: str, max_tokens: int, context: str, required: list[str]) -> dict:
    try:
        msg = _get_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise ExtractorError(f"API error during {context}: {exc}") from exc
    data = _parse_json(msg.content[0].text, context)
    _require(data, required, context)
    return data


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

EXTRACTION_FIELDS = ["entities", "relationships", "extraction_confidence"]


def extract_entities(text: str, form: str, filing_date: str) -> dict:
    """Extract owners, subsidiaries, and affiliates from one filing document."""
    prompt = f"""You are a corporate ownership analyst reading an SEC filing.

FORM TYPE: {form}
FILING DATE: {filing_date}
DOCUMENT TEXT:
{text}

Extract every company, fund, trust, and person mentioned as an owner,
subsidiary, affiliate, or beneficial owner. Ignore lawyers, auditors, and
transfer agents unless they appear as registered agents for entities.

Respond ONLY with valid JSON (no markdown fences, no explanation):
{{
  "entities": [
    {{
      "name": "exact name as written in the filing",
      "entity_type": "company|person|trust|fund",
      "jurisdiction": "state or country of organization, or null if not stated",
      "role": "owner|subsidiary|beneficial_owner|affiliate|registered_agent",
      "ownership_pct": 12.5,
      "address": "address if stated, else null"
    }}
  ],
  "relationships": [
    {{
      "owner": "name of the owning entity",
      "owned": "name of the owned entity",
      "ownership_pct": 12.5,
      "evidence_quote": "short verbatim quote from the document supporting this"
    }}
  ],
  "extraction_confidence": "high|medium|low"
}}

Use null for ownership_pct when the document does not state a percentage.
Confidence is "low" if the document was hard to parse or ownership statements
were vague."""
    return _call(prompt, EXTRACTION_MAX_TOKENS, "extract_entities", EXTRACTION_FIELDS)


# ---------------------------------------------------------------------------
# Entity match adjudication
# ---------------------------------------------------------------------------

ADJUDICATION_FIELDS = ["same_entity", "rationale"]


def adjudicate_match(name_a: str, name_b: str) -> dict:
    """Decide whether two name variants refer to the same legal entity.

    Called only for the ambiguous similarity band — the resolver handles
    clear matches and clear non-matches deterministically.
    """
    prompt = f"""Two entity names were extracted from SEC filings. Decide if they refer to
the SAME legal entity or DIFFERENT entities.

NAME A: {name_a}
NAME B: {name_b}

Consider: corporate suffix variations (LP vs L.P.) suggest same entity;
different jurisdictions in the name, or words like "Holdings" vs "Partners"
distinguishing parent from operating company, suggest different entities.
When genuinely uncertain, answer false — a wrong merge is worse than a
missed merge.

Respond ONLY with valid JSON:
{{"same_entity": true, "rationale": "one sentence"}}"""
    return _call(prompt, ADJUDICATION_MAX_TOKENS, "adjudicate_match", ADJUDICATION_FIELDS)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

REPORT_FIELDS = ["headline", "summary", "key_findings", "full_text"]


def generate_risk_report(
    company_name: str,
    risk_score: int,
    risk_level: str,
    findings: list[dict],
    entities: list[dict],
    links: list[dict],
) -> dict:
    """Write the narrative report. Scores and findings come from risk_engine —
    Claude explains them, it does not invent new ones."""
    findings_text = "\n".join(
        f"- [{f['rule']}] {f['detail']} (weight {f['weight']})" for f in findings
    ) or "- none"
    entities_text = "\n".join(
        f"- {e['canonical_name']} ({e.get('entity_type', 'company')}, "
        f"{e.get('jurisdiction') or 'jurisdiction unknown'})"
        for e in entities
    )
    links_text = "\n".join(
        f"- {l['owner']} owns "
        f"{str(l['ownership_pct']) + '%' if l.get('ownership_pct') is not None else 'UNDISCLOSED %'}"
        f" of {l['owned']}"
        for l in links
    ) or "- none"

    prompt = f"""You are a financial intelligence analyst writing an ownership risk
assessment for {company_name}.

RULE-BASED RISK SCORE: {risk_score}/100 — {risk_level}

RISK FINDINGS (produced by deterministic rules — explain these, do NOT
invent additional findings or change the score):
{findings_text}

RESOLVED ENTITY NETWORK:
{entities_text}

OWNERSHIP LINKS:
{links_text}

Respond ONLY with valid JSON:
{{
  "headline": "One-sentence bottom line",
  "summary": "One paragraph plain-language summary of the ownership structure and why it scored {risk_level}",
  "key_findings": ["3-5 findings, most significant first, grounded in the rule findings above"],
  "full_text": "Full assessment in plain text with sections: OWNERSHIP STRUCTURE, RISK INDICATORS, ASSESSMENT, RECOMMENDED FOLLOW-UP. Formal analytic style. State clearly that the score is rule-based and which rules fired."
}}"""
    return _call(prompt, REPORT_MAX_TOKENS, "generate_risk_report", REPORT_FIELDS)
