"""deep_trace.py — Bounded agentic loop for deeper ownership investigation.

Claude is given the initial trace summary and two tools: investigate_entity
searches EDGAR for a named company and extracts its ownership graph;
search_cached_filings does a keyword search over filing text already in the
SQLite cache. The loop runs until Claude is satisfied (end_turn) or the
DEEP_TRACE_MAX_TOOL_CALLS budget is spent.

The budget is the only hard cost control. Claude decides which entities to
pursue; the caller decides how many times it can go back to the well.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    DEEP_TRACE_MAX_TOKENS,
    DEEP_TRACE_MAX_TOOL_CALLS,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DeepTraceError(Exception):
    pass


_TOOLS = [
    {
        "name": "investigate_entity",
        "description": (
            "Search SEC EDGAR for a company by name and extract its ownership "
            "relationships from public filings. Use this to follow up on suspicious "
            "entities in the ownership network — shell companies, secrecy-jurisdiction "
            "funds, entities with undisclosed ownership stakes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "The company or fund name to look up in EDGAR.",
                },
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "search_cached_filings",
        "description": (
            "Search the cached SEC filing text for mentions of a company, person, "
            "address, or any keyword. Use this to find cross-references between "
            "entities already in the filing cache without making new EDGAR requests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — entity name, address, or keywords.",
                },
            },
            "required": ["query"],
        },
    },
]


def _execute_tool(name: str, inputs: dict, db: "Session") -> dict:
    if name == "investigate_entity":
        return _investigate_entity(inputs.get("entity_name", ""), db)
    if name == "search_cached_filings":
        return _search_cached_filings(inputs.get("query", ""), db)
    return {"error": f"Unknown tool: {name}"}


def _investigate_entity(entity_name: str, db: "Session") -> dict:
    from claude_extractor import ExtractorError, extract_entities
    from edgar_client import (
        EdgarError,
        fetch_document_text,
        find_exhibit_21,
        get_company_candidates,
        get_filing_documents,
        get_target_filings,
    )
    from models import Filing

    entity_name = entity_name.strip()
    if not entity_name:
        return {"error": "entity_name is required"}

    try:
        candidates = get_company_candidates(entity_name, limit=3)
    except EdgarError as exc:
        return {"status": "edgar_error", "detail": str(exc)}

    if not candidates:
        return {"status": "not_found", "entity_name": entity_name, "entities": [], "links": []}

    candidate = candidates[0]
    cik = candidate["cik"]

    try:
        filings = get_target_filings(cik)[:3]
    except EdgarError as exc:
        return {"status": "edgar_error", "cik": cik, "detail": str(exc)}

    entities_found: list[dict] = []
    links_found: list[dict] = []

    for filing in filings:
        acc = filing["accession_number"]
        form = filing["form"]
        date = filing["filing_date"]

        cached = db.query(Filing).filter_by(accession_number=acc).first()
        if cached:
            text = cached.text
        else:
            try:
                # get_filing_documents returns list[str] of filenames
                doc_names = get_filing_documents(cik, acc)
                if not doc_names:
                    continue
                doc_name = find_exhibit_21(doc_names) if form == "10-K" else None
                if not doc_name:
                    doc_name = doc_names[0]
                text = fetch_document_text(cik, acc, doc_name)
            except Exception:
                continue

        try:
            extracted = extract_entities(text, form, date)
            entities_found.extend(extracted.get("entities", []))
            links_found.extend(extracted.get("relationships", []))
        except ExtractorError:
            continue

    return {
        "status": "found",
        "entity_name": entity_name,
        "cik": cik,
        "matched_name": candidate["name"],
        "entities": entities_found[:8],
        "links": links_found[:15],
    }


def _search_cached_filings(query: str, db: "Session") -> dict:
    from vector_store import get_store

    if not query.strip():
        return {"error": "query is required"}
    try:
        results = get_store().search(query)
        return {"results": results[:5]}
    except Exception as exc:
        return {"error": str(exc)}


def _build_prompt(
    company_name: str,
    entities: list[dict],
    links: list[dict],
    risk_score: int,
    risk_level: str,
    findings: list[dict],
) -> str:
    entity_lines = "\n".join(
        f"- {e.get('canonical_name', e.get('name', '?'))} "
        f"({e.get('entity_type', 'company')}, "
        f"{e.get('jurisdiction') or 'jurisdiction unknown'})"
        for e in entities
    ) or "- none"

    link_lines = "\n".join(
        f"- {l.get('owner', l.get('owner_name', '?'))} → "
        f"{l.get('owned', l.get('owned_name', '?'))}"
        + (f" ({l['ownership_pct']}%)" if l.get("ownership_pct") else " (undisclosed %)")
        for l in links
    ) or "- none"

    finding_lines = "\n".join(
        f"- [{f['rule']}] {f['detail']}" for f in findings
    ) or "- none"

    return f"""You are a corporate intelligence analyst running a Deep Trace on {company_name}.

INITIAL TRACE RESULTS:
  Risk level: {risk_level} ({risk_score}/100)

RISK FINDINGS (rule-based):
{finding_lines}

ENTITIES IN OWNERSHIP NETWORK:
{entity_lines}

OWNERSHIP LINKS:
{link_lines}

YOUR TASK: Use the tools available to investigate suspicious entities further.
Prioritise: secrecy-jurisdiction vehicles (Cayman Islands, BVI, Panama, etc.),
entities with undisclosed ownership stakes, and names that suggest shell
structures. You have at most {DEEP_TRACE_MAX_TOOL_CALLS} tool calls — spend
them on the highest-value targets.

After your investigation, write a synthesis explaining what you found, what
remains uncertain, and whether the initial risk assessment should be escalated."""


def run_deep_trace(
    company_name: str,
    entities: list[dict],
    links: list[dict],
    risk_score: int,
    risk_level: str,
    findings: list[dict],
    db: "Session",
) -> dict:
    """Run the bounded Deep Trace agentic loop.

    Returns a dict with keys: tool_calls_used, max_calls, tool_call_log,
    synthesis. Never raises — errors are captured in the returned dict so
    a Deep Trace failure cannot take down the trace detail page.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = _build_prompt(company_name, entities, links, risk_score, risk_level, findings)

    messages: list[dict] = [{"role": "user", "content": prompt}]
    calls_used = 0
    tool_call_log: list[dict] = []
    synthesis = ""

    # MAX_TOOL_CALLS tool-use rounds + up to 2 extra turns (synthesis request,
    # final response) — generous ceiling so the loop never silently truncates.
    max_iterations = DEEP_TRACE_MAX_TOOL_CALLS + 3

    try:
        for _iter in range(max_iterations):
            # Remove tools from the call once the budget is spent — this forces
            # Claude to respond with text rather than another tool call.
            api_kwargs: dict = {}
            if calls_used < DEEP_TRACE_MAX_TOOL_CALLS:
                api_kwargs["tools"] = _TOOLS

            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=DEEP_TRACE_MAX_TOKENS,
                messages=messages,
                **api_kwargs,
            )
            messages.append({"role": "assistant", "content": response.content})

            # Capture any text block as the running synthesis
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    synthesis = block.text

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                break

            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_blocks:
                break

            tool_results = []
            for tb in tool_blocks:
                if calls_used < DEEP_TRACE_MAX_TOOL_CALLS:
                    result_data = _execute_tool(tb.name, tb.input, db)
                    calls_used += 1
                    tool_call_log.append({
                        "tool": tb.name,
                        "input": tb.input,
                        "result_preview": str(result_data)[:300],
                    })
                else:
                    # Budget already spent on a previous block in this same response;
                    # return a signal so Claude knows to stop requesting tools.
                    result_data = {
                        "status": "budget_exhausted",
                        "message": "Tool call limit reached. Provide your final synthesis now.",
                    }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": json.dumps(result_data),
                })

            messages.append({"role": "user", "content": tool_results})

            # If the budget is now exhausted, append an explicit synthesis prompt
            # so the next iteration (with no tools offered) generates the report.
            if calls_used >= DEEP_TRACE_MAX_TOOL_CALLS:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool call budget reached ({calls_used}/{DEEP_TRACE_MAX_TOOL_CALLS}). "
                        "Write your final Deep Trace synthesis now: what did you find, "
                        "what remains uncertain, and should the risk level be escalated?"
                    ),
                })

    except anthropic.APIError as exc:
        return {
            "tool_calls_used": calls_used,
            "max_calls": DEEP_TRACE_MAX_TOOL_CALLS,
            "tool_call_log": tool_call_log,
            "synthesis": synthesis or f"Deep Trace interrupted by API error: {exc}",
            "error": str(exc),
        }

    return {
        "tool_calls_used": calls_used,
        "max_calls": DEEP_TRACE_MAX_TOOL_CALLS,
        "tool_call_log": tool_call_log,
        "synthesis": synthesis,
    }
