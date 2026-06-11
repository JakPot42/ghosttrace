"""main.py — FastAPI application for GhostTrace."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from claude_extractor import ExtractorError, extract_entities, generate_risk_report
from config import APP_TITLE, DEMO_BANNER, DEMO_MODE, MAX_FILINGS_PER_TRACE
from database import SessionLocal, get_db, init_db
from edgar_client import (
    fetch_document_text,
    find_exhibit_21,
    get_company_candidates,
    get_filing_documents,
    get_target_filings,
)
from entity_resolver import normalize_name, resolve_entities, rewrite_links
from graph_builder import build_graph_png
from models import Entity, OwnershipLink, Trace
from risk_engine import assess, jurisdiction_category
from seed_data import load_seed_data

import os


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    os.makedirs("static/graphs", exist_ok=True)
    db = SessionLocal()
    try:
        load_seed_data(db)
    finally:
        db.close()
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _template(request: Request, name: str, ctx: dict) -> HTMLResponse:
    ctx.update({"app_title": APP_TITLE, "demo_mode": DEMO_MODE, "demo_banner": DEMO_BANNER})
    return templates.TemplateResponse(request, name, ctx)


# ---------------------------------------------------------------------------
# Home — company search
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    recent = db.query(Trace).order_by(Trace.created_at.desc()).limit(10).all()
    return _template(request, "index.html", {"recent_traces": recent})


# ---------------------------------------------------------------------------
# Company search — returns candidate list or redirects directly
# ---------------------------------------------------------------------------

@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...), db: Session = Depends(get_db)):
    query = query.strip()
    if not query:
        return RedirectResponse("/", status_code=303)
    candidates = get_company_candidates(query)
    if not candidates:
        return _template(request, "index.html", {
            "recent_traces": db.query(Trace).order_by(Trace.created_at.desc()).limit(10).all(),
            "search_error": f"No SEC registrants found for '{query}'.",
            "query": query,
        })
    # Exact ticker match → skip disambiguation
    exact = next((c for c in candidates if c.get("ticker", "").upper() == query.upper()), None)
    if exact and len(candidates) == 1:
        return RedirectResponse(f"/trace?cik={exact['cik']}&name={exact['name']}", status_code=303)
    return _template(request, "index.html", {
        "recent_traces": db.query(Trace).order_by(Trace.created_at.desc()).limit(10).all(),
        "candidates": candidates,
        "query": query,
    })


# ---------------------------------------------------------------------------
# Live trace — fetch EDGAR filings, extract, score, store
# ---------------------------------------------------------------------------

@app.get("/trace", response_class=HTMLResponse)
def run_trace(
    request: Request,
    cik: int,
    name: str,
    db: Session = Depends(get_db),
):
    # Check for existing non-demo trace for this CIK
    existing = db.query(Trace).filter_by(cik=cik, is_demo=False).order_by(
        Trace.created_at.desc()
    ).first()
    if existing:
        return RedirectResponse(f"/trace/{existing.id}", status_code=303)

    filings = get_target_filings(cik)[:MAX_FILINGS_PER_TRACE]
    if not filings:
        return _template(request, "index.html", {
            "recent_traces": db.query(Trace).order_by(Trace.created_at.desc()).limit(10).all(),
            "search_error": f"No relevant SEC filings found for '{name}' (CIK {cik}).",
            "query": name,
        })

    # Fetch and extract from each filing
    raw_entities: list[dict] = []
    raw_links: list[dict] = []

    for filing in filings:
        acc = filing["accession_number"]
        form = filing["form"]
        date = filing["filing_date"]

        docs = get_filing_documents(cik, acc)
        # For 10-K, prefer Exhibit 21; for others, take the first substantive doc
        if form == "10-K":
            doc_name = find_exhibit_21([d["name"] for d in docs])
            if not doc_name and docs:
                doc_name = docs[0]["name"]
        else:
            doc_name = docs[0]["name"] if docs else None

        if not doc_name:
            continue

        text = fetch_document_text(cik, acc, doc_name)
        if not text:
            continue

        try:
            extracted = extract_entities(text, form, date)
        except ExtractorError:
            continue

        for e in extracted.get("entities", []):
            e["sources"] = [acc]
            raw_entities.append(e)
        for link in extracted.get("relationships", []):
            link["source"] = acc
            raw_links.append(link)

    if not raw_entities:
        return _template(request, "index.html", {
            "recent_traces": db.query(Trace).order_by(Trace.created_at.desc()).limit(10).all(),
            "search_error": f"Could not extract ownership data for '{name}'. Filings may be in an unsupported format.",
            "query": name,
        })

    # Resolve entity name variants, rewire links
    def _adjudicator(a: str, b: str) -> bool:
        from claude_extractor import adjudicate_match
        try:
            return adjudicate_match(a, b)["same_entity"]
        except ExtractorError:
            return False

    resolved, alias_map = resolve_entities(raw_entities, adjudicator=_adjudicator)
    links = rewrite_links(raw_links, alias_map)

    # Score
    result = assess(resolved, links, name)

    # Generate narrative report
    try:
        report = generate_risk_report(
            name, result["score"], result["level"], result["findings"],
            resolved, links,
        )
    except ExtractorError:
        report = {
            "headline": f"Risk assessment for {name} — {result['level']}",
            "summary": "Report narrative unavailable (API error).",
            "key_findings": [f['detail'] for f in result['findings']],
            "full_text": "",
        }

    # Persist
    trace = Trace(
        company_name=name,
        cik=cik,
        is_demo=False,
        risk_score=result["score"],
        risk_level=result["level"],
        headline=report["headline"],
        summary=report["summary"],
        full_text=report["full_text"],
    )
    trace.findings = result["findings"]
    trace.key_findings = report["key_findings"]
    db.add(trace)
    db.flush()

    for e in resolved:
        ent = Entity(
            trace_id=trace.id,
            canonical_name=e["canonical_name"],
            entity_type=e.get("entity_type"),
            jurisdiction=e.get("jurisdiction"),
            jurisdiction_category=jurisdiction_category(e.get("jurisdiction")),
            address=e.get("address"),
            is_focal=(normalize_name(e["canonical_name"]) == normalize_name(name)),
        )
        ent.aliases = e.get("aliases", [])
        ent.sources = e.get("sources", [])
        db.add(ent)

    for link in links:
        db.add(OwnershipLink(
            trace_id=trace.id,
            owner_name=link["owner"],
            owned_name=link["owned"],
            ownership_pct=link.get("ownership_pct"),
            evidence_quote=link.get("evidence_quote"),
            source_accession=link.get("source"),
        ))

    try:
        path = os.path.join("static/graphs", f"trace_{trace.id}.png")
        build_graph_png(resolved, links, name, path)
        trace.graph_image_path = "/" + path.replace(os.sep, "/")
    except Exception:
        pass

    db.commit()
    return RedirectResponse(f"/trace/{trace.id}", status_code=303)


# ---------------------------------------------------------------------------
# Trace detail page
# ---------------------------------------------------------------------------

@app.get("/trace/{trace_id}", response_class=HTMLResponse)
def trace_detail(request: Request, trace_id: int, db: Session = Depends(get_db)):
    trace = db.query(Trace).filter_by(id=trace_id).first()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return _template(request, "trace_detail.html", {"trace": trace})


# ---------------------------------------------------------------------------
# Seed / health routes
# ---------------------------------------------------------------------------

@app.get("/seed")
def seed(db: Session = Depends(get_db)):
    result = load_seed_data(db)
    trace = db.query(Trace).filter_by(company_name="Harborview Capital Partners LP").first()
    if trace:
        return RedirectResponse(f"/trace/{trace.id}", status_code=303)
    return JSONResponse(result)


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    trace_count = db.query(Trace).count()
    entity_count = db.query(Entity).count()
    return {"status": "ok", "traces": trace_count, "entities": entity_count,
            "demo_mode": DEMO_MODE}
