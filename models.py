"""models.py — SQLAlchemy ORM for GhostTrace."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Filing(Base):
    """Fetched filing documents. Doubles as the EDGAR cache: the orchestrator
    checks here before hitting the network, and reports cite rows from here."""

    __tablename__ = "filings"
    __table_args__ = (UniqueConstraint("accession_number", "document_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cik: Mapped[int] = mapped_column(Integer, index=True)
    accession_number: Mapped[str] = mapped_column(String(30))
    document_name: Mapped[str] = mapped_column(String(200))
    form: Mapped[str] = mapped_column(String(20))
    filing_date: Mapped[str] = mapped_column(String(10))
    text: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    @property
    def edgar_url(self) -> str:
        acc = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{acc}/{self.document_name}"


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_name: Mapped[str] = mapped_column(String(300))
    cik: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # null for seed data
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    risk_score: Mapped[int] = mapped_column(Integer, default=0)
    risk_level: Mapped[str] = mapped_column(String(10), default="LOW")  # HIGH | MEDIUM | LOW
    _findings: Mapped[Optional[str]] = mapped_column("findings_json", Text, nullable=True)

    headline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    _key_findings: Mapped[Optional[str]] = mapped_column("key_findings_json", Text, nullable=True)
    full_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    graph_image_path: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)

    entities: Mapped[list["Entity"]] = relationship(
        "Entity", back_populates="trace", cascade="all, delete-orphan"
    )
    links: Mapped[list["OwnershipLink"]] = relationship(
        "OwnershipLink", back_populates="trace", cascade="all, delete-orphan"
    )

    @property
    def findings(self) -> list[dict]:
        return json.loads(self._findings) if self._findings else []

    @findings.setter
    def findings(self, val: list[dict]) -> None:
        self._findings = json.dumps(val)

    @property
    def key_findings(self) -> list[str]:
        return json.loads(self._key_findings) if self._key_findings else []

    @key_findings.setter
    def key_findings(self, val: list[str]) -> None:
        self._key_findings = json.dumps(val)

    @property
    def risk_badge_class(self) -> str:
        return {"HIGH": "badge-red", "MEDIUM": "badge-yellow", "LOW": "badge-green"}.get(
            self.risk_level, "badge-green"
        )


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[int] = mapped_column(Integer, ForeignKey("traces.id"))
    canonical_name: Mapped[str] = mapped_column(String(300))
    entity_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    jurisdiction: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    jurisdiction_category: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)
    is_focal: Mapped[bool] = mapped_column(Boolean, default=False)
    _aliases: Mapped[Optional[str]] = mapped_column("aliases_json", Text, nullable=True)
    _sources: Mapped[Optional[str]] = mapped_column("sources_json", Text, nullable=True)

    trace: Mapped[Trace] = relationship("Trace", back_populates="entities")

    @property
    def aliases(self) -> list[str]:
        return json.loads(self._aliases) if self._aliases else []

    @aliases.setter
    def aliases(self, val: list[str]) -> None:
        self._aliases = json.dumps(val)

    @property
    def sources(self) -> list[str]:
        return json.loads(self._sources) if self._sources else []

    @sources.setter
    def sources(self, val: list[str]) -> None:
        self._sources = json.dumps(val)

    @property
    def category_badge_class(self) -> str:
        return {"adversary": "badge-red", "secrecy": "badge-yellow"}.get(
            self.jurisdiction_category or "", "badge-gray"
        )


class OwnershipLink(Base):
    __tablename__ = "ownership_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[int] = mapped_column(Integer, ForeignKey("traces.id"))
    owner_name: Mapped[str] = mapped_column(String(300))
    owned_name: Mapped[str] = mapped_column(String(300))
    ownership_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evidence_quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_accession: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    trace: Mapped[Trace] = relationship("Trace", back_populates="links")
