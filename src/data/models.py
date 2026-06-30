"""SQLAlchemy 2.x ORM models for PriorArt.

Schema contract (docs/PHASE-1.md §1.3)
--------------------------------------
- ``Company`` is the metadata table that the YC JSONL snapshot lands in.
  Source of truth for: id, name, description, batch, status, url, tags,
  source, snapshot_date.
- ``CompanyEmbedding`` holds a single ``vector(1024)`` column per
  ``(company_id, model_version)`` row. Bge-m3 emits 1024-dim vectors;
  pinning the dim here means a model change is a schema change, not a
  silent breakage.
- Idempotency on ingest is enforced by the unique constraint on
  ``(company_id, model_version)`` — the ingest script uses an
  ``INSERT ... ON CONFLICT DO UPDATE`` upsert.
- HNSW index: ``m=16, ef_construction=64`` per the phase plan. Operator
  ``vector_cosine_ops`` so the ANN search (Phase 1.4) can use cosine
  similarity without re-normalising at query time.

Why HNSW with m=16 / ef_construction=64
----------------------------------------
From the pgvector README: ``m`` is the max number of connections per
layer (typical 16, 32); ``ef_construction`` is the candidate list size
during build (typical 64, 128). With ~6K YC rows we don't need 32/128
— 16/64 is a sensible balance between build time, index size, and
recall. Higher ef_search can be set at query time in 1.4.

Notes
-----
- ``Company.id`` is a ``BigInteger`` so we never run out of rows even
  after multiple re-scrapes of the YC directory. The id is assigned by
  a sequence (``companies_id_seq``) on insert.
- ``tags`` is a ``ARRAY(String)`` — pg-native. We don't filter on tags
  in Phase 1 but we want to preserve them for the UI in 1.9.
- ``status`` is a plain ``String`` (not enum) because the YC snapshot
  uses free-form strings ("Active", "Public", "Acquired", "Inactive",
  "Closed"…). Phase 2 may add a derived enum.
- ``source`` records where the row came from — useful when the corpus
  expands to Product Hunt + HN in Phase 2. For Phase 1 it's always
  ``"yc:<scrape_date>"``.
- ``snapshot_date`` is the date the JSONL was written, not the date
  this row landed in Postgres. Lets us reconstruct "what was in the
  corpus on date X" even after a re-ingest.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Project-wide declarative base. All models inherit from this."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Company(Base):
    """One company (YC / Product Hunt / HN) per row.

    Phase 2.7 (docs/PHASE-2.md §2.7) added three sources — ``yc``,
    ``producthunt``, ``hn`` — and switched the dedup key from
    ``(name, batch)`` to ``(source, external_id)`` so cross-source
    dedup works.

    Why ``(source, external_id)`` and not ``name``
    -----------------------------------------------
    PH and HN companies have no YC-style batch; some YC names collide
    with PH launches ("Dora AI (Alpha)" on PH is a different product
    than "Dora" on YC) so ``name`` is not a stable key. Each source
    has a natural primary key we can use:

    - ``yc``          → ``url`` (the canonical YC directory slug)
    - ``producthunt`` → ``id`` (PH's integer-as-string id)
    - ``hn``          → ``object_id`` (Algolia's per-post id)

    The idempotency contract is then ``INSERT ... ON CONFLICT (source,
    external_id) DO UPDATE`` — re-running the same JSONL is a no-op.
    """

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # ``batch`` is YC-only in practice; for PH/HN it's the launch
    # year/period (e.g. "PH 2024", "HN 2025") so the column stays
    # meaningful across all three sources. Nullable so future sources
    # can omit it.
    batch: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="Unknown")
    url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), nullable=False, default=list
    )

    # Provenance — Phase 2.7 changed the format from "yc:2026-06-08"
    # to just the source prefix ("yc" / "producthunt" / "hn"). The
    # scrape date lives in ``snapshot_date``. Existing rows were
    # backfilled by ``src.data.migrate``.
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)

    # The source's natural primary key (YC url, PH id, HN object_id).
    # Nullable so the schema can be created on an empty database
    # before any rows land; ``NOT NULL`` is enforced via the backfill
    # in ``src.data.migrate``.
    external_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    embedding: Mapped[Optional["CompanyEmbedding"]] = relationship(
        "CompanyEmbedding",
        back_populates="company",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Cross-source idempotency: re-running the same JSONL is a
        # no-op for the company table.
        UniqueConstraint("source", "external_id", name="uq_companies_source_external_id"),
        Index("ix_companies_batch", "batch"),
        Index("ix_companies_snapshot_date", "snapshot_date"),
        Index("ix_companies_source", "source"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Company id={self.id} source={self.source!r} "
            f"name={self.name!r} external_id={self.external_id!r}>"
        )


class CompanyEmbedding(Base):
    """One bge-m3 embedding per (company, model_version).

    The unique constraint is the idempotency key for the ingest
    pipeline: re-running with the same model version just updates the
    row. Bumping the model version creates a new row alongside the old
    one — useful when comparing model quality in Phase 2.
    """

    __tablename__ = "company_embeddings"

    # Synthetic PK so we can have multiple chunks per (company, model).
    # Phase 1 emits 1 chunk per YC company on average, but a few
    # long-tail descriptions split into 2–3 — the schema has to allow
    # that without giving up the (company, model_version, chunk_index)
    # idempotency key.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
    )

    embedding = mapped_column(Vector(1024), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
    )

    # Chunk metadata. Phase 1: usually one chunk per company; the
    # long tail (multi-paragraph descriptions) splits into 2–3.
    chunk_index: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    company: Mapped["Company"] = relationship("Company", back_populates="embedding")

    __table_args__ = (
        UniqueConstraint(
            "company_id", "model_version", "chunk_index",
            name="uq_company_embeddings_company_model_chunk",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<CompanyEmbedding company_id={self.company_id} "
            f"model={self.model_version!r} chunk={self.chunk_index}/{self.chunk_count}>"
        )


# ---------------------------------------------------------------------------
# Schema bootstrap helpers
# ---------------------------------------------------------------------------


#: HNSW index DDL — kept as a string so we can apply it idempotently
#: from the ingest script (``CREATE INDEX IF NOT EXISTS``) and from
#: tests (``drop + create``).
HNSW_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_company_embeddings_embedding_hnsw "
    "ON company_embeddings "
    "USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
)
