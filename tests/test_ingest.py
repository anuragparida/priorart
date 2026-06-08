"""Integration tests for the ingest pipeline.

These hit a real Postgres+pgvector instance. The embedder is mocked
so the suite stays fast and doesn't pull a 1.5 GB model.
"""

from __future__ import annotations

from datetime import date
from typing import List

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.data.chunking import chunk_text
from src.data.embedder import Embedder
from src.data.ingest import (
    CompanyRecord,
    IngestStats,
    _upsert_company,
    _upsert_embedding,
    ingest,
)
from src.data.models import Company, CompanyEmbedding


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeEmbedder(Embedder):
    """Deterministic embedder that returns hash-derived 1024-dim vectors.

    Doesn't inherit the lazy model load — overrides ``embed_batch``
    directly so the test never touches sentence-transformers.
    """

    def __init__(self, dim: int = 1024) -> None:
        super().__init__(model_name="BAAI/bge-m3")
        self._dim = dim
        self.calls: List[List[str]] = []

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim

    def embed_batch(self, texts):  # type: ignore[override]
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * self._dim
            # Use a stable hash of the text to spread the vector —
            # the value doesn't have to be meaningful, just
            # deterministic.
            h = abs(hash(t)) % (self._dim - 1)
            v[h] = 1.0
            out.append(v)
        return out


def _make_records(n: int = 3) -> List[CompanyRecord]:
    return [
        CompanyRecord(
            name=f"Acme{i}",
            description=(
                "Bookkeeping, compliance and tax for founders. "
                "We file in 50 states. "
                "Our team includes ex-Stripe engineers."
            ),
            batch="W21",
            status="Active",
            url=f"https://example.com/{i}",
            tags=["SaaS", "Fintech"],
            source="yc:2026-06-08",
            snapshot_date=date(2026, 6, 8),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Ingest pipeline
# ---------------------------------------------------------------------------


def test_ingest_creates_companies_and_embeddings(pg_session: Session) -> None:
    records = _make_records(3)
    embedder = FakeEmbedder()

    stats = ingest(pg_session, records, embedder=embedder)
    pg_session.commit()

    assert stats.companies_total == 3
    assert stats.companies_inserted == 3
    assert stats.companies_updated == 0
    # Each description is multi-sentence but < 480 chars → 1 chunk
    # per company.
    assert stats.chunks_total == 3
    assert stats.embeddings_inserted == 3

    # Verify the rows
    n_companies = pg_session.execute(select(func.count()).select_from(Company)).scalar_one()
    n_embeddings = pg_session.execute(select(func.count()).select_from(CompanyEmbedding)).scalar_one()
    assert n_companies == 3
    assert n_embeddings == 3


def test_ingest_is_idempotent_on_re_run(pg_session: Session) -> None:
    records = _make_records(2)
    embedder = FakeEmbedder()

    first = ingest(pg_session, records, embedder=embedder)
    pg_session.commit()
    assert first.companies_inserted == 2

    # Re-run with a fresh embedder (same model_version → idempotent).
    embedder2 = FakeEmbedder()
    second = ingest(pg_session, records, embedder=embedder2)
    pg_session.commit()

    assert second.companies_total == 2
    assert second.companies_inserted == 0
    assert second.companies_updated == 2
    # Embeddings are upserted, so the count stays at 2.
    n_embeddings = pg_session.execute(
        select(func.count()).select_from(CompanyEmbedding)
    ).scalar_one()
    assert n_embeddings == 2


def test_ingest_with_multi_chunk_description(pg_session: Session) -> None:
    """A description that splits into >1 chunks produces >1 embedding rows."""
    long_desc = " ".join(
        ["This is a long sentence about " + ("alpha " * 30) + "."] * 5
    )
    records = [
        CompanyRecord(
            name="BigCo",
            description=long_desc,
            batch="S22",
            status="Active",
            url="",
            tags=[],
            source="yc:2026-06-08",
            snapshot_date=date(2026, 6, 8),
        )
    ]
    embedder = FakeEmbedder()
    stats = ingest(pg_session, records, embedder=embedder)
    pg_session.commit()

    assert stats.companies_total == 1
    # Use the same target_chars the ingest pipeline uses (default).
    n_chunks = len(chunk_text(long_desc))
    assert stats.chunks_total == n_chunks
    assert stats.chunks_total >= 2

    # Verify the (company_id, model_version, chunk_index) rows
    rows = pg_session.execute(
        select(CompanyEmbedding).where(CompanyEmbedding.chunk_count == n_chunks)
    ).scalars().all()
    assert len(rows) == n_chunks


def test_ingest_handles_empty_descriptions(pg_session: Session) -> None:
    records = [
        CompanyRecord(
            name="NoDesc",
            description="",
            batch="W21",
            status="Active",
            url="",
            tags=[],
            source="yc:2026-06-08",
            snapshot_date=date(2026, 6, 8),
        )
    ]
    embedder = FakeEmbedder()
    stats = ingest(pg_session, records, embedder=embedder)
    pg_session.commit()

    # The chunker emits one placeholder chunk for empty text, so
    # we still get one embedding row — never zero.
    assert stats.chunks_total == 1
    assert stats.embeddings_inserted == 1


def test_ingest_batches_embedder_calls(pg_session: Session) -> None:
    """Verify the embedder is called with chunks-of-32 batches, not 1-by-1."""
    records = _make_records(5)  # 5 single-chunk descriptions
    embedder = FakeEmbedder()
    ingest(pg_session, records, embedder=embedder)
    pg_session.commit()

    # 5 chunks total → embedder called once with batch of 5
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 5


def test_upsert_company_creates_then_updates(pg_session: Session) -> None:
    rec = _make_records(1)[0]
    cid, was_insert = _upsert_company(pg_session, rec)
    pg_session.commit()
    assert was_insert
    first_id = cid

    # Same natural key → no new row, fresh id returned
    cid2, was_insert2 = _upsert_company(pg_session, rec)
    pg_session.commit()
    assert cid2 == first_id
    assert not was_insert2


def test_upsert_embedding_does_not_duplicate(pg_session: Session) -> None:
    rec = _make_records(1)[0]
    cid, _ = _upsert_company(pg_session, rec)
    pg_session.commit()
    chunk = chunk_text(rec.description)[0]
    vec = [0.0] * 1024

    _upsert_embedding(
        pg_session,
        company_id=cid,
        model_version="BAAI/bge-m3",
        chunk=chunk,
        vector=vec,
    )
    pg_session.commit()
    # Upsert again — should not error and should not create a 2nd row
    _upsert_embedding(
        pg_session,
        company_id=cid,
        model_version="BAAI/bge-m3",
        chunk=chunk,
        vector=vec,
    )
    pg_session.commit()

    n = pg_session.execute(
        select(func.count()).select_from(CompanyEmbedding)
    ).scalar_one()
    assert n == 1
