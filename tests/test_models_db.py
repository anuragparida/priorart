"""Integration tests for the SQLAlchemy 2.x models + HNSW index DDL.

These hit a real Postgres+pgvector instance (the docker-compose
service) inside a per-test schema so they don't pollute the live
ingest.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.data.models import Company, CompanyEmbedding, HNSW_INDEX_SQL


def test_company_table_created_with_expected_columns(pg_session: Session) -> None:
    inspector = inspect(pg_session.bind)
    cols = {c["name"]: c for c in inspector.get_columns("companies")}
    for name in ("id", "name", "description", "batch", "status", "url", "tags", "source", "snapshot_date"):
        assert name in cols, f"missing column companies.{name}"


def test_company_embedding_table_created_with_vector_column(pg_session: Session) -> None:
    inspector = inspect(pg_session.bind)
    cols = {c["name"]: c for c in inspector.get_columns("company_embeddings")}
    for name in ("id", "company_id", "embedding", "model_version", "created_at", "chunk_index", "chunk_count", "chunk_text"):
        assert name in cols, f"missing column company_embeddings.{name}"


def test_company_embedding_id_is_autoincrement(pg_session: Session) -> None:
    """The synthetic ``id`` PK is what enables multiple chunks per company."""
    company = Company(
        name="Acme",
        description="d",
        batch="W21",
        status="Active",
        url="",
        tags=[],
        source="yc:2026-06-08",
        snapshot_date=date(2026, 6, 8),
    )
    pg_session.add(company)
    pg_session.commit()
    pg_session.refresh(company)

    e1 = CompanyEmbedding(
        company_id=company.id, embedding=[0.0]*1024, model_version="BAAI/bge-m3",
        chunk_index=0, chunk_count=2, chunk_text="a",
    )
    e2 = CompanyEmbedding(
        company_id=company.id, embedding=[0.0]*1024, model_version="BAAI/bge-m3",
        chunk_index=1, chunk_count=2, chunk_text="b",
    )
    pg_session.add_all([e1, e2])
    pg_session.commit()
    pg_session.refresh(e1)
    pg_session.refresh(e2)
    assert e1.id is not None
    assert e2.id is not None
    assert e1.id != e2.id


def test_hnsw_index_present_on_embedding_column(pg_session: Session) -> None:
    inspector = inspect(pg_session.bind)
    indexes = inspector.get_indexes("company_embeddings")
    hnsw = [i for i in indexes if "hnsw" in (i.get("name") or "").lower()]
    # Some versions of SQLAlchemy report the index under different
    # keys — fall back to the raw PG index list if needed.
    if not hnsw:
        from sqlalchemy import text

        rows = pg_session.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'company_embeddings' "
                "AND indexname LIKE '%hnsw%'"
            )
        ).fetchall()
        assert rows, f"no HNSW index found; indexes={indexes}"


def test_company_unique_constraint_name_batch(pg_session: Session) -> None:
    company = Company(
        name="Acme",
        description="d",
        batch="W21",
        status="Active",
        url="https://example.com",
        tags=["AI"],
        source="yc:2026-06-08",
        snapshot_date=date(2026, 6, 8),
    )
    pg_session.add(company)
    pg_session.commit()
    pg_session.refresh(company)
    assert company.id is not None

    # Re-adding the same (name, batch) hits the unique constraint.
    dup = Company(
        name="Acme",
        description="different",
        batch="W21",
        status="Active",
        url="",
        tags=[],
        source="yc:2026-06-08",
        snapshot_date=date(2026, 6, 8),
    )
    pg_session.add(dup)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()


def test_embedding_roundtrip_preserves_vector(pg_session: Session) -> None:
    company = Company(
        name="Acme",
        description="d",
        batch="W21",
        status="Active",
        url="",
        tags=[],
        source="yc:2026-06-08",
        snapshot_date=date(2026, 6, 8),
    )
    pg_session.add(company)
    pg_session.commit()
    pg_session.refresh(company)

    vec = [0.0] * 1023 + [1.0]  # last dim = 1
    emb = CompanyEmbedding(
        company_id=company.id,
        embedding=vec,
        model_version="BAAI/bge-m3",
        chunk_index=0,
        chunk_count=1,
        chunk_text="d",
    )
    pg_session.add(emb)
    pg_session.commit()

    # Read back through a fresh query so we hit the DB, not the
    # identity map.
    pg_session.expire_all()
    row = pg_session.execute(
        select(CompanyEmbedding).where(CompanyEmbedding.company_id == company.id)
    ).scalar_one()
    assert len(row.embedding) == 1024
    assert row.embedding[-1] == pytest.approx(1.0, abs=1e-5)


def test_hnsw_ddl_string_is_idempotent(pg_engine) -> None:
    """Running the HNSW DDL twice should not raise.

    CREATE INDEX IF NOT EXISTS is the test we want; the schema
    fixture already ran it once, so we just re-execute.
    """
    from sqlalchemy import text

    with pg_engine.begin() as conn:
        # Idempotent run
        conn.execute(text(HNSW_INDEX_SQL))
        # And a third time, just to be paranoid
        conn.execute(text(HNSW_INDEX_SQL))
