"""Test for the /healthz response schema and corpus_count behaviour.

The shape is part of the Phase 1.8 contract — see PHASE-1.md §1.8.
A failing import here means the response model drifted from the spec.
"""

from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.app import HealthStatus
from src.api import app as app_module
from src.data.embedder import Embedder
from src.data.ingest import CompanyRecord, ingest


def test_health_status_schema_fields_present() -> None:
    """HealthStatus must carry status, db, model, corpus_count,
    sources, langfuse_enabled.

    Phase 2.3 added ``langfuse_enabled`` so operators can confirm
    tracing is wired without opening the Langfuse UI. Phase 2.7
    added ``sources`` so the merged corpus per-source breakdown is
    visible without opening psql. All fields are additive — old
    clients that only read the original four keep working.
    """
    fields = set(HealthStatus.model_fields.keys())
    assert fields == {
        "status",
        "db",
        "model",
        "corpus_count",
        "sources",
        "langfuse_enabled",
    }


def test_health_status_accepts_optional_corpus_count() -> None:
    """corpus_count is None when the table is missing/unreadable."""
    h = HealthStatus(status="ok", db="ok", model="BAAI/bge-m3", corpus_count=None)
    assert h.corpus_count is None
    assert h.model == "BAAI/bge-m3"
    assert h.sources == {}  # default empty dict


def test_health_status_accepts_int_corpus_count() -> None:
    """corpus_count is an int when ingest has run."""
    h = HealthStatus(status="ok", db="ok", model="BAAI/bge-m3", corpus_count=5949)
    assert h.corpus_count == 5949
    assert isinstance(h.corpus_count, int)


def test_health_status_accepts_per_source_breakdown() -> None:
    """Phase 2.7: ``sources`` carries the merged corpus breakdown."""
    h = HealthStatus(
        status="ok",
        db="ok",
        model="BAAI/bge-m3",
        corpus_count=11000,
        sources={"yc": 5949, "producthunt": 4000, "hn": 1051},
    )
    assert h.sources["yc"] == 5949
    assert h.sources["producthunt"] == 4000


def test_healthz_reports_real_corpus_count_after_ingest(pg_engine) -> None:
    """End-to-end: ingest a few rows, /healthz should report the count.

    We swap the FastAPI app's engine dependency for a per-test
    engine bound to the test schema. The test schema is created +
    dropped by the ``pg_engine`` fixture in conftest.py.
    """

    class _ZeroEmbedder(Embedder):
        def embed_batch(self, texts):  # type: ignore[override]
            return [[0.0] * 1024 for _ in texts]

    # 1. Ingest 3 rows into the test schema
    with Session(bind=pg_engine) as session:
        records = [
            CompanyRecord(
                name=f"Acme{i}",
                description="A description.",
                batch="W21",
                status="Active",
                url=f"https://example.com/{i}",
                tags=[],
                # Phase 2.7 — source is now just the prefix, and
                # external_id is the source's natural id (YC url).
                source="yc",
                external_id=f"https://example.com/{i}",
                snapshot_date=date(2026, 6, 8),
            )
            for i in range(3)
        ]
        ingest(session, records, embedder=_ZeroEmbedder())
        session.commit()

    # 2. Override the engine dependency and hit /healthz
    def _override_engine():
        return pg_engine

    app_module.app.dependency_overrides[app_module.get_engine] = _override_engine
    try:
        with TestClient(app_module.app) as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["db"] == "ok"
            assert body["model"] == "BAAI/bge-m3"
            assert body["corpus_count"] == 3
            # Phase 2.7 — the breakdown reports the per-source row count.
            assert body["sources"] == {"yc": 3}
    finally:
        app_module.app.dependency_overrides.clear()
