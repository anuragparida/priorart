"""Test for the /healthz response schema.

The shape is part of the Phase 1.8 contract — see PHASE-1.md §1.8.
A failing import here means the response model drifted from the spec.
"""

from __future__ import annotations

from src.api.app import HealthStatus


def test_health_status_schema_fields_present() -> None:
    """HealthStatus must carry status, db, model, corpus_count."""
    fields = set(HealthStatus.model_fields.keys())
    assert fields == {"status", "db", "model", "corpus_count"}


def test_health_status_accepts_optional_corpus_count() -> None:
    """corpus_count is None until ingest (Phase 1.3) lands."""
    h = HealthStatus(status="ok", db="ok", model="BAAI/bge-m3", corpus_count=None)
    assert h.corpus_count is None
    assert h.model == "BAAI/bge-m3"
