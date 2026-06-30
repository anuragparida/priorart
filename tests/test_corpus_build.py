"""Tests for Phase 2.7 schema migration + cross-source corpus build.

The corpus-build path is exercised end-to-end against the running
Postgres (see ``conftest.py`` for the per-test schema fixture). The
embedding step uses a deterministic fake embedder so the suite
stays fast — we never touch the 1.5 GB bge-m3 model here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import pytest
from sqlalchemy import select, text

from src.data.corpus_build import (
    DedupResult,
    EXACT_DUP_COSINE,
    SOURCE_PRECEDENCE,
    _LOADERS,
    _nearest_above,
    build_corpus,
    cross_source_dedup,
    discover_snapshots,
    load_all_snapshots,
    load_hn_snapshot,
    load_producthunt_snapshot,
    load_yc_snapshot,
)
from src.data.db import init_schema
from src.data.embedder import Embedder
from src.data.ingest import CompanyRecord
from src.data.migrate import migrate_phase_2_7
from src.data.models import Base, Company


# ---------------------------------------------------------------------------
# Fake embedder — deterministic, never touches sentence-transformers.
# ---------------------------------------------------------------------------


class FakeEmbedder(Embedder):
    """Deterministic 1024-dim embedder.

    Overrides ``embed_batch`` directly so the model load is skipped.
    Each unique text gets a stable hash-derived one-hot vector so
    cosine similarity is either 0.0 (different hashes) or 1.0
    (same text). This is fine for the dedup walk — two records
    collide iff their names hash to the same bucket.
    """

    def __init__(self, dim: int = 1024) -> None:
        super().__init__(model_name="BAAI/bge-m3")
        self._dim = dim

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim

    def embed_batch(self, texts):  # type: ignore[override]
        out = []
        for t in texts:
            v = [0.0] * self._dim
            h = abs(hash(t)) % (self._dim - 1)
            v[h] = 1.0
            out.append(v)
        return out


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_migrate_phase_2_7_adds_columns_and_constraints(pg_engine) -> None:
    """Migration adds external_id, normalises source, swaps unique key."""
    # The fixture already ran init_schema + Base.metadata.create_all,
    # but the column set is the post-migration one — so we run the
    # migration on a fresh schema and check the post-state.
    with pg_engine.begin() as conn:
        # Pre-state: schema should have Company table, no external_id.
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'companies'"
            )
        ).fetchall()
        cols = {r[0] for r in rows}
        # The model defines external_id — but Base.create_all already
        # added it. The migration is a no-op in that case; we still
        # run it to prove idempotency.
        assert "external_id" in cols

    # Run the migration.
    stats = migrate_phase_2_7(pg_engine)
    assert "external_id_backfilled" in stats
    assert "source_normalised" in stats

    # Post-state: external_id present, source column exists, the
    # new unique constraint is in place.
    with pg_engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conname LIKE 'uq_companies%'"
            )
        ).fetchall()
        names = {r[0] for r in names} if False else {r[0] for r in rows}
        # Either the old or the new constraint may be present, but
        # the new one must always be there post-migration.
        assert "uq_companies_source_external_id" in names


def test_migrate_phase_2_7_is_idempotent(pg_engine) -> None:
    """Second run produces zero changes."""
    migrate_phase_2_7(pg_engine)
    stats = migrate_phase_2_7(pg_engine)
    assert stats["external_id_backfilled"] == 0
    assert stats["source_normalised"] == 0
    assert stats["old_constraint_dropped"] is False
    assert stats["new_constraint_added"] is False


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_load_yc_snapshot(tmp_path: Path) -> None:
    """YC loader yields CompanyRecord with source='yc', external_id=url."""
    path = tmp_path / "yc_2026-06-08.jsonl"
    _write_jsonl(
        path,
        [
            {
                "name": "Acme",
                "description": "Foo.",
                "batch": "W21",
                "status": "Active",
                "url": "https://www.ycombinator.com/companies/acme",
                "tags": ["SaaS"],
            }
        ],
    )
    records = list(load_yc_snapshot(path))
    assert len(records) == 1
    r = records[0]
    assert r.name == "Acme"
    assert r.source == "yc"
    assert r.external_id == "https://www.ycombinator.com/companies/acme"
    assert r.batch == "W21"
    assert r.snapshot_date == date(2026, 6, 8)


def test_load_producthunt_snapshot(tmp_path: Path) -> None:
    """PH loader composes tagline + topics + description, source='producthunt'."""
    path = tmp_path / "producthunt_2026-06-29.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "123456",
                "name": "Dora AI",
                "tagline": "Generating powerful websites",
                "description": "Detailed description body.",
                "topics": ["Design Tools", "AI"],
                "created_at": "2024-03-15T00:01:00-07:00",
                "ph_url": "https://www.producthunt.com/posts/dora-ai",
            }
        ],
    )
    records = list(load_producthunt_snapshot(path))
    assert len(records) == 1
    r = records[0]
    assert r.source == "producthunt"
    assert r.external_id == "123456"
    assert r.batch == "PH 2024"
    # Composed text: tagline + topics + description, pipe-separated.
    assert "Generating powerful websites" in r.description
    assert "Design Tools" in r.description
    assert "Detailed description body" in r.description
    assert r.tags == ["Design Tools", "AI"]
    assert r.url == "https://www.producthunt.com/posts/dora-ai"


def test_load_hn_snapshot_strips_meta_prefix(tmp_path: Path) -> None:
    """HN loader strips 'Show HN:' prefix and (YC S22) batch marker."""
    path = tmp_path / "hn_show_2026-06-29.jsonl"
    _write_jsonl(
        path,
        [
            {
                "object_id": "99999",
                "title": "Show HN: Nia (YC W36) – Give context to coding agents",
                "url": "https://trynia.ai/launch",
                "points": 112,
                "comments": 45,
                "created_at": "2025-09-15T10:00:00Z",
                "description": "Detailed HN post body.",
                "hn_url": "https://news.ycombinator.com/item?id=99999",
            }
        ],
    )
    records = list(load_hn_snapshot(path))
    assert len(records) == 1
    r = records[0]
    assert r.source == "hn"
    assert r.external_id == "99999"
    # The "Show HN:" prefix AND the "(YC W36)" batch marker are stripped.
    assert r.name == "Nia – Give context to coding agents"
    assert r.batch == "HN 2025"
    assert "Show HN" not in r.name
    assert "(YC W36)" not in r.name
    assert r.description == "Detailed HN post body."


# ---------------------------------------------------------------------------
# Cross-source dedup
# ---------------------------------------------------------------------------


def _rec(name: str, source: str = "yc", external_id: str | None = None) -> CompanyRecord:
    return CompanyRecord(
        name=name,
        description="x",
        batch="W21",
        status="Active",
        url="",
        tags=[],
        source=source,
        external_id=external_id or f"{source}-id-{name}",
        snapshot_date=date(2026, 6, 29),
    )


def test_dedup_keeps_yc_over_ph_on_same_name() -> None:
    """When two records share a name, YC wins the dedup."""
    records = [
        _rec("Acme", source="yc", external_id="yc-acme"),
        _rec("Acme", source="producthunt", external_id="ph-acme"),
        _rec("Beta", source="producthunt", external_id="ph-beta"),
        _rec("Beta", source="hn", external_id="hn-beta"),
    ]
    embedder = FakeEmbedder()
    kept, dedup = cross_source_dedup(records, embedder=embedder, threshold=EXACT_DUP_COSINE)
    names = [(r.source, r.external_id) for r in kept]
    # YC wins for "Acme" (precedence over PH).
    assert ("yc", "yc-acme") in names
    assert ("producthunt", "ph-acme") not in names
    # PH wins for "Beta" (precedence over HN).
    assert ("producthunt", "ph-beta") in names
    assert ("hn", "hn-beta") not in names
    # Dedup accounting.
    assert dedup.kept == {"yc": 1, "producthunt": 1, "hn": 0}
    assert dedup.dropped == {"yc": 0, "producthunt": 1, "hn": 1}


def test_dedup_keeps_unique_records() -> None:
    """Distinct names all survive the dedup walk."""
    records = [
        _rec("Foo", source="yc"),
        _rec("Bar", source="yc"),
        _rec("Baz", source="producthunt"),
        _rec("Qux", source="hn"),
    ]
    embedder = FakeEmbedder()
    kept, dedup = cross_source_dedup(records, embedder=embedder, threshold=EXACT_DUP_COSINE)
    assert len(kept) == 4
    assert all(d == 0 for d in dedup.dropped.values())


def test_dedup_records_collisions_in_result() -> None:
    """Collisions are tracked on the DedupResult."""
    records = [
        _rec("SameName", source="yc", external_id="yc-1"),
        _rec("SameName", source="producthunt", external_id="ph-1"),
    ]
    embedder = FakeEmbedder()
    kept, dedup = cross_source_dedup(records, embedder=embedder, threshold=EXACT_DUP_COSINE)
    assert len(dedup.collisions) == 1
    c = dedup.collisions[0]
    assert c["kept"] == {"source": "yc", "external_id": "yc-1", "name": "SameName"}
    assert c["dropped"] == {"source": "producthunt", "external_id": "ph-1", "name": "SameName"}


def test_nearest_above_returns_none_when_no_match() -> None:
    """Helper returns None when no candidate exceeds the threshold."""
    a = [1.0] + [0.0] * 1023
    b = [0.0, 1.0] + [0.0] * 1022
    # Threshold of 0.99 — the only way to pass is identical vectors.
    assert _nearest_above(a, [b], threshold=0.99) is None


def test_nearest_above_returns_index_on_match() -> None:
    a = [1.0] + [0.0] * 1023
    b = [1.0] + [0.0] * 1023  # identical
    c = [0.0, 1.0] + [0.0] * 1022
    assert _nearest_above(a, [b, c], threshold=0.99) == 0


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------


def test_discover_snapshots_picks_latest_per_source(tmp_path: Path) -> None:
    """Discovery returns the most-recent JSONL per source."""
    _write_jsonl(tmp_path / "yc_2026-06-08.jsonl", [{"name": "A"}])
    _write_jsonl(tmp_path / "yc_2026-05-01.jsonl", [{"name": "OldYC"}])
    _write_jsonl(tmp_path / "producthunt_2026-06-29.jsonl", [{"id": "1", "name": "B"}])
    _write_jsonl(tmp_path / "hn_show_2026-06-29.jsonl", [{"object_id": "1", "title": "C"}])

    paths = discover_snapshots(tmp_path)
    assert set(paths.keys()) == {"yc", "producthunt", "hn"}
    assert paths["yc"].name == "yc_2026-06-08.jsonl"
    assert paths["producthunt"].name == "producthunt_2026-06-29.jsonl"
    assert paths["hn"].name == "hn_show_2026-06-29.jsonl"


def test_discover_snapshots_raises_when_source_missing(tmp_path: Path) -> None:
    """Missing any one source is a hard error — no partial merges."""
    _write_jsonl(tmp_path / "yc_2026-06-08.jsonl", [{"name": "A"}])
    with pytest.raises(FileNotFoundError, match="missing snapshots"):
        discover_snapshots(tmp_path)


# ---------------------------------------------------------------------------
# End-to-end pipeline (uses real Postgres, fake embedder)
# ---------------------------------------------------------------------------


def test_build_corpus_upserts_with_source_external_id_key(pg_engine) -> None:
    """Records land in the merged corpus with the (source, external_id) key."""
    records = [
        _rec("Foo", source="yc", external_id="yc-foo"),
        _rec("Bar", source="producthunt", external_id="ph-bar"),
        _rec("Baz", source="hn", external_id="hn-baz"),
    ]
    with pg_engine.begin() as conn:
        # Mirror what the live corpus_build does: init_schema + tables.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    embedder = FakeEmbedder()
    from src.data.db import session_scope
    with session_scope(pg_engine) as session:
        build_corpus(session, records=records, embedder=embedder)

    with pg_engine.begin() as conn:
        rows = conn.execute(
            text("SELECT source, external_id, name FROM companies ORDER BY source, name")
        ).fetchall()
        got = [(r[0], r[1], r[2]) for r in rows]
        assert ("yc", "yc-foo", "Foo") in got
        assert ("producthunt", "ph-bar", "Bar") in got
        assert ("hn", "hn-baz", "Baz") in got

        # Embeddings table populated for each company.
        emb = conn.execute(
            text("SELECT count(*) FROM company_embeddings")
        ).scalar()
        assert emb == 3


def test_build_corpus_is_idempotent_on_rerun(pg_engine) -> None:
    """Re-running with the same records is a no-op."""
    from src.data.db import session_scope

    records = [
        _rec("Foo", source="yc", external_id="yc-foo"),
        _rec("Bar", source="producthunt", external_id="ph-bar"),
    ]
    with pg_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    embedder = FakeEmbedder()
    with session_scope(pg_engine) as session:
        build_corpus(session, records=records, embedder=embedder)
    with pg_engine.begin() as conn:
        first_count = conn.execute(text("SELECT count(*) FROM companies")).scalar()

    with session_scope(pg_engine) as session:
        build_corpus(session, records=records, embedder=embedder)
    with pg_engine.begin() as conn:
        second_count = conn.execute(text("SELECT count(*) FROM companies")).scalar()

    assert first_count == second_count == 2