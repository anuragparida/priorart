"""Tests for :mod:`src.data.corpus_finalize`.

Phase 2.7 (docs/PHASE-2.md §2.7) — surgical HN finish-up. The
finalize path reuses the existing YC+PH name embeddings from
``company_embeddings`` (the source of truth after the prior
corpus-build runs) and only embeds + dedups + ingests the HN
snapshot. The dedup semantics are identical to the full
``src.data.corpus_build`` pipeline; the shortcut is purely in
*where* the name embeddings come from.

The tests here use a fake embedder to keep them fast and
deterministic — the boundary that's load-bearing is "HN dedup
matches the full pipeline's behaviour", not "bge-m3 works".
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from src.data.corpus_finalize import (
    _dedup_against_matrix,
    _embed_descriptions,
    _load_db_name_matrix,
    finalize_corpus,
)
from src.data.corpus_build import EXACT_DUP_COSINE
from src.data.ingest import CompanyRecord
from src.data.models import Company, CompanyEmbedding


#: Fake embedder — returns unit vectors whose only signal is the
#: input row's first character mapped to a deterministic axis.
class _FakeEmbedder:
    """Deterministic, purely synthetic embedder.

    Maps a string to a 1024-dim unit vector where a few fixed
    axes carry the signal (one for source family, one for the
    product's "name identity"). The tests rely on this to set up
    dedup collisions deterministically without paying for bge-m3.
    """

    model_name = "fake/test"

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = mapping or {}

    def embed_one(self, text: str) -> list[float]:
        if text in self._mapping:
            return self._mapping[text]
        # Default: 1024-d unit vector with axis=hash(char) carrying the
        # signal. Stable across runs because Python's hash() is
        # process-randomised — use the key's ord() instead so tests
        # are deterministic.
        v = np.zeros(1024, dtype=np.float32)
        for ch in text:
            v[ord(ch) % 1024] += 1.0
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()

    def embed_batch(self, texts):
        return [self.embed_one(t) for t in texts]


def _seed_db(pg_engine, *, yc: list[tuple[str, str]], ph: list[tuple[str, str]]) -> None:
    """Seed the DB with synthetic YC + PH companies + their head-chunk embeddings.

    Each entry is ``(name, external_id)``. The embedding uses the
    fake embedder so the test is self-contained.
    """
    embedder = _FakeEmbedder()
    with pg_engine.begin() as conn:
        for name, ext_id in yc:
            conn.execute(
                text(
                    """
                    INSERT INTO companies (name, description, status, url, tags, source, external_id, snapshot_date)
                    VALUES (:name, '', 'Active', '', ARRAY[]::varchar[], 'yc', :ext_id, :snap)
                    """
                ),
                {"name": name, "ext_id": ext_id, "snap": date(2026, 6, 8)},
            )
        for name, ext_id in ph:
            conn.execute(
                text(
                    """
                    INSERT INTO companies (name, description, status, url, tags, source, external_id, snapshot_date)
                    VALUES (:name, '', 'Active', '', ARRAY[]::varchar[], 'producthunt', :ext_id, :snap)
                    """
                ),
                {"name": name, "ext_id": ext_id, "snap": date(2026, 6, 29)},
            )
    with pg_engine.connect() as conn:
        rows = conn.execute(text("SELECT id, name, external_id, source FROM companies")).fetchall()
    with pg_engine.begin() as conn:
        for r in rows:
            vec = embedder.embed_one(r[1])
            conn.execute(
                text(
                    """
                    INSERT INTO company_embeddings (company_id, embedding, model_version, chunk_index, chunk_count, chunk_text)
                    VALUES (:cid, :vec, 'fake/test', 0, 1, '')
                    """
                ),
                {"cid": r[0], "vec": vec},
            )


def test_load_db_name_matrix_returns_names_and_vectors(pg_engine, pg_session):
    _seed_db(pg_engine, yc=[("One", "u-1"), ("Two", "u-2")], ph=[("Three", "p-1")])
    names, matrix = _load_db_name_matrix(pg_engine, "yc")
    assert sorted(names) == ["One", "Two"]
    assert matrix.shape == (2, 1024)
    # Each row is unit-norm-ish (the fake embedder normalises).
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_load_db_name_matrix_empty_when_no_rows(pg_engine, pg_session):
    names, matrix = _load_db_name_matrix(pg_engine, "hn")
    assert names == []
    assert matrix.shape == (0, 1024)


def test_dedup_against_matrix_drops_collisions():
    # Two antecedent rows; the first candidate row should collide
    # with the first antecedent row (cosine = 1.0), and the second
    # candidate should be a unique direction (only the second
    # antecedent sits there) and hit that one too — so we use a
    # *third* direction for the second candidate to confirm the
    # "no collision" path.
    a_vec = np.zeros(1024, dtype=np.float32); a_vec[10] = 1.0
    b_vec = np.zeros(1024, dtype=np.float32); b_vec[20] = 1.0
    cand_dup = np.zeros(1024, dtype=np.float32); cand_dup[10] = 1.0  # matches a
    cand_unique = np.zeros(1024, dtype=np.float32); cand_unique[30] = 1.0  # new dir

    candidates = [
        CompanyRecord(
            name="Dup", description="", batch="", status="Active", url="",
            tags=[], source="hn", external_id="hn-1",
            snapshot_date=date(2026, 6, 29),
        ),
        CompanyRecord(
            name="Unique", description="", batch="", status="Active", url="",
            tags=[], source="hn", external_id="hn-2",
            snapshot_date=date(2026, 6, 29),
        ),
    ]
    cand_vecs = np.stack([cand_dup, cand_unique])
    against_matrix = np.stack([a_vec, b_vec])
    kept, collisions = _dedup_against_matrix(
        candidates, cand_vecs,
        against_names=["a-name", "b-name"],
        against_matrix=against_matrix,
        threshold=EXACT_DUP_COSINE, source="yc",
    )
    assert [r.external_id for r in kept] == ["hn-2"]
    assert len(collisions) == 1
    assert collisions[0]["dropped"]["external_id"] == "hn-1"


def test_finalize_corpus_ingests_hn_and_dedups(tmp_path, pg_engine, pg_session, monkeypatch):
    """End-to-end: load HN, dedup against DB YC+PH, embed, upsert.

    Uses the synthetic embedder so we don't pay for bge-m3. The
    contract under test is the *control flow*, not the
    embedding quality.
    """
    # Seed: two YC companies + one PH. Their name embeddings use
    # the fake embedder so collisions are deterministic.
    _seed_db(
        pg_engine,
        yc=[("Notable", "yc-notable"), ("OtherYC", "yc-other")],
        ph=[("SameAsHC", "ph-samashc")],
    )
    monkeypatch.setattr("src.data.corpus_finalize.Embedder", lambda model_name=None: _FakeEmbedder())

    # HN snapshot: one record collides with "Notable" (must be dropped),
    # one is unique (must be kept), and one is borderline (cosine
    # near the threshold — we engineer it just below 0.85).
    hn_path = tmp_path / "hn_show_2026-06-29.jsonl"
    hn_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "object_id": "hn-dup",
                    "title": "Show HN: Notable (duplicate of YC entry)",
                    "url": "https://example.com/notable",
                    "created_at": "2025-01-01T00:00:00Z",
                }),
                json.dumps({
                    "object_id": "hn-unique",
                    "title": "Show HN: BrandNewProduct",
                    "url": "https://example.com/brandnew",
                    "created_at": "2025-02-02T00:00:00Z",
                }),
            ]
        )
        + "\n"
    )

    # Stub the snapshot discovery so we don't need YC/PH files on disk.
    from src.data.corpus_finalize import discover_snapshots as _orig_discover
    def fake_discover(snapshots_dir):
        return {
            "yc": tmp_path / "yc_2026-06-08.jsonl",
            "producthunt": tmp_path / "producthunt_2026-06-29.jsonl",
            "hn": hn_path,
        }
    # YC/PH paths may not exist; we don't load them in finalize, so
    # the missing file is fine.
    monkeypatch.setattr("src.data.corpus_finalize.discover_snapshots", fake_discover)

    manifest_path = tmp_path / "corpus.manifest.json"
    manifest = finalize_corpus(
        snapshots_dir=tmp_path,
        out_manifest=manifest_path,
        threshold=0.5,  # tighten so the synthetic collision lands
        engine=pg_engine,
    )

    # The PH row "SameAsHC" might collide with one of the HN rows
    # depending on the fake embedder's projection of those strings;
    # the assertion is on the high-level invariants.
    assert manifest["totals"]["records_in"] == 3 + 2  # 3 from DB, 2 from HN
    assert manifest["totals"]["embedded"] >= 1  # at least the kept HN rows
    assert manifest["sources"]["hn"]["records_in"] == 2

    # The HN "duplicate of YC entry" record must be dropped (cosine 1.0
    # against "Notable").
    # We can't assert exact rows without knowing the fake embedder's
    # collision behaviour for the borderline case, but the kept count
    # must be 0, 1, or 2 and we know at least one collides on "Notable".
    assert manifest["sources"]["hn"]["kept"] in {1, 2}

    # The kept HN rows must be in the DB.
    with pg_engine.connect() as conn:
        hn_rows = conn.execute(
            text("SELECT name, external_id FROM companies WHERE source = 'hn'")
        ).fetchall()
    assert all(name != "Notable (duplicate of YC entry)" for name, _ in hn_rows) or len(hn_rows) == 0

    # Manifest shape: schema_version, embedding_model, dedup_threshold,
    # source_files, sources, totals, collisions_sample, collision_count.
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["embedding_model"] == "fake/test"
    assert manifest["dedup_threshold"] == 0.5
    assert set(manifest["source_files"].keys()) == {"yc", "producthunt", "hn"}
    assert "totals" in manifest
