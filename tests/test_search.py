"""Tests for the /search endpoint (Phase 1.4).

What this covers
----------------
- Request schema: validation of ``query`` and ``top_k``.
- Response schema: shape, ordering, and field ranges.
- Pipeline: the query is embedded, ANN-searched, deduplicated by
  company, and ordered by descending similarity.
- Edge cases: empty corpus, missing table, query with no match.
- Numerical correctness: the ``confidence`` field is ``(sim + 1) / 2``,
  and the raw cosine similarity is in ``[-1, 1]``.

We do NOT exercise the real bge-m3 model — the test suite uses a
``_ConstantEmbedder`` (one fixed vector per text) so the test runs
in <1 s and doesn't depend on a 1.5 GB model download. The live
acceptance check is done outside pytest against the running
``uvicorn src.api.app:app`` instance on port 18001.
"""

from __future__ import annotations

import math
from datetime import date
from typing import List

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import app as app_module
from src.api import search as search_module
from src.api.search import (
    SearchHit,
    SearchRequest,
    SearchResponse,
    _cosine_to_confidence,
    search_corpus,
)
from src.data.embedder import Embedder
from src.data.ingest import CompanyRecord, ingest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _ConstantEmbedder(Embedder):
    """Embedder that returns a constant unit vector for every input.

    Every text becomes the *same* vector — so any difference in the
    ranked result is due to the *underlying* stored vector in the
    table, not the query. This is what we want for "did the search
    work" assertions: we can pin the query vector to e.g. the X axis
    and assert the closest stored vector is the one we put on the X
    axis.
    """

    def __init__(self, vector: Sequence[float]):
        # Bypass Embedder.__init__'s default EMBEDDING_MODEL
        # pinning — the test's vector may have any dimension.
        self._model_name = "test-constant-embedder"
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            raise ValueError("zero vector has no defined cosine direction")
        self._vector = [x / norm for x in vector]

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._model_name

    @property
    def dim(self) -> int:  # type: ignore[override]
        return len(self._vector)

    def embed_batch(self, texts):  # type: ignore[override]
        return [list(self._vector) for _ in texts]


class _PerTextEmbedder(Embedder):
    """Embedder that returns a deterministic vector per *text*.

    Maps each text to an integer hash, then projects that hash onto
    the unit circle in 4-dim space (sin/cos). Two different texts
    will (almost always) get different vectors; identical texts get
    identical vectors. Dim 4 is enough to make the tests work and
    small enough that the SQL is readable in the failing test output.

    The dim is small (4) so we have to update the ``company_embeddings``
    table to use vector(4) for the test schema. We do that by patching
    the schema-creation DDL inside the fixture.
    """

    VECTORS_BY_TEXT: dict[str, list[float]] = {}

    def __init__(self) -> None:
        self._model_name = "test-pertext-embedder"
        self._dim = 4

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._model_name

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim

    def embed_batch(self, texts):  # type: ignore[override]
        out: List[List[float]] = []
        for t in texts:
            if t not in self.VECTORS_BY_TEXT:
                # Stable hash → 4-d unit vector
                h = abs(hash(t))
                a = (h % 1000) / 1000.0 * 2 * math.pi
                b = ((h // 1000) % 1000) / 1000.0 * 2 * math.pi
                v = [math.cos(a) * math.cos(b), math.cos(a) * math.sin(b),
                     math.sin(a) * math.cos(b), math.sin(a) * math.sin(b)]
                norm = math.sqrt(sum(x * x for x in v))
                self.VECTORS_BY_TEXT[t] = [x / norm for x in v]
            out.append(list(self.VECTORS_BY_TEXT[t]))
        return out


def _ingest_with_per_text_embedder(
    session: Session, records: List[CompanyRecord], embedder: Embedder
) -> None:
    """Drop the bge-m3 column type, recreate as vector(4), then ingest.

    The default schema uses vector(1024) for bge-m3 — the test's
    4-dim vectors would be rejected by the column type. We swap the
    column type for vector(4) for the test, ingest via raw SQL (the
    SQLAlchemy ``Vector(1024)`` type is hard-pinned to 1024 and would
    raise on a 4-dim value), then re-create the HNSW index on the
    new column.
    """
    # 1. Drop + recreate company_embeddings as vector(4).
    session.execute(text("DROP TABLE IF EXISTS company_embeddings CASCADE"))
    session.execute(
        text(
            "CREATE TABLE company_embeddings ("
            "  id SERIAL PRIMARY KEY,"
            "  company_id BIGINT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,"
            "  embedding vector(4) NOT NULL,"
            "  model_version VARCHAR(128) NOT NULL,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  chunk_index BIGINT NOT NULL DEFAULT 0,"
            "  chunk_count BIGINT NOT NULL DEFAULT 1,"
            "  chunk_text TEXT NOT NULL DEFAULT ''"
            ")"
        )
    )
    session.execute(
        text(
            "CREATE UNIQUE INDEX uq_company_embeddings_company_model_chunk "
            "ON company_embeddings (company_id, model_version, chunk_index)"
        )
    )
    session.execute(
        text(
            "CREATE INDEX ix_company_embeddings_embedding_hnsw "
            "ON company_embeddings "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
    )
    session.commit()

    # 2. Upsert companies (production path) — these don't touch the
    #    vector column, so the production ingest helper is fine.
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from src.data.models import Company

    for rec in records:
        session.execute(
            pg_insert(Company)
            .values(
                name=rec.name,
                description=rec.description,
                batch=rec.batch,
                status=rec.status,
                url=rec.url,
                tags=rec.tags,
                source=rec.source,
                external_id=rec.external_id,
                snapshot_date=rec.snapshot_date,
            )
            .on_conflict_do_update(
                # Phase 2.7 — dedup key is (source, external_id).
                index_elements=["source", "external_id"],
                set_={"description": rec.description},
            )
        )
    session.commit()

    # 3. Embed each company's description and insert the vectors
    #    via raw SQL — pgvector accepts a string literal, and we
    #    sidestep the SQLAlchemy Vector(1024) type.
    for rec in records:
        company_id = int(
            session.execute(
                text(
                    "SELECT id FROM companies "
                    "WHERE source = :s AND external_id = :eid"
                ),
                {"s": rec.source, "eid": rec.external_id},
            ).scalar_one()
        )
        vec = embedder.embed_one(rec.description)
        vec_str = "[" + ",".join(f"{x}" for x in vec) + "]"
        session.execute(
            text(
                "INSERT INTO company_embeddings "
                "(company_id, embedding, model_version, chunk_index, chunk_count, chunk_text) "
                "VALUES (:cid, CAST(:vec AS vector), :mv, 0, 1, :txt) "
                "ON CONFLICT (company_id, model_version, chunk_index) "
                "DO UPDATE SET embedding = EXCLUDED.embedding"
            ),
            {
                "cid": company_id,
                "vec": vec_str,
                "mv": embedder.model_name,
                "txt": rec.description,
            },
        )
    session.commit()


@pytest.fixture
def client_with_empty_engine(pg_engine):
    """A TestClient whose engine dep points at a per-test schema.

    No data ingested yet. The ``/search`` endpoint should return
    200 with an empty list.
    """
    def _override_engine():
        return pg_engine

    def _override_embedder():
        return _ConstantEmbedder(vector=[1.0, 0.0, 0.0, 0.0])

    app_module.app.dependency_overrides[app_module.get_engine] = _override_engine
    app_module.app.dependency_overrides[search_module.get_embedder] = _override_embedder
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.app.dependency_overrides.clear()


@pytest.fixture
def client_with_indexed_corpus(pg_engine):
    """A TestClient with a 3-company corpus indexed via a per-text embedder.

    Companies: "Alpha" (vector ~ X), "Beta" (vector ~ Y), "Gamma"
    (vector ~ Z). Querying with a vector close to X should put Alpha
    first.
    """
    # Reset _PerTextEmbedder state so the test is hermetic.
    _PerTextEmbedder.VECTORS_BY_TEXT = {}

    def _override_engine():
        return pg_engine

    def _override_embedder():
        return _PerTextEmbedder()

    app_module.app.dependency_overrides[app_module.get_engine] = _override_engine
    app_module.app.dependency_overrides[search_module.get_embedder] = _override_embedder
    try:
        # Build a tiny 3-company corpus using the per-text embedder.
        embedder = _PerTextEmbedder()
        records = [
            CompanyRecord(
                name="Alpha Co",
                description="Alpha builds AI for legal contract review.",
                batch="W21",
                status="Active",
                url="",
                tags=["AI", "LegalTech"],
                source="yc",
                external_id="alpha-test",
                snapshot_date=date(2026, 6, 8),
            ),
            CompanyRecord(
                name="Beta Co",
                description="Beta makes a CRM for small businesses.",
                batch="S22",
                status="Active",
                url="",
                tags=["SaaS", "CRM"],
                source="yc",
                external_id="beta-test",
                snapshot_date=date(2026, 6, 8),
            ),
            CompanyRecord(
                name="Gamma Co",
                description="Gamma is a marketplace for vintage typewriters.",
                batch="W23",
                status="Active",
                url="",
                tags=["Marketplace"],
                source="yc",
                external_id="gamma-test",
                snapshot_date=date(2026, 6, 8),
            ),
        ]
        with Session(bind=pg_engine) as session:
            _ingest_with_per_text_embedder(session, records, embedder)
        with TestClient(app_module.app) as client:
            yield {
                "client": client,
                "alpha_vec": _PerTextEmbedder.VECTORS_BY_TEXT[
                    "Alpha builds AI for legal contract review."
                ],
                "beta_vec": _PerTextEmbedder.VECTORS_BY_TEXT[
                    "Beta makes a CRM for small businesses."
                ],
                "gamma_vec": _PerTextEmbedder.VECTORS_BY_TEXT[
                    "Gamma is a marketplace for vintage typewriters."
                ],
            }
    finally:
        app_module.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_cosine_to_confidence_zero():
    """Cosine = -1 (anti-parallel) maps to confidence 0."""
    assert _cosine_to_confidence(-1.0) == 0.0


def test_cosine_to_confidence_one():
    """Cosine = +1 (identical) maps to confidence 1."""
    assert _cosine_to_confidence(1.0) == 1.0


def test_cosine_to_confidence_orthogonal():
    """Cosine = 0 (orthogonal) maps to confidence 0.5."""
    assert _cosine_to_confidence(0.0) == 0.5


def test_cosine_to_confidence_typical_bge_m3_high():
    """A 'very similar' cosine (0.85) maps to ~0.925 — a high confidence.

    Documents the intended behaviour for downstream thresholding: a
    threshold of 0.65 in the *confidence* space corresponds to
    cosine ≈ 0.30, which is on the low end for a meaningful match.
    """
    assert _cosine_to_confidence(0.85) == pytest.approx(0.925)


def test_search_request_validates_empty_query():
    """An empty query is a 422 — Pydantic raises on construction."""
    with pytest.raises(Exception):  # pydantic_core.ValidationError
        SearchRequest(query="", top_k=10)


def test_search_request_validates_top_k_bounds():
    """top_k must be in [1, 200]."""
    with pytest.raises(Exception):
        SearchRequest(query="ok", top_k=0)
    with pytest.raises(Exception):
        SearchRequest(query="ok", top_k=201)


def test_search_request_accepts_valid_payload():
    """Smoke test: a minimal valid request serialises without error."""
    r = SearchRequest(query="hello", top_k=5)
    assert r.query == "hello"
    assert r.top_k == 5


def test_search_hit_field_ranges():
    """SearchHit enforces similarity ∈ [-1, 1] and confidence ∈ [0, 1]."""
    with pytest.raises(Exception):
        SearchHit(id=1, name="X", description="y", similarity=1.5, confidence=0.5)
    with pytest.raises(Exception):
        SearchHit(id=1, name="X", description="y", similarity=0.5, confidence=-0.1)
    with pytest.raises(Exception):
        SearchHit(id=1, name="X", description="y", similarity=0.5, confidence=1.5)


# ---------------------------------------------------------------------------
# Endpoint tests — empty corpus
# ---------------------------------------------------------------------------


def test_search_returns_empty_list_when_corpus_is_empty(client_with_empty_engine):
    """/search on an unindexed schema returns 200 with an empty list."""
    resp = client_with_empty_engine.post(
        "/search", json={"query": "anything", "top_k": 10}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"] == []
    assert body["corpus_count"] == 0
    assert body["query"] == "anything"
    assert body["top_k"] == 10


def test_search_validates_oversized_query(client_with_empty_engine):
    """/search rejects queries > 4096 chars with a 422."""
    resp = client_with_empty_engine.post(
        "/search", json={"query": "x" * 4097, "top_k": 10}
    )
    assert resp.status_code == 422


def test_search_uses_request_top_k_default(client_with_empty_engine):
    """/search defaults to top_k=20 when the field is omitted."""
    resp = client_with_empty_engine.post("/search", json={"query": "x"})
    assert resp.status_code == 200
    assert resp.json()["top_k"] == 20


# ---------------------------------------------------------------------------
# Endpoint tests — indexed corpus, end-to-end
# ---------------------------------------------------------------------------


def test_search_returns_ranked_hits_with_similarity_and_confidence(
    client_with_indexed_corpus,
):
    """An indexed corpus returns the 3 companies, ranked by similarity."""
    c = client_with_indexed_corpus["client"]
    resp = c.post(
        "/search",
        json={"query": "Alpha builds AI for legal contract review.", "top_k": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_count"] == 3
    assert len(body["hits"]) == 3

    # Each hit has the documented shape.
    for hit in body["hits"]:
        assert set(hit.keys()) == {
            "id", "name", "description", "similarity", "confidence",
        }
        assert -1.0 <= hit["similarity"] <= 1.0
        assert 0.0 <= hit["confidence"] <= 1.0
        # confidence must be the (sim+1)/2 mapping — this is a
        # contract downstream consumers (1.6, 1.8) rely on.
        assert hit["confidence"] == pytest.approx(
            (hit["similarity"] + 1.0) / 2.0
        )

    # Hits are sorted by descending similarity.
    sims = [h["similarity"] for h in body["hits"]]
    assert sims == sorted(sims, reverse=True)


def test_search_finds_company_with_closest_vector_in_top_1(
    client_with_indexed_corpus,
):
    """Sanity: a query whose text matches a stored description returns it first.

    We send the verbatim description of Alpha Co as the query. Since
    the embedder is deterministic (same text → same vector) and
    Alpha Co's stored vector equals the query's, cosine = 1.0 for
    that row.
    """
    c = client_with_indexed_corpus["client"]
    resp = c.post(
        "/search",
        json={"query": "Alpha builds AI for legal contract review.", "top_k": 3},
    )
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits, "expected non-empty hits"
    # Top hit is Alpha Co with similarity ≈ 1.0
    assert hits[0]["name"] == "Alpha Co"
    assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert hits[0]["confidence"] == pytest.approx(1.0, abs=1e-6)


def test_search_respects_top_k(client_with_indexed_corpus):
    """top_k=2 returns 2 hits; top_k=1 returns 1."""
    c = client_with_indexed_corpus["client"]
    r1 = c.post("/search", json={"query": "Alpha builds AI for legal contract review.", "top_k": 1})
    r2 = c.post("/search", json={"query": "Alpha builds AI for legal contract review.", "top_k": 2})
    assert len(r1.json()["hits"]) == 1
    assert len(r2.json()["hits"]) == 2


def test_search_dedups_multi_chunk_companies(pg_engine):
    """A company with multiple chunks returns ONE hit — the best chunk's sim.

    We ingest one company with 3 chunks of *different* vectors, then
    search with a query vector that's close to chunk 1, far from
    chunk 2, and orthogonal to chunk 3. The dedup contract says
    exactly one row per company, and the returned similarity is
    chunk 1's (the best match).
    """
    # Reset _PerTextEmbedder state so the test is hermetic.
    _PerTextEmbedder.VECTORS_BY_TEXT = {}

    # Recreate company_embeddings as vector(4) — _PerTextEmbedder
    # produces 4-dim vectors, not the 1024-dim the bge-m3 schema has.
    with pg_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS company_embeddings CASCADE"))
        conn.execute(
            text(
                "CREATE TABLE company_embeddings ("
                "  id SERIAL PRIMARY KEY,"
                "  company_id BIGINT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,"
                "  embedding vector(4) NOT NULL,"
                "  model_version VARCHAR(128) NOT NULL,"
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  chunk_index BIGINT NOT NULL DEFAULT 0,"
                "  chunk_count BIGINT NOT NULL DEFAULT 1,"
                "  chunk_text TEXT NOT NULL DEFAULT ''"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX uq_company_embeddings_company_model_chunk "
                "ON company_embeddings (company_id, model_version, chunk_index)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX ix_company_embeddings_embedding_hnsw "
                "ON company_embeddings "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )

    # Insert one company via raw SQL (avoids the Vector(1024) type)
    # and three chunks with three different vectors.
    chunk_vecs = [
        [1.0, 0.0, 0.0, 0.0],   # chunk 0: aligned with X
        [0.0, 1.0, 0.0, 0.0],   # chunk 1: aligned with Y
        [0.0, 0.0, 0.0, 1.0],   # chunk 2: aligned with Z
    ]
    with pg_engine.begin() as conn:
        cid_row = conn.execute(
            text(
                "INSERT INTO companies "
                "(name, description, batch, status, url, tags, source, snapshot_date) "
                "VALUES (:n, :d, 'W21', 'Active', '', '{}', 'yc:test', '2026-06-08') "
                "RETURNING id"
            ),
            {"n": "Multi Co", "d": "A description that gets split into chunks."},
        )
        company_id = int(cid_row.scalar_one())
        for i, vec in enumerate(chunk_vecs):
            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
            conn.execute(
                text(
                    "INSERT INTO company_embeddings "
                    "(company_id, embedding, model_version, chunk_index, chunk_count, chunk_text) "
                    "VALUES (:cid, CAST(:vec AS vector), 'test-embedder', :i, 3, :txt) "
                    "ON CONFLICT (company_id, model_version, chunk_index) "
                    "DO UPDATE SET embedding = EXCLUDED.embedding"
                ),
                {"cid": company_id, "vec": vec_str, "i": i, "txt": f"chunk {i}"},
            )

    # Query with a vector aligned with X (chunk 0). The embedder
    # returns whatever _PerTextEmbedder produces for the query text —
    # but we control the stored vectors explicitly above. To make
    # the test deterministic, we use a ConstantEmbedder for the
    # query, set to the X axis. Then chunk 0's similarity = 1.0,
    # chunk 1's = 0.0, chunk 2's = 0.0 — so the "best" is chunk 0.
    class _XAxisEmbedder(_ConstantEmbedder):
        pass

    query_embedder = _XAxisEmbedder(vector=[1.0, 0.0, 0.0, 0.0])
    hits = search_corpus(
        pg_engine, query_embedder, query="anything", top_k=10
    )

    assert len(hits) == 1, f"expected 1 hit (dedup), got {len(hits)}: {hits}"
    assert hits[0].name == "Multi Co"
    # Best chunk (chunk 0, aligned with X) is the one returned.
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert hits[0].confidence == pytest.approx(1.0, abs=1e-6)
