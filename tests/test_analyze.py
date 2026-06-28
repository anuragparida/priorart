"""Tests for the /ideas/analyze endpoint (Phase 1.8).

What this covers
----------------
- The import contract: ``analyze_endpoint`` and ``AnalyzeRequest`` /
  ``AnalyzeError`` are importable from ``src.api.analyze``.
- The library layer: ``analyze_endpoint`` orchestrates embed →
  search → compare correctly (mocked LLM, real search via
  per-test schema). Top-K is passed through; cosine similarity
  is preserved.
- The HTTP layer (TestClient):
  - Happy path: 200 + a valid ``IdeaVerdict`` body.
  - Empty corpus: 200 + ``{"error": "no_competitors", ...}``.
  - Schema-violation: 200 + ``{"error": "schema_violation", ...}``.
  - Missing API key: 503 + structured body.
  - Validation error: 422 for empty / oversized ``idea``.
- Cost-control: exactly one LLM call per request.
- ``top_k`` is forwarded from the request to the LLM (and to the
  ANN search).
- The ``no_competitors`` 200 body has the ``corpus_count`` field.

What this does NOT cover
------------------------
- A live LLM call. That's the live smoke test against the running
  uvicorn on port 18001; here we mock the CompareClient so the
  test runs in <1 s without a real Anthropic call.

Test isolation
--------------
We use the same ``pg_engine`` fixture as ``test_search.py`` — a
per-test schema that's dropped on teardown. The tests do NOT touch
the live corpus.

Why we mock ``compare_topk`` rather than patching the instructor
client
------------------------------------------------------------------
``compare_topk`` is the smallest unit that owns the LLM contract.
Mocking one function gives us full control over the response shape
(we can return a known ``IdeaVerdict`` to assert on, or a
``SchemaViolationError`` to exercise the error path). Patching the
instructor client directly would couple the test to internal
instructor behavior — fragile when the SDK changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import analyze as analyze_module
from src.api import app as app_module
from src.api import search as search_module
from src.api.analyze import (
    AnalyzeError,
    AnalyzeRequest,
    analyze_endpoint,
)
from src.data.embedder import Embedder
from src.data.ingest import CompanyRecord
from src.llm.compare import (
    LLMTransportError,
    MissingAPIKeyError,
    SchemaViolationError,
)
from src.llm.schemas import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    IdeaVerdict,
    MarketScope,
)

# ---------------------------------------------------------------------------
# Embedder fixture (shared with test_search.py — copied here so this
# module is self-contained).
# ---------------------------------------------------------------------------


class _PerTextEmbedder(Embedder):
    """Per-text deterministic embedder, dim=4 (test-only).

    See the same class in ``test_search.py`` for the rationale.
    """

    VECTORS_BY_TEXT: dict[str, list[float]] = {}

    def __init__(self) -> None:
        self._model_name = "test-pertext-embedder"
        self._dim = 4

    @property
    def model_name(self) -> None:  # type: ignore[override]
        return self._model_name

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if t not in self.VECTORS_BY_TEXT:
                h = abs(hash(t))
                a = (h % 1000) / 1000.0 * 2 * math.pi
                b = ((h // 1000) % 1000) / 1000.0 * 2 * math.pi
                v = [
                    math.cos(a) * math.cos(b),
                    math.cos(a) * math.sin(b),
                    math.sin(a) * math.cos(b),
                    math.sin(a) * math.sin(b),
                ]
                norm = math.sqrt(sum(x * x for x in v))
                self.VECTORS_BY_TEXT[t] = [x / norm for x in v]
            out.append(list(self.VECTORS_BY_TEXT[t]))
        return out


def _ingest_with_per_text_embedder(
    session: Session, records: list[CompanyRecord], embedder: Embedder
) -> None:
    """Swap the company_embeddings column to vector(4) and ingest.

    The default schema uses vector(1024) for bge-m3 — the test's
    4-dim vectors would be rejected by the column type. We swap
    the column type for vector(4) for the test, ingest via raw SQL
    (the SQLAlchemy ``Vector(1024)`` type is hard-pinned to 1024),
    then re-create the HNSW index on the new column.
    """
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
                snapshot_date=rec.snapshot_date,
            )
            .on_conflict_do_update(
                index_elements=["name", "batch"],
                set_={"description": rec.description},
            )
        )
    session.commit()

    for rec in records:
        company_id = int(
            session.execute(
                text("SELECT id FROM companies WHERE name = :n AND batch = :b"),
                {"n": rec.name, "b": rec.batch},
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


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_indexed_corpus(pg_engine):
    """A TestClient with a 3-company corpus indexed via _PerTextEmbedder."""
    _PerTextEmbedder.VECTORS_BY_TEXT = {}

    def _override_engine():
        return pg_engine

    def _override_embedder():
        return _PerTextEmbedder()

    app_module.app.dependency_overrides[app_module.get_engine] = _override_engine
    app_module.app.dependency_overrides[search_module.get_embedder] = _override_embedder
    try:
        embedder = _PerTextEmbedder()
        records = [
            CompanyRecord(
                name="Alpha Co",
                description="Alpha builds AI for legal contract review.",
                batch="W21",
                status="Active",
                url="",
                tags=["AI", "LegalTech"],
                source="yc:test",
                snapshot_date=date(2026, 6, 8),
            ),
            CompanyRecord(
                name="Beta Co",
                description="Beta makes a CRM for small businesses.",
                batch="S22",
                status="Active",
                url="",
                tags=["SaaS", "CRM"],
                source="yc:test",
                snapshot_date=date(2026, 6, 8),
            ),
            CompanyRecord(
                name="Gamma Co",
                description="Gamma is a marketplace for vintage typewriters.",
                batch="W23",
                status="Active",
                url="",
                tags=["Marketplace"],
                source="yc:test",
                snapshot_date=date(2026, 6, 8),
            ),
        ]
        with Session(bind=pg_engine) as session:
            _ingest_with_per_text_embedder(session, records, embedder)
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.app.dependency_overrides.clear()


@pytest.fixture
def client_with_empty_corpus(pg_engine):
    """A TestClient whose corpus is empty (no rows in company_embeddings)."""
    _PerTextEmbedder.VECTORS_BY_TEXT = {}

    def _override_engine():
        return pg_engine

    def _override_embedder():
        return _PerTextEmbedder()

    app_module.app.dependency_overrides[app_module.get_engine] = _override_engine
    app_module.app.dependency_overrides[search_module.get_embedder] = _override_embedder
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_analyze_request_field_set() -> None:
    """``AnalyzeRequest`` exposes exactly the documented fields."""
    fields = set(AnalyzeRequest.model_fields.keys())
    assert fields == {"idea", "top_k"}
    assert AnalyzeRequest.model_fields["idea"].is_required()
    assert AnalyzeRequest.model_fields["top_k"].default == DEFAULT_TOP_K
    # top_k must be bounded by [1, MAX_TOP_K] — assert via JSON schema.
    schema = AnalyzeRequest.model_json_schema()
    top_k = schema["properties"]["top_k"]
    assert top_k["minimum"] == 1
    assert top_k["maximum"] == MAX_TOP_K
    assert top_k["default"] == DEFAULT_TOP_K
    # idea must be length-bounded [1, 4096].
    idea = schema["properties"]["idea"]
    assert idea["minLength"] == 1
    assert idea["maxLength"] == 4096


def test_analyze_error_field_set() -> None:
    """``AnalyzeError`` exposes ``error`` and ``details``."""
    e = AnalyzeError(error="schema_violation", details={"x": 1})
    assert e.error == "schema_violation"
    assert e.details == {"x": 1}
    e2 = AnalyzeError(error="no_competitors")
    assert e2.details is None


# ---------------------------------------------------------------------------
# Library-layer tests (no FastAPI)
# ---------------------------------------------------------------------------


def test_analyze_endpoint_returns_verdict_on_happy_path(client_with_indexed_corpus, pg_engine):
    """``analyze_endpoint`` calls ``compare_topk`` once with the right top-K
    and returns the verdict."""
    embedder = _PerTextEmbedder()
    # Pre-warm VECTORS_BY_TEXT so the query embedding is deterministic.
    _ = embedder.embed_one("AI for SMB legal contract review")

    fake_verdict = IdeaVerdict(
        idea="AI for SMB legal contract review",
        top_competitors=[
            # Filled in below; compare_topk builds them from top_k dicts,
            # not from the verdict, so we just need a valid shape here.
        ],
        market_scope=MarketScope.CROWDED_BUT_GROWING,
        market_scope_rationale="3 similar YC launches in legaltech, none dominant",
        supporting_evidence=[],
    )

    with patch.object(
        analyze_module, "compare_topk", return_value=fake_verdict
    ) as mock_compare:
        result = analyze_endpoint(
            AnalyzeRequest(idea="AI for SMB legal contract review", top_k=3),
            engine=pg_engine,
            embedder=embedder,
        )

    assert result is fake_verdict
    assert mock_compare.call_count == 1
    # compare_topk received a non-empty top_k list with the right shape.
    call_args = mock_compare.call_args
    assert call_args.kwargs["idea"] == "AI for SMB legal contract review"
    top_k_arg = call_args.kwargs["top_k"]
    assert len(top_k_arg) <= 3
    for entry in top_k_arg:
        assert set(entry.keys()) >= {
            "company_id",
            "name",
            "description",
            "similarity",
        }
        # similarity in [-1, 1].
        assert -1.0 <= entry["similarity"] <= 1.0


def test_analyze_endpoint_top_k_is_respected(client_with_indexed_corpus, pg_engine):
    """``analyze_endpoint`` trims top_k to the requested depth."""
    embedder = _PerTextEmbedder()
    _ = embedder.embed_one("legal contract review")

    fake_verdict = IdeaVerdict(
        idea="x",
        top_competitors=[],
        market_scope=MarketScope.WIDE_OPEN,
        market_scope_rationale="stub",
        supporting_evidence=[],
    )

    with patch.object(
        analyze_module, "compare_topk", return_value=fake_verdict
    ) as mock_compare:
        analyze_endpoint(
            AnalyzeRequest(idea="legal contract review", top_k=1),
            engine=pg_engine,
            embedder=embedder,
        )

    # top_k=1 in the request ⇒ one entry forwarded to compare_topk.
    assert len(mock_compare.call_args.kwargs["top_k"]) <= 1


def test_analyze_endpoint_empty_corpus_returns_no_competitors(pg_engine):
    """Empty corpus → ``AnalyzeError("no_competitors", ...)`` (no LLM call)."""
    embedder = _PerTextEmbedder()
    _ = embedder.embed_one("anything")
    # corpus has tables but no rows → search_corpus returns []

    with patch.object(analyze_module, "compare_topk") as mock_compare:
        result = analyze_endpoint(
            AnalyzeRequest(idea="anything", top_k=3),
            engine=pg_engine,
            embedder=embedder,
        )

    assert isinstance(result, AnalyzeError)
    assert result.error == "no_competitors"
    assert mock_compare.call_count == 0  # no LLM call on empty corpus


# ---------------------------------------------------------------------------
# HTTP-layer tests (TestClient)
# ---------------------------------------------------------------------------


def test_ideas_analyze_happy_path(client_with_indexed_corpus):
    """Happy path: 200 + a valid IdeaVerdict JSON body."""
    fake_verdict_dict = {
        "idea": "AI for SMB legal contract review",
        "top_competitors": [
            {
                "company_id": 1,
                "name": "Alpha Co",
                "similarity_axes": ["AI contract drafting", "SMB market"],
                "key_differences": ["different focus"],
                "likely_failure_modes": ["strong distribution"],
                "evidence_links": [],
                "confidence": 0.75,
            }
        ],
        "market_scope": "crowded_but_growing",
        "market_scope_rationale": "3 similar YC launches in legaltech, none dominant",
        "supporting_evidence": [],
    }
    fake_verdict = IdeaVerdict.model_validate(fake_verdict_dict)

    with patch.object(
        analyze_module, "compare_topk", return_value=fake_verdict
    ):
        resp = client_with_indexed_corpus.post(
            "/ideas/analyze",
            json={"idea": "AI for SMB legal contract review", "top_k": 1},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["idea"] == "AI for SMB legal contract review"
    assert body["market_scope"] == "crowded_but_growing"
    assert len(body["top_competitors"]) == 1
    assert body["top_competitors"][0]["name"] == "Alpha Co"


def test_ideas_analyze_empty_corpus_returns_no_competitors(client_with_empty_corpus):
    """Empty corpus: 200 + structured no_competitors error."""
    with patch.object(analyze_module, "compare_topk") as mock_compare:
        resp = client_with_empty_corpus.post(
            "/ideas/analyze",
            json={"idea": "anything"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"] == "no_competitors"
    assert body["details"]["corpus_count"] == 0
    # We never made an LLM call.
    assert mock_compare.call_count == 0


def test_ideas_analyze_schema_violation_returns_structured_error(
    client_with_indexed_corpus,
):
    """SchemaViolationError from the LLM → 200 + structured error body."""
    with patch.object(
        analyze_module,
        "compare_topk",
        side_effect=SchemaViolationError(
            "LLM returned garbage",
            details=[{"loc": ("top_competitors",), "msg": "missing"}],
        ),
    ):
        resp = client_with_indexed_corpus.post(
            "/ideas/analyze",
            json={"idea": "AI for SMB legal contract review", "top_k": 3},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"] == "schema_violation"
    assert body["details"] == [{"loc": ["top_competitors"], "msg": "missing"}]


def test_ideas_analyze_llm_transport_error_returns_structured_error(
    client_with_indexed_corpus,
):
    """LLMTransportError → 200 + structured error body."""
    with patch.object(
        analyze_module,
        "compare_topk",
        side_effect=LLMTransportError("Anthropic call failed: timeout"),
    ):
        resp = client_with_indexed_corpus.post(
            "/ideas/analyze",
            json={"idea": "AI for SMB legal contract review", "top_k": 3},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"] == "llm_transport"
    assert "timeout" in body["details"]["message"]


def test_ideas_analyze_missing_api_key_returns_503(client_with_indexed_corpus):
    """MissingAPIKeyError → 503 + structured error body (still parseable)."""
    with patch.object(
        analyze_module,
        "compare_topk",
        side_effect=MissingAPIKeyError(
            "Anthropic API key not found. Set $ANTHROPIC_API_KEY."
        ),
    ):
        resp = client_with_indexed_corpus.post(
            "/ideas/analyze",
            json={"idea": "AI for SMB legal contract review", "top_k": 3},
        )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    # FastAPI wraps HTTPException(detail=...) under "detail".
    assert body["detail"]["error"] == "llm_unconfigured"
    assert "ANTHROPIC_API_KEY" in body["detail"]["details"]["message"]


def test_ideas_analyze_validation_error_on_empty_idea(client_with_indexed_corpus):
    """Empty ``idea`` → 422 (FastAPI default validation handler)."""
    resp = client_with_indexed_corpus.post(
        "/ideas/analyze",
        json={"idea": ""},
    )
    assert resp.status_code == 422, resp.text


def test_ideas_analyze_validation_error_on_oversized_idea(client_with_indexed_corpus):
    """Idea longer than 4096 chars → 422."""
    resp = client_with_indexed_corpus.post(
        "/ideas/analyze",
        json={"idea": "x" * 4097},
    )
    assert resp.status_code == 422, resp.text


def test_ideas_analyze_validation_error_on_top_k_too_large(client_with_indexed_corpus):
    """top_k > MAX_TOP_K → 422."""
    resp = client_with_indexed_corpus.post(
        "/ideas/analyze",
        json={"idea": "x", "top_k": MAX_TOP_K + 1},
    )
    assert resp.status_code == 422, resp.text


def test_ideas_analyze_default_top_k(client_with_indexed_corpus):
    """Omitting ``top_k`` defaults to DEFAULT_TOP_K (3)."""
    fake_verdict = IdeaVerdict(
        idea="x",
        top_competitors=[],
        market_scope=MarketScope.WIDE_OPEN,
        market_scope_rationale="x",
        supporting_evidence=[],
    )
    with patch.object(
        analyze_module, "compare_topk", return_value=fake_verdict
    ) as mock_compare:
        resp = client_with_indexed_corpus.post(
            "/ideas/analyze",
            json={"idea": "AI for SMB legal contract review"},
        )
    assert resp.status_code == 200, resp.text
    # Default top_k = 3 ⇒ forwarded list size ≤ 3.
    assert len(mock_compare.call_args.kwargs["top_k"]) <= DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# Cost-control: exactly one LLM call per /ideas/analyze request
# ---------------------------------------------------------------------------


def test_ideas_analyze_makes_exactly_one_llm_call(client_with_indexed_corpus):
    """PHASE-1.md §1.7 cost-control rule: one LLM call per request, not per competitor."""
    fake_verdict = IdeaVerdict(
        idea="x",
        top_competitors=[],
        market_scope=MarketScope.WIDE_OPEN,
        market_scope_rationale="x",
        supporting_evidence=[],
    )
    with patch.object(
        analyze_module, "compare_topk", return_value=fake_verdict
    ) as mock_compare:
        resp = client_with_indexed_corpus.post(
            "/ideas/analyze",
            json={"idea": "AI for SMB legal contract review", "top_k": 3},
        )
    assert resp.status_code == 200, resp.text
    assert mock_compare.call_count == 1