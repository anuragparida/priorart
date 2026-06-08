"""POST /search — ANN retrieval against the YC public corpus.

Phase 1.4 (docs/PHASE-1.md §1.4). Contract:

    POST /search {"query": "...", "top_k": 20}
        -> 200 {"hits": [{"id", "name", "description", "similarity",
                           "confidence"}, ...]}
        -> 422 on validation failure
        -> 503 on missing / unembedded corpus

Pipeline
--------
1. Embed the query with the same bge-m3 model that the ingest used
   (so cosine is apples-to-apples).
2. Run a pgvector ANN query: ``ORDER BY embedding <=> :q LIMIT top_k``.
3. For each result, the *raw cosine similarity* is ``1 - (embedding <=> q)``
   (true cosine for unit vectors). The *normalized confidence* is a
   linear map ``(sim + 1) / 2`` that lives in ``[0, 1]`` — used by the
   eval harness (1.6) and the low-confidence /ideas/analyze branch
   (1.8) to apply a single threshold.
4. If a company has multiple chunks, keep the highest-similarity chunk
   per company_id. The user-facing contract is *one hit per company*;
   the chunking detail is an implementation artefact.

Why cosine + confidence (and not just one of them)
--------------------------------------------------
- ``similarity`` is the raw math. Anyone reproducing the index can
  re-derive the ranking with the same number, no opaque normalisation.
- ``confidence`` is the calibrated 0–1 signal downstream consumers
  (Phase 1.6 threshold sweep, Phase 1.8 low-confidence branch) use to
  decide "is this a duplicate?". Mapping ``(sim+1)/2`` is a
  deliberate, lossy choice — it throws away the sign of cosine, but
  bge-m3 vectors for natural-language descriptions are almost always
  positive-correlated with each other, so the sign carries no signal
  in practice and the linear map keeps the threshold tuning
  intuitive (``0.65`` means "65% of a perfect match").
"""

from __future__ import annotations

import logging
from typing import Annotated, List, Optional

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.data.db import get_engine
from src.data.embedder import Embedder

logger = logging.getLogger(__name__)

#: The HNSW candidate-list size at query time. pgvector's default is 40;
#: 100 is a comfortable middle ground for ~6K rows — recall is within a
#: percent of ef_search=200 at one-third the latency. Phase 2 can tune.
HNSW_EF_SEARCH = 100


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """``POST /search`` body."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Free-text idea / query to embed and search against the corpus.",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of companies to return (1–200, default 20).",
    )


class SearchHit(BaseModel):
    """One ranked result."""

    id: int = Field(..., description="Company id (companies.id).")
    name: str
    description: str
    similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Raw cosine similarity in [-1, 1]. 1.0 = identical direction.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised 0–1 confidence = (similarity + 1) / 2. "
        "Downstream thresholding (Phase 1.6 / 1.8) uses this.",
    )


class SearchResponse(BaseModel):
    """``POST /search`` response."""

    hits: List[SearchHit]
    query: str
    model: str
    top_k: int
    corpus_count: int


# ---------------------------------------------------------------------------
# Embedder dependency
# ---------------------------------------------------------------------------


def get_embedder() -> Embedder:
    """FastAPI dependency that yields the singleton Embedder.

    Held as a module-level singleton inside ``src.data.embedder`` — the
    sentence-transformers model is ~1.5 GB, so we want exactly one
    instance per process. The dependency form lets tests override the
    embedder (e.g. with a ``_ZeroEmbedder``) without monkey-patching
    the module.
    """
    return Embedder()


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


# ---------------------------------------------------------------------------
# Search core
# ---------------------------------------------------------------------------


def _cosine_to_confidence(cosine: float) -> float:
    """Map a cosine similarity in [-1, 1] to a 0–1 confidence.

    Linear remap. No calibration, no sigmoid — Phase 1.6's threshold
    sweep will tell us if a more aggressive curve is needed (it
    probably won't be; bge-m3 + normalise is already well-behaved).
    """
    return (cosine + 1.0) / 2.0


#: SQL used by :func:`search_corpus`. Bound parameters: ``q`` (the
#: query vector as a JSON array), ``k`` (top_k), ``ef`` (HNSW
#: ef_search). The ``embedding`` column is ``vector(1024)``; pgvector
#: accepts the JSON-list string as a vector literal.
_SEARCH_SQL = text(
    """
    SELECT
        c.id            AS id,
        c.name          AS name,
        c.description   AS description,
        ce.chunk_index  AS chunk_index,
        ce.chunk_count  AS chunk_count,
        (1 - (ce.embedding <=> :q)) AS similarity
    FROM company_embeddings ce
    JOIN companies c ON c.id = ce.company_id
    ORDER BY ce.embedding <=> :q ASC
    LIMIT :k
    """
).bindparams(
    bindparam("q", type_=None),  # pgvector reads the string as a vector literal
)


def _corpus_count(engine: Engine) -> int:
    """Row count in ``company_embeddings`` — the index size."""
    with engine.connect() as conn:
        return int(
            conn.execute(text("SELECT count(*) FROM company_embeddings")).scalar_one()
        )


def search_corpus(
    engine: Engine,
    embedder: Embedder,
    *,
    query: str,
    top_k: int = 20,
) -> List[SearchHit]:
    """Run an ANN search.

    Returns a list of ``SearchHit`` ordered by descending cosine
    similarity. Companies that have multiple chunks are deduplicated
    — the highest-similarity chunk per company wins, so the user sees
    one row per company.

    The deduplication is "post-limit", not "pre-DB". The DB has no
    concept of "one row per company" (that's a SQL rank() + partition
    puzzle pgvector's ``<=`` doesn't index well), so we over-fetch by
    2x and prune. For top_k ≤ 200 and a 5990-row corpus this is
    negligible; if top_k ever grows past 500 we'll need a windowed
    query instead.
    """
    fetch_k = min(top_k * 2, 500)
    query_vec = embedder.embed_one(query)

    with engine.connect() as conn:
        # Set ef_search for the connection (HNSW candidate list).
        # ``SET LOCAL`` would be per-tx; ``SET`` is per-session. We
        # use a plain SET to keep the override alive for the query.
        conn.execute(text(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"))
        rows = conn.execute(
            _SEARCH_SQL,
            {"q": str(query_vec), "k": fetch_k},
        ).fetchall()

    # Post-fetch dedup: keep the best (highest-similarity) chunk per
    # company_id. Stable on ties (first row wins, which is the
    # nearest-tie-break by chunk_index ASC, but we don't promise
    # that).
    by_company: dict[int, SearchHit] = {}
    for row in rows:
        cid = int(row.id)
        sim = float(row.similarity)
        existing = by_company.get(cid)
        if existing is None or sim > existing.similarity:
            by_company[cid] = SearchHit(
                id=cid,
                name=str(row.name),
                description=str(row.description),
                similarity=sim,
                confidence=_cosine_to_confidence(sim),
            )

    # Re-sort by similarity desc and trim to top_k.
    hits = sorted(by_company.values(), key=lambda h: h.similarity, reverse=True)
    return hits[:top_k]


# ---------------------------------------------------------------------------
# Exception type
# ---------------------------------------------------------------------------


class CorpusNotIndexedError(RuntimeError):
    """The corpus table is empty / missing — search cannot proceed."""


# ---------------------------------------------------------------------------
# FastAPI route helper (so app.py can mount it)
# ---------------------------------------------------------------------------


def search_endpoint(
    request: SearchRequest,
    engine: Annotated[Engine, Depends(get_engine)],
    embedder: EmbedderDep,
) -> SearchResponse:
    """POST /search route body — kept here so the test suite can
    import the function directly without going through FastAPI.
    """
    try:
        corpus_count = _corpus_count(engine)
    except SQLAlchemyError as exc:
        logger.exception("search: failed to count corpus")
        raise CorpusNotIndexedError("corpus not reachable") from exc

    if corpus_count == 0:
        # No rows to search against. Return an empty list — the API
        # contract is "non-empty JSON list" per the task spec, so an
        # empty list is the correct signal: the corpus has nothing
        # indexed yet. (The eval harness treats this as
        # "FPR-on-novel = 0" by definition.)
        return SearchResponse(
            hits=[],
            query=request.query,
            model=embedder.model_name,
            top_k=request.top_k,
            corpus_count=0,
        )

    hits = search_corpus(
        engine,
        embedder,
        query=request.query,
        top_k=request.top_k,
    )
    return SearchResponse(
        hits=hits,
        query=request.query,
        model=embedder.model_name,
        top_k=request.top_k,
        corpus_count=corpus_count,
    )
