"""FastAPI app entrypoint.

Phase 1.3: ``/healthz`` returns the real ``corpus_count`` from
``company_embeddings`` once ingest has run.
Phase 1.4: ``POST /search`` — ANN retrieval against the corpus.
The /ideas/analyze route lands in Phase 1.8.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.api.search import (
    CorpusNotIndexedError,
    EmbedderDep,
    SearchRequest,
    SearchResponse,
    search_endpoint,
)
from src.config import DATABASE_URL, EMBEDDING_MODEL
from src.data.models import CompanyEmbedding

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PriorArt",
    description=(
        "Startup-idea deduplication & competitor-research service. "
        "Takes a free-text idea, returns ranked similar YC launches + "
        "structured comparison + market-scope signal."
    ),
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_engine() -> Engine:
    """One engine per process. Cheap to create, fine for Phase 1.

    Phase 2 should swap this for a connection pool sized to the
    Temporal worker's expected concurrency.
    """
    return create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    status: str
    db: str
    model: str
    corpus_count: int | None  # None only when the table is missing or unreadable


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _count_corpus(engine: Engine) -> int | None:
    """Count rows in ``company_embeddings``.

    Returns None when the table doesn't exist yet (Phase 1.3
    pre-ingest) or when the DB is unreachable. The API treats
    None and 0 differently: 0 means "ingest has run on an empty
    snapshot", None means "ingest hasn't been wired up".
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(select(func.count()).select_from(CompanyEmbedding))
            return int(result.scalar_one())
    except SQLAlchemyError:
        return None


@app.get("/healthz", response_model=HealthStatus, status_code=status.HTTP_200_OK)
def healthz(engine: EngineDep) -> HealthStatus:
    """Liveness + dependency check.

    Returns 200 only when postgres is reachable. ``corpus_count`` is
    the row count of ``company_embeddings`` — the number of embedded
    chunks in the index. ``None`` means the table doesn't exist yet
    (run the ingest pipeline).
    """
    db_status = "ok"
    corpus_count: int | None = None
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        corpus_count = _count_corpus(engine)
    except SQLAlchemyError:
        db_status = "down"

    overall = "ok" if db_status == "ok" else "degraded"

    return HealthStatus(
        status=overall,
        db=db_status,
        model=EMBEDDING_MODEL,
        corpus_count=corpus_count,
    )


# ---------------------------------------------------------------------------
# Search route (Phase 1.4)
# ---------------------------------------------------------------------------


@app.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Ranked list of similar companies (may be empty)."},
        422: {"description": "Validation error (empty / oversized query)."},
        503: {"description": "Corpus not reachable."},
    },
)
def search(
    request: SearchRequest,
    engine: EngineDep,
    embedder: EmbedderDep,
) -> SearchResponse:
    """Ranked ANN search against the embedded YC corpus.

    Returns the top-``top_k`` companies (deduplicated, so one row per
    company even if the description was chunked) ordered by cosine
    similarity to the query embedding. Each hit carries both the raw
    cosine similarity and a normalised 0–1 confidence for downstream
    thresholding.

    Empty corpus: returns 200 with an empty list. The eval harness
    treats this as "nothing to dedup against"; the API treats it as
    "ingest hasn't run, but the system is up".
    """
    try:
        return search_endpoint(request, engine=engine, embedder=embedder)
    except CorpusNotIndexedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="corpus not reachable",
        )
