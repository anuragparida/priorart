"""FastAPI app entrypoint.

Phase 1.3: ``/healthz`` returns the real ``corpus_count`` from
``company_embeddings`` once ingest has run.
Phase 1.4: ``POST /search`` — ANN retrieval against the corpus.
Phase 1.8: ``POST /ideas/analyze`` — orchestrates embed → ANN search
        → LLM compare → ``IdeaVerdict``.
Phase 2.3: ``/healthz`` returns ``langfuse_enabled`` (boolean) so
        operators can confirm Langfuse tracing is wired without
        opening the Langfuse UI. ``init_langfuse`` runs in the
        startup hook so the first /ideas/analyze request already
        has a client.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.api.analyze import (
    AnalyzeError,
    AnalyzeRequest,
    analyze_endpoint,
)
from src.api.db import get_engine
from src.api.search import (
    CorpusNotIndexedError,
    EmbedderDep,
    SearchRequest,
    SearchResponse,
    search_endpoint,
)
from src.config import EMBEDDING_MODEL
from src.data.models import CompanyEmbedding
from src.llm.compare import (
    LLMTransportError,
    MissingAPIKeyError,
    SchemaViolationError,
)
from src.observability.langfuse import init_langfuse, is_tracing_enabled

# ---------------------------------------------------------------------------
# App factory + lifespan (Phase 2.3 Langfuse init)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Process-wide startup / shutdown hooks (Phase 2.3).

    Replaces ``@app.on_event("startup")`` (deprecated in
    FastAPI 0.115). We use the modern ``lifespan`` context
    manager — on entry we init Langfuse; on exit we have
    nothing to flush because the Langfuse SDK uses a
    background-thread flush model and the process is
    shutting down anyway.
    """
    init_langfuse()
    logging.getLogger(__name__).info(
        "langfuse tracing=%s",
        "enabled" if is_tracing_enabled() else "noop (no keys configured)",
    )
    yield


app = FastAPI(
    title="PriorArt",
    description=(
        "Startup-idea deduplication & competitor-research service. "
        "Takes a free-text idea, returns ranked similar YC launches + "
        "structured comparison + market-scope signal."
    ),
    version="0.3.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


#: Re-exported so existing tests that monkeypatch ``app.get_engine``
#: keep working. The real factory lives in ``src.api.db`` so both
#: ``/search`` and ``/ideas/analyze`` can import it without a
#: circular dependency through ``app.py``.
__all__ = ["app", "get_engine"]


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    status: str
    db: str
    model: str
    corpus_count: int | None  # None only when the table is missing or unreadable
    # Phase 2.3 — Langfuse tracing status. ``True`` when real keys are
    # configured + the SDK authenticates against the self-hosted server;
    # ``False`` when keys are missing/placeholder or the SDK failed to
    # init. The route layer reads this via ``is_tracing_enabled()``.
    langfuse_enabled: bool = False


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
        langfuse_enabled=is_tracing_enabled(),
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
    except CorpusNotIndexedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="corpus not reachable",
        ) from exc


# ---------------------------------------------------------------------------
# /ideas/analyze route (Phase 1.8)
# ---------------------------------------------------------------------------


#: Response model is ``Union[IdeaVerdict, AnalyzeError]``; both are
#: 200s. FastAPI picks the right ``response_model`` for serialisation
#: based on the return type. We document the four ``AnalyzeError``
#: shapes in the route's ``responses`` block so OpenAPI reflects
#: them.
@app.post(
    "/ideas/analyze",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Either a validated IdeaVerdict (success) or a "
                "structured AnalyzeError. See the AnalyzeError schema "
                "for the error variants."
            ),
        },
        422: {"description": "Validation error (empty / oversized idea)."},
        503: {
            "description": (
                "Corpus is missing (table unreachable), or "
                "ANTHROPIC_API_KEY is not configured. The latter is "
                "a 503 because the endpoint is useless without an "
                "LLM — clients should retry after configuring."
            ),
        },
    },
)
def analyze(
    request: AnalyzeRequest,
    engine: EngineDep,
    embedder: EmbedderDep,
):
    """Orchestrate: embed → ANN search → LLM compare → IdeaVerdict.

    The happy path returns a Pydantic-validated ``IdeaVerdict`` with
    the top-K competitors, market-scope signal, and supporting
    evidence.

    The error contract (all 200s, not 500s) covers four cases:

    - ``no_competitors`` — corpus is empty / no hits. This is a
      legitimate "genuinely novel" signal, not a failure.
    - ``schema_violation`` — the LLM returned something instructor
      couldn't coerce into ``IdeaVerdict`` after retries. ``details``
      carries the Pydantic error dict so the client can debug.
    - ``llm_transport`` — Anthropic SDK raised a non-validation error
      (timeout, network, 5xx). ``details`` carries the exception
      class + message.
    - ``llm_unconfigured`` — ``ANTHROPIC_API_KEY`` is missing. This is
      also a 503 (see the ``responses`` block) — the endpoint can't
      function without an LLM, but we still emit a structured
      AnalyzeError body so the client has something parseable.

    A real 503 (corpus missing table) is the only case where we
    raise — there's no IdeaVerdict-shaped response to return, just
    "the corpus is broken, retry later".
    """
    try:
        result = analyze_endpoint(request, engine=engine, embedder=embedder)
    except CorpusNotIndexedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="corpus not reachable",
        ) from exc
    except MissingAPIKeyError as exc:
        # 503 + structured body. We use HTTPException for the status
        # code and smuggle the AnalyzeError shape in the ``detail``
        # field — clients should still get something parseable.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llm_unconfigured",
                "details": {"message": str(exc)},
            },
        ) from exc
    except SchemaViolationError as exc:
        # Spec: "No 500s on schema-violation — surface as
        # {"error": "schema_violation", "details": ...}" with 200.
        return AnalyzeError(error="schema_violation", details=exc.details)
    except LLMTransportError as exc:
        # Same shape as schema_violation — structured 200 body so
        # the client can tell "the LLM is sick" from "the LLM
        # returned garbage". The cause is in details.
        return AnalyzeError(
            error="llm_transport",
            details={"message": str(exc), "type": type(exc).__name__},
        )
    return result
