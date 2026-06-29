"""FastAPI app entrypoint.

Phase 1.3: ``/healthz`` returns the real ``corpus_count`` from
``company_embeddings`` once ingest has run.
Phase 1.4: ``POST /search`` — ANN retrieval against the corpus.
Phase 1.8: ``POST /ideas/analyze`` — orchestrates embed → ANN search
        → LLM compare → ``IdeaVerdict`` (legacy synchronous path).
Phase 2.1: ``POST /ideas/analyze`` is now a Temporal client — it
        starts an ``IdeaAnalysisWorkflow`` and returns the workflow
        handle (workflow_id, run_id, status). The synchronous
        ``analyze_endpoint`` library function is preserved for
        direct callers (tests, notebooks, the Phase 1.8
        backwards-compat shim) but no longer reaches an LLM from
        the FastAPI process. ``GET /workflows/{id}`` describes a
        workflow's status + result; ``GET /workflows/{id}/result``
        is the block-poll convenience route.
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

from src.api.analyze import AnalyzeRequest
from src.api.db import get_engine
from src.api.search import (
    CorpusNotIndexedError,
    EmbedderDep,
    SearchRequest,
    SearchResponse,
    search_endpoint,
)
from src.api.workflows import (
    AnalyzeStartResponse,
    SignalReviewResponse,
    WorkflowStatusResponse,
    analyze_start_endpoint,
    workflow_result_endpoint,
    workflow_signal_review_endpoint,
    workflow_status_endpoint,
)
from src.config import EMBEDDING_MODEL
from src.data.models import CompanyEmbedding
from src.observability.langfuse import init_langfuse, is_tracing_enabled
from src.workflow.shared import IdeaAnalysisInput, ReviewSignal

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
# /ideas/analyze route (Phase 2.1 — Temporal client)
# ---------------------------------------------------------------------------
#
# Phase 2.1 contract: ``POST /ideas/analyze`` no longer runs the
# embed → ANN → LLM pipeline inline. It starts an
# ``IdeaAnalysisWorkflow`` on Temporal and returns the handle. The
# verdict (or a structured error) lives at
# ``GET /workflows/{id}`` / ``GET /workflows/{id}/result``.
#
# Why a module-level reference to ``analyze_start_endpoint`` (instead
# of an inline ``await client.start_workflow(...)``)? The
# HTTP-layer tests in ``tests/test_analyze.py`` patch
# ``src.api.app.analyze_start_endpoint`` with an ``AsyncMock`` to
# assert the wire shape without standing up Temporal. The function
# must therefore be importable from the ``src.api.app`` namespace
# — we import-and-rename it at module load.
@app.post(
    "/ideas/analyze",
    response_model=AnalyzeStartResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Workflow handle (workflow_id + run_id + status=running). "
                "Poll ``GET /workflows/{id}`` for completion; read the "
                "verdict at ``GET /workflows/{id}/result``."
            ),
        },
        422: {
            "description": "Validation error (empty / oversized idea, top_k out of range).",
        },
        503: {
            "description": (
                "Temporal is unreachable (``temporal_unavailable``) — "
                "the workflow never started. Retry after the Temporal "
                "server is back up."
            ),
        },
    },
)
async def analyze(request: AnalyzeRequest) -> AnalyzeStartResponse:
    """Start an ``IdeaAnalysisWorkflow`` and return its handle.

    The synchronous Phase 1.8 path is preserved in
    ``src.api.analyze.analyze_endpoint`` for direct callers (the
    library-function tests in ``tests/test_analyze.py`` still
    exercise that surface). The HTTP layer here is Temporal-only;
    no LLM call happens in the FastAPI process for this route.

    Returns
    -------
    AnalyzeStartResponse
        ``{workflow_id, run_id, status: "running", task_queue}``.

    Raises
    ------
    HTTPException(503)
        Temporal client is unreachable. The detail carries
        ``{"error": "temporal_unavailable", "details": {...}}``.
    """
    workflow_input = IdeaAnalysisInput(
        idea=request.idea,
        top_k=request.top_k,
        request_id=None,
        enable_web_fallback=request.enable_web_fallback,
        web_fallback_threshold=request.web_fallback_threshold,
        enable_low_confidence_review=request.enable_low_confidence_review,
    )
    return await analyze_start_endpoint(workflow_input)


# ---------------------------------------------------------------------------
# /workflows/{id} route (Phase 2.1 — Temporal describe)
# ---------------------------------------------------------------------------


@app.get(
    "/workflows/{workflow_id}",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Workflow status (RUNNING / COMPLETED / FAILED / "
                "TIMED_OUT / CANCELLED / TERMINATED / CONTINUED_AS_NEW), "
                "the in-flight ``phase`` from the ``get_status`` query, "
                "and — for completed workflows — the final "
                "``IdeaVerdict`` in ``result``."
            ),
        },
        404: {
            "description": "Unknown workflow id.",
        },
        503: {
            "description": "Temporal is unreachable.",
        },
    },
)
async def workflows_get(workflow_id: str) -> WorkflowStatusResponse:
    """Describe a Temporal workflow.

    The wire shape is ``WorkflowStatusResponse`` (see
    ``src.api.workflows``). For completed workflows the ``result``
    field is populated with the ``IdeaVerdict``; for failed
    workflows the ``failure`` field carries the Temporal error
    metadata. Workflows that have not yet finished return
    ``status: "RUNNING"`` + a ``phase`` string from the
    ``get_status`` query handler in the workflow code.
    """
    return await workflow_status_endpoint(workflow_id)


# ---------------------------------------------------------------------------
# /workflows/{id}/result route (Phase 2.1 — block-poll convenience)
# ---------------------------------------------------------------------------


@app.get(
    "/workflows/{workflow_id}/result",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Block-polls the workflow until it reaches a terminal "
                "state, then returns either the ``IdeaVerdict`` or a "
                "structured failure body."
            ),
        },
        404: {"description": "Unknown workflow id."},
        503: {"description": "Temporal is unreachable."},
    },
)
async def workflows_result(workflow_id: str) -> dict:
    """Block-poll a workflow until it finishes.

    Convenience wrapper around ``workflow_status_endpoint`` for
    callers that don't want to manage the polling loop themselves.
    Caps at ``RESULT_POLL_TIMEOUT_SECONDS`` (30 s) so a hung
    workflow doesn't wedge the FastAPI worker; clients that need a
    longer wait can fall back to polling ``/workflows/{id}``
    directly.

    The response body shape is intentionally a plain dict (not a
    Pydantic model) because the verdict has many variants — a
    successful ``IdeaVerdict``, a structured failure, or a
    timeout-exceeded body. Callers parse ``status`` first.
    """
    return await workflow_result_endpoint(workflow_id)


# ---------------------------------------------------------------------------
# /workflows/{id}/signal/review route (Phase 2.2 — low-confidence channel)
# ---------------------------------------------------------------------------


@app.post(
    "/workflows/{workflow_id}/signal/review",
    response_model=SignalReviewResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Signal delivered. The caller polls "
                "``GET /workflows/{id}`` afterwards to observe the "
                "workflow's terminal state (``COMPLETED`` for "
                "``confirm`` / ``override``, ``FAILED`` for ``reject``)."
            ),
        },
        404: {"description": "Unknown workflow id."},
        422: {
            "description": (
                "Validation error (missing ``decision``, "
                "``override`` without ``corrected_verdict``, etc.)."
            ),
        },
        503: {"description": "Temporal is unreachable."},
    },
)
async def workflows_signal_review(
    workflow_id: str, signal: ReviewSignal
) -> SignalReviewResponse:
    """Send a review signal to a workflow parked on a low-confidence verdict.

    PHASE-2.md §2.2 acceptance step 3:

        Same novel idea → workflow parks; the status endpoint
        shows "running (waiting for signal)".
        curl -X POST http://localhost:18001/workflows/<id>/signal/review \\
          -H "Content-Type: application/json" \\
          -d '{"verdict": "<corrected>"}'
        Workflow completes. /workflows/<id> shows "completed".

    The body shape is a :class:`ReviewSignal` Pydantic model with
    three branches: ``confirm`` (keep verdict), ``override``
    (swap in ``corrected_verdict``), ``reject`` (fail the
    workflow with a structured reason).
    """
    return await workflow_signal_review_endpoint(workflow_id, signal)
