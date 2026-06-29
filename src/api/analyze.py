"""POST /ideas/analyze — orchestrate embed → ANN search → LLM compare.

Phase 1.8 (docs/PHASE-1.md §1.8). Glue layer. Contract:

    POST /ideas/analyze {"idea": "...", "top_k": 3}
        -> 200 <IdeaVerdict JSON>
        -> 200 {"error": "schema_violation", "details": ...}
               on a Pydantic-validation failure from the LLM
        -> 200 {"error": "no_competitors", "details": ...}
               when the corpus is empty or returns zero hits (still 200,
               not a 500 — the API treats this as "genuinely novel")
        -> 422 on request validation failure
        -> 503 on missing / unembedded corpus
        -> 503 on missing ANTHROPIC_API_KEY (the compare call can't run
               without one — we surface it explicitly rather than 500)

Pipeline
--------
1. Embed the idea with the same bge-m3 model the corpus uses.
2. Run ``search_corpus`` (Phase 1.4) → top-K companies.
3. Build the ``top_k`` dicts that ``compare_topk`` (Phase 1.7) expects.
4. Run ``compare_topk`` (one LLM call — PHASE-1.md §1.7 cost-control rule).
5. Return the validated ``IdeaVerdict``.

Error model
-----------
The spec is explicit: "No 500s on schema-violation — surface as
``{"error": "schema_violation", "details": ...}``." We return 200 with
that body so the client gets a parseable, machine-readable error
instead of an opaque stack trace. Same shape for the "no competitors"
branch — a 200 with a structured error, not a 503 or 500.

Why a separate module (not inlined into app.py)
-----------------------------------------------
``app.py`` is the assembly point — routes + dependency wiring + the
one healthz helper. The analyze pipeline is large enough (a few
hundred lines of glue + helpers) that inlining would obscure both.
Splitting also mirrors the Phase 1.4 layout (``src/api/search.py`` +
the ``/search`` route in ``app.py``).

Why we don't use FastAPI's exception handlers for schema violation
------------------------------------------------------------------
We *could* register a handler that catches ``SchemaViolationError``,
returns 200 + the structured error, and the route body would shrink
to three lines. We don't, because:

- The handler would also catch SchemaViolationErrors raised *outside*
  the analyze route (e.g. from a future endpoint). That's a leaky
  abstraction — the handler doesn't know whether the caller wants a
  200-with-error or a 500.
- An explicit ``try/except`` makes the error contract obvious to a
  reader of the route. Future endpoints that want a different
  contract (a 500, say) can decide locally.

The trade-off is one extra try/except per request — a non-issue
compared to the LLM latency we're already paying.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine

from src.api.db import get_engine
from src.api.search import (
    SearchHit,
    search_corpus,
)
from src.data.embedder import Embedder
from src.llm.compare import (
    compare_topk,
)
from src.llm.schemas import DEFAULT_TOP_K, MAX_TOP_K, IdeaVerdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """``POST /ideas/analyze`` body."""

    idea: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description=(
            "Free-text startup idea. Will be embedded and searched against "
            "the YC corpus, then sent to the LLM for structured comparison."
        ),
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description=(
            f"Number of competitors to include in the LLM verdict. "
            f"Default {DEFAULT_TOP_K}, max {MAX_TOP_K}. The corpus is "
            f"searched at this depth; lower values are cheaper."
        ),
    )
    # Phase 2.2 — opt-in flags plumbed through to the Temporal
    # ``IdeaAnalysisInput``. We *don't* enforce the request
    # shape here beyond what's in the workflow's input Pydantic
    # model; the route converts ``AnalyzeRequest`` →
    # ``IdeaAnalysisInput`` explicitly (see ``app.py``).
    enable_web_fallback: bool = Field(
        default=True,
        description=(
            "Phase 2.2 — when True, the workflow runs a SearXNG-"
            "backed web search if the corpus returns nothing above "
            "the threshold. Default True; pass False to opt out."
        ),
    )
    web_fallback_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Phase 2.2 — minimum top-1 cosine similarity required "
            "from the corpus before the web fallback is skipped."
        ),
    )
    enable_low_confidence_review: bool = Field(
        default=True,
        description=(
            "Phase 2.2 — when True, a low-confidence verdict parks "
            "the workflow on a signal channel for human review. "
            "Default True; pass False to skip the wait."
        ),
    )


class AnalyzeError(BaseModel):
    """Structured error returned by ``/ideas/analyze``.

    Always returned with HTTP 200 (per spec) so the client can
    distinguish a "the system has no answer for this" from a
    "the system is broken" (which would still come through as
    a real 500 / 422 / 503 from FastAPI's default handlers).
    """

    error: str = Field(
        ...,
        description=(
            "One of: 'schema_violation', 'no_competitors', "
            "'llm_transport', 'llm_unconfigured'. See the README "
            "for the contract of each."
        ),
    )
    details: object | None = Field(
        default=None,
        description=(
            "Optional structured details. For schema_violation, "
            "the Pydantic error dict. For llm_transport, the "
            "exception class + message. None otherwise."
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _hits_to_top_k(hits: list[SearchHit], top_k: int) -> list[dict]:
    """Convert a list of ``SearchHit`` to the dict shape ``compare_topk`` expects.

    ``compare_topk`` requires ``company_id``, ``name``, ``description``,
    and ``similarity``. ``SearchHit`` already has all four (id, name,
    description, similarity). We just rename ``id`` → ``company_id`` and
    pass through.

    The LLM cares about cosine similarity because the system prompt
    explicitly mentions it (so the model can decide "high cosine +
    different market" vs "low cosine + same market"). The normalised
    ``confidence`` is *not* sent — it's a downstream-thresholding
    signal we use in Phase 1.6, not a model signal.
    """
    return [
        {
            "company_id": int(h.id),
            "name": h.name,
            "description": h.description,
            "similarity": float(h.similarity),
        }
        for h in hits[:top_k]
    ]


# ---------------------------------------------------------------------------
# The route body — kept importable so tests can exercise it without uvicorn
# ---------------------------------------------------------------------------


def analyze_endpoint(
    request: AnalyzeRequest,
    engine: Engine,
    embedder: Embedder,
) -> IdeaVerdict | AnalyzeError:
    """``POST /ideas/analyze`` route body.

    Pipeline: embed → ANN search → ``compare_topk`` → ``IdeaVerdict``.

    Returns
    -------
    IdeaVerdict
        On success — a Pydantic-validated structured comparison.

    AnalyzeError
        On any of:
        - the corpus is empty / unreachable (``no_competitors``)
        - the LLM call returned something instructor couldn't coerce
          into IdeaVerdict (``schema_violation``)
        - the LLM transport raised a non-validation error
          (``llm_transport``)
        - ``ANTHROPIC_API_KEY`` is not configured (``llm_unconfigured``)

    Raises
    ------
    CorpusNotIndexedError
        When the corpus table is unreachable (not empty — *missing*).
        The route layer maps that to 503.
    """
    # Step 1+2: embed + ANN search. The corpus count check lives inside
    # search_corpus's helper upstream (in /search); here we call the
    # core directly and handle the empty-corpus case ourselves because
    # /ideas/analyze has a different empty-state contract: it returns
    # an empty verdict, not an empty search hit list.
    hits = search_corpus(
        engine,
        embedder,
        query=request.idea,
        top_k=request.top_k,
    )

    if not hits:
        # Genuinely-novel signal. Return 200 + structured error so the
        # client knows this isn't a system failure — it's "the corpus
        # has nothing close to your idea, so we can't compare". This
        # is also what /search does on an empty corpus, except the
        # spec for /ideas/analyze explicitly says "no 500s".
        return AnalyzeError(
            error="no_competitors",
            details={
                "message": (
                    "No similar launches found in the YC corpus. "
                    "This is genuinely novel — or outside the YC market."
                ),
                "corpus_count": 0,
            },
        )

    # Step 3: shape the top-K for compare_topk.
    top_k_for_llm = _hits_to_top_k(hits, request.top_k)

    # Step 4: one LLM call. We let the LLMTransportError +
    # MissingAPIKeyError + SchemaViolationError exceptions propagate
    # into the route handler's try/except block (which lives in app.py
    # for the HTTP layer; tests exercise this function directly).
    return compare_topk(
        idea=request.idea,
        top_k=top_k_for_llm,
    )


# ---------------------------------------------------------------------------
# FastAPI dependency wrapper
# ---------------------------------------------------------------------------


#: ``EngineDep`` lives here for the FastAPI route signature. The
#: underlying factory is ``get_engine`` in ``src.api.db`` — shared
#: with ``/search`` so both routes use the same engine instance per
#: process.
EngineDep = Annotated[Engine, Depends(get_engine)]