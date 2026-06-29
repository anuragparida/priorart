"""Temporal activities — the per-step work of ``IdeaAnalysisWorkflow``.

Phase 2.1 (docs/PHASE-2.md §2.1): **port the Phase 1.8 logic into
Temporal activities verbatim — no behavior changes in this step.**

Each activity is a thin wrapper around the Phase 1 module that
implements that step:

- ``embed_idea``        → ``src.data.embedder.Embedder.embed_one``
- ``ann_search``        → ``src.api.search.search_corpus``
- ``llm_compare_topk``  → ``src.llm.compare.compare_topk``
- ``market_scope_signal`` → derived from the LLM verdict (no
  separate call)
- ``assemble_verdict``  → identity / pass-through for Phase 2.1
"""

from __future__ import annotations

import logging
from typing import Any

from temporalio import activity

from src.api.search import search_corpus
from src.config import EMBEDDING_MODEL
from src.data.embedder import Embedder
from src.data.models import CompanyEmbedding
from src.llm.compare import compare_topk
from src.llm.schemas import IdeaVerdict
from src.workflow.shared import (
    AnnSearchHit,
    AnnSearchResult,
    IdeaAnalysisInput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    """Lazy import + cache of the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        from src.api.db import get_engine

        _engine = get_engine()
    return _engine


def _reset_engine_for_tests() -> None:
    """Test hook — drop the cached engine so tests can swap it."""
    global _engine
    _engine = None


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@activity.defn(name="embed_idea")
async def embed_idea(idea: str) -> list[float]:
    """Embed the user's idea text with bge-m3.

    Phase 2.1 keeps this as a separate activity from ``ann_search``
    for two reasons:

    1. **Cleaner trace shape** — the Temporal UI shows
       ``embed_idea`` as its own row with a latency number, separate
       from the pgvector query. Phase 2.3 (Langfuse) hooks into this
       boundary to record the embedding latency.
    2. **Verbatim port** — the Phase 1.8 pipeline did embed →
       ANN search as two logical steps; the Temporal port preserves
       that shape so a reviewer can trace the workflow against the
       Phase 1 route.

    The embedding return value is *not* forwarded into
    ``ann_search``; the corpus search re-embeds the query itself
    via ``search_corpus``. That's wasteful (one extra encode) but
    matches the Phase 1.8 behaviour exactly, which is what the
    spec asks for in this step.

    Marked ``async`` because temporalio's worker rejects sync
    activities by default (the ``activity_executor`` config is
    opt-in). Async is the idiomatic shape — even for CPU-bound
    work like embedding, the event-loop hop is cheap (the bge-m3
    encode call runs synchronously inside the async function).
    """
    embedder = Embedder()
    return embedder.embed_one(idea)


@activity.defn(name="ann_search")
async def ann_search(idea: str, top_k: int) -> AnnSearchResult:
    """Run ANN search against the embedded YC corpus.

    The activity takes the original idea text (not the pre-computed
    embedding) because ``search_corpus`` re-embeds internally —
    the search pipeline owns the embedding->pgvector boundary.
    Passing the idea text here keeps the activity's signature
    stable regardless of how the workflow chooses to time the
    embed step.
    """
    engine = _get_engine()
    embedder = Embedder()
    hits = search_corpus(engine, embedder, query=idea, top_k=top_k)

    with engine.connect() as conn:
        from sqlalchemy import func, select

        corpus_count = int(
            conn.execute(select(func.count()).select_from(CompanyEmbedding)).scalar_one()
        )

    return AnnSearchResult(
        hits=[
            AnnSearchHit(
                company_id=int(h.id),
                name=str(h.name),
                description=str(h.description),
                similarity=float(h.similarity),
            )
            for h in hits
        ],
        corpus_count=corpus_count,
    )


@activity.defn(name="llm_compare_topk")
async def llm_compare_topk(
    idea: str,
    top_k_payload: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    """One structured-comparison LLM call (Phase 1.7)."""
    verdict: IdeaVerdict = compare_topk(
        idea=idea,
        top_k=top_k_payload,
        max_companies=top_k,
    )
    return verdict.model_dump(mode="json")


@activity.defn(name="market_scope_signal")
async def market_scope_signal(llm_verdict: dict[str, Any]) -> dict[str, Any]:
    """Pass-through for Phase 2.1 — the LLM verdict already carries
    ``market_scope`` + ``market_scope_rationale``."""
    return {
        "market_scope": llm_verdict.get("market_scope"),
        "market_scope_rationale": llm_verdict.get("market_scope_rationale", ""),
    }


@activity.defn(name="assemble_verdict")
async def assemble_verdict(
    llm_verdict: dict[str, Any],
    _market_scope: dict[str, Any],
    _ann_result: AnnSearchResult,
) -> dict[str, Any]:
    """Identity pass-through for Phase 2.1."""
    return llm_verdict


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "Embedder",
    "EMBEDDING_MODEL",
    "IdeaAnalysisInput",
    "ann_search",
    "assemble_verdict",
    "embed_idea",
    "llm_compare_topk",
    "market_scope_signal",
]


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _reset_all_for_tests() -> None:
    """Test hook — drop the cached engine + embedder model."""
    _reset_engine_for_tests()
    from src.data import embedder as _embedder

    _embedder.reset_model_for_tests()