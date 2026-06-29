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

Phase 2.2 adds ``web_fallback_if_empty`` — a SearXNG-backed
fallback path that fires when the corpus returns nothing above
the configured cosine threshold. See the per-activity docstring
for the loop shape (search → scrape → embed → second-pass ANN).
"""

from __future__ import annotations

import asyncio
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
from src.workflow.web_fallback import (
    WebFallbackClient,
    WebFallbackDoc,
    WebFallbackTransportError,
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
# Phase 2.2 — web fallback activity
# ---------------------------------------------------------------------------


@activity.defn(name="web_fallback_if_empty")
async def web_fallback_if_empty(
    idea: str,
    current_ann_result: AnnSearchResult,
    top_k: int = 3,
    threshold: float = 0.7,
) -> AnnSearchResult:
    """SearXNG-backed fallback when the corpus returned nothing above ``threshold``.

    PHASE-2.md §2.2 specifies the loop:

    > If top-K from the corpus returns nothing above the configured
    > threshold (default 0.7 cosine), run a SearXNG search, scrape
    > the top-3 results via your self-hosted Firecrawl, embed them,
    > re-run ANN search.

    Implementation shape
    --------------------
    1. **Skip when corpus already has hits above threshold.** If
       ``current_ann_result.hits[0].similarity >= threshold``, we
       return ``current_ann_result`` unchanged and the activity
       logs ``web_fallback_skipped=True`` for the metric. This
       keeps the fallback "fires < 10% of the time on the eval
       set" pitfall honest — the curated YC corpus is
       higher-quality than the open web, so we don't want the
       fallback silently becoming the primary path.

    2. **SearXNG search** via the self-hosted Firecrawl
       ``/v2/search`` proxy. ``WebFallbackClient.search`` returns
       up to ``top_k`` ``{url, title, description}`` candidates.
       We tolerate an empty list — "SearXNG searched but nothing
       matched" is a legitimate outcome that means the idea is
       genuinely novel.

    3. **Per-URL scrape** via Firecrawl ``/v1/scrape``. Each
       scrape returns clean markdown (no HTML noise). We trim to
       4000 chars (bge-m3 has an 8192-token context; 4k chars
       ~1k tokens, plenty for a "what is this product" blurb).

    4. **bge-m3 embed** the scraped markdown (or the SearXNG
       description if markdown is empty). We use the same
       ``Embedder`` the rest of the workflow uses so the
       vector space matches the corpus exactly.

    5. **Second-pass ANN search.** The scraped docs don't
       replace the corpus — they're "virtual candidates" we
       rank against the existing HNSW index. We do this by:

       a. **Calling ``search_corpus`` again** with each scraped
          doc as a one-item corpus, then sorting the union of
          (corpus-hits + scraped-hits) by similarity. The scraped
          hits *aren't* in the corpus — they're ranked only
          against the original query embedding. So we synthesise
          a "scraped-as-competitor" AnnSearchHit by running the
          bge-m3 encoder on the *idea* (we already have it from
          ``embed_idea``) and computing cosine against each
          scraped doc's embedding. The top-K by cosine is the
          fallback's contribution.

       b. If the corpus already had hits (we entered this
          activity because the *first* hit's cosine was below
          threshold), we union those hits with the scraped hits
          and re-rank. Otherwise the result is the scraped hits
          alone.

    Failure model
    -------------
    - Transport / non-2xx errors from Firecrawl → log + return
      the *original* ``current_ann_result`` with
      ``web_fallback_fired=True`` (we did try). The workflow
      treats this as "fallback attempted but SearXNG returned
      nothing usable" and surfaces it as a low-confidence
      verdict (not a workflow failure).
    - Scrape failure on a single URL → log + skip that URL; the
      remaining URLs are still embedded and ranked.
    - bge-m3 embed failure → log + return the original
      ``current_ann_result``.

    Returns
    -------
    AnnSearchResult
        Either the original result (when the corpus already met
        the threshold, or the fallback found nothing usable) or
        a freshly-ranked result mixing corpus hits and scraped
        hits. The workflow's ``assemble_verdict`` activity and
        the ``web_fallback_fired`` workflow metric both rely on
        the ``AnnSearchResult.corpus_count`` field (kept as the
        corpus count, even when the result has scraped hits) and
        the workflow inspects ``len(self._ann_result.hits) > 0``
        to decide whether to fire the fallback path.
    """
    # Temporal's Pydantic data converter can serialise the
    # previous activity's ``AnnSearchResult`` as a plain dict on
    # the receiving side. We coerce defensively so the activity
    # works regardless of which ``result_type`` the workflow's
    # ``execute_activity`` call passes.
    if isinstance(current_ann_result, dict):
        current_ann_result = AnnSearchResult.model_validate(current_ann_result)

    # 1. Threshold check — fast-path skip
    if current_ann_result.hits:
        top_similarity = max(h.similarity for h in current_ann_result.hits)
        if top_similarity >= threshold:
            logger.info(
                "web_fallback_if_empty: corpus top_similarity=%.3f >= threshold=%.3f; skipping",
                top_similarity,
                threshold,
            )
            # We deliberately do *not* set web_fallback_fired here.
            # The metric key on the workflow is "fires < 10% of the
            # time on the eval set" — a skip-without-trying
            # shouldn't count toward the denominator.
            return current_ann_result

    logger.info(
        "web_fallback_if_empty: corpus empty or below threshold; running SearXNG search (top_k=%d)",
        top_k,
    )

    # 2. SearXNG search via Firecrawl proxy (async so the gather
    # below can parallelise the per-URL scrapes).
    client = WebFallbackClient()
    try:
        try:
            candidates = await client.asearch(idea, limit=top_k)
        except WebFallbackTransportError as exc:
            logger.warning(
                "web_fallback_if_empty: Firecrawl search failed: %s",
                exc,
            )
            return current_ann_result

        if not candidates:
            logger.info(
                "web_fallback_if_empty: SearXNG returned 0 results for idea=%r",
                idea[:80],
            )
            return current_ann_result

        # 3. Per-URL scrape + 4. embed (in parallel via asyncio.gather).
        # Each scrape uses the async HTTP client so the gather
        # actually runs them concurrently. A sync ``httpx.Client``
        # inside a coroutine would block the event loop and
        # serialise the scrapes — a subtle gotcha that bit the
        # first live run.
        embedder = Embedder()

        async def _scrape_one(cand: dict[str, str]) -> WebFallbackDoc | None:
            try:
                markdown = await client.ascrape(cand["url"])
            except WebFallbackTransportError as exc:
                logger.warning(
                    "web_fallback_if_empty: scrape failed for %s: %s",
                    cand["url"],
                    exc,
                )
                return None
            text_for_embedding = (
                markdown
                if markdown
                else (cand.get("description") or cand.get("title") or "")
            )
            if not text_for_embedding:
                return None
            try:
                # bge-m3 ``encode`` is CPU-bound and releases the
                # GIL via torch's backend, but a sync call inside
                # a coroutine still blocks the event loop. We
                # push it onto the default executor so the gather
                # can keep the other scrapes moving in parallel.
                embedding = await asyncio.to_thread(
                    embedder.embed_one, text_for_embedding
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "web_fallback_if_empty: embed failed for %s: %s",
                    cand["url"],
                    exc,
                )
                return None
            return WebFallbackDoc(
                url=cand["url"],
                title=cand.get("title", ""),
                description=cand.get("description", ""),
                markdown=markdown,
                embedding=embedding,
            )

        scraped_results = await asyncio.gather(
            *(_scrape_one(cand) for cand in candidates),
        )
        docs: list[WebFallbackDoc] = [
            r for r in scraped_results if r is not None
        ]

        if not docs:
            logger.info(
                "web_fallback_if_empty: 0 docs after scrape+embed; returning original result",
            )
            return current_ann_result

        # 5. Second-pass ranking: cosine(query_embedding, doc.embedding).
        # The workflow has already embedded the idea via ``embed_idea``;
        # re-embed here is acceptable (bge-m3 is <1s on warm path).
        try:
            query_embedding = await asyncio.to_thread(embedder.embed_one, idea)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "web_fallback_if_empty: embed(query) failed: %s; returning original",
                exc,
            )
            return current_ann_result

        scored: list[tuple[float, WebFallbackDoc]] = []
        for doc in docs:
            sim = _cosine(query_embedding, doc.embedding)
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, top_k)]

        # Build AnnSearchHit entries. IDs are negative to make it
        # obvious these are *virtual* rows (not corpus rows) — a
        # downstream consumer that tries to look them up in the
        # ``companies`` table will hit a clean "not found" path.
        scraped_hits = [
            AnnSearchHit(
                company_id=-1000 - i,
                name=doc.title or doc.url,
                description=(doc.description or doc.markdown[:200]),
                similarity=float(sim),
            )
            for i, (sim, doc) in enumerate(top)
        ]

        # Union with the original corpus hits (re-ranked by similarity)
        all_hits = list(current_ann_result.hits) + scraped_hits
        all_hits.sort(key=lambda h: h.similarity, reverse=True)
        final_hits = all_hits[: max(1, top_k)]

        logger.info(
            "web_fallback_if_empty: returning %d hits (corpus=%d, scraped=%d)",
            len(final_hits),
            len(current_ann_result.hits),
            len(scraped_hits),
        )

        return AnnSearchResult(
            hits=final_hits,
            corpus_count=current_ann_result.corpus_count,
        )
    finally:
        client.close()


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain-Python cosine similarity (both inputs unit-norm from bge-m3).

    bge-m3's ``encode(..., normalize_embeddings=True)`` returns
    unit-norm vectors, so cosine == dot product. We use a tiny
    pure-Python implementation to avoid dragging numpy into the
    Temporal worker (numpy isn't on the worker's critical path).
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # Pad / truncate to the shorter length — defensive against
        # a future embedding-model dimension change. A warning is
        # appropriate because this would silently degrade ranking.
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        aa += x * x
        bb += y * y
    denom_sq = aa * bb
    if denom_sq <= 0.0:
        return 0.0
    return dot / (denom_sq ** 0.5)


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
    "web_fallback_if_empty",
]


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _reset_all_for_tests() -> None:
    """Test hook — drop the cached engine + embedder model."""
    _reset_engine_for_tests()
    from src.data import embedder as _embedder

    _embedder.reset_model_for_tests()