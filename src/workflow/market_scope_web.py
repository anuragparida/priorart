"""SearXNG + Firecrawl augmentation for sparse market-scope directions (Phase 4.3).

Why this module exists
----------------------
PHASE-4.md §4.3 asks for a web-augmentation step that fires when the
local corpus is too thin to be quantitative — i.e. when the Phase 4.2
deterministic rules land on ``wide_open`` or ``niche_but_real`` with
``competitor_count < 10``. In that regime the corpus can't tell us
whether the direction is genuinely open or just under-represented in
our 11K-company snapshot, so we reach for the open web.

The path is the *same* one Phase 2.10's ``web_fallback_if_empty`` uses:

    SearXNG (meta-search, no API key)  →  Firecrawl ``/v2/search``
       ↓ (returns candidate URLs + descriptions)
    Firecrawl ``/v1/scrape``          →  clean markdown snippet
       ↓ (per kept, non-duplicate result)
    ``[{url, title, snippet, as_of}]``

We reuse ``WebFallbackClient`` for the HTTP transport (connection
pool, timeouts, structured ``WebFallbackTransportError``) rather than
re-implementing an ``httpx`` client — the market-scope path only adds
the *interpretation* layer on top: a fixed query template, URL
dedup, a distinct-domain "search-volume proxy", and the mapping onto
the Phase 4.1 ``MarketScopeEvidence`` shape.

What this module does NOT do
----------------------------
- **No LLM call.** This is a pure data fetch. The LLM direction
  synthesis over these snippets is Phase 4.4's job — it consumes this
  module's output as input.
- **No envelope assembly.** It returns a small dataclass; the 4.4
  activity folds ``evidence`` / ``confidence`` / ``search_volume_proxy``
  into the ``MarketScopeSignal`` envelope.
- **No corpus SQL.** The corpus density computation is Phase 4.2.

Cost / abuse guard
------------------
Two gates protect the external HTTP call:

1. **Direction gate** — ``should_augment(direction, competitor_count)``
   returns ``True`` only for ``wide_open`` / ``niche_but_real`` with
   ``competitor_count < 10``. The caller (the 4.4 activity) checks this
   before invoking ``augment_market_scope_web``. Everything else stays
   corpus-only.
2. **Offline mode gate** — ``augment_market_scope_web`` short-circuits
   to ``directional`` confidence and makes **zero** web calls when the
   offline flag is set. CI's ``eval-regression`` workflow sets
   ``$PRIORART_MARKET_SCOPE_OFFLINE=1`` so the eval harness never
   touches the network (matches the Phase 3.6.2 offline-eval pattern).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from src.llm.schemas import MarketScope, MarketScopeConfidence, MarketScopeEvidence
from src.workflow.web_fallback import (
    WebFallbackClient,
    WebFallbackError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: The single SearXNG query template. PHASE-4.md §4.3 pins the exact
#: wording — ``"{idea} startup competitors"`` — so the query is
#: reproducible and reviewable. We ``.strip()`` the idea before
#: interpolation so a trailing newline doesn't leak into the query.
MARKET_SCOPE_QUERY_TEMPLATE = "{idea} startup competitors"

#: Number of non-duplicate results to keep and scrape. PHASE-4.md §4.3
#: says "the top 3 non-duplicate results". Bump via
#: ``$PRIORART_MARKET_SCOPE_WEB_TOP_N`` (kept small — each scrape is
#: ~5s on the self-hosted Firecrawl stack).
WEB_AUG_TOP_N = int(os.getenv("PRIORART_MARKET_SCOPE_WEB_TOP_N", "3"))

#: How many raw search candidates to request before dedup. We over-
#: fetch 3x so that after dropping duplicate URLs we still have enough
#: to fill ``WEB_AUG_TOP_N``.
WEB_AUG_SEARCH_LIMIT = int(
    os.getenv("PRIORART_MARKET_SCOPE_WEB_SEARCH_LIMIT", str(WEB_AUG_TOP_N * 3))
)

#: Max characters of scraped markdown to keep as the evidence snippet.
#: A market-scope snippet is a "why this result was kept" quote, not a
#: full page — 280 chars (~a tweet) is plenty and keeps the evidence
#: payload small when it flows through Temporal + into the API response.
WEB_AUG_SNIPPET_MAX_CHARS = int(
    os.getenv("PRIORART_MARKET_SCOPE_WEB_SNIPPET_MAX_CHARS", "280")
)

#: Env var that forces offline mode. When set to a truthy value, the
#: augmentation short-circuits to ``directional`` confidence and makes
#: no web calls. CI's ``eval-regression`` workflow sets this so the
#: eval harness is self-contained (Phase 3.6.2 pattern). Any of
#: ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive) counts as set.
OFFLINE_ENV_VAR = "PRIORART_MARKET_SCOPE_OFFLINE"

#: The ``competitor_count`` ceiling below which the direction gate lets
#: the web augmentation fire (PHASE-4.md §4.3). Encoded as a module-
#: level constant per the Phase 3 type-level-guardrail rule, not a
#: config value.
COMPETITOR_COUNT_AUGMENT_CEILING = 10

#: The directions for which the corpus is "thin enough" that a web
#: augmentation is worth the external call (PHASE-4.md §4.3).
_AUGMENTABLE_DIRECTIONS: frozenset[MarketScope] = frozenset(
    {MarketScope.WIDE_OPEN, MarketScope.NICHE_BUT_REAL}
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketScopeWebResult:
    """One scraped, non-duplicate web result for the market-scope signal.

    ``url`` + ``title`` come from the SearXNG search card; ``snippet``
    is the scraped markdown (truncated to ``WEB_AUG_SNIPPET_MAX_CHARS``),
    falling back to the search ``description`` when the scrape returns
    empty. ``as_of`` is the wall-clock at which the result was captured.
    """

    url: str
    title: str
    snippet: str
    as_of: datetime


@dataclass(frozen=True)
class MarketScopeWebAugmentation:
    """The output of ``augment_market_scope_web``.

    Pure data — the 4.4 activity folds these fields into the
    ``MarketScopeSignal`` envelope. ``fired`` is ``True`` only when the
    web path actually ran (i.e. not offline-short-circuited), so the
    caller can log a "web augmentation fired" metric honestly.
    """

    results: list[MarketScopeWebResult] = field(default_factory=list)
    evidence: list[MarketScopeEvidence] = field(default_factory=list)
    confidence: MarketScopeConfidence = "directional"
    search_volume_proxy: int | None = None
    fired: bool = False


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def is_offline_mode() -> bool:
    """Return ``True`` when the offline env flag is set to a truthy value."""
    raw = os.getenv(OFFLINE_ENV_VAR, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def should_augment(direction: MarketScope, competitor_count: int) -> bool:
    """Direction gate for the web augmentation (PHASE-4.md §4.3).

    Fires only when the deterministic 4.2 rules land on ``wide_open``
    or ``niche_but_real`` AND the corpus neighborhood is thin
    (``competitor_count < 10``). The caller checks this before spending
    an external HTTP call; everything else stays corpus-only.
    """
    return (
        direction in _AUGMENTABLE_DIRECTIONS
        and competitor_count < COMPETITOR_COUNT_AUGMENT_CEILING
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registrable_domain(url: str) -> str:
    """Return a lower-cased ``host`` for a URL, stripping a leading ``www.``.

    Used both for URL dedup and the distinct-domain search-volume proxy.
    We deliberately keep this simple (host minus ``www.``) rather than
    pulling in a public-suffix-list dependency — the proxy is a coarse
    "how many distinct sources mention this" count, not a precise
    registrable-domain calculation.
    """
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _dedupe_by_url(
    candidates: list[dict[str, str]], *, limit: int
) -> list[dict[str, str]]:
    """Keep the first occurrence of each distinct URL, up to ``limit``.

    "Non-duplicate" in PHASE-4.md §4.3 means distinct URLs — two cards
    pointing at the same page collapse to one. We preserve SearXNG's
    ranking order (first seen wins).
    """
    seen: set[str] = set()
    kept: list[dict[str, str]] = []
    for entry in candidates:
        url = str(entry.get("url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        kept.append(entry)
        if len(kept) >= limit:
            break
    return kept


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def augment_market_scope_web(
    idea: str,
    *,
    client: WebFallbackClient | None = None,
    offline: bool | None = None,
    top_n: int = WEB_AUG_TOP_N,
) -> MarketScopeWebAugmentation:
    """Fetch a web-augmented evidence layer for a sparse market-scope direction.

    Runs one SearXNG query (``"{idea} startup competitors"``) via
    Firecrawl's ``/v2/search`` proxy, dedupes to the top ``top_n``
    distinct URLs, scrapes each for a snippet, and returns the results
    plus a distinct-domain ``search_volume_proxy``.

    Parameters
    ----------
    idea:
        The free-text startup idea. Interpolated into
        ``MARKET_SCOPE_QUERY_TEMPLATE``.
    client:
        A ``WebFallbackClient`` to reuse (tests inject a stub). When
        ``None``, a fresh client is created and closed before return.
    offline:
        Force offline mode. When ``None`` (default), reads the
        ``$PRIORART_MARKET_SCOPE_OFFLINE`` env flag. When truthy, the
        function makes **no** web calls and returns ``directional``
        confidence with empty evidence.
    top_n:
        Number of non-duplicate results to keep. Defaults to
        ``WEB_AUG_TOP_N`` (3).

    Returns
    -------
    MarketScopeWebAugmentation
        - ``confidence == "evidence_backed"`` when ≥1 web result was
          kept; ``"directional"`` when offline or nothing was found.
        - ``search_volume_proxy`` = count of distinct domains across
          the kept results (``None`` when offline / no results).
        - ``evidence`` = one ``MarketScopeEvidence(source="web", ...)``
          per kept result, carrying the *real* scraped URL — never an
          LLM-invented one (PHASE-4.md hard rule).
        - ``fired`` = ``True`` only when the web path actually ran.

    Notes
    -----
    Transport failures are swallowed into a ``directional`` /
    empty-evidence result (with ``fired=True``, since we did try) —
    the market-scope pipeline degrades to the corpus-only signal
    rather than failing the whole ``/ideas/analyze`` call. This mirrors
    the ``web_fallback_if_empty`` failure model.
    """
    if offline is None:
        offline = is_offline_mode()

    if offline:
        logger.info(
            "augment_market_scope_web: offline mode set (%s); short-circuiting to directional",
            OFFLINE_ENV_VAR,
        )
        return MarketScopeWebAugmentation(confidence="directional", fired=False)

    if not idea or not idea.strip():
        raise ValueError("idea must be a non-empty string")
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")

    query = MARKET_SCOPE_QUERY_TEMPLATE.format(idea=idea.strip())
    search_limit = max(WEB_AUG_SEARCH_LIMIT, top_n)

    owns_client = client is None
    if client is None:
        client = WebFallbackClient()

    try:
        try:
            candidates = client.search(query, limit=search_limit)
        except WebFallbackError as exc:
            logger.warning(
                "augment_market_scope_web: SearXNG search failed (%s); "
                "degrading to directional",
                exc,
            )
            return MarketScopeWebAugmentation(confidence="directional", fired=True)

        kept = _dedupe_by_url(candidates, limit=top_n)
        if not kept:
            logger.info(
                "augment_market_scope_web: SearXNG returned no usable results for %r; "
                "degrading to directional",
                query,
            )
            return MarketScopeWebAugmentation(confidence="directional", fired=True)

        as_of = datetime.now(timezone.utc)
        results: list[MarketScopeWebResult] = []
        for entry in kept:
            url = str(entry.get("url", "") or "").strip()
            title = str(entry.get("title", "") or "")
            description = str(entry.get("description", "") or "")

            # Scrape the page for a clean-markdown snippet. A scrape
            # failure is non-fatal — we fall back to the SearXNG
            # description so the result is still usable.
            snippet = ""
            try:
                markdown = client.scrape(url, max_chars=WEB_AUG_SNIPPET_MAX_CHARS)
                snippet = markdown.strip()
            except WebFallbackError as exc:
                logger.info(
                    "augment_market_scope_web: scrape failed for %s (%s); "
                    "using search description as snippet",
                    url,
                    exc,
                )
            if not snippet:
                snippet = description.strip()
            if len(snippet) > WEB_AUG_SNIPPET_MAX_CHARS:
                snippet = snippet[:WEB_AUG_SNIPPET_MAX_CHARS]

            results.append(
                MarketScopeWebResult(
                    url=url, title=title, snippet=snippet, as_of=as_of
                )
            )
    finally:
        if owns_client:
            client.close()

    # Distinct-domain proxy: how many independent sources mention the
    # idea's competitive space. PHASE-4.md §4.3: "3 = niche-but-real,
    # 10+ = wide-open-with-known-players".
    distinct_domains = {
        d for d in (_registrable_domain(r.url) for r in results) if d
    }
    search_volume_proxy = len(distinct_domains)

    evidence = [
        MarketScopeEvidence(
            source="web",
            url=r.url,
            company_id=None,
            snippet=r.snippet or None,
            as_of=r.as_of,
        )
        for r in results
    ]

    logger.info(
        "augment_market_scope_web: kept %d results, %d distinct domains -> evidence_backed",
        len(results),
        search_volume_proxy,
    )

    return MarketScopeWebAugmentation(
        results=results,
        evidence=evidence,
        confidence="evidence_backed",
        search_volume_proxy=search_volume_proxy,
        fired=True,
    )


# ---------------------------------------------------------------------------
# Public-API aliases for the Phase 4.4 integration
# ---------------------------------------------------------------------------
#
# The Phase 4.4 activity (``market_scope_signal`` in ``activities.py``)
# consumes this module under two shorter names. We expose them as
# aliases so the integration point has a stable contract regardless of
# the internal ``WEB_AUG_*`` naming:
#
#   ``augment_market_scope``  →  ``augment_market_scope_web``
#   ``MARKET_SCOPE_WEB_TOP_N`` →  ``WEB_AUG_TOP_N``
#
# ``is_offline_mode`` / ``should_augment`` are already the names the
# caller imports.
augment_market_scope = augment_market_scope_web
MARKET_SCOPE_WEB_TOP_N = WEB_AUG_TOP_N


__all__ = [
    "COMPETITOR_COUNT_AUGMENT_CEILING",
    "MARKET_SCOPE_QUERY_TEMPLATE",
    "MARKET_SCOPE_WEB_TOP_N",
    "OFFLINE_ENV_VAR",
    "WEB_AUG_SNIPPET_MAX_CHARS",
    "WEB_AUG_TOP_N",
    "MarketScopeWebAugmentation",
    "MarketScopeWebResult",
    "augment_market_scope",
    "augment_market_scope_web",
    "is_offline_mode",
    "should_augment",
]
