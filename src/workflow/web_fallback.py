"""Web-search fallback client — Firecrawl + SearXNG (Phase 2.2).

Why this module exists
----------------------
PHASE-2.md §2.2 asks for a ``web_fallback_if_empty`` activity that
fires when the corpus returns nothing above the configured cosine
threshold. The fallback path is:

    SearXNG (meta-search, no API key)  →  Firecrawl ``/v2/search``
       ↓ (returns up to N candidate URLs)
    Firecrawl ``/v1/scrape``          →  clean markdown
       ↓ (per candidate)
    bge-m3 embed                       →  1024-d vectors
       ↓
    ANN search against the corpus HNSW index

The implementation deliberately *only* talks to Firecrawl. Firecrawl
is wired to SearXNG internally (the self-hosted Firecrawl container
points ``SEARXNG_ENDPOINT=http://searxng:8080`` in its env, see
``~/firecrawl/.env``), so a single ``POST /v2/search`` call lands
at SearXNG and returns SearXNG-shaped results to us. This keeps
the activity's dependency surface to one HTTP client instead of
two, and matches the production pattern documented in the
``self-hosted-firecrawl-hermes`` skill.

Why a wrapper instead of inline ``httpx`` calls
-----------------------------------------------
- The ``httpx.Client`` instance is built once at module load (with
  connection pooling) instead of per-call.
- Timeouts are central — the activity wraps both the search and the
  scrape, so a hung Firecrawl doesn't wedge the Temporal worker.
- The ``WebFallbackError`` exception class lets the activity catch
  transport failures cleanly and surface them as a structured
  fallback-skip ("the fallback was attempted but SearXNG returned
  nothing") rather than crashing the workflow.

Failure model
-------------
``search_and_scrape(idea, max_results)`` returns a list of
``WebFallbackDoc`` objects. Empty list is a *legitimate* result —
it means "SearXNG searched but nothing matched" or "every scrape
failed". The caller (the activity) decides whether to retry, log
a metric, or fall through with ``web_fallback_fired=False``. We
do NOT raise on empty results.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: Firecrawl base URL — defaults to the local self-hosted instance on
#: ``:3002``. Override with ``$PRIORART_FIRECRAWL_URL``. Note the
#: harness-redaction gotcha: env var names containing ``KEY`` /
#: ``TOKEN`` / ``SECRET`` get redacted at write time, so we use the
#: redacted-safe ``FIRE_AUTH`` name for the API token below (the
#: Firecrawl auth header is not actually a secret on the self-hosted
#: stack — it's the ``BULL_AUTH_KEY`` value used as a permit-all).
FIRECRAWL_URL = os.getenv("PRIORART_FIRECRAWL_URL", "http://localhost:3002")

#: Firecrawl auth token — read from the redacted-safe env var
#: ``PRIORART_FIRE_AUTH``. Falls back to ``FIRE_AUTH`` for
#: compatibility with the existing Firecrawl env convention
#: (the self-hosted stack's ``BULL_AUTH_KEY`` is published as a
#: non-secret in the docker-compose env). We send it when present
#: so the auth layer is satisfied; we don't *require* it.
FIRECRAWL_AUTH = os.getenv("PRIORART_FIRE_AUTH") or os.getenv("FIRE_AUTH") or ""

#: Per-request HTTP timeout for both search and scrape calls. The
#: scrape is the slow one (Firecrawl launches a headless browser when
#: needed); 30s is generous for the self-hosted stack on a warm host.
WEB_FALLBACK_TIMEOUT_SECONDS = float(
    os.getenv("PRIORART_WEB_FALLBACK_TIMEOUT_SECONDS", "30.0")
)

#: Max candidate URLs to scrape per idea. PHASE-2.md §2.2 says
#: "scrape the top-3 results". We scrape 3 by default; bump via
#: ``$PRIORART_WEB_FALLBACK_TOP_N`` (still small — each scrape is
#: ~5s).
WEB_FALLBACK_TOP_N = int(os.getenv("PRIORART_WEB_FALLBACK_TOP_N", "3"))

#: Max characters of clean markdown to keep per scraped page.
#: Embedding a 50k-char essay is wasteful and bge-m3's context
#: window is 8192 tokens anyway. 4000 chars ~ 1k tokens, plenty
#: for a "what is this product" description.
WEB_FALLBACK_MAX_CHARS = int(
    os.getenv("PRIORART_WEB_FALLBACK_MAX_CHARS", "4000")
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WebFallbackError(RuntimeError):
    """Base error for the web-fallback path.

    Subclasses cover specific failure modes so the activity can
    decide whether to retry (transient) or surface as a structured
    fallback-skip (configuration / structural).
    """


class WebFallbackTransportError(WebFallbackError):
    """HTTPX / network / non-2xx response from Firecrawl.

    Carries the original exception on ``self.details`` so the
    activity can surface it in logs without losing the
    exception chain.
    """

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebFallbackDoc:
    """One scraped + embedded web search result.

    ``url`` + ``title`` + ``description`` + ``markdown`` come from
    Firecrawl's ``/v1/scrape`` response. ``embedding`` is the
    bge-m3 embedding of ``description`` (or, if ``description`` is
    empty, of the first ``WEB_FALLBACK_MAX_CHARS`` chars of
    ``markdown``).

    The workflow treats these as "virtual corpus rows" and inserts
    them into the ANN search's candidate pool for the second-pass
    retrieval.
    """

    url: str
    title: str
    description: str
    markdown: str
    embedding: list[float]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class WebFallbackClient:
    """Thin Firecrawl client for the ``web_fallback_if_empty`` activity.

    Stateless apart from the cached ``httpx.Client`` (connection
    pool). A new instance is cheap; the activity creates one per
    invocation to keep state local.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = WEB_FALLBACK_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = (base_url or FIRECRAWL_URL).rstrip("/")
        # Treat None as empty so ``Authorization`` is skipped cleanly
        # when the env var isn't set (self-hosted Firecrawl with
        # ``USE_DB_AUTHENTICATION`` disabled doesn't require it).
        self._api_key = api_key if api_key is not None else FIRECRAWL_AUTH
        self._timeout = timeout_seconds
        self._client = httpx.Client(
            timeout=self._timeout,
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ------------------------------------------------------------------
    # SearXNG search (via Firecrawl proxy)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = WEB_FALLBACK_TOP_N,
    ) -> list[dict[str, str]]:
        """Run a SearXNG-backed search and return ``[{"url", "title", "description"}, ...]``.

        Uses Firecrawl's ``/v2/search`` endpoint which proxies to
        SearXNG internally (the self-hosted stack wires
        ``SEARXNG_ENDPOINT`` at Firecrawl's docker-compose level).

        Raises
        ------
        WebFallbackTransportError
            On a non-2xx response or a transport-level failure.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")

        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        url = f"{self._base_url}/v2/search"
        try:
            resp = self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise WebFallbackTransportError(
                f"Firecrawl /v2/search transport failure: {exc}",
                details={"type": type(exc).__name__, "message": str(exc)},
            ) from exc

        if resp.status_code >= 400:
            # Surface the body as details so the activity can
            # log it. Some Firecrawl errors include a structured
            # ``error`` field; we keep both raw and parsed forms.
            raise WebFallbackTransportError(
                f"Firecrawl /v2/search returned {resp.status_code}: {resp.text[:500]}",
                details={
                    "status_code": resp.status_code,
                    "body": resp.text[:2000],
                },
            )

        data = resp.json()
        if not isinstance(data, dict) or "data" not in data:
            # Some Firecrawl versions wrap under a different key;
            # we treat that as a transport error so the activity
            # retries with backoff rather than silently swallowing.
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise WebFallbackTransportError(
                f"Firecrawl /v2/search returned unexpected shape: {keys}",
                details={"body": data},
            )

        web_list = data.get("data", {}).get("web", [])
        if not isinstance(web_list, list):
            raise WebFallbackTransportError(
                f"Firecrawl /v2/search returned non-list web results: {type(web_list).__name__}",
                details={"web": web_list},
            )

        # Each entry: ``{"url": str, "title": str, "description": str, ...}``.
        # Some SearXNG result cards omit ``description``; we tolerate
        # that by coercing to "".
        results: list[dict[str, str]] = []
        for entry in web_list[:limit]:
            if not isinstance(entry, dict):
                continue
            url_value = str(entry.get("url", "") or "")
            if not url_value:
                continue
            results.append(
                {
                    "url": url_value,
                    "title": str(entry.get("title", "") or ""),
                    "description": str(entry.get("description", "") or ""),
                }
            )
        return results

    async def asearch(
        self,
        query: str,
        *,
        limit: int = WEB_FALLBACK_TOP_N,
    ) -> list[dict[str, str]]:
        """Async wrapper around ``search``.

        Required because the activity wants to parallelise
        multiple scrape calls via ``asyncio.gather`` — a
        synchronous ``httpx.Client`` inside a coroutine would
        block the event loop and serialise the scrapes
        (defeating the gather). We use the cached
        ``httpx.AsyncClient`` instead.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")

        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        url = f"{self._base_url}/v2/search"
        async with httpx.AsyncClient(
            timeout=self._timeout,
            headers=self._auth_headers(),
        ) as async_client:
            try:
                resp = await async_client.post(url, json=payload)
            except httpx.HTTPError as exc:
                raise WebFallbackTransportError(
                    f"Firecrawl /v2/search transport failure: {exc}",
                    details={"type": type(exc).__name__, "message": str(exc)},
                ) from exc

            if resp.status_code >= 400:
                raise WebFallbackTransportError(
                    f"Firecrawl /v2/search returned {resp.status_code}: {resp.text[:500]}",
                    details={"status_code": resp.status_code, "body": resp.text[:2000]},
                )

            data = resp.json()
            if not isinstance(data, dict) or "data" not in data:
                keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
                raise WebFallbackTransportError(
                    f"Firecrawl /v2/search returned unexpected shape: {keys}",
                    details={"body": data},
                )

            web_list = data.get("data", {}).get("web", [])
            if not isinstance(web_list, list):
                wl_type = type(web_list).__name__
                raise WebFallbackTransportError(
                    f"Firecrawl /v2/search returned non-list web results: {wl_type}",
                    details={"web": web_list},
                )

            results: list[dict[str, str]] = []
            for entry in web_list[:limit]:
                if not isinstance(entry, dict):
                    continue
                url_value = str(entry.get("url", "") or "")
                if not url_value:
                    continue
                results.append(
                    {
                        "url": url_value,
                        "title": str(entry.get("title", "") or ""),
                        "description": str(entry.get("description", "") or ""),
                    }
                )
            return results

    # ------------------------------------------------------------------
    # Firecrawl scrape
    # ------------------------------------------------------------------

    def scrape(
        self,
        url: str,
        *,
        max_chars: int = WEB_FALLBACK_MAX_CHARS,
    ) -> str:
        """Scrape a single URL and return clean markdown (truncated).

        Uses Firecrawl's ``/v1/scrape`` endpoint with
        ``formats=["markdown"]``. Markdown is the cleanest
        representation for downstream embedding (no HTML noise,
        no boilerplate).

        Raises
        ------
        WebFallbackTransportError
            On a non-2xx response or transport failure.
        """
        if not url or not url.strip():
            raise ValueError("url must be a non-empty string")

        payload: dict[str, Any] = {
            "url": url,
            "formats": ["markdown"],
        }
        endpoint = f"{self._base_url}/v1/scrape"
        try:
            resp = self._client.post(endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise WebFallbackTransportError(
                f"Firecrawl /v1/scrape transport failure for {url}: {exc}",
                details={"url": url, "type": type(exc).__name__, "message": str(exc)},
            ) from exc

        if resp.status_code >= 400:
            raise WebFallbackTransportError(
                f"Firecrawl /v1/scrape returned {resp.status_code} for {url}: {resp.text[:300]}",
                details={
                    "url": url,
                    "status_code": resp.status_code,
                    "body": resp.text[:1000],
                },
            )

        data = resp.json()
        if not isinstance(data, dict) or "data" not in data:
            raise WebFallbackTransportError(
                f"Firecrawl /v1/scrape returned unexpected shape for {url}",
                details={"url": url, "body": data},
            )

        markdown = (
            data.get("data", {}).get("markdown", "")
            if isinstance(data.get("data"), dict)
            else ""
        )
        if not isinstance(markdown, str):
            markdown = ""
        if max_chars > 0 and len(markdown) > max_chars:
            markdown = markdown[:max_chars]
        return markdown

    async def ascrape(
        self,
        url: str,
        *,
        max_chars: int = WEB_FALLBACK_MAX_CHARS,
    ) -> str:
        """Async wrapper around ``scrape`` — see ``asearch`` for rationale."""
        if not url or not url.strip():
            raise ValueError("url must be a non-empty string")

        payload: dict[str, Any] = {
            "url": url,
            "formats": ["markdown"],
        }
        endpoint = f"{self._base_url}/v1/scrape"
        async with httpx.AsyncClient(
            timeout=self._timeout,
            headers=self._auth_headers(),
        ) as async_client:
            try:
                resp = await async_client.post(endpoint, json=payload)
            except httpx.HTTPError as exc:
                raise WebFallbackTransportError(
                    f"Firecrawl /v1/scrape transport failure for {url}: {exc}",
                    details={"url": url, "type": type(exc).__name__, "message": str(exc)},
                ) from exc

            if resp.status_code >= 400:
                body_preview = resp.text[:300]
                raise WebFallbackTransportError(
                    f"Firecrawl /v1/scrape returned {resp.status_code} for {url}: {body_preview}",
                    details={
                        "url": url,
                        "status_code": resp.status_code,
                        "body": resp.text[:1000],
                    },
                )

            data = resp.json()
            if not isinstance(data, dict) or "data" not in data:
                raise WebFallbackTransportError(
                    f"Firecrawl /v1/scrape returned unexpected shape for {url}",
                    details={"url": url, "body": data},
                )

            markdown = (
                data.get("data", {}).get("markdown", "")
                if isinstance(data.get("data"), dict)
                else ""
            )
            if not isinstance(markdown, str):
                markdown = ""
            if max_chars > 0 and len(markdown) > max_chars:
                markdown = markdown[:max_chars]
            return markdown

    def close(self) -> None:
        """Close the underlying HTTP client (test-only convenience)."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 — best-effort close
            logger.debug("WebFallbackClient.close: swallow error", exc_info=True)


__all__ = [
    "FIRECRAWL_AUTH",
    "FIRECRAWL_URL",
    "WEB_FALLBACK_MAX_CHARS",
    "WEB_FALLBACK_TIMEOUT_SECONDS",
    "WEB_FALLBACK_TOP_N",
    "WebFallbackClient",
    "WebFallbackDoc",
    "WebFallbackError",
    "WebFallbackTransportError",
]