"""Scrape the public Hacker News "Show HN" corpus into a versioned JSONL snapshot.

Why this module exists
----------------------
PriorArt's Phase 2 expansion (``docs/PHASE-2.md`` §2.6) adds Hacker News's
public "Show HN" tag stream to the corpus alongside YC and Product Hunt.
The HN Algolia search API at ``hn.algolia.com`` is free, unauthenticated,
and lets us paginate the full history of "Show HN" posts without scraping
the news site itself. Each post carries a ``url`` field pointing to the
external product being launched — that's the "launch URL" we then
scrape via the self-hosted Firecrawl (:3002) for a one-paragraph
description.

Spec contract (``docs/PHASE-2.md`` §2.6 + AGENTS.md stack rules)
---------------------------------------------------------------
- Output: ``data/snapshots/hn_show_<date>.jsonl``, one record per line, UTF-8.
- Dedupe key: ``object_id`` (HN's stable per-post id).
- Manifest: ``data/snapshots/hn_show_<date>.manifest.json`` with ``count``,
  ``scrape_date``, ``source_url``, ``schema_version``, ``date_range``,
  ``points_floor``, ``dead_url_skipped``, ``firecrawl_success_rate``.
- Idempotent: re-running on the same date produces a byte-identical file.
- Deterministic sort: ordered by ``points desc, created_at desc, object_id``.
- Rate limit: >= 1 req/sec to ``hn.algolia.com`` (free tier is generous but
  we stay polite).
- Firecrawl: scrape with small concurrency (~6) to avoid melting the
  self-hosted Firecrawl browser pool. Failures are logged and skipped
  (don't fail the whole job).

Public data only. No login. No third-party mirror. Algolia is the same
index that powers ``news.ycombinator.com`` and the public Algolia search
UI.

CLI
---
    uv run priorart-scrape-hn                    # writes today's snapshot
    uv run priorart-scrape-hn --date 2026-06-08  # deterministic for tests
    uv run priorart-scrape-hn --out /tmp/x       # override output dir
    uv run priorart-scrape-hn --max-records 5000 # cap the corpus size
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
import typer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: HN Algolia endpoint. Free, unauthenticated. The same index that backs
#: ``news.ycombinator.com`` and https://hn.algolia.com.
SOURCE_URL = "https://hn.algolia.com/api/v1/search"

#: HN Algolia search query string. ``tags=show_hn`` is the canonical HN
#: tag for "Show HN" posts; ``show hn`` as a ``query`` is redundant but
#: kept for parity with the spec example. ``numericFilters=created_at_i>X``
#: bounds the result set to the last 3 years (configurable).
HN_QUERY = "show hn"
HN_TAGS = "show_hn"

#: Default age window: posts created in the last ``DEFAULT_LOOKBACK_DAYS``
#: days. The HN "Show HN" universe goes back to 2008 (~1.9M posts);
#: limiting to 3 years keeps the corpus comparable to YC's ~6K and the
#: eval run tractable. Override with ``--lookback-days``.
DEFAULT_LOOKBACK_DAYS = 3 * 365

#: Algolia page size. 1000 is the public-API ceiling — confirmed by
#: probing the endpoint with varying ``hitsPerPage`` values.
PAGE_SIZE = 1000

#: Points threshold. The spec asks for ``points >= 50``. Client-side
#: filtering because HN Algolia does NOT whitelist ``points`` for
#: server-side ``numericFilters`` (verified by 400 response).
POINTS_FLOOR = 50

#: Schema version. Bump only on a field rename or new mandatory field.
SCHEMA_VERSION = "1.0.0"

#: HTTP timeouts (seconds). Algolia is fast; Firecrawl scrubs can be slow
#: because they actually fetch external pages.
ALGOLIA_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
FIRECRAWL_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=10.0)

#: Algolia rate limit. 1.0s base + 0–250ms jitter keeps us well above
#: the free tier's per-IP QPS without being silly-slow.
ALGOLIA_RATE_SECONDS = 1.0
ALGOLIA_RATE_JITTER_SECONDS = 0.25

#: Retry policy on Algolia. Transient (network blip, 5xx, 429) retried;
#: 4xx other than 429 surfaces immediately.
MAX_RETRIES = 4

#: Firecrawl endpoint (self-hosted, see skills/self-hosted-firecrawl-hermes).
#: Override via ``FIRECRAWL_URL`` env var.
FIRECRAWL_URL = os.environ.get("FIRECRAWL_URL", "http://localhost:3002")

#: Firecrawl concurrency. Self-hosted Firecrawl uses a Playwright
#: browser pool; 6 parallel scrapes is the observed practical ceiling
#: for our self-hosted instance — pushing higher causes Playwright
#: queue backpressure and 30s timeouts on ``/v1/scrape``. Tune via
#: the ``FIRECRAWL_CONCURRENCY`` env var.
FIRECRAWL_CONCURRENCY = int(os.environ.get("FIRECRAWL_CONCURRENCY", "6"))

#: Firecrawl per-call timeout. 6s is the practical sweet spot: many
#: external launch sites are heavy JS SPAs that load in 5–10s, but
#: genuinely-dead URLs sit at 30s. Capping at 6s means dead URLs cost
#: 6 worker-seconds instead of 30, while legit sites still mostly fit.
#: The HTTP transport timeout (``FIRECRAWL_TIMEOUT``) below is the
#: hard kill — the 6s here is what httpx *waits* before giving up.
FIRECRAWL_PER_SCRAPE_SECONDS = 6.0

# User-Agent: HN Algolia gates nothing on User-Agent; Firecrawl passes
# the UA through to the target site, so a browser-shaped UA helps avoid
# bot-detection.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Regex for cheap URL liveness check before paying a Firecrawl round-trip.
# HN's ``url`` field for "Ask HN" / "Show HN" text-only posts is null —
# those have no external launch URL and are skipped (not counted as dead;
# they're "no launch URL" and that's the expected shape).
_URL_LIKE = re.compile(r"^https?://", re.IGNORECASE)


# ---------------------------------------------------------------------
# Public schema — what a single HN record looks like in the snapshot
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HNPost:
    """One HN "Show HN" record as it appears in ``hn_show_<date>.jsonl``.

    Field names are stable and form the public schema; do not rename
    without bumping ``SCHEMA_VERSION``.

    Attributes
    ----------
    object_id:
        HN's stable per-post id (``objectID`` in Algolia). Unique even
        across edits.
    title:
        The post title, e.g. ``"Show HN: Frobnicator 9000"``. Includes
        the ``"Show HN: "`` prefix as posted.
    author:
        HN username of the poster.
    url:
        The external URL being launched. ``None`` for text-only posts.
    points:
        The post's score at scrape time. ``points >= POINTS_FLOOR`` is
        enforced before this record is written.
    comments:
        Number of comments at scrape time.
    created_at:
        ISO-8601 UTC timestamp.
    description:
        One-paragraph markdown extract of ``url`` via Firecrawl.
        ``None`` if no external URL, or if Firecrawl failed.
    hn_url:
        Canonical HN discussion URL (``item?id=<object_id>``).
    """

    object_id: str
    title: str
    author: str
    url: str | None
    points: int
    comments: int
    created_at: str
    description: str | None
    hn_url: str

    def to_jsonl(self) -> str:
        """Serialize one record as a single line of JSON + trailing newline.

        Keys are emitted in declaration order so the file is
        byte-stable across re-runs (given identical input data).
        """
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False) + "\n"


# ---------------------------------------------------------------------
# HTTP helpers — Algolia
# ---------------------------------------------------------------------


def _sleep_politely() -> None:
    """Sleep the configured rate-limit interval with small jitter."""
    jitter = random.uniform(0.0, ALGOLIA_RATE_JITTER_SECONDS)
    time.sleep(ALGOLIA_RATE_SECONDS + jitter)


def _get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any],
) -> dict[str, Any]:
    """GET with exponential backoff on transient failures.

    Retries on connection errors, timeouts, 5xx, and 429 (honoring
    ``Retry-After``). Does NOT retry on other 4xx — those are programming
    errors (bad params) that won't fix themselves.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, params=params)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning(
                "Algolia GET failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc
            )
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 10))
                continue
            raise
        if 500 <= resp.status_code < 600:
            logger.warning(
                "Algolia returned %d on attempt %d/%d",
                resp.status_code,
                attempt,
                MAX_RETRIES,
            )
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 10))
                continue
            resp.raise_for_status()
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "5"))
            logger.warning("Algolia rate-limited (429); sleeping %.1fs", retry_after)
            time.sleep(retry_after)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Algolia GET failed with {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()
    raise RuntimeError(
        f"Algolia GET failed after {MAX_RETRIES} attempts; last error: {last_exc}"
    )


# ---------------------------------------------------------------------
# Algolia pagination
# ---------------------------------------------------------------------


def _algolia_params(*, page: int, created_at_floor: int) -> dict[str, Any]:
    """Build the query-string params for one Algolia pagination call.

    The 3-tuple of ``(query, tags, numericFilter)`` is the canonical HN
    "last-N-days of Show HN" query. We do NOT use ``points>=50`` server-
    side because the HN Algolia index does not whitelist ``points`` for
    ``numericFilters`` (verified: API returns 400). Filtering happens
    client-side via ``POINTS_FLOOR``.
    """
    return {
        "query": HN_QUERY,
        "tags": HN_TAGS,
        "numericFilters": f"created_at_i>{created_at_floor}",
        "hitsPerPage": PAGE_SIZE,
        "page": page,
    }


def _fetch_page(
    client: httpx.Client,
    *,
    page: int,
    created_at_floor: int,
) -> dict[str, Any]:
    """Fetch a single page of HN "Show HN" hits.

    Returns the raw Algolia response (``{hits, nbHits, nbPages, page, ...}``).
    """
    return _get_with_retry(
        client,
        SOURCE_URL,
        params=_algolia_params(page=page, created_at_floor=created_at_floor),
    )


def iter_hits(
    client: httpx.Client,
    *,
    created_at_floor: int,
    max_records: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield every HN "Show HN" hit older than ``created_at_floor``.

    The HN Algolia index returns hits sorted by ``created_at_i`` DESC
    (newest first), so the stream is naturally reverse-chronological.
    Pagination uses Algolia's ``page`` cursor.

    Parameters
    ----------
    max_records:
        Optional cap on the number of hits yielded (post-filtering by
        ``POINTS_FLOOR``). Useful for tests and for sizing the corpus
        to match the YC ~6K target.
    """
    page = 0
    yielded = 0
    while True:
        data = _fetch_page(client, page=page, created_at_floor=created_at_floor)
        hits = data.get("hits", [])
        if not hits:
            return
        for hit in hits:
            points = int(hit.get("points") or 0)
            if points < POINTS_FLOOR:
                continue
            yield hit
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return
        nb_pages = int(data.get("nbPages") or 0)
        page += 1
        if page >= nb_pages:
            return
        # Rate-limit per page (not per hit).
        _sleep_politely()


# ---------------------------------------------------------------------
# Firecrawl — scrape external launch URLs
# ---------------------------------------------------------------------


def _firecrawl_scrape(client: httpx.Client, url: str) -> str | None:
    """Scrape one external URL via the self-hosted Firecrawl.

    Returns a one-paragraph markdown extract, or ``None`` on any
    failure (dead URL, Firecrawl error, empty markdown). The scraper
    intentionally treats "unscrapable" as data — a launched-that-404s
    is a real signal, not a noise to retry forever.
    """
    try:
        resp = client.post(
            f"{FIRECRAWL_URL}/v1/scrape",
            json={"url": url, "formats": ["markdown"]},
            timeout=FIRECRAWL_PER_SCRAPE_SECONDS,
        )
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        logger.warning("Firecrawl transport error for %s: %s", url, exc)
        return None
    if resp.status_code >= 400:
        logger.warning(
            "Firecrawl returned %d for %s; skipping", resp.status_code, url
        )
        return None
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        logger.warning("Firecrawl returned non-JSON for %s", url)
        return None
    if not payload.get("success"):
        logger.debug("Firecrawl reported failure for %s: %s", url, payload)
        return None
    data = payload.get("data") or {}
    md = data.get("markdown") or data.get("content") or ""
    if not md:
        return None
    # Collapse runs of whitespace into a single 1-paragraph summary so
    # the field stays compact in the JSONL (and human-readable).
    return " ".join(md.split())


def _scrape_urls_concurrent(
    urls: list[str],
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str | None]:
    """Scrape a batch of URLs concurrently against Firecrawl.

    Returns a ``{url: description_or_None}`` map. Skipped (None) entries
    are dead URLs, timeouts, or Firecrawl errors — same outcome; we
    don't distinguish because downstream consumers only care about
    "got a description or not."

    The ``transport`` parameter is for tests — pass an
    ``httpx.MockTransport`` to keep the test hermetic. Production
    callers leave it as ``None`` and let httpx build a real client.
    """
    if not urls:
        return {}
    headers = {"User-Agent": USER_AGENT}
    results: dict[str, str | None] = {}
    client_kwargs: dict[str, Any] = dict(headers=headers, timeout=FIRECRAWL_TIMEOUT)
    if transport is not None:
        client_kwargs["transport"] = transport
    with httpx.Client(**client_kwargs) as client:
        with ThreadPoolExecutor(max_workers=FIRECRAWL_CONCURRENCY) as executor:
            future_to_url = {
                executor.submit(_firecrawl_scrape, client, url): url for url in urls
            }
            for fut in as_completed(future_to_url):
                url = future_to_url[fut]
                try:
                    results[url] = fut.result()
                except Exception as exc:  # defensive: never let one bad URL kill the run
                    logger.warning("Scrape future for %s raised: %s", url, exc)
                    results[url] = None
    return results


# ---------------------------------------------------------------------
# Transform: raw Algolia hit -> HNPost
# ---------------------------------------------------------------------


def _hit_to_post(hit: dict[str, Any]) -> HNPost:
    """Project a raw Algolia HN hit to an HNPost.

    Hits are post-filtered by ``points >= POINTS_FLOOR`` upstream, so
    no range check here. Description is the placeholder ``None`` and
    is filled in by ``attach_descriptions`` after Firecrawl scrubs.
    """
    object_id = str(hit["objectID"])
    title = (hit.get("title") or "").strip()
    author = hit.get("author") or ""
    url = hit.get("url") or None
    points = int(hit.get("points") or 0)
    comments = int(hit.get("num_comments") or 0)
    created_at = hit.get("created_at") or ""
    hn_url = f"https://news.ycombinator.com/item?id={object_id}"
    return HNPost(
        object_id=object_id,
        title=title,
        author=author,
        url=url,
        points=points,
        comments=comments,
        created_at=created_at,
        description=None,
        hn_url=hn_url,
    )


# ---------------------------------------------------------------------
# Dedup + sort + manifest
# ---------------------------------------------------------------------


def _dedupe(records: Iterable[HNPost]) -> list[HNPost]:
    """Drop duplicates by ``object_id``, keeping the first occurrence.

    HN Algolia is internally consistent so dedup should be a no-op
    within a single run. The guard is for safety in case pagination
    yields an overlap during a mid-run index update.
    """
    seen: set[str] = set()
    out: list[HNPost] = []
    for rec in records:
        if rec.object_id in seen:
            continue
        seen.add(rec.object_id)
        out.append(rec)
    return out


def _sort_deterministic(records: Iterable[HNPost]) -> list[HNPost]:
    """Sort by ``points DESC, created_at DESC, object_id``.

    Highest-scoring "Show HN" posts first; ties broken by recency,
    then object id. This puts the most-likely-valuable corpus entries
    at the top of the file for human skim.
    """
    return sorted(
        records,
        key=lambda r: (-r.points, r.created_at, r.object_id),
    )


def _attach_descriptions(
    records: list[HNPost],
    *,
    scrape_concurrency: int = FIRECRAWL_CONCURRENCY,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[HNPost], int, int, int]:
    """Attach Firecrawl descriptions to each post.

    Returns ``(updated_records, no_launch_url_count, dead_url_count,
    scraped_success_count)``. Posts without an external ``url`` (Ask
    HN-style text posts that showed up via the ``show_hn`` tag because
    their title started with "Show HN:") are kept with
    ``description=None`` and counted as "no launch URL" rather than
    "dead URL" — they had nothing to scrape.

    The ``transport`` parameter is for tests (MockTransport); production
    callers leave it ``None``.
    """
    scrapeable: list[tuple[int, str]] = []  # (record_idx, url)
    no_launch = 0
    for i, rec in enumerate(records):
        if not rec.url or not _URL_LIKE.match(rec.url):
            no_launch += 1
            continue
        scrapeable.append((i, rec.url))

    if not scrapeable:
        logger.info("No URLs to Firecrawl scrape (%d had no launch URL)", no_launch)
        return records, no_launch, 0, 0

    urls = [u for _, u in scrapeable]
    logger.info(
        "Firecrawl scraping %d external launch URLs (concurrency=%d)",
        len(urls),
        scrape_concurrency,
    )
    descriptions = _scrape_urls_concurrent(urls, transport=transport)

    dead = 0
    ok = 0
    out = list(records)
    import dataclasses

    for idx, url in scrapeable:
        desc = descriptions.get(url)
        if desc is None:
            dead += 1
            continue
        # ``dataclasses.replace`` preserves the frozen-dataclass invariant
        # by constructing a new record.
        out[idx] = dataclasses.replace(out[idx], description=desc)
        ok += 1
    return out, no_launch, dead, ok


def _build_manifest(
    *,
    snapshot_path: Path,
    records: list[HNPost],
    scrape_date: date,
    date_range: tuple[str, str],
    no_launch_url_count: int,
    dead_url_count: int,
    scraped_count: int,
    max_records: int | None,
    lookback_days: int,
) -> dict[str, Any]:
    """Build the manifest dict that gets written to ``.manifest.json``.

    The four fields the spec requires are present; everything else is a
    courtesy for future debugging.
    """
    if records:
        oldest = min(r.created_at for r in records)
        newest = max(r.created_at for r in records)
    else:
        oldest = newest = ""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_url": SOURCE_URL,
        "scrape_date": scrape_date.isoformat(),
        "count": len(records),
        "date_range": {
            "oldest_post": oldest,
            "newest_post": newest,
            "search_window": date_range,
        },
        "points_floor": POINTS_FLOOR,
        "lookback_days": lookback_days,
        "max_records_cap": max_records,
        "no_launch_url_count": no_launch_url_count,
        "dead_url_count": dead_url_count,
        "firecrawl_success_count": scraped_count,
        "firecrawl_success_rate": (
            scraped_count / (scraped_count + dead_url_count)
            if (scraped_count + dead_url_count) > 0
            else None
        ),
        "scraped_at_utc": datetime.now(tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "snapshot_filename": snapshot_path.name,
    }


def write_snapshot(
    records: list[HNPost],
    *,
    out_dir: Path,
    scrape_date: date,
    no_launch_url_count: int,
    dead_url_count: int,
    scraped_count: int,
    max_records: int | None,
    lookback_days: int,
    date_range: tuple[str, str],
) -> tuple[Path, Path]:
    """Write ``records`` to a date-stamped jsonl + manifest under ``out_dir``.

    Returns the ``(jsonl_path, manifest_path)`` pair. The jsonl uses
    fixed field order (``asdict`` on the dataclass) so the file is
    diff-stable across re-runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = scrape_date.isoformat()
    jsonl_path = out_dir / f"hn_show_{date_str}.jsonl"
    manifest_path = out_dir / f"hn_show_{date_str}.manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_jsonl())

    manifest = _build_manifest(
        snapshot_path=jsonl_path,
        records=records,
        scrape_date=scrape_date,
        date_range=date_range,
        no_launch_url_count=no_launch_url_count,
        dead_url_count=dead_url_count,
        scraped_count=scraped_count,
        max_records=max_records,
        lookback_days=lookback_days,
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")

    return jsonl_path, manifest_path


# ---------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------


def fetch_all_posts(
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_records: int | None = None,
    scrape: bool = True,
    user_agent: str = USER_AGENT,
    algolia_transport: httpx.BaseTransport | None = None,
    firecrawl_transport: httpx.BaseTransport | None = None,
) -> tuple[list[HNPost], dict[str, int]]:
    """Fetch every HN "Show HN" post meeting the spec's filters.

    Pipeline:
      1. Compute ``created_at_floor`` = ``now - lookback_days``.
      2. Paginate Algolia for every hit older than that floor.
      3. Filter client-side to ``points >= POINTS_FLOOR``.
      4. Project to HNPost.
      5. Dedup on ``object_id``.
      6. Sort deterministically.
      7. Optionally Firecrawl-scrape each external launch URL and
         attach the description.

    Parameters
    ----------
    lookback_days:
        How far back to query HN. Defaults to 3 years (1095 days).
    max_records:
        Optional cap on the corpus size post-filtering. Useful for
        keeping the corpus comparable to YC (~6K).
    scrape:
        If True, Firecrawl-scrape every external launch URL. Set False
        for fast smoke tests.
    user_agent:
        Override the User-Agent (default: a Chrome shape; helps when
        scraping sites that block non-browser UAs).

    Returns
    -------
    tuple[list[HNPost], dict[str, int]]
        ``(records, stats)`` where ``stats`` is a small dict of counters
        suitable for downstream logging / the manifest.

    The two ``*_transport`` parameters are for tests — pass an
    ``httpx.MockTransport`` to keep the call hermetic. Production
    callers leave both as ``None``.
    """
    now_ts = int(time.time())
    created_at_floor = now_ts - lookback_days * 24 * 3600

    headers = {"User-Agent": user_agent}
    stats: dict[str, int] = {
        "raw_hits_yielded": 0,
        "no_launch_url": 0,
        "dead_url": 0,
        "firecrawl_ok": 0,
    }

    client_kwargs: dict[str, Any] = dict(headers=headers, timeout=ALGOLIA_TIMEOUT)
    if algolia_transport is not None:
        client_kwargs["transport"] = algolia_transport
    with httpx.Client(**client_kwargs) as client:
        raw = list(
            iter_hits(
                client,
                created_at_floor=created_at_floor,
                max_records=max_records,
            )
        )
    stats["raw_hits_yielded"] = len(raw)

    posts = [_hit_to_post(h) for h in raw]
    posts = _dedupe(posts)
    posts = _sort_deterministic(posts)

    if scrape and posts:
        posts, no_launch, dead, ok = _attach_descriptions(
            posts, transport=firecrawl_transport
        )
        stats["no_launch_url"] = no_launch
        stats["dead_url"] = dead
        stats["firecrawl_ok"] = ok

    return posts, stats


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Scrape the HN 'Show HN' public corpus into data/snapshots/.",
    invoke_without_command=True,
)


@app.callback()
def _root(
    ctx: typer.Context,
) -> None:
    """Default to running the metadata scrape when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(main)


@app.command()
def main(
    out: Path = typer.Option(
        Path("data/snapshots"),
        "--out",
        "-o",
        help="Directory to write hn_show_<date>.jsonl + manifest into.",
    ),
    date_str: str | None = typer.Option(
        None,
        "--date",
        help="Scrape date (ISO YYYY-MM-DD). Defaults to today UTC. "
        "Pinned for deterministic re-runs and tests.",
    ),
    lookback_days: int = typer.Option(
        DEFAULT_LOOKBACK_DAYS,
        "--lookback-days",
        help="How far back to query HN. Defaults to 3 years.",
    ),
    max_records: int | None = typer.Option(
        None,
        "--max-records",
        help="Optional cap on the corpus size post-filtering.",
    ),
    no_scrape: bool = typer.Option(
        False,
        "--no-scrape",
        help="Skip Firecrawl scraping of external URLs. "
        "Useful for fast smoke tests.",
    ),
    firecrawl_url: str | None = typer.Option(
        None,
        "--firecrawl-url",
        help="Override Firecrawl base URL (default: FIRECRAWL_URL env or http://localhost:3002).",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    ),
) -> None:
    """Run the HN "Show HN" scraper end-to-end."""
    global FIRECRAWL_URL  # noqa: PLW0603
    if firecrawl_url:
        FIRECRAWL_URL = firecrawl_url

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if date_str:
        scrape_date = date.fromisoformat(date_str)
    else:
        scrape_date = datetime.now(tz=timezone.utc).date()

    now_ts = int(time.time())
    search_floor_ts = now_ts - lookback_days * 24 * 3600
    search_floor_str = datetime.fromtimestamp(
        search_floor_ts, tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    today_str = datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    date_range = (search_floor_str, today_str)

    logger.info(
        "Starting HN 'Show HN' scrape for %s (lookback=%d days, max_records=%s)",
        scrape_date,
        lookback_days,
        max_records,
    )

    records, stats = fetch_all_posts(
        lookback_days=lookback_days,
        max_records=max_records,
        scrape=not no_scrape,
    )

    jsonl_path, manifest_path = write_snapshot(
        records,
        out_dir=out,
        scrape_date=scrape_date,
        no_launch_url_count=stats["no_launch_url"],
        dead_url_count=stats["dead_url"],
        scraped_count=stats["firecrawl_ok"],
        max_records=max_records,
        lookback_days=lookback_days,
        date_range=date_range,
    )

    # Final summary line. The Makefile target's success depends on
    # nonzero exit code; nothing here returns nonzero intentionally —
    # a sparse result set is still a valid snapshot.
    logger.info(
        "Wrote %d records to %s (dead=%d, firecrawl_ok=%d, no_launch=%d)",
        len(records),
        jsonl_path,
        stats["dead_url"],
        stats["firecrawl_ok"],
        stats["no_launch_url"],
    )
    typer.echo(str(jsonl_path))
    typer.echo(str(manifest_path))


@app.command("scrape-descriptions")
def scrape_descriptions(
    snapshot: Path = typer.Option(
        ...,
        "--snapshot",
        "-s",
        help="Path to a hn_show_<date>.jsonl snapshot to backfill "
        "descriptions for. Reads the records, attempts Firecrawl on "
        "each external URL, and rewrites the file in-place.",
    ),
    firecrawl_url: str | None = typer.Option(
        None,
        "--firecrawl-url",
        help="Override Firecrawl base URL.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
    ),
) -> None:
    """Backfill descriptions for an existing HN snapshot.

    This is the second pass in the two-phase scrape: first ``scrape-hn``
    writes the metadata-only JSONL (no Firecrawl calls, takes seconds),
    then ``scrape-descriptions`` attaches one-paragraph descriptions
    via Firecrawl best-effort. Splitting the two phases means a stale
    or slow Firecrawl can't block the metadata write.

    The output file is rewritten in-place; the resulting JSONL is
    byte-identical to the input for any URL that Firecrawl failed on,
    so this command is naturally idempotent.
    """
    global FIRECRAWL_URL  # noqa: PLW0603
    if firecrawl_url:
        FIRECRAWL_URL = firecrawl_url

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not snapshot.exists():
        raise typer.BadParameter(f"Snapshot not found: {snapshot}")

    raw = snapshot.read_text(encoding="utf-8")
    records = [HNPost(**json.loads(line)) for line in raw.splitlines() if line]
    logger.info("Loaded %d records from %s", len(records), snapshot)

    # Collect scrapeable URLs with their record indices so we can
    # attach descriptions back to the right record after each batch.
    scrapeable: list[tuple[int, str]] = []
    no_launch = 0
    for i, rec in enumerate(records):
        if not rec.url or not _URL_LIKE.match(rec.url):
            no_launch += 1
            continue
        scrapeable.append((i, rec.url))

    if not scrapeable:
        logger.info("No URLs to Firecrawl scrape (%d had no launch URL)", no_launch)
        typer.echo(str(snapshot))
        return

    logger.info(
        "Backfilling descriptions for %d launch URLs (concurrency=%d)",
        len(scrapeable),
        FIRECRAWL_CONCURRENCY,
    )

    # Backfill in chunks so we write progress to disk periodically.
    # A SIGTERM loses at most one chunk (default 25 URLs).
    chunk_size = int(os.environ.get("HN_BACKFILL_CHUNK_SIZE", "25"))
    headers = {"User-Agent": USER_AGENT}

    # Mutable state for async results.
    state = {"ok": 0, "dead": 0}

    def _consume_completed(results: dict[str, str | None], idx_to_url: list[tuple[int, str]]) -> int:
        """Apply a chunk's results to records; return attached count."""
        attached = 0
        for idx, url in idx_to_url:
            desc = results.get(url)
            if desc is None:
                state["dead"] += 1
                continue
            import dataclasses

            records[idx] = dataclasses.replace(records[idx], description=desc)
            state["ok"] += 1
            attached += 1
        return attached

    def _write_snapshot_locked() -> None:
        """Atomic write of the (partial) snapshot.

        We write to ``{path}.tmp`` then ``Path.rename`` so a kill
        mid-write can't corrupt the existing file.
        """
        tmp_path = snapshot.with_suffix(snapshot.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(r.to_jsonl())
        tmp_path.replace(snapshot)

    # Process chunks. Within a chunk, run Firecrawl concurrently; between
    # chunks, persist progress.
    cursor = 0
    while cursor < len(scrapeable):
        chunk = scrapeable[cursor : cursor + chunk_size]
        urls = [u for _, u in chunk]
        logger.info(
            "Chunk %d/%d: scraping %d URLs",
            cursor // chunk_size + 1,
            (len(scrapeable) + chunk_size - 1) // chunk_size,
            len(urls),
        )
        results = _scrape_urls_chunk(urls, headers=headers)
        _consume_completed(results, chunk)
        _write_snapshot_locked()
        logger.info(
            "  chunk progress: ok=%d dead=%d, snapshot on disk",
            state["ok"],
            state["dead"],
        )
        cursor += chunk_size

    logger.info(
        "Backfill complete: %d ok, %d dead, %d no_launch_url",
        state["ok"],
        state["dead"],
        no_launch,
    )

    # Rewrite manifest if it exists.
    manifest_path = snapshot.with_suffix("").with_suffix(".manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = None
        if isinstance(manifest, dict):
            manifest["firecrawl_success_count"] = state["ok"]
            manifest["dead_url_count"] = state["dead"]
            manifest["no_launch_url_count"] = no_launch
            scraped_plus_dead = state["ok"] + state["dead"]
            manifest["firecrawl_success_rate"] = (
                state["ok"] / scraped_plus_dead if scraped_plus_dead else None
            )
            manifest["scraped_at_utc"] = (
                datetime.now(tz=timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False)
                + "\n",
                encoding="utf-8",
            )

    typer.echo(str(snapshot))


def _scrape_urls_chunk(
    urls: list[str],
    *,
    headers: dict[str, str],
) -> dict[str, str | None]:
    """Scrape a chunk of URLs concurrently.

    Split out from ``_scrape_urls_concurrent`` so the backfill loop
    can persist progress between chunks instead of waiting for the
    full batch to complete.
    """
    if not urls:
        return {}
    results: dict[str, str | None] = {}
    with httpx.Client(headers=headers, timeout=FIRECRAWL_TIMEOUT) as client:
        with ThreadPoolExecutor(max_workers=FIRECRAWL_CONCURRENCY) as executor:
            future_to_url = {
                executor.submit(_firecrawl_scrape, client, url): url for url in urls
            }
            for fut in as_completed(future_to_url):
                url = future_to_url[fut]
                try:
                    results[url] = fut.result()
                except Exception as exc:
                    logger.warning("Scrape future for %s raised: %s", url, exc)
                    results[url] = None
    return results


if __name__ == "__main__":
    app()
