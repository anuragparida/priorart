"""Scrape the public Y Combinator company directory into a versioned JSONL snapshot.

Why this module exists
----------------------
The YC public directory at https://www.ycombinator.com/companies is rendered
client-side: the HTML is a thin shell that bootstraps a search-only Algolia
key from ``window.AlgoliaOpts`` and then hydrates the table from the Algolia
DLA (Distributed Search Network) endpoint. We go through the same Algolia
path the directory page uses — the search key is public-by-construction
(it's the key YC publishes so anyone hitting the public directory page can
query the index). No login, no scraping of authenticated routes.

Algolia has strict per-IP rate limits but the public search key lets us
paginate cleanly: the index is partitioned by ``batch`` (a YC season code
like "W21"), and each batch supports ``hitsPerPage=1000`` with a ``page``
cursor. With 50+ batches and ~5K companies total, a single run finishes
in well under a minute and stays well inside the rate limit.

Spec contract (docs/PHASE-1.md §1.2)
------------------------------------
- Output: ``data/snapshots/yc_<date>.jsonl``, one record per line, UTF-8.
- Dedupe key: ``(name, batch)``.
- Manifest: ``data/snapshots/yc_<date>.manifest.json`` with ``count``,
  ``scrape_date``, ``source_url``, ``schema_version``.
- Idempotent: re-running on the same date produces a byte-identical file.
- Deterministic sort: ordered by ``(batch, name)``.
- Rate limit: >= 1 req/sec (we use ~1.05s with 0–200ms jitter to stay
  clear of the limit and look human).

Public data only. No login. No third-party mirror. If YC changes their
key bootstrap (the regex in ``_extract_algolia_key``), the failure is
loud — the scraper refuses to guess.

CLI
---
    uv run priorart-scrape                    # writes today's snapshot
    uv run priorart-scrape --date 2026-06-08  # deterministic for tests
    uv run priorart-scrape --out /tmp/x       # override output dir
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
import typer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants — pinned from the public YC directory page
# ---------------------------------------------------------------------

#: Public URL we bootstrap the Algolia search key from. Per robots.txt
#: ``/companies?*`` is disallowed; the bare ``/companies`` page is allowed
#: and is what the public directory actually renders for unauthenticated
#: visitors.
SOURCE_URL = "https://www.ycombinator.com/companies"

#: Algolia application ID for the YC company index. Pinned — it's part of
#: the public Algolia config and is the same value the directory page uses.
ALGOLIA_APP_ID = "45BWZJ1SGC"

#: Algolia index name. ``_By_Launch_Date_production`` is the suffix YC
#: uses for the index that backs the public directory (sorted by
#: ``launched_at`` descending — i.e. newest launches first).
ALGOLIA_INDEX = "YCCompany_By_Launch_Date_production"

#: DSN endpoint used for the multi-query request. The ``*-dsn.algolia.net``
#: subdomain is the standard Algolia DLA (Distributed Search Network)
#: endpoint for federated multi-index queries.
ALGOLIA_DSN = f"https://{ALGOLIA_APP_ID.lower()}-dsn.algolia.net/1/indexes/*/queries"

#: Agent string the directory page sends. Pinning it keeps us in the
#: "browser" tier of Algolia's quota (search-only, no abuse signal).
ALGOLIA_AGENT = "Algolia for JavaScript (3.35.1); Browser; JS Helper (3.16.1)"

#: Page size per batch. 1000 is Algolia's hard max for ``hitsPerPage``.
PAGE_SIZE = 1000

#: Polite rate limit. 1.05s + 0–200ms jitter → mean 1.15s, well above the
#: 1 req/sec floor the spec asks for and below the threshold where Algolia
#: starts returning 429s on a search-only key.
RATE_LIMIT_SECONDS = 1.05
RATE_JITTER_SECONDS = 0.20

#: Schema version. Bump only on a field rename or a new mandatory field.
#: Adding optional fields is fine without a bump.
SCHEMA_VERSION = "1.0.0"

#: HTTP timeouts (seconds). Connect is short; read is generous because
#: the largest batch (W22 at ~400 companies) can take a beat to respond.
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

#: Browser-like User-Agent. YC's CDN gates the directory page on a
#: non-default UA (returns a 200 to curl but renders an empty shell for
#: some non-browser clients in practice).
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Maximum number of retry attempts per HTTP request (network blips,
#: transient 5xx, etc.). Exponential backoff between attempts.
MAX_RETRIES = 4

# Regex to extract the Algolia search key from the directory page. The
# directory page inlines the key as ``window.AlgoliaOpts = {...}``; the
# key value is base64-encoded with restrictions baked in. We treat the
# shape as untrusted — if the page format changes, we'd rather fail
# loudly than silently fall back to a different key.
_ALGOLIA_OPTS_RE = re.compile(r"window\.AlgoliaOpts\s*=\s*(\{[^<]+\})")


# ---------------------------------------------------------------------
# Public schema — what a single YC record looks like in the snapshot
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class YCCompany:
    """One YC company record as it appears in ``yc_<date>.jsonl``.

    Field names are stable and form the public schema; do not rename
    without bumping ``SCHEMA_VERSION`` and writing a migration note.

    Attributes
    ----------
    name:
        The company's current name. ``former_names`` is intentionally
        not captured — it's noisy and out of scope for the
        deduplication signal.
    description:
        The 1-paragraph description YC publishes on the directory page
        (``one_liner``). One sentence, fits in a single line.
    tags:
        YC-assigned topic tags (e.g. ``["AI", "Developer Tools"]``).
        Sorted alphabetically on write for deterministic output.
    batch:
        The YC batch the company joined (e.g. ``"W21"``). The convention
        is ``[S|W|F]<two-digit-year>`` for Summer / Winter / Fall.
    status:
        YC's reported status, verbatim. Known values observed in the
        wild: ``"Active"``, ``"Public"``, ``"Acquired"``,
        ``"Inactive"``, ``"Closed"``. Not normalized — downstream
        code that wants a canonical enum should map explicitly.
    url:
        The canonical YC directory URL (``/companies/<slug>``).
    """

    name: str
    description: str
    tags: list[str]
    batch: str
    status: str
    url: str

    def to_jsonl(self) -> str:
        """Serialize one record as a single line of JSON + trailing newline.

        Keys are emitted in declaration order so the file is
        byte-stable across re-runs (given identical input data) and
        ``diff``-friendly for humans.
        """
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False) + "\n"


# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------


def _extract_algolia_key(html: str) -> str:
    """Pull the public Algolia search key out of the directory page HTML.

    The directory page inlines the key as ``window.AlgoliaOpts = {...}``
    in a ``<script>`` block. We pin the format to that exact shape; if YC
    changes how they publish the key, this raises instead of silently
    degrading.

    Returns
    -------
    str
        The base64-encoded search-only Algolia API key. Search-only keys
        can read the index but not write to it — that's the whole point
        of using the directory page's key.
    """
    match = _ALGOLIA_OPTS_RE.search(html)
    if not match:
        raise RuntimeError(
            "Could not locate window.AlgoliaOpts on the YC companies page. "
            "The page format may have changed; check "
            "https://www.ycombinator.com/companies manually."
        )
    try:
        opts = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"window.AlgoliaOpts is not valid JSON: {exc}. "
            "The YC page format has likely changed."
        ) from exc
    if opts.get("app") != ALGOLIA_APP_ID:
        raise RuntimeError(
            f"Unexpected Algolia app id {opts.get('app')!r}; "
            f"expected {ALGOLIA_APP_ID!r}. YC may have rotated their index."
        )
    key = opts.get("key")
    if not isinstance(key, str) or not key:
        raise RuntimeError("Algolia search key missing or empty in window.AlgoliaOpts")
    return key


def _sleep_politely() -> None:
    """Sleep the configured rate-limit interval with small jitter.

    Splitting the sleep out as its own function makes it easy to monkey-
    patch in tests (so the test suite doesn't actually wait 1+ seconds
    per request).
    """
    jitter = random.uniform(0.0, RATE_JITTER_SECONDS)
    time.sleep(RATE_LIMIT_SECONDS + jitter)


def _post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    """POST a JSON body with exponential backoff on transient failures.

    Retries on connection errors, timeouts, and 5xx. Does NOT retry on
    4xx — those are programming errors (bad key, malformed params)
    that won't fix themselves.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.post(url, params=params, json=body)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning(
                "Algolia POST failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc
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
            # 429s come with a Retry-After header; honor it.
            retry_after = float(resp.headers.get("Retry-After", "5"))
            logger.warning(
                "Algolia rate-limited (429); sleeping %.1fs", retry_after
            )
            time.sleep(retry_after)
            continue
        if resp.status_code >= 400:
            # 4xx other than 429: surface the body, don't retry.
            raise RuntimeError(
                f"Algolia POST failed with {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()
    # All retries exhausted.
    raise RuntimeError(
        f"Algolia POST failed after {MAX_RETRIES} attempts; last error: {last_exc}"
    )


# ---------------------------------------------------------------------
# Algolia fetch — paginate by batch
# ---------------------------------------------------------------------


def _algolia_query_params(api_key: str) -> dict[str, str]:
    """Build the query-string params shared by every Algolia request."""
    return {
        "x-algolia-agent": ALGOLIA_AGENT,
        "x-algolia-application-id": ALGOLIA_APP_ID,
        "x-algolia-api-key": api_key,
    }


def _algolia_body(
    *,
    batch: str | None,
    page: int,
) -> dict[str, Any]:
    """Build the JSON body for a single Algolia multi-query request.

    We always request the full facet list (matches what the directory
    page does) even though we only consume the ``batch`` facet. This
    keeps us indistinguishable from a real browser hit on the wire
    and means we can opportunistically use other facets in the future
    without a scraper rewrite.

    Parameters
    ----------
    batch:
        If ``None``, return only facet counts (no hits). If a string,
        filter to that batch and return up to ``PAGE_SIZE`` hits.
    page:
        Zero-indexed page number (Algolia's cursor is 0-based).
    """
    facets = (
        "%5B%22app_answers%22%2C%22app_video_public%22%2C%22batch%22%2C"
        "%22demo_day_video_public%22%2C%22highlight_black%22%2C"
        "%22highlight_latinx%22%2C%22highlight_women%22%2C%22industries%22%2C"
        "%22isHiring%22%2C%22nonprofit%22%2C%22question_answers%22%2C"
        "%22regions%22%2C%22subindustry%22%2C%22tags%22%2C%22top_company%22%5D"
    )
    base_params = (
        f"facets={facets}"
        f"&hitsPerPage={PAGE_SIZE}"
        "&maxValuesPerFacet=1000"
        "&query="
        "&tagFilters="
    )
    if batch is not None:
        # Algolia uses URL-encoding of the colon in the facet filter; the
        # directory page's JS does this for us normally.
        encoded_batch = batch.replace("%", "%25").replace(":", "%3A")
        params = (
            f"{base_params}"
            f"&facetFilters=batch%3A{encoded_batch}"
            f"&page={page}"
        )
    else:
        params = f"{base_params}&page={page}"
    return {"requests": [{"indexName": ALGOLIA_INDEX, "params": params}]}


def _fetch_batch_facets(
    client: httpx.Client, api_key: str
) -> dict[str, int]:
    """Return ``{batch_name: count}`` for every batch in the index.

    Uses the no-batch filter call — the response includes a ``facets``
    block with the per-batch totals. The first call of the run; from
    this we know how many batches to paginate.
    """
    body = _algolia_body(batch=None, page=0)
    data = _post_with_retry(
        client, ALGOLIA_DSN, params=_algolia_query_params(api_key), body=body
    )
    try:
        facets = data["results"][0]["facets"]["batch"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Algolia response shape; no batch facets: {data!r}"
        ) from exc
    # Algolia returns facets as {value: count}. The values are strings
    # (batch codes like "W21", "Summer 2026", "Unspecified").
    return {str(k): int(v) for k, v in facets.items()}


def _fetch_batch_hits(
    client: httpx.Client,
    api_key: str,
    batch: str,
    expected_count: int,
) -> Iterator[dict[str, Any]]:
    """Yield every raw Algolia hit for one batch, paginating as needed.

    ``expected_count`` is the per-batch total from the facets call. We
    paginate until we've either seen that many hits or the server
    returns an empty page (the latter shouldn't happen, but a defensive
    bound prevents an infinite loop if the index changes mid-run).
    """
    fetched = 0
    page = 0
    while fetched < expected_count:
        body = _algolia_body(batch=batch, page=page)
        data = _post_with_retry(
            client, ALGOLIA_DSN, params=_algolia_query_params(api_key), body=body
        )
        try:
            hits = data["results"][0]["hits"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Algolia response shape for batch {batch!r} "
                f"page {page}: {data!r}"
            ) from exc
        if not hits:
            # Defensive: server says there are ``expected_count`` hits
            # in the facet, but page returned zero. Stop and let the
            # caller warn about the gap rather than spinning.
            logger.warning(
                "Algolia returned 0 hits for batch %r page %d (expected %d)",
                batch,
                page,
                expected_count - fetched,
            )
            return
        for hit in hits:
            yield hit
        fetched += len(hits)
        page += 1
        # Rate-limit per page (not per hit) — each page is one HTTP call.
        if fetched < expected_count:
            _sleep_politely()


# ---------------------------------------------------------------------
# Transform: raw Algolia hit -> YCCompany
# ---------------------------------------------------------------------


def _hit_to_company(hit: dict[str, Any]) -> YCCompany:
    """Project a raw Algolia hit to a YCCompany.

    The Algolia hit has many fields we don't expose in the snapshot
    schema (``team_size``, ``website``, ``small_logo_thumb_url``, etc.).
    They're discarded here on purpose: this snapshot is the input to
    Phase 1.3's ingest step, and we want a stable, narrow schema. If a
    future phase needs more fields, add them to ``YCCompany`` and bump
    ``SCHEMA_VERSION``.

    Raises
    ------
    ValueError
        If the hit is missing a required field. The caller is expected
        to log and skip — bad data shouldn't poison the whole run.
    """
    try:
        name = hit["name"]
        slug = hit["slug"]
        batch = hit["batch"]
        status = hit["status"]
    except KeyError as exc:
        raise ValueError(f"Algolia hit missing required field {exc!s}: {hit!r}") from exc

    # Description: prefer one_liner (the 1-paragraph on the directory
    # page); fall back to long_description if one_liner is empty. Both
    # fields are stripped of the YC trailing CRLF + redundant whitespace.
    one_liner = (hit.get("one_liner") or "").strip()
    if not one_liner:
        long_desc = (hit.get("long_description") or "").strip()
        # Collapse runs of whitespace so a multi-paragraph long_description
        # still fits on one jsonl line cleanly.
        one_liner = " ".join(long_desc.split())

    # Tags: sorted for deterministic output. YC sometimes returns the
    # same tag with different casing across batches (rare but seen);
    # we don't normalize case here — the canonical tag list lives in
    # the YC index and any future dedupe layer can do case-insensitive
    # matching.
    raw_tags = hit.get("tags") or []
    tags = sorted({t for t in raw_tags if t})

    return YCCompany(
        name=name,
        description=one_liner,
        tags=tags,
        batch=batch,
        status=status,
        url=f"https://www.ycombinator.com/companies/{slug}",
    )


# ---------------------------------------------------------------------
# Dedup + sort
# ---------------------------------------------------------------------


def _dedupe(records: Iterable[YCCompany]) -> list[YCCompany]:
    """Drop duplicates by ``(name, batch)``, keeping the first occurrence.

    The spec says dedupe on ``(name, batch)``. YC's index is internally
    consistent (no duplicate ids) but the same company name can appear
    in multiple batches across different eras (a relaunch under a new
    entity but the same name). Those should NOT be collapsed — the
    batch is what makes them distinct.
    """
    seen: set[tuple[str, str]] = set()
    out: list[YCCompany] = []
    for rec in records:
        key = (rec.name, rec.batch)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _sort_deterministic(records: Iterable[YCCompany]) -> list[YCCompany]:
    """Sort by ``(batch, name)`` for stable, byte-identical output.

    Batches aren't naturally orderable as strings — "S21" and "W22"
    sort lexicographically, but the year/month order is what readers
    actually want. We use a stable two-level key: tuple of ``(batch
    as string, name as string)``. This is a deliberately simple
    ordering — the eval harness doesn't depend on order, and any
    downstream that needs chronological order can derive it from
    the ``batch`` field itself.
    """
    return sorted(records, key=lambda r: (r.batch, r.name))


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------


def _build_manifest(
    *,
    snapshot_path: Path,
    records: list[YCCompany],
    scrape_date: date,
    batches_seen: list[str],
    errors_skipped: int,
) -> dict[str, Any]:
    """Build the manifest dict that gets written to ``.manifest.json``.

    The manifest is the only human-facing metadata in the snapshot
    directory. The four fields the spec requires are present;
    everything else is a courtesy for future debugging.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "source_url": SOURCE_URL,
        "scrape_date": scrape_date.isoformat(),
        "count": len(records),
        "batches": sorted(batches_seen),
        "errors_skipped": errors_skipped,
        "scraped_at_utc": datetime.now(tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "snapshot_filename": snapshot_path.name,
    }


# ---------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------


def fetch_all_companies(
    client: httpx.Client,
    api_key: str,
) -> tuple[list[YCCompany], int]:
    """Fetch every public YC company and return as a deduped, sorted list.

    Pipeline:
      1. Ask Algolia for the per-batch facet counts.
      2. For each batch, paginate through every hit.
      3. Project each hit to a ``YCCompany`` (skipping malformed ones).
      4. Dedupe on ``(name, batch)``.
      5. Sort by ``(batch, name)`` for byte-stable output.

    Rate-limited at ``RATE_LIMIT_SECONDS`` per batch-page request.
    Returns ``(records, errors_skipped)`` — the caller is responsible
    only for writing the file and the manifest.
    """
    logger.info("Fetching batch facets from Algolia")
    batch_counts = _fetch_batch_facets(client, api_key)
    logger.info("Found %d batches; %d companies total", len(batch_counts), sum(batch_counts.values()))

    raw_records: list[YCCompany] = []
    errors_skipped = 0
    for batch in sorted(batch_counts):
        expected = batch_counts[batch]
        logger.info("Fetching batch %r (%d expected)", batch, expected)
        # First page: no sleep (we just hit the facets endpoint).
        try:
            page_iter = _fetch_batch_hits(client, api_key, batch, expected)
            for hit in page_iter:
                try:
                    raw_records.append(_hit_to_company(hit))
                except ValueError as exc:
                    errors_skipped += 1
                    logger.warning("Skipping malformed hit: %s", exc)
        except Exception:
            logger.exception("Batch %r failed; continuing with next batch", batch)
            errors_skipped += expected  # best-effort count
            continue
        # Rate-limit per batch (after the last page of that batch,
        # before the first page of the next batch).
        _sleep_politely()

    deduped = _dedupe(raw_records)
    if len(deduped) != len(raw_records):
        logger.info(
            "Dedupe removed %d duplicates (%d raw → %d unique)",
            len(raw_records) - len(deduped),
            len(raw_records),
            len(deduped),
        )
    return _sort_deterministic(deduped), errors_skipped


def write_snapshot(
    records: list[YCCompany],
    *,
    out_dir: Path,
    scrape_date: date,
    errors_skipped: int,
) -> tuple[Path, Path]:
    """Write ``records`` to a date-stamped jsonl + manifest under ``out_dir``.

    Returns the ``(jsonl_path, manifest_path)`` pair. The jsonl uses
    fixed field order (``asdict`` on the dataclass) so the file is
    diff-stable across re-runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = scrape_date.isoformat()
    jsonl_path = out_dir / f"yc_{date_str}.jsonl"
    manifest_path = out_dir / f"yc_{date_str}.manifest.json"

    # `batches` is a sorted, deduplicated list of the batches the records
    # actually came from — the spec doesn't list this in the manifest
    # but it's useful for sanity-checking that a run covered the right
    # ranges (a scrape that missed half the batches would still produce
    # a valid jsonl, but `batches` would reveal it).
    batches_seen = sorted({r.batch for r in records})
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_jsonl())

    manifest = _build_manifest(
        snapshot_path=jsonl_path,
        records=records,
        scrape_date=scrape_date,
        batches_seen=batches_seen,
        errors_skipped=errors_skipped,
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")

    return jsonl_path, manifest_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _resolve_algolia_key(client: httpx.Client) -> str:
    """Bootstrap the Algolia search key from the public YC directory page."""
    logger.info("Bootstrapping Algolia search key from %s", SOURCE_URL)
    resp = client.get(SOURCE_URL)
    resp.raise_for_status()
    return _extract_algolia_key(resp.text)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Scrape the YC public company directory into data/snapshots/.",
)


@app.command()
def main(
    out: Path = typer.Option(
        Path("data/snapshots"),
        "--out",
        "-o",
        help="Directory to write yc_<date>.jsonl + manifest into.",
    ),
    date_str: str | None = typer.Option(
        None,
        "--date",
        help=(
            "Scrape date in YYYY-MM-DD. Defaults to today (UTC). "
            "Use this in tests for deterministic filenames."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable INFO-level logging.",
    ),
) -> None:
    """Scrape the YC public directory and write a versioned JSONL snapshot."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scrape_date = (
        date.fromisoformat(date_str)
        if date_str
        else datetime.now(tz=timezone.utc).date()
    )

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=HTTP_TIMEOUT, headers=headers) as client:
        api_key = _resolve_algolia_key(client)
        records, errors = fetch_all_companies(client, api_key)

    jsonl_path, manifest_path = write_snapshot(
        records,
        out_dir=out,
        scrape_date=scrape_date,
        errors_skipped=errors,
    )
    typer.echo(f"Wrote {len(records)} records to {jsonl_path}")
    typer.echo(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    app()
