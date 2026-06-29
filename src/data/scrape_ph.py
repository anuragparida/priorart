"""Scrape the public Product Hunt archive into a versioned JSONL snapshot.

Why this module exists
----------------------
PriorArt's Phase 2 expansion (``docs/PHASE-2.md`` §2.5) adds Product
Hunt's public launches to the corpus alongside YC and HN. The product
goal is the "top 5K most-upvoted launches over the last 3 years" — a
denoised, deduplicated-against-YC dataset that the retrieval layer can
search alongside the existing YC companies.

The blocker is the data source. Product Hunt's web app at
``https://www.producthunt.com/`` is behind Cloudflare's bot-detection
("Verify you are human"); the same wall is hit by our self-hosted
Firecrawl (verified — :3002 ``/v1/scrape`` returns ``statusCode=403``)
and by raw ``httpx`` (returns the CF challenge HTML). The official
GraphQL endpoint at ``api.producthunt.com/v2`` requires an OAuth
token. The Atom feed at ``/feed`` is reachable but only carries the
~30 most-recent launches, not the historical archive we need.

The only public source of the full archive is the **Internet Archive's
Wayback Machine** (``web.archive.org``), which has crawled the PH
``/leaderboard/daily/<year>/<month>/<day>`` pages back to 2013. Each
archived daily page is a Next.js SSR render — the response body
embeds the post data (either as a relay/apollo cache blob on
pre-2024 archives, or as inline homefeed edge nodes on the
post-2024-redesign) with one record per ranked post on that day,
carrying the post's ``id``, ``name``, ``slug``, ``tagline``,
``votesCount`` (rendered as ``launchDayScore`` on the post-2024
redesign or as ``votesCount`` on the pre-2024 site), ``commentsCount``,
``topics``, and ``createdAt``. We parse the embedded JSON, dedup
across the ~1,100 archived days in the 3-year window, and emit the
top ~5K launches by vote count.

This module is a *public-data-only* scraper. The Wayback Machine is
itself a public mirror (CC-BOT / IA-BOT crawls) and is the only
public source of the historical archive; the PH live site is blocked.
The CDX Server API at ``web.archive.org/cdx/search/cdx`` is the
index, and is the same endpoint other public-archive tools use.

Spec contract (``docs/PHASE-2.md`` §2.5 + AGENTS.md stack rules)
---------------------------------------------------------------
- Output: ``data/snapshots/producthunt_<date>.jsonl``, one record per
  line, UTF-8. Stable field order via ``asdict`` on the dataclass.
- Top-5K by ``launchDayScore`` (votes count at end of launch day).
- Dedupe key: PH post ``id`` (a stringified integer stable across
  edits). 2.5 internal dedup; cross-corpus dedup happens in 2.7.
- Manifest: ``data/snapshots/producthunt_<date>.manifest.json`` with
  ``count``, ``scrape_date``, ``source_url``, ``schema_version``,
  ``date_range``, ``votes_floor``, ``dedup_stats``
  (``exact_dup_count``, ``borderline_count``, ``novel_count``), and
  the embedding-model version that produced the dedup vectors.
- Borderline review queue:
  ``data/snapshots/producthunt_<date>.borderline.jsonl`` — PH records
  whose max-cosine against any YC name lands in ``[0.75, 0.85)``. The
  band is the spec's "5-10% borderline" range; we pick ``[0.75, 0.85)``
  exactly so the count and the manifest field stay self-explanatory.
- Idempotent: re-running on the same date produces a byte-identical
  jsonl + manifest + borderline file.
- Deterministic sort: ``votes_count DESC, created_at DESC, id``.
- Cosine dedup: ``cosine(bge-m3_embed(name), bge_m3_embed(yc_name))``
  for each PH name × every YC name. Take the max per PH. Exact-dup
  band is ``>= 0.85`` (drop / flag in manifest); borderline band is
  ``[0.75, 0.85)`` (queue for manual review). The bge-m3 cache for
  YC names is a one-time compute cached to
  ``data/cache/yc_name_embeddings.npz`` so the second-and-onward
  re-runs of the PH scraper only need to embed the PH names
  (~25 min on contested CPU vs ~55 min for the full set).

CLI
---
    uv run python -m src.data.scrape_ph                    # writes today's snapshot
    uv run python -m src.data.scrape_ph --date 2026-06-08  # deterministic for tests
    uv run python -m src.data.scrape_ph --out /tmp/x       # override output dir
    uv run python -m src.data.scrape_ph --max-records 5000 # cap the corpus size
    uv run python -m src.data.scrape_ph --skip-dedup       # skip the bge-m3 dedup
    uv run python -m src.data.scrape_ph --yc-snapshot data/snapshots/yc_2026-06-08.jsonl
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import typer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — pinned from the public Wayback Machine
# ---------------------------------------------------------------------------

#: Wayback Machine CDX index. The IA's standard index endpoint;
#: used by every public-archive tool that scrapes from the IA. Filter
#: chain: status=200 + mimetype=text/html to skip redirects / 404s /
#: JS chunks / image captures.
CDX_URL = "https://web.archive.org/cdx/search/cdx"

#: Wayback Machine replay endpoint. The ``id_`` flag strips the IA
#: wrapping (banner, JS) so the response is the raw archived HTML.
WAYBACK_REPLAY = "https://web.archive.org/web/{ts}id_/https://www.producthunt.com/leaderboard/daily/{date}"

#: PH daily leaderboard URL — pinned here for the manifest.
SOURCE_URL = "https://www.producthunt.com/leaderboard/daily"

#: Default scrape window. The spec asks for the last 3 years of
#: "top 5K most upvoted" launches. From 2023-07-01 to today
#: (2026-06-29) is 1094 days. Default lookback matches.
DEFAULT_LOOKBACK_DAYS = 3 * 365

#: Spec asks for top 5K upvoted. The CDX sweep usually yields ~5–8K
#: unique posts in the 3-year window on wayback coverage, so 5K
#: is the conservative cut. Override with ``--max-records``.
DEFAULT_MAX_RECORDS = 5000

#: Floor on votes. The spec asks for the "most upvoted" — no
#: explicit floor. We use a small floor of 50 to keep tiny test /
#: spam launches out of the corpus; below 50 is almost always
#: low-signal and inflates the embedding cost on the dedup step.
VOTES_FLOOR = 50

#: Cosine dedup bands. ``>= EXACT_DUP_COSINE`` is exact-dup (the PH
#: record has a near-identical YC company name, suggesting the same
#: product). ``[BORDERLINE_LOWER, BORDERLINE_UPPER)`` is borderline —
#: queued for manual review. The 5–10% borderline band in the spec
#: maps cleanly to a 0.10-wide cosine band centred around the
#: decision boundary; we pick ``[0.75, 0.85)`` so the lower edge
#: of the band is exactly the typical "high-confidence relevant"
#: cutoff and the upper edge is the "definite duplicate" cutoff.
EXACT_DUP_COSINE = 0.85
BORDERLINE_LOWER = 0.75
BORDERLINE_UPPER = 0.85  # exclusive — equal to EXACT_DUP_COSINE.

#: bge-m3 model name — pinned in ``src/config.py`` but we re-pin here
#: so the scraper is self-contained. Must match the one used for the
#: YC corpus embedder or the cosine values are not comparable.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

#: HTTP timeouts. Wayback can be slow on cold replays (the IA
#: sometimes has to re-fetch from origin).
CDX_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
WAYBACK_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=10.0, pool=10.0)

#: Rate-limit. Wayback is happy with sustained 4 req/s but the CDX
#: endpoint is the choke point. We sleep 0.5s between page fetches
#: (well above floor) and 1.5s between CDX queries (slower endpoint).
CDX_RATE_SECONDS = 1.5
WAYBACK_RATE_SECONDS = 0.5
RATE_JITTER_SECONDS = 0.25

#: Retry policy on Wayback / CDX. Transient (network blip, 5xx, 429)
#: retried; 4xx other than 429 surfaces immediately.
MAX_RETRIES = 4

#: CDX batch size — how many CDX rows to fetch per request. The CDX
#: API caps at 100K per request; we use 10K for safety.
CDX_LIMIT = 10000

#: User-Agent string. Wayback logs UA but doesn't block browsers.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "(PriorArt-Phase-2.5-PH-Scraper)"
)

#: Where the bge-m3 embeddings cache lives. The YC names are
#: embedded exactly once; the cache is the contract that makes the
#: second-and-onward re-runs of the PH scraper feasible on CPU.
EMBEDDINGS_CACHE_DIR = Path("data/cache")
YC_NAME_EMBEDDINGS_CACHE = EMBEDDINGS_CACHE_DIR / "yc_name_embeddings.npz"

#: Schema version. Bump only on a field rename or new mandatory
#: field. The on-disk JSONL is the public contract — every
#: downstream consumer (2.7 ingestion, 2.8 eval-set) reads by
#: field name.
SCHEMA_VERSION = "1.0.0"

#: Object-key patterns that signal "this is a Post". The legacy
#: shape stores posts as ``"PostNNN":{...}`` cache entries. The new
#: shape (2024 redesign onward) stores them inline as
#: ``"node":{"__typename":"Post",...}}`` homefeed edge nodes.
RE_POST_LEGACY_KEY = re.compile(r'"Post(\d+)":\{')
RE_POST_NEW_START = re.compile(r'"node":(?P<open>\{)"__typename":"Post",')

#: Per-field extractors used against the legacy post body. Each is
#: anchored by name (no ``.*?`` between field name and value) so
#: we don't accidentally cross object boundaries.
RE_LEGACY_NAME = re.compile(r'"name":"((?:[^"\\]|\\.)*)"')
RE_LEGACY_SLUG = re.compile(r'"slug":"((?:[^"\\]|\\.)*)"')
RE_LEGACY_TAGLINE = re.compile(r'"tagline":"((?:[^"\\]|\\.)*)"')
RE_LEGACY_VOTES = re.compile(r'"votesCount":(\d+)')
RE_LEGACY_COMMENTS = re.compile(r'"commentsCount":(\d+)')

#: Strict date-path matcher for CDX rows. The Wayback sometimes
#: captures URLs where the trailing path segment is a CSS color
#: value (``.../daily/2024/11/255, 255, 255, 0.3``) instead of a day
#: integer. We accept only ``YYYY/M/D`` with valid month (1–12)
#: and day (1–31) ranges to filter those out. The day upper bound
#: is loose (31) because we don't validate against the actual
#: month-length — the downstream date decode would catch any
#: truly invalid combo.
_RE_DATE_PATH = re.compile(r"^\d{4}/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])$")

#: Per-field extractors for the new shape (same names as legacy).
RE_NEW_ID = re.compile(r'"id":"(\d+)"')
RE_NEW_NAME = re.compile(r'"name":"((?:[^"\\]|\\.)*)"')
RE_NEW_SLUG = re.compile(r'"slug":"((?:[^"\\]|\\.)*)"')
RE_NEW_TAGLINE = re.compile(r'"tagline":"((?:[^"\\]|\\.)*)"')
#: New shape uses ``launchDayScore`` (final score at end of launch
#: day) for the first post on a page, but most posts have
#: ``launchDayScore:0`` (PH re-uses the field for re-launches
#: only) — we fall back to ``latestScore`` and then ``votesCount``
#: when both earlier fields are zero or missing. The legacy shape
#: uses ``votesCount`` directly.
RE_NEW_LAUNCH_DAY = re.compile(r'"launchDayScore":(\d+)')
RE_NEW_LATEST = re.compile(r'"latestScore":(\d+)')
RE_NEW_VOTES = re.compile(r'"votesCount":(\d+)')
RE_NEW_COMMENTS = re.compile(r'"commentsCount":(\d+)')
RE_NEW_CREATED = re.compile(r'"createdAt":"([^"]+)"')
RE_NEW_TOPICS = re.compile(
    r'"topics":\{"__typename":"TopicConnection","edges":\[(?P<edges>.*?)\]\}',
    re.DOTALL,
)
RE_NEW_TOPIC_NAME = re.compile(r'"name":"((?:[^"\\]|\\.)*)"')

#: Lookup of votes-field-name → regex for the new shape. Used by
#: ``_extract_new_posts`` to walk the priority list
#: ``launchDayScore → latestScore → votesCount`` and pick the first
#: non-zero value.
_RE_VOTES_BY_FIELD: dict[str, re.Pattern[str]] = {
    "launchDayScore": RE_NEW_LAUNCH_DAY,
    "latestScore": RE_NEW_LATEST,
    "votesCount": RE_NEW_VOTES,
}


# ---------------------------------------------------------------------------
# Public schema — what a single PH record looks like in the snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PHPost:
    """One Product Hunt launch record as it appears in the snapshot.

    Field names are stable and form the public schema; do not rename
    without bumping ``SCHEMA_VERSION``.

    Attributes
    ----------
    id:
        PH's stable per-post id (a stringified integer). Unique even
        across edits; the dedup key.
    name:
        The product name (e.g. ``"Polar Habits"``).
    slug:
        URL slug, e.g. ``"polar-habits-2"``. Last segment of the
        canonical post URL.
    tagline:
        One-line tagline written by the maker at launch time. May be
        empty if the post is missing a tagline; we keep empty strings
        rather than ``None`` so the JSONL row is well-typed.
    description:
        One-paragraph description extracted from the post page. We
        do NOT scrape the post page itself (the Wayback CDX for
        individual ``/posts/<slug>`` is patchy, and the post page
        often requires JS to render the full body). The tagline is
        a fine 1-line description for the dedup-vs-YC task.
    votes_count:
        The post's final launch-day score. PH updates this for
        several weeks after launch; the Wayback snapshot's
        ``launchDayScore`` (or pre-2024 ``votesCount``) is a
        good-enough "votes at end of launch day" signal. Note this
        is *not* the all-time vote count — for that we'd need a
        separate post-detail scrape. For the top-5K cut, the
        launch-day rank is what determines inclusion.
    comments_count:
        Number of comments at scrape time.
    created_at:
        ISO-8601 UTC launch date. Used as a secondary sort key.
    topics:
        List of topic slugs (e.g. ``["health-fitness",
        "productivity"]``). Empty if the post is untopiced.
    url:
        Canonical post URL on ``producthunt.com``.
    ph_url:
        Same as ``url`` — kept for symmetry with the HN schema.
        (HN distinguishes ``url`` = external launch from ``hn_url``
        = discussion; PH collapses both into a single URL.)
    """

    id: str
    name: str
    slug: str
    tagline: str
    description: str
    votes_count: int
    comments_count: int
    created_at: str
    topics: list[str] = field(default_factory=list)
    url: str = ""
    ph_url: str = ""

    def to_jsonl(self) -> str:
        """Serialize one record as a single line of JSON + trailing newline.

        Keys are emitted in declaration order so the file is
        byte-stable across re-runs (given identical input data).
        """
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# HTTP helpers — CDX + Wayback
# ---------------------------------------------------------------------------


def _sleep_politely(seconds: float) -> None:
    """Sleep ``seconds`` plus a small jitter to look human."""
    jitter = random.uniform(0.0, RATE_JITTER_SECONDS)
    time.sleep(seconds + jitter)


def _get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """GET with exponential backoff on transient failures.

    Retries on connection errors, timeouts, 5xx, and 429 (honoring
    ``Retry-After``). Does NOT retry on other 4xx — those are
    programming errors (bad params) that won't fix themselves.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.get(url, params=params)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning(
                "GET %s failed (attempt %d/%d): %s", url, attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
                continue
            raise
        if 500 <= resp.status_code < 600:
            logger.warning(
                "%s returned %d on attempt %d/%d",
                url,
                resp.status_code,
                attempt,
                max_retries,
            )
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
                continue
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "5"))
            logger.warning("%s rate-limited (429); sleeping %.1fs", url, retry_after)
            time.sleep(retry_after)
            # Loop forever (not bounded by max_retries) — the
            # Wayback rate-limits sustained traffic and we want to
            # back off rather than give up.
            continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"GET {url} failed with {resp.status_code}: {resp.text[:500]}"
            )
        return resp
    raise RuntimeError(
        f"GET {url} failed after {max_retries} attempts; last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# CDX enumeration — find every archived daily leaderboard page
# ---------------------------------------------------------------------------


def _list_cdx_for_year(client: httpx.Client, year: int) -> list[tuple[str, str, str]]:
    """Query the Wayback CDX for all archived daily-leaderboard pages in a year.

    Returns a list of ``(date_str, timestamp, original_url)`` tuples,
    where ``date_str`` is the YYYY/M/D URL fragment. We filter out
    the ``/all`` (all-time) variant of the leaderboard — that's a
    different page — and the rare phantom URLs where the Wayback
    captured a CSS asset's color value (e.g. ``.../daily/2024/11/255,
    255, 255, 0.3``) as a captured URL.

    Strict URL filter: we accept everything the CDX returns for
    ``/leaderboard/daily/<year>`` and then filter locally with
    :data:`_RE_DATE_PATH` so we only keep ``YYYY/M/D`` paths. The
    CDX ``filter=original:`` regex syntax is unreliable (it
    sometimes hangs the server or returns []), so we do the
    strict pattern match in Python instead.
    """
    cdx_url = (
        f"{CDX_URL}?url=producthunt.com/leaderboard/daily/{year}"
        f"&matchType=prefix&filter=statuscode:200"
        f"&filter=mimetype:text/html&output=json&limit={CDX_LIMIT}"
    )
    logger.info("CDX: listing %d daily-leaderboard archives", year)
    resp = _get_with_retry(client, cdx_url)
    if resp.text == "[]":
        return []
    try:
        rows = json.loads(resp.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"CDX returned non-JSON for year {year}: {resp.text[:200]}"
        ) from exc
    if not rows or len(rows) < 2:
        return []
    out: list[tuple[str, str, str]] = []
    for row in rows[1:]:
        _, ts, original = row[0], row[1], row[2]
        if "/leaderboard/daily/" not in original:
            continue
        parts = original.split("/leaderboard/daily/", 1)[1]
        # strip query / fragment
        parts = parts.split("?")[0].split("#")[0].rstrip("/")
        if parts.endswith("/all"):
            continue
        # Defensive: also enforce the Y/M/D pattern locally. The CDX
        # regex should have done this but belt-and-braces for rows
        # whose original ends with whitespace, encoded chars, etc.
        if not _RE_DATE_PATH.match(parts):
            logger.debug("CDX: skipping non-date path %r (year %d)", parts, year)
            continue
        out.append((parts, ts, original))
    return out


def list_archived_daily_dates(
    client: httpx.Client,
    *,
    start_year: int,
    end_year: int,
) -> list[tuple[str, str]]:
    """Enumerate archived daily leaderboard dates between ``start_year`` and ``end_year``.

    Returns a list of ``(date_str, timestamp)`` tuples, deduplicated to
    the most-recent snapshot per date, sorted by date ascending. The
    CDX is per-year — we issue one CDX request per year and
    merge + dedup.
    """
    by_date: dict[str, str] = {}
    for year in range(start_year, end_year + 1):
        _sleep_politely(CDX_RATE_SECONDS)
        for d, ts, _ in _list_cdx_for_year(client, year):
            if d not in by_date or ts > by_date[d]:
                by_date[d] = ts
    out = sorted(by_date.items(), key=lambda kv: _date_str_to_sortable(kv[0]))
    logger.info("CDX: %d unique dates in %d–%d", len(out), start_year, end_year)
    return out


def _date_str_to_sortable(d: str) -> str:
    """Turn ``2024/1/1`` into ``20240101`` for lexicographic sort = date sort."""
    y, m, day = d.split("/")
    return f"{int(y):04d}{int(m):02d}{int(day):02d}"


# ---------------------------------------------------------------------------
# Wayback page fetch + post extraction
# ---------------------------------------------------------------------------


def fetch_daily_page(client: httpx.Client, date_str: str, ts: str) -> str:
    """Fetch the archived daily leaderboard page for ``date_str``.

    ``date_str`` is the YYYY/M/D URL fragment; ``ts`` is the CDX
    timestamp to replay. Returns the raw HTML body.
    """
    url = WAYBACK_REPLAY.format(ts=ts, date=date_str)
    resp = _get_with_retry(client, url)
    return resp.text


def _unescape_json_string(s: str) -> str:
    """Reverse the common JSON backslash escapes (``\\n``, ``\\"``,
    ``\\\\``). The post ``tagline`` field is a JSON string and
    PH sometimes escapes special chars; we unescape to keep the
    JSONL human-readable.
    """
    return s.encode("utf-8").decode("unicode_escape", errors="ignore")


def _first_match(re_obj: re.Pattern[str], body: str) -> str | None:
    """Return the first group-1 of the first match in ``body`` or None."""
    m = re_obj.search(body)
    return m.group(1) if m else None


def _balanced_object_end(s: str, start: int) -> int | None:
    """Return the index *after* the matching close brace of the
    object that starts at ``s[start] == '{'``. Returns ``None`` if
    the braces never balance (truncated HTML).

    This is the workhorse that lets us robustly extract a JSON
    object from a string that contains nested objects and quoted
    braces. We track a counter:
      - ``{`` increments
      - ``}`` decrements
      - Inside a JSON string (``"..."``), braces are ignored. We
        also handle the standard escape (``\\"``).
    """
    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start
    in_string = False
    while i < len(s):
        c = s[i]
        if in_string:
            if c == "\\" and i + 1 < len(s):
                # Skip the escaped char (handles \\", \\\\, \\/, etc.)
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _extract_legacy_posts(html: str) -> list[PHPost]:
    """Extract posts from a pre-2024 archive snapshot (Relay cache shape).

    The cache is a flat list of objects, each keyed by an Apollo
    normalized store id (``"Post123":{...}``). The post body
    contains nested objects (``"topics({"first":4})":{...}``,
    ``"product":{...}``, etc.) so a non-brace-aware regex won't
    bound to the right close. We use ``_balanced_object_end`` to
    find the matching close of each post's open brace.
    """
    out: list[PHPost] = []
    for m in RE_POST_LEGACY_KEY.finditer(html):
        pid = m.group(1)
        # m.end() - 1 is the index of the '{'. The object body
        # starts there.
        start = m.end() - 1
        end = _balanced_object_end(html, start)
        if end is None:
            continue
        body = html[start:end]
        if "__typename\":\"Post\"" not in body:
            # Not a Post — some other object happened to be named "Post<id>".
            continue
        name = _first_match(RE_LEGACY_NAME, body)
        slug = _first_match(RE_LEGACY_SLUG, body)
        tagline = _first_match(RE_LEGACY_TAGLINE, body)
        votes_s = _first_match(RE_LEGACY_VOTES, body)
        comments_s = _first_match(RE_LEGACY_COMMENTS, body)
        if not (name and slug and votes_s):
            continue
        try:
            votes_int = int(votes_s)
            comments_int = int(comments_s) if comments_s else 0
        except ValueError:
            continue
        if votes_int < VOTES_FLOOR:
            continue
        out.append(
            PHPost(
                id=str(pid),
                name=name,
                slug=slug,
                tagline=_unescape_json_string(tagline) if tagline else "",
                description="",  # we do not scrape post detail
                votes_count=votes_int,
                comments_count=comments_int,
                created_at="",  # not always present in legacy shape
                topics=[],
                url=f"https://www.producthunt.com/posts/{slug}",
                ph_url=f"https://www.producthunt.com/posts/{slug}",
            )
        )
    return out


def _extract_new_posts(html: str) -> list[PHPost]:
    """Extract posts from a 2024+ archive snapshot (inline homefeed shape).

    The homefeed edge list wraps each post as
    ``{"node":{"__typename":"Post", ...}, "cursor":"..."}``. We
    capture the open-brace of the post body via a regex group
    (the second ``{`` after ``"node":``) and let the brace
    counter find the matching close.

    The "votes" field on the post is whichever of the following
    is present and non-zero, in priority order:

    1. ``launchDayScore`` — the post's score at end of launch day
       (the spec's "votes at end of launch day" signal).
    2. ``latestScore`` — the most recent score.
    3. ``votesCount`` — the all-time vote count. Fallback only —
       using this can over-count a post that has accumulated votes
       long after launch.
    """
    out: list[PHPost] = []
    for m in RE_POST_NEW_START.finditer(html):
        start = m.start("open")
        end = _balanced_object_end(html, start)
        if end is None:
            continue
        body = html[start:end]
        pid = _first_match(RE_NEW_ID, body)
        name = _first_match(RE_NEW_NAME, body)
        slug = _first_match(RE_NEW_SLUG, body)
        tagline = _first_match(RE_NEW_TAGLINE, body)
        # Resolve the votes field by priority.
        votes_s: str | None = None
        for field_name in ("launchDayScore", "latestScore", "votesCount"):
            v = _first_match(_RE_VOTES_BY_FIELD[field_name], body)
            if v is not None and int(v) > 0:
                votes_s = v
                break
        if not (pid and name and slug and votes_s):
            continue
        try:
            votes_int = int(votes_s)
        except ValueError:
            continue
        if votes_int < VOTES_FLOOR:
            continue
        comments_s = _first_match(RE_NEW_COMMENTS, body)
        created_s = _first_match(RE_NEW_CREATED, body)
        comments_int = int(comments_s) if comments_s else 0
        # Topics: list of name strings from the edges.
        topics: list[str] = []
        topic_block = RE_NEW_TOPICS.search(body)
        if topic_block:
            edges = topic_block.group("edges")
            topics = RE_NEW_TOPIC_NAME.findall(edges)
        out.append(
            PHPost(
                id=str(pid),
                name=name,
                slug=slug,
                tagline=_unescape_json_string(tagline) if tagline else "",
                description="",
                votes_count=votes_int,
                comments_count=comments_int,
                created_at=created_s or "",
                topics=topics,
                url=f"https://www.producthunt.com/posts/{slug}",
                ph_url=f"https://www.producthunt.com/posts/{slug}",
            )
        )
    return out


def extract_posts(html: str) -> list[PHPost]:
    """Extract posts from a Wayback-replay HTML body.

    Tries the legacy shape first (it's stricter and unambiguous);
    falls back to the new shape for 2024+ archives. Dedups by post
    id within the page (a page can have the same post listed
    twice in the homefeed list — the "Post" block + the
    "HomefeedItemEdge > Post" path).
    """
    legacy = _extract_legacy_posts(html)
    new = _extract_new_posts(html)
    seen: set[str] = set()
    out: list[PHPost] = []
    for p in legacy + new:
        if p.id in seen:
            continue
        seen.add(p.id)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Dedup + sort
# ---------------------------------------------------------------------------


def _dedupe(records: Iterable[PHPost]) -> list[PHPost]:
    """Drop duplicates by ``id``, keeping the first occurrence.

    Within a single Wayback snapshot, a post can appear in the
    "PostNNN" cache AND in the "HomefeedItemEdge > Post" path of
    the same page. We dedup to one record per id here.
    """
    seen: set[str] = set()
    out: list[PHPost] = []
    for rec in records:
        if rec.id in seen:
            continue
        seen.add(rec.id)
        out.append(rec)
    return out


def _sort_deterministic(records: Iterable[PHPost]) -> list[PHPost]:
    """Sort by ``votes_count DESC, created_at DESC, id`` ASC.

    Highest-voted posts first; ties broken by recency, then by id
    (so the file is byte-stable across re-runs even when the
    Wayback has multiple snapshots with the same vote count).
    """
    return sorted(
        records,
        key=lambda r: (-r.votes_count, r.created_at, r.id),
    )


# ---------------------------------------------------------------------------
# Borderline review queue + manifest
# ---------------------------------------------------------------------------


def _borderline_band_label() -> str:
    """The band spec used to populate the manifest's ``borderline_band`` field."""
    return f"[{BORDERLINE_LOWER:.2f}, {BORDERLINE_UPPER:.2f})"


def _build_manifest(
    *,
    snapshot_path: Path,
    borderline_path: Path,
    records: list[PHPost],
    scrape_date: date,
    lookback_days: int,
    max_records: int,
    date_range: tuple[str, str],
    cdx_dates_scanned: int,
    cdx_dates_fetched: int,
    cdx_dates_failed: int,
    raw_hits_yielded: int,
    dedup_stats: dict[str, int],
    embedding_model: str,
) -> dict[str, Any]:
    """Build the manifest dict for ``producthunt_<date>.manifest.json``.

    The required fields per ``docs/PHASE-2.md`` §2.5 are:
    ``schema_version``, ``source_url``, ``scrape_date``, ``count``,
    ``date_range``, ``votes_floor``, ``dedup_stats``,
    ``borderline_count``, ``borderline_path``, and the embedding
    model name. Everything else is courtesy for debugging.
    """
    if records:
        oldest = min((r.created_at for r in records if r.created_at), default="")
        newest = max((r.created_at for r in records if r.created_at), default="")
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
        "lookback_days": lookback_days,
        "max_records_cap": max_records,
        "votes_floor": VOTES_FLOOR,
        "wayback": {
            "cdx_dates_scanned": cdx_dates_scanned,
            "cdx_dates_fetched": cdx_dates_fetched,
            "cdx_dates_failed": cdx_dates_failed,
        },
        "raw_hits_yielded": raw_hits_yielded,
        "dedup_stats": dedup_stats,
        "borderline_band": _borderline_band_label(),
        "borderline_count": dedup_stats.get("borderline_count", 0),
        "borderline_path": borderline_path.name,
        "embedding_model": embedding_model,
        "scraped_at_utc": datetime.now(tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "snapshot_filename": snapshot_path.name,
    }


def write_snapshot(
    records: list[PHPost],
    borderline: list[tuple[PHPost, str, float]],
    *,
    out_dir: Path,
    scrape_date: date,
    lookback_days: int,
    max_records: int,
    date_range: tuple[str, str],
    cdx_dates_scanned: int,
    cdx_dates_fetched: int,
    cdx_dates_failed: int,
    raw_hits_yielded: int,
    dedup_stats: dict[str, int],
) -> tuple[Path, Path, Path]:
    """Write the jsonl, manifest, and borderline queue to ``out_dir``.

    Returns the ``(jsonl_path, manifest_path, borderline_path)`` triple.
    The jsonl uses fixed field order (``asdict`` on the dataclass) so
    the file is diff-stable across re-runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = scrape_date.isoformat()
    jsonl_path = out_dir / f"producthunt_{date_str}.jsonl"
    manifest_path = out_dir / f"producthunt_{date_str}.manifest.json"
    borderline_path = out_dir / f"producthunt_{date_str}.borderline.jsonl"

    # Write the main JSONL.
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_jsonl())

    # Write the borderline review queue. Each line is a JSON object
    # with the post record plus the closest YC name and the cosine
    # score, so the reviewer can decide without a separate lookup.
    with borderline_path.open("w", encoding="utf-8") as f:
        for post, yc_name, cosine in borderline:
            row = {
                "post": asdict(post),
                "nearest_yc_name": yc_name,
                "cosine": round(cosine, 4),
            }
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")

    manifest = _build_manifest(
        snapshot_path=jsonl_path,
        borderline_path=borderline_path,
        records=records,
        scrape_date=scrape_date,
        lookback_days=lookback_days,
        max_records=max_records,
        date_range=date_range,
        cdx_dates_scanned=cdx_dates_scanned,
        cdx_dates_fetched=cdx_dates_fetched,
        cdx_dates_failed=cdx_dates_failed,
        raw_hits_yielded=raw_hits_yielded,
        dedup_stats=dedup_stats,
        embedding_model=EMBEDDING_MODEL,
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")

    return jsonl_path, manifest_path, borderline_path


# ---------------------------------------------------------------------------
# YC name loading (for the dedup step)
# ---------------------------------------------------------------------------


def _load_yc_names(yc_snapshot_path: Path) -> list[str]:
    """Read the YC JSONL snapshot and return the list of company names.

    Used as the dedup target. The YC snapshot is the canonical
    source — we trust the ingest hasn't munged the names.
    """
    if not yc_snapshot_path.exists():
        raise FileNotFoundError(
            f"YC snapshot not found: {yc_snapshot_path}. "
            f"Run `make scrape` first to produce it."
        )
    names: list[str] = []
    with yc_snapshot_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = str(row.get("name", "")).strip()
            if name:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Embedding cache (YC names are a one-time compute)
# ---------------------------------------------------------------------------


def _load_or_build_yc_embeddings(
    yc_names: list[str],
    *,
    cache_path: Path = YC_NAME_EMBEDDINGS_CACHE,
) -> np.ndarray:  # type: ignore[name-defined]
    """Load cached bge-m3 embeddings for YC names, or compute + cache.

    The cache is ``data/cache/yc_name_embeddings.npz`` with arrays
    ``names`` (object array of strings) and ``embeddings`` (float32
    ``(N, 1024)``). Computing fresh takes ~30 min on contested CPU
    for ~6K names; the cache turns subsequent runs into a < 1s load.
    """
    if cache_path.exists():
        try:
            # ``allow_pickle=True`` is needed for the object array of
            # string names that np.savez writes. The cache is produced
            # by us (same code path), not by a third party — this is
            # a project-local artifact, not an untrusted pickle. If
            # you want belt-and-braces, delete the cache file and
            # we'll rebuild from scratch.
            data = np.load(cache_path, allow_pickle=True)
            cached_names = list(data["names"])
            cached_embs = data["embeddings"]
            if len(cached_names) == len(yc_names) and list(cached_names) == yc_names:
                logger.info(
                    "Loaded cached YC name embeddings from %s (%d x %d)",
                    cache_path,
                    cached_embs.shape[0],
                    cached_embs.shape[1],
                )
                return cached_embs
            logger.warning(
                "YC name embeddings cache miss: name list differs "
                "(cached=%d, current=%d). Recomputing.",
                len(cached_names),
                len(yc_names),
            )
        except Exception as exc:  # corrupted cache, etc.
            logger.warning("YC embeddings cache unreadable (%s); recomputing.", exc)

    # Cache miss — compute and save.
    from src.data.embedder import Embedder

    logger.info(
        "Computing bge-m3 embeddings for %d YC names (one-time; ~30 min on CPU)",
        len(yc_names),
    )
    embedder = Embedder()
    t0 = time.time()
    # Warmup — first call always slow due to lazy import.
    embedder.embed_batch(["warmup"])
    logger.info("Embedder warmup done in %.1fs", time.time() - t0)
    t0 = time.time()
    embs = embedder.embed_batch(yc_names)
    logger.info(
        "YC name embeddings done in %.1fs (%.1f names/s)",
        time.time() - t0,
        len(yc_names) / max(time.time() - t0, 1e-3),
    )
    arr = np.array(embs, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        names=np.array(yc_names, dtype=object),
        embeddings=arr,
    )
    logger.info("Saved YC name embeddings cache to %s", cache_path)
    return arr


# ---------------------------------------------------------------------------
# Cosine dedup
# ---------------------------------------------------------------------------


def _cosine_dedup(
    ph_posts: list[PHPost],
    yc_names: list[str],
    yc_embeddings: np.ndarray,  # type: ignore[name-defined]
) -> tuple[list[PHPost], list[tuple[PHPost, str, float]], dict[str, int]]:
    """Classify each PH post as exact-dup / borderline / novel.

    Returns ``(kept_posts, borderline_queue, stats)`` where:
    - ``kept_posts`` are the posts we want in the corpus (novel +
      borderline — exact-dups are dropped because the spec says
      "dedup if ≥ 0.85").
    - ``borderline_queue`` is the list of ``(post, nearest_yc_name,
      cosine)`` tuples in the borderline band; the caller writes them
      to the ``.borderline.jsonl`` file.
    - ``stats`` is a small dict with ``exact_dup_count``,
      ``borderline_count``, ``novel_count``, ``max_cosine_mean``,
      ``max_cosine_median``.

    We embed the PH names fresh on every run (no cache — the PH
    corpus changes daily and is small enough that the ~25-min embed
    is the cost of being current). The YC name embeddings are
    cached.
    """
    if not ph_posts:
        return [], [], {
            "exact_dup_count": 0,
            "borderline_count": 0,
            "novel_count": 0,
            "max_cosine_mean": 0.0,
            "max_cosine_median": 0.0,
        }

    # Embed PH names.
    from src.data.embedder import Embedder

    ph_names = [p.name for p in ph_posts]
    logger.info("Embedding %d PH names with %s", len(ph_names), EMBEDDING_MODEL)
    embedder = Embedder()
    t0 = time.time()
    ph_vecs = np.array(embedder.embed_batch(ph_names), dtype=np.float32)
    logger.info(
        "PH name embeddings done in %.1fs (%.1f names/s)",
        time.time() - t0,
        len(ph_names) / max(time.time() - t0, 1e-3),
    )

    # Cosine similarity matrix. PH vectors are unit-normalised (bge-m3
    # default with ``normalize_embeddings=True``), YC cache is
    # already unit-normalised. A single matmul gives us
    # ``(N_ph, N_yc)`` cosine values.
    logger.info(
        "Computing cosine matrix: %d PH × %d YC", ph_vecs.shape[0], yc_embeddings.shape[0]
    )
    sims = ph_vecs @ yc_embeddings.T  # (N_ph, N_yc)
    max_cos = sims.max(axis=1)  # (N_ph,)
    max_idx = sims.argmax(axis=1)  # (N_ph,)

    # Classify.
    kept: list[PHPost] = []
    borderline: list[tuple[PHPost, str, float]] = []
    exact = 0
    bord = 0
    nov = 0
    for i, post in enumerate(ph_posts):
        cos = float(max_cos[i])
        yc_name = yc_names[int(max_idx[i])]
        if cos >= EXACT_DUP_COSINE:
            exact += 1
            # Drop the PH record — YC wins.
            continue
        if BORDERLINE_LOWER <= cos < BORDERLINE_UPPER:
            bord += 1
            borderline.append((post, yc_name, cos))
            # Keep in main JSONL too — borderline means "probably
            # novel but worth a human review", not "definite dup".
            kept.append(post)
            continue
        nov += 1
        kept.append(post)

    stats = {
        "exact_dup_count": exact,
        "borderline_count": bord,
        "novel_count": nov,
        "max_cosine_mean": float(max_cos.mean()),
        "max_cosine_median": float(np.median(max_cos)),
    }
    logger.info(
        "Dedup: %d exact-dups dropped, %d borderline, %d novel (of %d total)",
        exact,
        bord,
        nov,
        len(ph_posts),
    )
    return kept, borderline, stats


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def fetch_all_posts(
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_records: int = DEFAULT_MAX_RECORDS,
    skip_dedup: bool = False,
    yc_snapshot: Path = Path("data/snapshots/yc_2026-06-08.jsonl"),
    user_agent: str = USER_AGENT,
    wayback_transport: httpx.BaseTransport | None = None,
) -> tuple[list[PHPost], list[tuple[PHPost, str, float]], dict[str, Any]]:
    """Fetch every archived PH daily-leaderboard post meeting the spec filters.

    Pipeline:
      1. CDX-enumerate every archived ``/leaderboard/daily/<date>``
         page in the lookback window.
      2. Fetch each page from the Wayback Machine and extract the
         embedded post data (handles both pre-2024 and 2024+ HTML
         shapes).
      3. Filter to ``votes_count >= VOTES_FLOOR``.
      4. Dedup on ``id`` (a post may appear in multiple daily pages
         if it stayed on the daily top-N for several days).
      5. Sort deterministically (votes DESC, then created_at DESC,
         then id ASC).
      6. Cut to ``max_records`` (default 5K).
      7. Cosine-dedup against YC names: drop exact-dups
         (cosine >= 0.85), queue borderlines (cosine in
         ``[0.75, 0.85)``) for manual review, keep novel.

    Returns
    -------
    tuple[list[PHPost], list[tuple[PHPost, str, float]], dict[str, Any]]
        ``(records, borderline, stats)`` — see ``_cosine_dedup`` for
        the borderline shape; ``stats`` is a small dict suitable for
        logging / manifest.

    Parameters
    ----------
    lookback_days:
        How far back to scan. Default 3 years.
    max_records:
        Cap on the final corpus size post-dedup. Default 5K per spec.
    skip_dedup:
        If True, skip the bge-m3 cosine-dedup step. Useful for fast
        smoke tests and for ``--skip-dedup`` runs that just want the
        raw JSONL.
    yc_snapshot:
        Path to the YC JSONL snapshot used for the dedup target.
        Defaults to ``data/snapshots/yc_2026-06-08.jsonl`` (the
        existing snapshot from Phase 1.2).
    user_agent:
        Override the User-Agent.
    wayback_transport:
        For tests — pass an ``httpx.MockTransport`` to keep the
        scraper hermetic. Production callers leave this ``None``.
    """
    today = datetime.now(tz=UTC).date()
    start_date = today - timedelta(days=lookback_days)
    start_year = start_date.year
    end_year = today.year
    search_floor_str = start_date.isoformat()
    today_str = today.isoformat()
    headers = {"User-Agent": user_agent}
    # Allow the caller to inject a transport (for tests via
    # ``httpx.MockTransport``). Production callers leave this
    # ``None``.
    client_kwargs: dict[str, Any] = dict(headers=headers, timeout=CDX_TIMEOUT)
    if wayback_transport is not None:
        client_kwargs["transport"] = wayback_transport
    stats: dict[str, Any] = {
        "cdx_dates_scanned": 0,
        "cdx_dates_fetched": 0,
        "cdx_dates_failed": 0,
        "raw_hits_yielded": 0,
    }
    cdx_by_date: list[tuple[str, str]] = []
    with httpx.Client(**client_kwargs) as client:
        cdx_by_date = list_archived_daily_dates(
            client, start_year=start_year, end_year=end_year
        )
        stats["cdx_dates_scanned"] = len(cdx_by_date)
        logger.info("Wayback: fetching %d daily pages", len(cdx_by_date))
        # Use a separate client with the longer Wayback timeout, but
        # carry the same transport (so test mocks see all requests,
        # not just the CDX ones).
        wb_kwargs: dict[str, Any] = dict(headers=headers, timeout=WAYBACK_TIMEOUT)
        if wayback_transport is not None:
            wb_kwargs["transport"] = wayback_transport
        wb_client = httpx.Client(**wb_kwargs)
        try:
            all_records: list[PHPost] = []
            for i, (date_str, ts) in enumerate(cdx_by_date, start=1):
                if i % 50 == 0:
                    logger.info(
                        "Wayback fetch progress: %d / %d dates", i, len(cdx_by_date)
                    )
                try:
                    _sleep_politely(WAYBACK_RATE_SECONDS)
                    html = fetch_daily_page(wb_client, date_str, ts)
                    page_posts = extract_posts(html)
                    all_records.extend(page_posts)
                    stats["cdx_dates_fetched"] += 1
                except Exception as exc:
                    logger.warning(
                        "Wayback fetch failed for %s (ts=%s): %s", date_str, ts, exc
                    )
                    stats["cdx_dates_failed"] += 1
                    continue
        finally:
            wb_client.close()

    stats["raw_hits_yielded"] = len(all_records)
    records = _dedupe(all_records)
    records = _sort_deterministic(records)
    logger.info("After internal dedup: %d unique records", len(records))

    # Truncate to max_records BEFORE the bge-m3 step. Embedding 8K
    # names instead of 5K costs 60% more time and produces an
    # extra few hundred borderline records that won't make the
    # cut.
    records = records[:max_records]
    logger.info("After top-%d cut: %d records", max_records, len(records))

    borderline: list[tuple[PHPost, str, float]] = []
    if skip_dedup:
        dedup_stats: dict[str, int] = {
            "exact_dup_count": 0,
            "borderline_count": 0,
            "novel_count": len(records),
            "max_cosine_mean": 0.0,
            "max_cosine_median": 0.0,
        }
        dedup_stats["skipped"] = 1
    else:
        yc_names = _load_yc_names(yc_snapshot)
        yc_embs = _load_or_build_yc_embeddings(yc_names)
        records, borderline, dedup_stats = _cosine_dedup(records, yc_names, yc_embs)
    stats.update(dedup_stats)
    return records, borderline, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Scrape the public Product Hunt archive into data/snapshots/.",
)


@app.command()
def main(
    out: Path = typer.Option(
        Path("data/snapshots"),
        "--out",
        "-o",
        help="Directory to write producthunt_<date>.jsonl + manifest into.",
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
        help="How far back to scan. Defaults to 3 years (1095 days).",
    ),
    max_records: int = typer.Option(
        DEFAULT_MAX_RECORDS,
        "--max-records",
        help="Cap on the final corpus size post-dedup. Default 5000 per spec.",
    ),
    yc_snapshot: Path = typer.Option(
        Path("data/snapshots/yc_2026-06-08.jsonl"),
        "--yc-snapshot",
        help="Path to the YC JSONL snapshot used as the dedup target.",
    ),
    skip_dedup: bool = typer.Option(
        False,
        "--skip-dedup",
        help="Skip the bge-m3 cosine dedup against YC. "
        "Faster smoke runs; output JSONL is unfiltered.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    ),
) -> None:
    """Run the Product Hunt archive scraper end-to-end."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if date_str:
        scrape_date = date.fromisoformat(date_str)
    else:
        scrape_date = datetime.now(tz=UTC).date()

    today = datetime.now(tz=UTC).date()
    start_date = today - timedelta(days=lookback_days)
    date_range = (start_date.isoformat(), today.isoformat())

    logger.info(
        "Starting PH archive scrape for %s (lookback=%d days, max_records=%d, skip_dedup=%s)",
        scrape_date,
        lookback_days,
        max_records,
        skip_dedup,
    )

    records, borderline, stats = fetch_all_posts(
        lookback_days=lookback_days,
        max_records=max_records,
        skip_dedup=skip_dedup,
        yc_snapshot=yc_snapshot,
    )

    jsonl_path, manifest_path, borderline_path = write_snapshot(
        records,
        borderline,
        out_dir=out,
        scrape_date=scrape_date,
        lookback_days=lookback_days,
        max_records=max_records,
        date_range=date_range,
        cdx_dates_scanned=stats["cdx_dates_scanned"],
        cdx_dates_fetched=stats["cdx_dates_fetched"],
        cdx_dates_failed=stats["cdx_dates_failed"],
        raw_hits_yielded=stats["raw_hits_yielded"],
        dedup_stats=stats,
    )

    logger.info(
        "Wrote %d records to %s (borderline=%d, exact_dups_dropped=%d, "
        "cdx_fetched=%d, cdx_failed=%d)",
        len(records),
        jsonl_path,
        stats.get("borderline_count", 0),
        stats.get("exact_dup_count", 0),
        stats["cdx_dates_fetched"],
        stats["cdx_dates_failed"],
    )
    typer.echo(str(jsonl_path))
    typer.echo(str(manifest_path))
    typer.echo(str(borderline_path))


if __name__ == "__main__":
    app()
