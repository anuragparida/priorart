"""Tests for ``src.data.scrape_hn``.

The scraper is split into:
  1. Algolia HTTP pagination (page-by-page, ``points`` filter client-side).
  2. Hit → HNPost projection.
  3. Firecrawl scraping (concurrent, failure-tolerant).
  4. Dedup + deterministic sort.
  5. JSONL + manifest file writing.

These tests are all unit tests with stubbed HTTP — no live network
calls. The end-to-end "does it produce ~5–10K records" check is in
the kanban task's acceptance criteria, exercised manually after the
scraper ships (card t_56b10368).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.data.scrape_hn import (
    HNPost,
    POINTS_FLOOR,
    SCHEMA_VERSION,
    SOURCE_URL,
    _algolia_params,
    _attach_descriptions,
    _dedupe,
    _firecrawl_scrape,
    _hit_to_post,
    _sort_deterministic,
    fetch_all_posts,
    write_snapshot,
)


# ---------------------------------------------------------------------
# Fixtures: canned Algolia hit shapes (mimic the real HN Algolia index)
# ---------------------------------------------------------------------


@pytest.fixture
def sample_hit_with_url() -> dict[str, Any]:
    """A realistic Algolia HN hit — has an external ``url`` and points."""

    return {
        "objectID": "46205632",
        "title": "Show HN: Gemini Pro 3 imagines the HN front page 10 years from now",
        "author": "dmarz",
        "url": "https://dosaygo-studio.github.io/hn-front-page-2035/news",
        "points": 3346,
        "num_comments": 965,
        "created_at": "2025-12-09T15:00:38Z",
        "created_at_i": 1733758838,
        "story_id": 46205632,
        "story_text": None,
        "story_title": None,
        "comment_text": None,
        "_tags": ["story", "author_dmarz", "show_hn"],
        "_highlightResult": {},
        "updated_at": "2025-12-12T08:42:11Z",
    }


@pytest.fixture
def sample_hit_text_only() -> dict[str, Any]:
    """A hit with no external ``url`` (text-only Show HN post)."""

    return {
        "objectID": "3742902",
        "title": "Show HN: This up votes itself",
        "author": "olalonde",
        "url": None,
        "points": 3531,
        "num_comments": 82,
        "created_at": "2012-03-23T00:40:39Z",
        "created_at_i": 1332463239,
    }


@pytest.fixture
def sample_hit_below_floor() -> dict[str, Any]:
    """A hit with points below POINTS_FLOOR — should be filtered out upstream."""

    return {
        "objectID": "12345",
        "title": "Show HN: Small toy project",
        "author": "somebody",
        "url": "https://example.com/toy",
        "points": 12,  # below POINTS_FLOOR
        "num_comments": 1,
        "created_at": "2025-06-01T00:00:00Z",
        "created_at_i": 1748736000,
    }


# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------


def test_points_floor_constant() -> None:
    """The points threshold is wired per the spec (PHASE-2 §2.6)."""
    assert POINTS_FLOOR == 50


def test_algolia_params_shape() -> None:
    """Query params match the canonical HN "Show HN" 3-year slice."""
    params = _algolia_params(page=2, created_at_floor=1609459200)
    assert params["query"] == "show hn"
    assert params["tags"] == "show_hn"
    assert params["hitsPerPage"] == 1000
    assert params["page"] == 2
    assert "created_at_i>1609459200" in params["numericFilters"]


def test_hit_to_post_with_url(sample_hit_with_url: dict[str, Any]) -> None:
    """Projection preserves the canonical Algolia fields."""
    post = _hit_to_post(sample_hit_with_url)
    assert post.object_id == "46205632"
    assert post.title.startswith("Show HN: Gemini Pro 3")
    assert post.author == "dmarz"
    assert post.url == "https://dosaygo-studio.github.io/hn-front-page-2035/news"
    assert post.points == 3346
    assert post.comments == 965
    assert post.created_at == "2025-12-09T15:00:38Z"
    assert post.description is None  # filled by _attach_descriptions
    assert post.hn_url == "https://news.ycombinator.com/item?id=46205632"


def test_hit_to_post_text_only(sample_hit_text_only: dict[str, Any]) -> None:
    """A hit with ``url=None`` projects with ``url=None`` and the HN discussion URL."""
    post = _hit_to_post(sample_hit_text_only)
    assert post.url is None
    assert post.hn_url == "https://news.ycombinator.com/item?id=3742902"


def test_dedupe_drops_duplicate_object_ids() -> None:
    """Same ``object_id`` keeps the first occurrence only."""
    p1 = HNPost(
        object_id="1", title="t1", author="a", url="u", points=10,
        comments=0, created_at="2025-01-01T00:00:00Z",
        description=None, hn_url="https://news.ycombinator.com/item?id=1",
    )
    p2 = HNPost(
        object_id="1", title="t2", author="b", url="u2", points=20,
        comments=1, created_at="2025-01-02T00:00:00Z",
        description=None, hn_url="https://news.ycombinator.com/item?id=1",
    )
    p3 = HNPost(
        object_id="2", title="t3", author="c", url="u3", points=5,
        comments=0, created_at="2025-01-03T00:00:00Z",
        description=None, hn_url="https://news.ycombinator.com/item?id=2",
    )
    result = _dedupe([p1, p2, p3])
    assert len(result) == 2
    assert result[0].object_id == "1"
    assert result[0].title == "t1"  # first occurrence wins
    assert result[1].object_id == "2"


def test_sort_deterministic_orders_by_points_then_recency_then_id() -> None:
    """Sort: points DESC, then created_at DESC, then object_id ASC."""
    posts = [
        HNPost(
            object_id="c", title="t", author="a", url="u", points=10,
            comments=0, created_at="2025-01-01T00:00:00Z",
            description=None, hn_url="x",
        ),
        HNPost(
            object_id="a", title="t", author="a", url="u", points=20,
            comments=0, created_at="2025-01-02T00:00:00Z",
            description=None, hn_url="x",
        ),
        HNPost(
            object_id="b", title="t", author="a", url="u", points=20,
            comments=0, created_at="2025-01-02T00:00:00Z",
            description=None, hn_url="x",
        ),
    ]
    sorted_posts = _sort_deterministic(posts)
    assert [p.object_id for p in sorted_posts] == ["a", "b", "c"]


def test_to_jsonl_emits_field_order_and_trailing_newline(
    sample_hit_with_url: dict[str, Any],
) -> None:
    """The JSONL line matches the declaration order of the dataclass."""
    post = _hit_to_post(sample_hit_with_url)
    line = post.to_jsonl()
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert list(parsed.keys()) == [
        "object_id",
        "title",
        "author",
        "url",
        "points",
        "comments",
        "created_at",
        "description",
        "hn_url",
    ]


# ---------------------------------------------------------------------
# Firecrawl — stubbed transport
# ---------------------------------------------------------------------


def _firecrawl_handler_success(url_to_markdown: dict[str, str]):
    """Build an httpx MockTransport that returns canned markdown per URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        url = body["url"]
        md = url_to_markdown.get(url)
        if md is None:
            return httpx.Response(
                404,
                json={"success": False, "error": f"Not found: {url}"},
            )
        return httpx.Response(
            200,
            json={"success": True, "data": {"markdown": md}},
        )

    return handler


def test_firecrawl_scrape_returns_collapsed_markdown() -> None:
    """Whitespace is collapsed to a single paragraph for human readability."""
    url = "https://example.com"
    handler = _firecrawl_handler_success(
        {url: "Title\n\nThis is a\nmultiline paragraph.\n\nAnd another."}
    )
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        desc = _firecrawl_scrape(client, url)
    assert desc is not None
    assert "\n" not in desc  # no newlines in collapsed output
    assert "multiline paragraph" in desc


def test_firecrawl_scrape_returns_none_on_404() -> None:
    """A 404 is data, not a crash."""
    handler = _firecrawl_handler_success({})  # empty — 404 everything
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        assert _firecrawl_scrape(client, "https://nope.example") is None


def test_firecrawl_scrape_returns_none_on_empty_markdown() -> None:
    """An empty markdown field counts as 'no description'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "data": {"markdown": ""}},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        assert _firecrawl_scrape(client, "https://example.com/empty") is None


def test_attach_descriptions_fills_matching_urls() -> None:
    """Attach is keyed by URL; description=None entries stay None."""
    p1 = HNPost(
        object_id="1", title="t1", author="a", url="https://a",
        points=10, comments=0, created_at="2025-01-01T00:00:00Z",
        description=None, hn_url="x",
    )
    p2 = HNPost(
        object_id="2", title="t2", author="b", url="https://b",
        points=10, comments=0, created_at="2025-01-01T00:00:00Z",
        description=None, hn_url="x",
    )
    p3 = HNPost(
        object_id="3", title="t3", author="c", url=None,
        points=10, comments=0, created_at="2025-01-01T00:00:00Z",
        description=None, hn_url="x",
    )

    handler = _firecrawl_handler_success(
        {"https://a": "alpha description", "https://b": ""}  # empty -> None
    )
    transport = httpx.MockTransport(handler)

    out, no_launch, dead, ok = _attach_descriptions(
        [p1, p2, p3], scrape_concurrency=2, transport=transport
    )

    assert ok == 1  # only p1 got a real description
    assert dead == 1  # p2 returned empty markdown
    assert no_launch == 1  # p3 had no url
    assert out[0].description == "alpha description"
    assert out[1].description is None
    assert out[2].description is None


# ---------------------------------------------------------------------
# Top-level pipeline + write_snapshot — stubbed Algolia transport
# ---------------------------------------------------------------------


def _algolia_handler(pages: list[dict[str, Any]]):
    """Build an httpx MockTransport that returns a canned page sequence.

    The handler matches GET requests to SOURCE_URL with ``page=N`` param
    and returns the corresponding page from ``pages``. On a page number
    beyond the list, return an empty page (terminating pagination).
    """
    pages_by_index = {p.get("page", i): p for i, p in enumerate(pages)}

    def handler(request: httpx.Request) -> httpx.Response:
        # Pull page from URL-encoded query.
        page = 0
        for k, v in request.url.params.items():
            if k == "page":
                page = int(v)
        if page in pages_by_index:
            return httpx.Response(200, json=pages_by_index[page])
        return httpx.Response(200, json={"hits": [], "nbHits": 0, "nbPages": 1})

    return handler


def test_fetch_all_posts_paginates_and_filters_by_points() -> None:
    """Pagination stops at the end of the page list; POINTS_FLOOR filters."""
    pages = [
        {
            "page": 0,
            "nbHits": 3,
            "nbPages": 2,
            "hits": [
                # First hit below floor — should be filtered client-side
                {
                    "objectID": "99",
                    "title": "Show HN: low score",
                    "author": "x",
                    "url": "https://x.example",
                    "points": 10,
                    "num_comments": 1,
                    "created_at": "2025-06-01T00:00:00Z",
                },
                # Keep
                {
                    "objectID": "11",
                    "title": "Show HN: top",
                    "author": "a",
                    "url": "https://a.example",
                    "points": 200,
                    "num_comments": 50,
                    "created_at": "2025-06-02T00:00:00Z",
                },
            ],
        },
        {
            "page": 1,
            "nbHits": 3,
            "nbPages": 2,
            "hits": [
                {
                    "objectID": "12",
                    "title": "Show HN: text-only",
                    "author": "b",
                    "url": None,
                    "points": 100,
                    "num_comments": 5,
                    "created_at": "2025-06-03T00:00:00Z",
                },
            ],
        },
    ]
    transport = httpx.MockTransport(_algolia_handler(pages))

    posts, stats = fetch_all_posts(
        lookback_days=3650,
        max_records=None,
        scrape=False,  # skip Firecrawl so the test is hermetic
        algolia_transport=transport,
    )

    # Two hits had points >= 50; one was below floor.
    assert len(posts) == 2
    # Sort: points DESC. Post 11 has 200 pts, post 12 has 100 pts.
    assert posts[0].object_id == "11"
    assert posts[1].object_id == "12"
    assert stats["raw_hits_yielded"] == 2  # 11 and 12; 99 was filtered


def test_fetch_all_posts_respects_max_records() -> None:
    """``max_records`` caps the post-filter yield."""
    pages = [
        {
            "page": 0,
            "nbHits": 5,
            "nbPages": 1,
            "hits": [
                {
                    "objectID": f"{i}",
                    "title": f"Show HN: p{i}",
                    "author": "a",
                    "url": f"https://x{i}.example",
                    "points": 100 + i,
                    "num_comments": 1,
                    "created_at": f"2025-05-{i + 10:02d}T00:00:00Z",
                }
                for i in range(1, 6)
            ],
        },
    ]
    transport = httpx.MockTransport(_algolia_handler(pages))

    posts, _ = fetch_all_posts(
        lookback_days=3650,
        max_records=3,
        scrape=False,
        algolia_transport=transport,
    )

    assert len(posts) == 3


# ---------------------------------------------------------------------
# write_snapshot — file shape
# ---------------------------------------------------------------------


def test_write_snapshot_emits_jsonl_and_manifest(tmp_path: Path) -> None:
    """The on-disk shape matches the spec (date-stamped filenames, valid manifest)."""
    records = [
        HNPost(
            object_id="11", title="Show HN: top", author="a",
            url="https://a.example", points=200, comments=50,
            created_at="2025-06-02T00:00:00Z",
            description="alpha description",
            hn_url="https://news.ycombinator.com/item?id=11",
        ),
        HNPost(
            object_id="12", title="Show HN: text-only", author="b",
            url=None, points=100, comments=5,
            created_at="2025-06-03T00:00:00Z",
            description=None,
            hn_url="https://news.ycombinator.com/item?id=12",
        ),
    ]
    jsonl_path, manifest_path = write_snapshot(
        records,
        out_dir=tmp_path,
        scrape_date=date(2026, 6, 8),
        no_launch_url_count=1,
        dead_url_count=0,
        scraped_count=1,
        max_records=None,
        lookback_days=1095,
        date_range=("2023-06-14T00:00:00Z", "2026-06-14T00:00:00Z"),
    )

    assert jsonl_path.name == "hn_show_2026-06-08.jsonl"
    assert manifest_path.name == "hn_show_2026-06-08.manifest.json"

    # JSONL: one record per line, valid JSON, deterministic order.
    with jsonl_path.open() as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert set(rec.keys()) == {
            "object_id",
            "title",
            "author",
            "url",
            "points",
            "comments",
            "created_at",
            "description",
            "hn_url",
        }

    # Manifest: required fields per docs/PHASE-2.md §2.6.
    with manifest_path.open() as f:
        manifest = json.load(f)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["source_url"] == SOURCE_URL
    assert manifest["scrape_date"] == "2026-06-08"
    assert manifest["count"] == 2
    assert manifest["snapshot_filename"] == "hn_show_2026-06-08.jsonl"
    assert manifest["points_floor"] == POINTS_FLOOR
    assert manifest["no_launch_url_count"] == 1
    assert manifest["dead_url_count"] == 0
    assert manifest["firecrawl_success_count"] == 1
    assert manifest["firecrawl_success_rate"] == 1.0


def test_idempotency_same_records_yield_byte_identical_jsonl(tmp_path: Path) -> None:
    """Same records → same JSONL bytes."""
    records = [
        HNPost(
            object_id="11", title="t", author="a", url=None,
            points=10, comments=0, created_at="2025-01-01T00:00:00Z",
            description=None, hn_url="x",
        ),
    ]
    jsonl_a, _ = write_snapshot(
        records,
        out_dir=tmp_path / "a",
        scrape_date=date(2026, 6, 8),
        no_launch_url_count=1, dead_url_count=0, scraped_count=0,
        max_records=None, lookback_days=1095,
        date_range=("x", "y"),
    )
    jsonl_b, _ = write_snapshot(
        records,
        out_dir=tmp_path / "b",
        scrape_date=date(2026, 6, 8),
        no_launch_url_count=1, dead_url_count=0, scraped_count=0,
        max_records=None, lookback_days=1095,
        date_range=("x", "y"),
    )
    assert jsonl_a.read_bytes() == jsonl_b.read_bytes()


def test_snapshot_path_pattern_for_backfill(tmp_path: Path) -> None:
    """``scrape-descriptions`` rewrites ``.jsonl`` (not ``.jsonl.tmp``).

    The atomic-write pattern rewrites via ``{path}.tmp`` then renames
    over the original. Test the path arithmetic directly so a typo in
    the ``with_suffix`` chain can't silently break backfill.
    """
    snapshot = tmp_path / "hn_show_2026-06-29.jsonl"
    tmp_path_for_atomic = snapshot.with_suffix(snapshot.suffix + ".tmp")
    assert tmp_path_for_atomic.name == "hn_show_2026-06-29.jsonl.tmp"
    # Manifest path strips ``.jsonl`` then adds ``.manifest.json``.
    manifest = snapshot.with_suffix("").with_suffix(".manifest.json")
    assert manifest.name == "hn_show_2026-06-29.manifest.json"
