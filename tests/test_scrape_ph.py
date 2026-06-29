"""Tests for ``src.data.scrape_ph``.

The scraper is split into:
  1. CDX enumeration (with strict YYYY/M/D filtering — Wayback
     sometimes captures phantom CSS-bleed URLs as if they were
     leaderboard pages).
  2. Wayback page fetch + post extraction (two HTML shapes:
     pre-2024 Relay cache, 2024+ inline homefeed edge list).
  3. Internal dedup + deterministic sort.
  4. Cosine dedup against the YC name embedding cache (with a fake
     embeddings cache).
  5. JSONL + manifest + borderline queue writing, idempotency.

All tests use ``httpx.MockTransport`` (or fake HTTP) — no live
network. The "does it produce ~5K records" end-to-end check is
exercised manually via ``make ph-scrape`` per the kanban acceptance
criteria.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from src.data.scrape_ph import (
    _RE_DATE_PATH,
    BORDERLINE_LOWER,
    BORDERLINE_UPPER,
    CDX_URL,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_RECORDS,
    EMBEDDING_MODEL,
    EXACT_DUP_COSINE,
    RE_LEGACY_NAME,
    SCHEMA_VERSION,
    SOURCE_URL,
    VOTES_FLOOR,
    PHPost,
    _balanced_object_end,
    _cosine_dedup,
    _date_str_to_sortable,
    _dedupe,
    _extract_legacy_posts,
    _extract_new_posts,
    _first_match,
    _load_or_build_yc_embeddings,
    _load_yc_names,
    _sort_deterministic,
    extract_posts,
    fetch_all_posts,
    write_snapshot,
)

# ---------------------------------------------------------------------------
# Canned HTML fixtures
# ---------------------------------------------------------------------------

# Minimal legacy-shape HTML: a single Post cache entry with name,
# slug, tagline, votes, comments. Mirrors what a Wayback replay of a
# pre-2024 daily leaderboard page looks like. The PH Relay cache
# serialises entries as ``"Post<id>":{...}`` with no whitespace
# between the colon and the open brace.
LEGACY_HTML_ONE_POST = """
<html><body>
<script>window.__APOLLO_STATE__ = {
  "Post462912":{"id":"462912","__typename":"Post","name":"Vlip","slug":"vlip","tagline":"Short videos, big laughs.","votesCount":426,"commentsCount":32,"url":"https://www.producthunt.com/posts/vlip"}
};
</script>
</body></html>
""".strip()

# Legacy HTML with two posts and a non-Post noise entry (to test
# that we don't pick it up).
LEGACY_HTML_TWO_POSTS = """
<html><body>
<script>
{"Post1":{"__typename":"Post","id":"1","name":"Alpha","slug":"alpha","tagline":"First.","votesCount":500,"commentsCount":10},"Topic42":{"__typename":"Topic","name":"Design"},"Post2":{"__typename":"Post","id":"2","name":"Beta","slug":"beta","tagline":"Second.","votesCount":300,"commentsCount":5},"Post3":{"__typename":"Post","id":"3","name":"Gamma","slug":"gamma","tagline":"Low votes.","votesCount":10,"commentsCount":1}}
</script>
</body></html>
""".strip()

# Inline-edge new-shape HTML (post-2024). Uses ``votesCount``
# (the lowest-priority fallback) to exercise the priority chain
# ``launchDayScore → latestScore → votesCount``.
NEW_HTML_ONE_POST = """
<html><body>
<script>
{"data":{"homefeed":{"edges":[
  {"node":{"__typename":"Post","id":"780048","name":"Currents AI","slug":"currents-ai","tagline":"Stay current.","votesCount":914,"commentsCount":120,"createdAt":"2025-02-28T08:00:00Z","topics":{"__typename":"TopicConnection","edges":[{"node":{"name":"AI"}},{"node":{"name":"Productivity"}}]}}},
  {"cursor":"abc"}
]}}}
</script>
</body></html>
""".strip()

# New-shape HTML with launchDayScore > 0 (re-launched post).
# launchDayScore (2000) is the highest priority.
NEW_HTML_LAUNCH_DAY = """
<html><body>
<script>
{"node":{"__typename":"Post","id":"100","name":"ReLaunch","slug":"relaunch","tagline":"v2.","launchDayScore":2000,"latestScore":1500,"votesCount":1200,"commentsCount":50,"createdAt":"2025-01-15T00:00:00Z","topics":{"__typename":"TopicConnection","edges":[]}}}
</script>
</body></html>
""".strip()


def _cdx_row(year: int, month: int, day: int, ts: str = "20250101000000") -> list[str]:
    """Build a fake CDX row matching what the real endpoint returns."""
    return [
        f"com,producthunt)/leaderboard/daily/{year}/{month}/{day}",
        ts,
        f"https://www.producthunt.com/leaderboard/daily/{year}/{month}/{day}",
        "text/html",
        "200",
        "DIGEST",
        "12345",
    ]


@pytest.fixture
def fake_cdx_2024() -> list[list[str]]:
    """A CDX response covering 2024 — leap year, 366 days."""
    rows = [["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]]
    for day in range(1, 367):
        m = (day - 1) // 31 + 1
        d = ((day - 1) % 31) + 1
        if m > 12 or d > 28:
            continue
        rows.append(_cdx_row(2024, m, d))
    return rows


@pytest.fixture
def fake_cdx_mixed() -> list[list[str]]:
    """A CDX response with phantom CSS-bleed rows that should be filtered out."""
    return [
        ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
        _cdx_row(2024, 1, 15),
        _cdx_row(2024, 1, 16),
        # Phantom CSS color captured as URL — must be filtered.
        [
            "com,producthunt)/leaderboard/daily/2024/11/255,%20255,%20255,%200.3",
            "20241120025559",
            "https://www.producthunt.com/leaderboard/daily/2024/11/255,%20255,%20255,%200.3",
            "text/html",
            "200",
            "D",
            "1",
        ],
        # /all aggregate pages — must be filtered.
        [
            "com,producthunt)/leaderboard/daily/2024/1/15/all",
            "20250513111332",
            "https://www.producthunt.com/leaderboard/daily/2024/1/15/all",
            "text/html",
            "200",
            "D",
            "1",
        ],
        # Query-suffix row that should be stripped to YYYY/M/D.
        [
            "com,producthunt)/leaderboard/daily/2024/3/1?ref=header_nav",
            "20240301000000",
            "https://www.producthunt.com/leaderboard/daily/2024/3/1?ref=header_nav",
            "text/html",
            "200",
            "D",
            "1",
        ],
    ]


@pytest.fixture
def fake_client() -> MagicMock:
    return MagicMock(spec=httpx.Client)


# ---------------------------------------------------------------------------
# Regex + small helpers
# ---------------------------------------------------------------------------


class TestDatePathRegex:
    """The CDX filter strips phantom URLs whose trailing path segment
    is a CSS color value rather than a day integer."""

    @pytest.mark.parametrize(
        "good",
        ["2024/1/15", "2024/11/30", "2025/3/1", "2023/6/15", "2026/12/31", "2024/2/29"],
    )
    def test_accepts_valid_dates(self, good: str) -> None:
        assert _RE_DATE_PATH.match(good), f"expected {good!r} to match"

    @pytest.mark.parametrize(
        "bad",
        [
            "2024/1/15/all",  # aggregate page
            "2024/11/255, 255, 255, 0.3",  # CSS color
            "2024/13/1",  # invalid month
            "2024/1/32",  # invalid day
            "2024/00/15",  # month zero
            "2024/1/00",  # day zero
            "abc/def/ghi",
            "",
            "2024/1",  # missing day
        ],
    )
    def test_rejects_invalid_dates(self, bad: str) -> None:
        assert not _RE_DATE_PATH.match(bad), f"expected {bad!r} to NOT match"


class TestBalancedObjectEnd:
    """Brace-counting helper for legacy post extraction."""

    def test_simple_object(self) -> None:
        s = '{"a":1,"b":2}'
        assert _balanced_object_end(s, 0) == len(s)

    def test_nested_object(self) -> None:
        s = '{"a":{"b":1},"c":2}'
        assert _balanced_object_end(s, 0) == len(s)

    def test_string_with_braces(self) -> None:
        # The string contains a `}` literal that should NOT close.
        s = '{"name":"a}b","x":1}'
        assert _balanced_object_end(s, 0) == len(s)

    def test_escaped_quote(self) -> None:
        s = r'{"name":"a\"b","x":1}'
        assert _balanced_object_end(s, 0) == len(s)

    def test_unterminated(self) -> None:
        s = '{"a":1'
        assert _balanced_object_end(s, 0) is None


class TestFirstMatch:
    def test_returns_first_group(self) -> None:
        assert _first_match(RE_LEGACY_NAME, '{"name":"hello"}') == "hello"

    def test_returns_none_when_missing(self) -> None:
        assert _first_match(RE_LEGACY_NAME, '{"other":1}') is None


class TestDateStrToSortable:
    def test_lex_sorts_chronologically(self) -> None:
        a = _date_str_to_sortable("2024/1/15")
        b = _date_str_to_sortable("2024/1/16")
        c = _date_str_to_sortable("2024/2/1")
        assert a < b < c


# ---------------------------------------------------------------------------
# PHPost dataclass — serialization shape
# ---------------------------------------------------------------------------


class TestPHPostSerialization:
    def test_to_jsonl_has_stable_field_order(self) -> None:
        p = PHPost(
            id="1",
            name="x",
            slug="x",
            tagline="",
            description="",
            votes_count=10,
            comments_count=0,
            created_at="2025-01-01",
        )
        line = p.to_jsonl()
        # Parsing round-trips
        roundtrip = json.loads(line)
        assert roundtrip["id"] == "1"
        assert roundtrip["name"] == "x"
        assert roundtrip["votes_count"] == 10
        # Field order: id, name, slug, tagline, description,
        # votes_count, comments_count, created_at, topics, url, ph_url
        expected_order = [
            "id",
            "name",
            "slug",
            "tagline",
            "description",
            "votes_count",
            "comments_count",
            "created_at",
            "topics",
            "url",
            "ph_url",
        ]
        assert list(roundtrip.keys()) == expected_order


# ---------------------------------------------------------------------------
# Post extraction (legacy + new shapes)
# ---------------------------------------------------------------------------


class TestExtractLegacyPosts:
    def test_one_post(self) -> None:
        posts = _extract_legacy_posts(LEGACY_HTML_ONE_POST)
        assert len(posts) == 1
        assert posts[0].id == "462912"
        assert posts[0].name == "Vlip"
        assert posts[0].slug == "vlip"
        assert posts[0].tagline == "Short videos, big laughs."
        assert posts[0].votes_count == 426
        assert posts[0].comments_count == 32

    def test_filters_votes_floor(self) -> None:
        # Gamma has 10 votes, below VOTES_FLOOR (50). Must be dropped.
        posts = _extract_legacy_posts(LEGACY_HTML_TWO_POSTS)
        names = sorted(p.name for p in posts)
        assert names == ["Alpha", "Beta"]

    def test_ignores_non_post_entries(self) -> None:
        posts = _extract_legacy_posts(LEGACY_HTML_TWO_POSTS)
        # Topic entry should be ignored even though it appears in the cache.
        for p in posts:
            assert p.name != "Design"


class TestExtractNewPosts:
    def test_inline_edge_post(self) -> None:
        posts = _extract_new_posts(NEW_HTML_ONE_POST)
        assert len(posts) == 1
        assert posts[0].id == "780048"
        assert posts[0].name == "Currents AI"
        assert posts[0].votes_count == 914
        assert posts[0].topics == ["AI", "Productivity"]
        assert posts[0].created_at == "2025-02-28T08:00:00Z"

    def test_prefers_launch_day_score(self) -> None:
        # launchDayScore (2000) wins over latestScore (1500).
        posts = _extract_new_posts(NEW_HTML_LAUNCH_DAY)
        assert len(posts) == 1
        assert posts[0].votes_count == 2000


class TestExtractPosts:
    def test_dedup_across_legacy_and_new(self) -> None:
        # Same id appears in both shapes; extract_posts dedupes.
        html = LEGACY_HTML_ONE_POST + "\n" + NEW_HTML_ONE_POST.replace(
            "780048", "462912"
        )
        posts = extract_posts(html)
        assert len(posts) == 1
        assert posts[0].id == "462912"


# ---------------------------------------------------------------------------
# Dedup + sort
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_drops_duplicate_ids(self) -> None:
        p = PHPost(
            id="1", name="x", slug="x", tagline="",
            description="", votes_count=10, comments_count=0,
            created_at="",
        )
        out = _dedupe([p, p, p])
        assert len(out) == 1


class TestSortDeterministic:
    def test_sorts_by_votes_desc_then_created_desc_then_id_asc(self) -> None:
        records = [
            PHPost(
                id="1", name="A", slug="a", tagline="",
                description="", votes_count=10, comments_count=0,
                created_at="2025-01-01",
            ),
            PHPost(
                id="2", name="B", slug="b", tagline="",
                description="", votes_count=20, comments_count=0,
                created_at="2025-01-01",  # ties with C on votes+date
            ),
            PHPost(
                id="3", name="C", slug="c", tagline="",
                description="", votes_count=20, comments_count=0,
                created_at="2025-01-01",  # same votes+date as B, higher id
            ),
        ]
        out = _sort_deterministic(records)
        # B and C tie on votes + created; lower id (B="2") wins → B first.
        # Then C, then A (lowest votes).
        assert [r.id for r in out] == ["2", "3", "1"]


# ---------------------------------------------------------------------------
# Manifest + snapshot writing
# ---------------------------------------------------------------------------


class TestWriteSnapshot:
    @pytest.fixture
    def records(self) -> list[PHPost]:
        return [
            PHPost(
                id=f"{i}",
                name=f"Post {i}",
                slug=f"post-{i}",
                tagline="hi",
                description="",
                votes_count=100 + i,
                comments_count=i,
                created_at=f"2025-01-{(i % 28) + 1:02d}",
                topics=["AI"],
                url=f"https://www.producthunt.com/posts/post-{i}",
                ph_url=f"https://www.producthunt.com/posts/post-{i}",
            )
            for i in range(1, 4)
        ]

    def test_writes_three_files(self, records: list[PHPost], tmp_path: Path) -> None:
        borderline: list[tuple[PHPost, str, float]] = [
            (records[0], "YC Co", 0.80),
        ]
        jsonl, manifest, borderline_path = write_snapshot(
            records,
            borderline,
            out_dir=tmp_path,
            scrape_date=date(2026, 6, 29),
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            max_records=DEFAULT_MAX_RECORDS,
            date_range=("2023-06-30", "2026-06-29"),
            cdx_dates_scanned=10,
            cdx_dates_fetched=9,
            cdx_dates_failed=1,
            raw_hits_yielded=100,
            dedup_stats={
                "exact_dup_count": 0,
                "borderline_count": 1,
                "novel_count": 2,
                "max_cosine_mean": 0.42,
                "max_cosine_median": 0.40,
            },
        )
        assert jsonl.exists()
        assert manifest.exists()
        assert borderline_path.exists()
        # JSONL has one record per line
        assert sum(1 for _ in jsonl.open()) == 3
        # Manifest has the spec-required fields
        m = json.loads(manifest.read_text())
        assert m["schema_version"] == SCHEMA_VERSION
        assert m["source_url"] == SOURCE_URL
        assert m["scrape_date"] == "2026-06-29"
        assert m["count"] == 3
        assert m["votes_floor"] == VOTES_FLOOR
        assert m["dedup_stats"]["borderline_count"] == 1
        assert m["borderline_count"] == 1
        assert m["borderline_band"] == f"[{BORDERLINE_LOWER:.2f}, {BORDERLINE_UPPER:.2f})"
        assert m["embedding_model"] == EMBEDDING_MODEL
        # Borderline has one record with all three fields
        bl_lines = [ln for ln in borderline_path.open() if ln.strip()]
        assert len(bl_lines) == 1
        bl = json.loads(bl_lines[0])
        assert "post" in bl
        assert bl["nearest_yc_name"] == "YC Co"
        assert 0.79 < bl["cosine"] < 0.81

    def test_idempotency_same_input_same_bytes(self, records: list[PHPost], tmp_path: Path) -> None:
        """Re-running with the same input produces the same JSONL bytes."""
        borderline: list[tuple[PHPost, str, float]] = []
        a_jsonl, a_manifest, a_bord = write_snapshot(
            records,
            borderline,
            out_dir=tmp_path / "a",
            scrape_date=date(2026, 6, 29),
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            max_records=DEFAULT_MAX_RECORDS,
            date_range=("2023-06-30", "2026-06-29"),
            cdx_dates_scanned=10,
            cdx_dates_fetched=10,
            cdx_dates_failed=0,
            raw_hits_yielded=100,
            dedup_stats={"exact_dup_count": 0, "borderline_count": 0, "novel_count": 3},
        )
        b_jsonl, b_manifest, b_bord = write_snapshot(
            records,
            borderline,
            out_dir=tmp_path / "b",
            scrape_date=date(2026, 6, 29),
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            max_records=DEFAULT_MAX_RECORDS,
            date_range=("2023-06-30", "2026-06-29"),
            cdx_dates_scanned=10,
            cdx_dates_fetched=10,
            cdx_dates_failed=0,
            raw_hits_yielded=100,
            dedup_stats={"exact_dup_count": 0, "borderline_count": 0, "novel_count": 3},
        )
        # JSONL bytes must match (deterministic field order via asdict).
        assert a_jsonl.read_bytes() == b_jsonl.read_bytes()
        # Manifest: only the timestamp field varies; pin it.
        ma = json.loads(a_manifest.read_text())
        mb = json.loads(b_manifest.read_text())
        assert ma == mb


# ---------------------------------------------------------------------------
# Cosine dedup (with fake YC embeddings)
# ---------------------------------------------------------------------------


class TestCosineDedup:
    @pytest.fixture
    def posts(self) -> list[PHPost]:
        return [
            PHPost(id="A", name="Vlip", slug="vlip", tagline="",
                   description="", votes_count=500, comments_count=0,
                   created_at="2025-01-01"),
            PHPost(id="B", name="Vlipe", slug="vlipe", tagline="",
                   description="", votes_count=400, comments_count=0,
                   created_at="2025-01-01"),
            PHPost(id="C", name="QuasarDB", slug="quasardb", tagline="",
                   description="", votes_count=300, comments_count=0,
                   created_at="2025-01-01"),
        ]

    @pytest.fixture
    def yc_names(self) -> list[str]:
        return ["Vlip", "Quasar DB", "Vlipe"]

    def test_classifies_exact_dup_borderline_and_novel(
        self,
        posts: list[PHPost],
        yc_names: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Layout (1-D unit vectors — cosine is just the product):
        # - Vlip (id A)    vs YC Vlip       → 0.95 (exact-dup, ≥ 0.85)
        # - Vlipe (id B)   vs all YC        → 0 (novel)
        # - QuasarDB (id C) vs YC Quasar DB → 0.80 (borderline, [0.75, 0.85))
        # YC names are ["Vlip", "Quasar DB", "Vlipe"]; yc_embs rows must
        # be in the same order — the i-th row is the embedding for the
        # i-th name.
        v_vlip = np.array([1.0])
        v_lipe = np.array([0.0])
        v_quasar = np.array([0.80])
        yc_vlip = np.array([0.95])  # cosine 1.0*0.95 = 0.95 with Vlip (exact-dup)
        yc_quasar = np.array([1.0])  # cosine 0.80*1.0 = 0.80 with QuasarDB (borderline)
        yc_lipe = np.array([0.0])  # cosine 0.0 with Vlipe (novel)
        # YC names in order, embeddings aligned to the names list:
        yc_embs = np.stack([yc_vlip, yc_quasar, yc_lipe]).astype(np.float32)

        ph_mapping = {
            "Vlip": v_vlip.tolist(),
            "Vlipe": v_lipe.tolist(),
            "QuasarDB": v_quasar.tolist(),
        }

        class FakeEmbedder:
            def embed_batch(self, names):  # type: ignore[no-untyped-def]
                return [ph_mapping[n] for n in names]

        monkeypatch.setattr("src.data.embedder.Embedder", FakeEmbedder)

        kept, borderline, stats = _cosine_dedup(posts, yc_names, yc_embs)

        # Vlip dropped (exact-dup); QuasarDB borderline; Vlipe novel.
        kept_ids = sorted(p.id for p in kept)
        assert kept_ids == ["B", "C"]  # Vlipe + QuasarDB kept
        assert stats["exact_dup_count"] == 1  # Vlip
        assert stats["borderline_count"] == 1  # QuasarDB
        assert stats["novel_count"] == 1  # Vlipe
        # Borderline queue has 1 entry: QuasarDB → nearest YC is Quasar DB.
        assert len(borderline) == 1
        post, yc_name, cos = borderline[0]
        assert post.id == "C"
        assert yc_name == "Quasar DB"
        assert 0.79 < cos < 0.81

    def test_empty_input(self) -> None:
        kept, borderline, stats = _cosine_dedup(
            [], [], np.zeros((0, 2), dtype=np.float32)
        )
        assert kept == []
        assert borderline == []
        assert stats["exact_dup_count"] == 0
        assert stats["borderline_count"] == 0


# ---------------------------------------------------------------------------
# CDX enumeration — phantom URL filtering
# ---------------------------------------------------------------------------


class TestListArchivedDailyDates:
    def test_filters_phantom_css_bleed_and_aggregate_pages(
        self, fake_cdx_mixed: list[list[str]]
    ) -> None:
        # Fake client whose get() returns a synthetic Response.
        fake_response = MagicMock(spec=httpx.Response)
        fake_response.text = json.dumps(fake_cdx_mixed)
        fake_response.status_code = 200

        client = MagicMock(spec=httpx.Client)
        client.get.return_value = fake_response

        # Wrap _get_with_retry to call our fake directly.
        # We test the filter path through _list_cdx_for_year, which
        # uses _get_with_retry.
        from src.data import scrape_ph

        original = scrape_ph._get_with_retry
        scrape_ph._get_with_retry = lambda c, url, **kw: fake_response  # type: ignore[assignment]
        try:
            rows = scrape_ph._list_cdx_for_year(client, 2024)
        finally:
            scrape_ph._get_with_retry = original

        # Three kept: 2024/1/15, 2024/1/16, 2024/3/1 (query stripped).
        # Two dropped: CSS-bleed + /all aggregate.
        paths = sorted(r[0] for r in rows)
        assert paths == ["2024/1/15", "2024/1/16", "2024/3/1"]


# ---------------------------------------------------------------------------
# YC name loading + embeddings cache
# ---------------------------------------------------------------------------


class TestLoadYCNames:
    def test_loads_names(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "yc.jsonl"
        snapshot.write_text(
            '{"name":"Vlip"}\n{"name":"QuasarDB"}\n{"name":""}\n'
        )
        assert _load_yc_names(snapshot) == ["Vlip", "QuasarDB"]


class TestLoadOrBuildYCEmbeddings:
    def test_cache_hit_returns_cached(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = tmp_path / "cache.npz"
        names = ["Vlip", "QuasarDB"]
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.savez(cache, names=np.array(names, dtype=object), embeddings=embs)

        # Even with a fake embedder, the cache path shouldn't be hit.
        class ShouldNotCall:
            def embed_batch(self, names):  # type: ignore[no-untyped-def]
                raise AssertionError("cache should have served the embeddings")

        monkeypatch.setattr("src.data.embedder.Embedder", ShouldNotCall)
        out = _load_or_build_yc_embeddings(names, cache_path=cache)
        np.testing.assert_array_equal(out, embs)

    def test_cache_miss_computes_and_saves(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = tmp_path / "cache.npz"
        names = ["Vlip", "QuasarDB"]

        computed = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class FakeEmbedder:
            _called = False

            def __init__(self) -> None:
                self._called = False

            def embed_batch(self, names):  # type: ignore[no-untyped-def]
                self._called = True
                # Return one vec per name
                return [list(row) for row in computed]

        monkeypatch.setattr("src.data.embedder.Embedder", FakeEmbedder)
        out = _load_or_build_yc_embeddings(names, cache_path=cache)
        np.testing.assert_array_equal(out, computed)
        assert cache.exists()


# ---------------------------------------------------------------------------
# End-to-end with httpx.MockTransport — small fake Wayback
# ---------------------------------------------------------------------------


class TestFetchAllPostsSkipDedup:
    """End-to-end with mocked CDX + Wayback. Uses a 3-day fake archive
    in 2024; we cap --max-records to keep the test small."""

    @pytest.fixture
    def mock_transport(
        self,
        fake_cdx_mixed: list[list[str]],
    ) -> httpx.MockTransport:
        # The Wayback replay URL is `/web/{ts}id_/...`. The CDX row
        # is the raw `/leaderboard/daily/<date>` URL. The mock
        # handler matches on substring — return the canned HTML for
        # either form.
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/cdx/search/cdx" in url:
                return httpx.Response(200, text=json.dumps(fake_cdx_mixed))
            if "/leaderboard/daily/" in url:
                return httpx.Response(200, text=LEGACY_HTML_TWO_POSTS)
            return httpx.Response(404, text="not found")

        return httpx.MockTransport(handler)

    def test_pipeline_emits_jsonl_manifest_borderline(
        self,
        mock_transport: httpx.MockTransport,
        tmp_path: Path,
    ) -> None:
        records, borderline, stats = fetch_all_posts(
            lookback_days=1,
            max_records=10,
            skip_dedup=True,
            yc_snapshot=tmp_path / "yc.jsonl",  # unused when skip_dedup
            wayback_transport=mock_transport,
        )

        # fake_cdx_mixed has 3 valid dates (1/15, 1/16, 3/1 after
        # query strip). Each wayback page yields Alpha + Beta (Gamma
        # below VOTES_FLOOR). 3 pages * 2 posts = 6 raw → dedup → 2.
        assert stats["raw_hits_yielded"] == 6
        assert len(records) == 2
        # Alpha has higher votes — sorts first.
        assert records[0].name == "Alpha"
        # CDX stats: 3 scanned, 3 fetched, 0 failed.
        assert stats["cdx_dates_scanned"] == 3
        assert stats["cdx_dates_fetched"] == 3
        assert stats["cdx_dates_failed"] == 0
        # --skip-dedup → no borderline.
        assert stats.get("skipped") == 1


# ---------------------------------------------------------------------------
# Constants pinning (sanity)
# ---------------------------------------------------------------------------


def test_schema_version_is_set() -> None:
    assert SCHEMA_VERSION == "1.0.0"


def test_cdx_url_is_wayback() -> None:
    assert "web.archive.org/cdx/search/cdx" in CDX_URL


def test_source_url_is_producthunt_leaderboard() -> None:
    assert "producthunt.com/leaderboard/daily" in SOURCE_URL


def test_votes_floor_is_at_least_50() -> None:
    """The 50 floor keeps tiny test launches out of the corpus."""
    assert VOTES_FLOOR >= 50


def test_dedup_bands_are_strictly_ordered() -> None:
    assert 0.0 <= BORDERLINE_LOWER < BORDERLINE_UPPER <= 1.0
    assert EXACT_DUP_COSINE == BORDERLINE_UPPER


def test_default_lookback_is_3_years() -> None:
    # Spec: last 3 years.
    assert DEFAULT_LOOKBACK_DAYS == 3 * 365


def test_default_max_records_is_5k() -> None:
    # Spec: top 5K upvoted.
    assert DEFAULT_MAX_RECORDS == 5000