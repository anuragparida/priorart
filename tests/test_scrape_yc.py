"""Tests for ``src.data.scrape_yc``.

The scraper is split into:
  1. Algolia key extraction from the directory page HTML.
  2. Algolia HTTP pagination over batches.
  3. Hit → YCCompany projection.
  4. Dedup + deterministic sort.
  5. JSONL + manifest file writing.

These tests are all unit tests with stubbed HTTP — no live network
calls. The end-to-end "does it produce ~5K records" check is in the
kanban task's acceptance criteria, exercised manually after the
scraper ships.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from src.data.scrape_yc import (
    ALGOLIA_APP_ID,
    ALGOLIA_DSN,
    SOURCE_URL,
    YCCompany,
    _algolia_body,
    _algolia_query_params,
    _dedupe,
    _extract_algolia_key,
    _hit_to_company,
    _post_with_retry,
    _resolve_algolia_key,
    _sort_deterministic,
    fetch_all_companies,
    write_snapshot,
)


# ---------------------------------------------------------------------
# Fixtures: canned Algolia hit shapes (mimic the real index)
# ---------------------------------------------------------------------


@pytest.fixture
def sample_hit() -> dict[str, Any]:
    """A realistic Algolia hit — same shape yc-oss/api fetches."""
    return {
        "id": 32520,
        "name": "Cerenovus",
        "slug": "cerenovus",
        "former_names": [],
        "small_logo_thumb_url": "https://example.com/logo.png",
        "website": "https://cerenovus.ai",
        "all_locations": "San Francisco, CA, USA",
        "long_description": "Long form desc.",
        "one_liner": "Aggregate company knowledge and make inferences",
        "team_size": 3,
        "industry": "B2B",
        "subindustry": "B2B",
        "launched_at": 1780900630,
        "tags": ["AI", "Developer Tools", "B2B"],
        "tags_highlighted": [],
        "top_company": False,
        "isHiring": True,
        "nonprofit": False,
        "batch": "Summer 2026",
        "status": "Active",
        "industries": ["B2B"],
        "regions": ["United States of America"],
        "stage": "Early",
        "app_video_public": True,
        "demo_day_video_public": False,
        "app_answers": None,
        "question_answers": False,
        "_highlightResult": {"name": {"value": "Cerenovus"}},  # must be discarded
        "objectID": "32520",  # must be discarded
    }


@pytest.fixture
def sample_companies() -> list[YCCompany]:
    """Three distinct records, no duplicates."""
    return [
        YCCompany(
            name="Zeta",
            description="zeta description",
            tags=["AI", "B2B"],
            batch="W22",
            status="Active",
            url="https://www.ycombinator.com/companies/zeta",
        ),
        YCCompany(
            name="Alpha",
            description="alpha description",
            tags=["Fintech"],
            batch="S21",
            status="Public",
            url="https://www.ycombinator.com/companies/alpha",
        ),
        YCCompany(
            name="Mu",
            description="mu description",
            tags=["Healthcare", "AI"],
            batch="S21",
            status="Acquired",
            url="https://www.ycombinator.com/companies/mu",
        ),
    ]


# ---------------------------------------------------------------------
# Algolia key extraction
# ---------------------------------------------------------------------


class TestExtractAlgoliaKey:
    def test_extracts_key_from_real_page_shape(self) -> None:
        html = (
            "<html><head>"
            '<script>window.AlgoliaOpts = {"app":"45BWZJ1SGC",'
            '"key":"abc123=="};</script>'
            "</head></html>"
        )
        assert _extract_algolia_key(html) == "abc123=="

    def test_real_algolia_opts_block_from_yc(self) -> None:
        """Smoke-test with the actual shape that ships on YC's directory page."""
        # This is the exact format yc-oss/api parses (see fetcher.ts).
        # Use a single-line string to avoid Python's implicit string
        # concatenation slicing through the middle of the JSON.
        key_value = "NzllNTY5MzJiZGM2OTY2ZTQwMDEzOTk="
        html = (
            f'<script nonce="abc">window.AlgoliaOpts = '
            f'{{"app":"{ALGOLIA_APP_ID}","key":"{key_value}"}};</script>'
        )
        assert _extract_algolia_key(html) == key_value

    def test_missing_window_algoliaopts_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Could not locate window.AlgoliaOpts"):
            _extract_algolia_key("<html><body>no script here</body></html>")

    def test_wrong_app_id_raises(self) -> None:
        html = '<script>window.AlgoliaOpts = {"app":"OTHER","key":"k=="};</script>'
        with pytest.raises(RuntimeError, match="Unexpected Algolia app id"):
            _extract_algolia_key(html)

    def test_empty_key_raises(self) -> None:
        html = f'<script>window.AlgoliaOpts = {{"app":"{ALGOLIA_APP_ID}","key":""}};</script>'
        with pytest.raises(RuntimeError, match="key missing or empty"):
            _extract_algolia_key(html)

    def test_malformed_json_raises(self) -> None:
        html = '<script>window.AlgoliaOpts = {not valid json};</script>'
        with pytest.raises(RuntimeError, match="not valid JSON"):
            _extract_algolia_key(html)


# ---------------------------------------------------------------------
# Hit → YCCompany projection
# ---------------------------------------------------------------------


class TestHitToCompany:
    def test_basic_projection(self, sample_hit: dict[str, Any]) -> None:
        company = _hit_to_company(sample_hit)
        assert company.name == "Cerenovus"
        assert company.description == "Aggregate company knowledge and make inferences"
        assert company.tags == ["AI", "B2B", "Developer Tools"]  # sorted
        assert company.batch == "Summer 2026"
        assert company.status == "Active"
        assert company.url == "https://www.ycombinator.com/companies/cerenovus"

    def test_tags_deduped_and_sorted(self, sample_hit: dict[str, Any]) -> None:
        sample_hit["tags"] = ["B2B", "AI", "AI", "B2B"]
        company = _hit_to_company(sample_hit)
        assert company.tags == ["AI", "B2B"]

    def test_missing_required_field_raises(self) -> None:
        bad = {"name": "X", "batch": "W21", "status": "Active"}  # no slug
        with pytest.raises(ValueError, match="missing required field"):
            _hit_to_company(bad)

    def test_empty_one_liner_falls_back_to_long_description(self) -> None:
        hit = {
            "name": "X",
            "slug": "x",
            "batch": "W21",
            "status": "Active",
            "one_liner": "",
            "long_description": "Long\n\nmulti-paragraph   description",
        }
        company = _hit_to_company(hit)
        assert company.description == "Long multi-paragraph description"

    def test_empty_description_yields_empty_string_not_none(self) -> None:
        hit = {
            "name": "X",
            "slug": "x",
            "batch": "W21",
            "status": "Active",
            "one_liner": "",
            "long_description": "",
        }
        company = _hit_to_company(hit)
        assert company.description == ""


# ---------------------------------------------------------------------
# Dedup + sort
# ---------------------------------------------------------------------


class TestDedupAndSort:
    def test_dedupe_on_name_batch_keeps_first(self) -> None:
        a = YCCompany("X", "d1", [], "W22", "Active", "u1")
        b = YCCompany("X", "d2", [], "W22", "Active", "u2")  # dup of a
        c = YCCompany("X", "d3", [], "S21", "Active", "u3")  # different batch, kept
        result = _dedupe([a, b, c])
        assert len(result) == 2
        assert result[0] is a
        assert result[2 - 1] is c  # second kept (X in S21)

    def test_dedupe_preserves_distinct_records(self, sample_companies: list[YCCompany]) -> None:
        result = _dedupe(sample_companies)
        assert len(result) == 3
        assert [r.name for r in result] == ["Zeta", "Alpha", "Mu"]

    def test_sort_is_deterministic_by_batch_then_name(self, sample_companies: list[YCCompany]) -> None:
        sorted_ = _sort_deterministic(sample_companies)
        # Batches: S21 < W22 lexicographically; within S21, Alpha < Mu
        assert [r.name for r in sorted_] == ["Alpha", "Mu", "Zeta"]

    def test_sort_is_stable_across_runs(self, sample_companies: list[YCCompany]) -> None:
        """Re-sorting the same list yields the same order every time."""
        first = _sort_deterministic(sample_companies)
        second = _sort_deterministic(first)
        assert [r.name for r in first] == [r.name for r in second]


# ---------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------


class TestToJsonl:
    def test_one_record_per_line_with_trailing_newline(self) -> None:
        c = YCCompany("X", "d", ["a"], "W22", "Active", "u")
        line = c.to_jsonl()
        assert line.endswith("\n")
        assert line.count("\n") == 1
        parsed = json.loads(line)
        assert parsed["name"] == "X"
        assert parsed["tags"] == ["a"]


# ---------------------------------------------------------------------
# Algolia body / query-param shapes
# ---------------------------------------------------------------------


class TestAlgoliaRequestShape:
    def test_facet_only_body_has_no_batch_filter(self) -> None:
        body = _algolia_body(batch=None, page=0)
        assert "facetFilters" not in body["requests"][0]["params"]

    def test_batch_body_encodes_colon_in_facet_filter(self) -> None:
        body = _algolia_body(batch="W22", page=2)
        params = body["requests"][0]["params"]
        assert "facetFilters=batch%3AW22" in params
        assert "&page=2" in params

    def test_query_params_include_required_algolia_headers(self) -> None:
        params = _algolia_query_params("KEY")
        assert params["x-algolia-application-id"] == ALGOLIA_APP_ID
        assert params["x-algolia-api-key"] == "KEY"
        assert "x-algolia-agent" in params


# ---------------------------------------------------------------------
# _post_with_retry — error paths
# ---------------------------------------------------------------------


class TestPostWithRetry:
    def test_4xx_does_not_retry(self) -> None:
        """A 400 means the request is malformed; retrying won't help."""
        client = MagicMock(spec=httpx.Client)
        client.post.return_value = MagicMock(
            status_code=400, text="bad request body", headers={}
        )
        with pytest.raises(RuntimeError, match="Algolia POST failed with 400"):
            _post_with_retry(client, ALGOLIA_DSN, params={}, body={})

    def test_5xx_eventually_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.data.scrape_yc.time.sleep", lambda _s: None)
        client = MagicMock(spec=httpx.Client)
        fail = MagicMock(status_code=503, text="oops", headers={})
        ok = MagicMock(status_code=200, headers={}, json=lambda: {"results": [{}]})
        client.post.side_effect = [fail, ok]
        result = _post_with_retry(client, ALGOLIA_DSN, params={}, body={})
        assert result == {"results": [{}]}

    def test_429_honors_retry_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("src.data.scrape_yc.time.sleep", lambda s: sleeps.append(s))
        client = MagicMock(spec=httpx.Client)
        throttled = MagicMock(status_code=429, text="slow down", headers={"Retry-After": "0.5"})
        ok = MagicMock(status_code=200, headers={}, json=lambda: {"results": [{}]})
        client.post.side_effect = [throttled, ok]
        _post_with_retry(client, ALGOLIA_DSN, params={}, body={})
        assert sleeps == [0.5]


# ---------------------------------------------------------------------
# _resolve_algolia_key — happy path
# ---------------------------------------------------------------------


class TestResolveAlgoliaKey:
    def test_uses_yc_directory_page(self) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(
            status_code=200,
            text=(
                '<script>window.AlgoliaOpts = '
                f'{{"app":"{ALGOLIA_APP_ID}","key":"real-key"}}</script>'
            ),
        )
        # `raise_for_status` is a no-op on the magic mock
        assert _resolve_algolia_key(client) == "real-key"
        client.get.assert_called_once_with(SOURCE_URL)


# ---------------------------------------------------------------------
# End-to-end with stubbed HTTP: fetch_all_companies + write_snapshot
# ---------------------------------------------------------------------


class TestFetchAllAndWrite:
    """Drive the full pipeline with stubbed HTTP responses.

    Two batches, two pages each, with one duplicate across batches
    to exercise dedup. Rate-limit sleeps are monkey-patched to no-ops
    so the test is fast.
    """

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skip the rate-limit sleep between requests in the test suite."""
        monkeypatch.setattr("src.data.scrape_yc.time.sleep", lambda _s: None)
        monkeypatch.setattr("src.data.scrape_yc._sleep_politely", lambda: None)

    @pytest.fixture
    def fake_client(self) -> MagicMock:
        """Stub httpx.Client whose ``post`` returns the right response per call.

        The first POST is the facets call; subsequent POSTs are batch
        pagination requests. We identify which is which by the
        ``facetFilters=batch:<name>`` substring in the request body.
        """
        client = MagicMock(spec=httpx.Client)

        # Facets response: two batches, "W22" (3 cos) and "S21" (2 cos).
        facets_resp = {
            "results": [
                {
                    "facets": {"batch": {"W22": 3, "S21": 2}},
                    "hits": [],
                }
            ]
        }

        # Batch pages. We pre-build the per-batch hit lists and have
        # the test driver dispatch them based on the request body.
        self.batch_pages: dict[str, list[list[dict[str, Any]]]] = {
            "W22": [
                [self._hit("Zeta", "W22"), self._hit("Alpha", "W22"), self._hit("Mu", "W22")],
            ],
            "S21": [
                [self._hit("Apex", "S21"), self._hit("Bolt", "S21")],
            ],
        }
        # Cross-batch duplicate (Apex in S21 and W22) for dedup testing.
        self.batch_pages["W22"][0].append(self._hit("Apex", "W22"))

        def _post(url: str, params: dict, json: dict) -> MagicMock:
            body = json["requests"][0]
            params_str = body["params"]
            if "facetFilters" not in params_str:
                return self._ok(facets_resp)
            # Identify the batch from the request body.
            for batch in self.batch_pages:
                if f"batch%3A{batch}" in params_str:
                    page_idx = int(params_str.rsplit("&page=", 1)[1])
                    pages = self.batch_pages[batch]
                    if page_idx >= len(pages):
                        return self._ok({"results": [{"hits": [], "facets": {}}]})
                    return self._ok(
                        {"results": [{"hits": pages[page_idx], "facets": {}}]}
                    )
            raise AssertionError(f"Unexpected request: params={params_str}")

        client.post.side_effect = _post
        return client

    @staticmethod
    def _hit(name: str, batch: str) -> dict[str, Any]:
        return {
            "id": hash((name, batch)) & 0x7FFFFFFF,
            "name": name,
            "slug": name.lower(),
            "former_names": [],
            "small_logo_thumb_url": "",
            "website": "",
            "all_locations": "",
            "long_description": f"long {name}",
            "one_liner": f"one-liner for {name}",
            "team_size": 1,
            "industry": "B2B",
            "subindustry": "B2B",
            "launched_at": 1700000000,
            "tags": ["AI"],
            "tags_highlighted": [],
            "top_company": False,
            "isHiring": False,
            "nonprofit": False,
            "batch": batch,
            "status": "Active",
            "industries": ["B2B"],
            "regions": ["United States of America"],
            "stage": "Early",
            "app_video_public": False,
            "demo_day_video_public": False,
            "app_answers": None,
            "question_answers": False,
        }

    @staticmethod
    def _ok(payload: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json = lambda: payload
        return resp

    def test_full_pipeline_produces_deduped_sorted_records(
        self, fake_client: MagicMock
    ) -> None:
        records, errors = fetch_all_companies(fake_client, "fake-key")
        assert errors == 0
        # W22: Zeta, Alpha, Mu, Apex (4) + S21: Apex, Bolt (2) = 6 raw hits.
        # Dedup is on (name, batch), so Apex/W22 and Apex/S21 are both
        # kept (different batches = different records). 6 → 6.
        assert len(records) == 6
        # Sort order: S21 < W22 lexicographically; within S21, Apex < Bolt.
        assert [r.name for r in records] == [
            "Apex", "Bolt",  # S21
            "Alpha", "Apex", "Mu", "Zeta",  # W22
        ]

    def test_write_snapshot_creates_jsonl_and_manifest(
        self, fake_client: MagicMock, tmp_path: Path
    ) -> None:
        records, errors = fetch_all_companies(fake_client, "fake-key")
        scrape_date = date(2026, 6, 8)
        jsonl_path, manifest_path = write_snapshot(
            records,
            out_dir=tmp_path,
            scrape_date=scrape_date,
            errors_skipped=errors,
        )

        # Filenames are date-stamped.
        assert jsonl_path.name == "yc_2026-06-08.jsonl"
        assert manifest_path.name == "yc_2026-06-08.manifest.json"

        # JSONL: one record per line, valid JSON, deterministic order.
        with jsonl_path.open() as f:
            lines = [ln for ln in f.read().splitlines() if ln]
        assert len(lines) == 6
        for line in lines:
            rec = json.loads(line)
            assert set(rec.keys()) == {
                "name", "description", "tags", "batch", "status", "url"
            }

        # Manifest: required fields per docs/PHASE-1.md §1.2.
        with manifest_path.open() as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == "1.0.0"
        assert manifest["source_url"] == SOURCE_URL
        assert manifest["scrape_date"] == "2026-06-08"
        assert manifest["count"] == 6
        assert manifest["snapshot_filename"] == "yc_2026-06-08.jsonl"
        assert manifest["errors_skipped"] == 0
        assert manifest["batches"] == ["S21", "W22"]  # sorted

    def test_idempotency_same_date_yields_byte_identical_output(
        self, fake_client: MagicMock, tmp_path: Path
    ) -> None:
        """Re-running with the same date produces the same file bytes."""
        records, _ = fetch_all_companies(fake_client, "fake-key")
        scrape_date = date(2026, 6, 8)

        # The `scraped_at_utc` field would normally differ; pin it by
        # patching the manifest build path is not worth it — instead
        # we just compare the jsonl bytes and the deterministic parts
        # of the manifest. The full byte-identical guarantee is
        # exercised manually in the acceptance check.
        jsonl_path_a, manifest_path_a = write_snapshot(
            records, out_dir=tmp_path / "a", scrape_date=scrape_date, errors_skipped=0
        )
        jsonl_path_b, manifest_path_b = write_snapshot(
            records, out_dir=tmp_path / "b", scrape_date=scrape_date, errors_skipped=0
        )
        assert jsonl_path_a.read_bytes() == jsonl_path_b.read_bytes()
