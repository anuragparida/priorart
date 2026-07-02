"""Tests for the Phase 4.3 SearXNG + Firecrawl market-scope augmentation.

Covers the card's ``## Verify`` checklist:

- Stubbed Firecrawl client returning 3 fake search results →
  ``evidence`` has 3 web entries, ``confidence == "evidence_backed"``,
  ``search_volume_proxy == 3``, and the evidence URLs are the exact
  URLs the stub returned (proves no LLM-invented URLs).
- Offline mode flag set → ``confidence == "directional"`` and **zero**
  web calls (the stub's counters stay at 0).
- The direction gate ``should_augment`` fires only for the sparse
  directions with ``competitor_count < 10``.
"""

from __future__ import annotations

import pytest

from src.llm.schemas import MarketScope
from src.workflow.market_scope_web import (
    COMPETITOR_COUNT_AUGMENT_CEILING,
    MARKET_SCOPE_QUERY_TEMPLATE,
    OFFLINE_ENV_VAR,
    augment_market_scope_web,
    is_offline_mode,
    should_augment,
)


# ---------------------------------------------------------------------------
# Stub Firecrawl client
# ---------------------------------------------------------------------------


class StubFirecrawlClient:
    """A ``WebFallbackClient`` look-alike that records calls and returns
    canned data. No network, no httpx. Counters let the offline test
    assert that *no* web call happened.
    """

    def __init__(
        self,
        *,
        search_results: list[dict[str, str]] | None = None,
        scrape_markdown: str = "This is a competitor landscape page.",
    ) -> None:
        self._search_results = search_results or []
        self._scrape_markdown = scrape_markdown
        self.search_calls = 0
        self.scrape_calls = 0
        self.closed = False
        self.last_query: str | None = None

    def search(self, query: str, *, limit: int = 3) -> list[dict[str, str]]:
        self.search_calls += 1
        self.last_query = query
        return list(self._search_results[:limit])

    def scrape(self, url: str, *, max_chars: int = 4000) -> str:
        self.scrape_calls += 1
        md = self._scrape_markdown
        if max_chars > 0 and len(md) > max_chars:
            md = md[:max_chars]
        return md

    def close(self) -> None:
        self.closed = True


THREE_RESULTS = [
    {
        "url": "https://techcrunch.com/underwater-basket-weaving-ai",
        "title": "AI for underwater basket weaving raises $2M",
        "description": "A startup applying AI to underwater basket weaving.",
    },
    {
        "url": "https://www.producthunt.com/posts/basketweave-ai",
        "title": "BasketWeave AI",
        "description": "Automated basket weaving with computer vision.",
    },
    {
        "url": "https://news.ycombinator.com/item?id=99999",
        "title": "Show HN: I built an AI basket weaver",
        "description": "Show HN post about the space.",
    },
]


# ---------------------------------------------------------------------------
# Direction gate
# ---------------------------------------------------------------------------


class TestShouldAugment:
    def test_wide_open_sparse_fires(self) -> None:
        assert should_augment(MarketScope.WIDE_OPEN, 3) is True

    def test_niche_but_real_sparse_fires(self) -> None:
        assert should_augment(MarketScope.NICHE_BUT_REAL, 9) is True

    def test_ceiling_is_exclusive(self) -> None:
        # competitor_count == ceiling (10) must NOT fire — the corpus is
        # dense enough to be quantitative.
        assert should_augment(MarketScope.WIDE_OPEN, COMPETITOR_COUNT_AUGMENT_CEILING) is False

    def test_saturated_never_fires(self) -> None:
        assert should_augment(MarketScope.SATURATED, 2) is False

    def test_crowded_never_fires(self) -> None:
        assert should_augment(MarketScope.CROWDED_BUT_GROWING, 2) is False


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------


class TestOfflineMode:
    def test_is_offline_reads_truthy_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv(OFFLINE_ENV_VAR, val)
            assert is_offline_mode() is True

    def test_is_offline_false_when_unset_or_falsey(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
        assert is_offline_mode() is False
        monkeypatch.setenv(OFFLINE_ENV_VAR, "0")
        assert is_offline_mode() is False
        monkeypatch.setenv(OFFLINE_ENV_VAR, "")
        assert is_offline_mode() is False

    def test_offline_flag_short_circuits_no_web_calls(self) -> None:
        """With offline set, the activity returns directional and makes
        NO web calls (the CI self-contained guard)."""
        stub = StubFirecrawlClient(search_results=THREE_RESULTS)
        out = augment_market_scope_web(
            "AI for underwater basket weaving", client=stub, offline=True
        )
        assert out.confidence == "directional"
        assert out.evidence == []
        assert out.search_volume_proxy is None
        assert out.fired is False
        # The load-bearing assertion: zero network activity.
        assert stub.search_calls == 0
        assert stub.scrape_calls == 0

    def test_offline_via_env_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
        stub = StubFirecrawlClient(search_results=THREE_RESULTS)
        # offline=None → read the env flag.
        out = augment_market_scope_web(
            "AI for underwater basket weaving", client=stub, offline=None
        )
        assert out.confidence == "directional"
        assert stub.search_calls == 0
        assert stub.scrape_calls == 0


# ---------------------------------------------------------------------------
# The happy path — 3 stubbed results
# ---------------------------------------------------------------------------


class TestAugmentHappyPath:
    def test_three_results_evidence_backed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
        stub = StubFirecrawlClient(search_results=THREE_RESULTS)
        out = augment_market_scope_web(
            "AI for underwater basket weaving", client=stub, offline=False
        )

        assert out.confidence == "evidence_backed"
        assert out.fired is True
        assert len(out.evidence) == 3
        # 3 distinct domains → proxy == 3 (niche-but-real per the spec).
        assert out.search_volume_proxy == 3

    def test_evidence_urls_are_the_scraped_urls(self) -> None:
        """No LLM-invented URLs — every web evidence URL must be one the
        (stubbed) search actually returned. PHASE-4.md hard rule."""
        stub = StubFirecrawlClient(search_results=THREE_RESULTS)
        out = augment_market_scope_web("some idea", client=stub, offline=False)

        returned_urls = {r["url"] for r in THREE_RESULTS}
        for ev in out.evidence:
            assert ev.source == "web"
            assert ev.url in returned_urls
            assert ev.company_id is None
            assert ev.as_of is not None

    def test_query_uses_the_pinned_template(self) -> None:
        stub = StubFirecrawlClient(search_results=THREE_RESULTS)
        augment_market_scope_web("meal kits", client=stub, offline=False)
        assert stub.last_query == MARKET_SCOPE_QUERY_TEMPLATE.format(idea="meal kits")

    def test_snippet_falls_back_to_description_on_empty_scrape(self) -> None:
        stub = StubFirecrawlClient(search_results=THREE_RESULTS, scrape_markdown="")
        out = augment_market_scope_web("x", client=stub, offline=False)
        # Empty scrape → snippet is the SearXNG description.
        assert out.evidence[0].snippet == THREE_RESULTS[0]["description"]

    def test_scrape_is_called_per_kept_result(self) -> None:
        stub = StubFirecrawlClient(search_results=THREE_RESULTS)
        augment_market_scope_web("x", client=stub, offline=False)
        assert stub.search_calls == 1
        assert stub.scrape_calls == 3

    def test_owns_client_closed_when_not_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no client is injected, the created one is closed."""
        created: dict[str, StubFirecrawlClient] = {}

        def _factory() -> StubFirecrawlClient:
            c = StubFirecrawlClient(search_results=THREE_RESULTS)
            created["c"] = c
            return c

        monkeypatch.setattr(
            "src.workflow.market_scope_web.WebFallbackClient", _factory
        )
        out = augment_market_scope_web("x", offline=False)
        assert out.confidence == "evidence_backed"
        assert created["c"].closed is True


# ---------------------------------------------------------------------------
# Dedup + proxy edge cases
# ---------------------------------------------------------------------------


class TestDedupAndProxy:
    def test_duplicate_urls_collapse(self) -> None:
        dupes = [
            THREE_RESULTS[0],
            dict(THREE_RESULTS[0]),  # same URL again
            THREE_RESULTS[1],
        ]
        stub = StubFirecrawlClient(search_results=dupes)
        out = augment_market_scope_web("x", client=stub, offline=False)
        # 2 distinct URLs kept.
        assert len(out.evidence) == 2
        assert out.search_volume_proxy == 2

    def test_www_prefix_treated_as_same_domain(self) -> None:
        same_domain = [
            {"url": "https://example.com/a", "title": "A", "description": "a"},
            {"url": "https://www.example.com/b", "title": "B", "description": "b"},
        ]
        stub = StubFirecrawlClient(search_results=same_domain)
        out = augment_market_scope_web("x", client=stub, offline=False)
        # 2 distinct URLs but only 1 distinct domain.
        assert len(out.evidence) == 2
        assert out.search_volume_proxy == 1

    def test_empty_search_degrades_to_directional(self) -> None:
        stub = StubFirecrawlClient(search_results=[])
        out = augment_market_scope_web("x", client=stub, offline=False)
        assert out.confidence == "directional"
        assert out.evidence == []
        assert out.fired is True  # we tried
        assert stub.scrape_calls == 0

    def test_transport_error_on_search_degrades(self) -> None:
        from src.workflow.web_fallback import WebFallbackTransportError

        class BoomClient(StubFirecrawlClient):
            def search(self, query: str, *, limit: int = 3):
                self.search_calls += 1
                raise WebFallbackTransportError("boom")

        stub = BoomClient()
        out = augment_market_scope_web("x", client=stub, offline=False)
        assert out.confidence == "directional"
        assert out.fired is True

    def test_empty_idea_rejected(self) -> None:
        with pytest.raises(ValueError):
            augment_market_scope_web("   ", client=StubFirecrawlClient(), offline=False)


# ---------------------------------------------------------------------------
# No-LLM guarantee (static)
# ---------------------------------------------------------------------------


def test_module_makes_no_llm_call() -> None:
    """The module must not import an LLM client — it's a pure data fetch."""
    import inspect

    from src.workflow import market_scope_web

    source = inspect.getsource(market_scope_web)
    for banned in ("instructor", "anthropic", "Anthropic", "compare_topk", "llm_"):
        assert banned not in source, f"market_scope_web must not reference {banned!r}"
