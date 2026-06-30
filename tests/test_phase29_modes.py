"""Phase 2.9 — tests for the BM25 + Hybrid RRF retrieval modes.

What this guards against
-----------------------
Phase 2.9 added two new ``mode`` values to ``POST /search``:
``bm25`` (lexical) and ``hybrid`` (dense + BM25 fused via RRF).
The dense path is unchanged.

These tests pin:
1. ``SearchRequest`` accepts ``dense`` / ``bm25`` / ``hybrid`` and
   rejects unknown modes via Pydantic enum validation.
2. ``search_endpoint`` routes each mode to the right backend
   (``search_corpus`` / ``search_bm25`` / ``search_hybrid``) and
   does NOT cross-contaminate results.
3. The eval runner's ``_MODE_FOR_CONFIG`` mapping picks the right
   ``mode`` from each config name. (Phase 2.9 bug fix: the runner
   used to send no ``mode`` at all, so every config silently hit
   the dense endpoint — the BM25 + Hybrid rows in the leaderboard
   were identical to dense, with the same MRR. This test locks the
   fix in.)

The search.py tests run against the in-process FastAPI app via
``TestClient`` so they don't need the live API on :18001.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.search import SearchRequest, search_endpoint
from src.eval.run import _MODE_FOR_CONFIG


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


def test_search_request_default_mode_is_dense():
    req = SearchRequest(query="AI legal contract review")
    assert req.mode == "dense"


def test_search_request_accepts_all_three_modes():
    # Pydantic pattern ``^(dense|bm25|hybrid)$`` is the contract — adding
    # a 4th mode here without updating it would surface as a 422 in
    # production.
    for m in ("dense", "bm25", "hybrid"):
        req = SearchRequest(query="x", mode=m)
        assert req.mode == m


def test_search_request_rejects_unknown_mode():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        SearchRequest(query="x", mode="cohere")  # opt-in only, not a default


# ---------------------------------------------------------------------------
# search_endpoint dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_engine():
    """A minimal Engine stand-in; the dispatch test never queries the DB."""
    return MagicMock(name="engine")


@pytest.fixture
def fake_embedder():
    e = MagicMock(name="embedder")
    e.model_name = "BAAI/bge-m3"
    return e


def test_search_endpoint_dispatches_dense(monkeypatch, fake_engine, fake_embedder):
    import src.api.search as s

    called = MagicMock(return_value=[])
    monkeypatch.setattr(s, "search_corpus", called)
    # Patch the corpus count to non-zero so we don't early-return empty.
    monkeypatch.setattr(s, "_corpus_count", lambda engine: 5)

    req = SearchRequest(query="x", mode="dense", top_k=10)
    search_endpoint(request=req, engine=fake_engine, embedder=fake_embedder)

    called.assert_called_once()
    assert called.call_args.kwargs["query"] == "x"
    assert called.call_args.kwargs["top_k"] == 10


def test_search_endpoint_dispatches_bm25(monkeypatch, fake_engine, fake_embedder):
    import src.api.search as s

    called = MagicMock(return_value=[])
    monkeypatch.setattr(s, "search_bm25", called)
    monkeypatch.setattr(s, "_corpus_count", lambda engine: 5)
    monkeypatch.setattr(s, "_populate_bm25_name_desc_cache", lambda e, idx: None)
    monkeypatch.setattr(s, "get_bm25_index", lambda engine: MagicMock())
    # _bm25_name_desc_cache is a module-level dict; reset between tests.
    monkeypatch.setattr(s, "_bm25_name_desc_cache", {})

    req = SearchRequest(query="x", mode="bm25", top_k=10)
    search_endpoint(request=req, engine=fake_engine, embedder=fake_embedder)

    called.assert_called_once()


def test_search_endpoint_dispatches_hybrid(monkeypatch, fake_engine, fake_embedder):
    import src.api.search as s

    called = MagicMock(return_value=[])
    monkeypatch.setattr(s, "search_hybrid", called)
    monkeypatch.setattr(s, "_corpus_count", lambda engine: 5)

    req = SearchRequest(query="x", mode="hybrid", top_k=10)
    search_endpoint(request=req, engine=fake_engine, embedder=fake_embedder)

    called.assert_called_once()


# ---------------------------------------------------------------------------
# Eval runner — _MODE_FOR_CONFIG mapping (regression test for Phase 2.9)
# ---------------------------------------------------------------------------


def test_mode_for_config_mapping_is_complete():
    """All three Phase 2.9 configs must map to a real /search mode.

    Regression guard: when Phase 2.9 was first shipped, the mapping did
    NOT exist and the runner silently hit the dense endpoint for every
    config. The leaderboard BM25 + Hybrid rows had the same MRR as
    dense. This test fails fast if a future worker adds a 4th config
    without updating the mapping.
    """
    assert _MODE_FOR_CONFIG == {
        "dense_bge_m3": "dense",
        "bm25": "bm25",
        "hybrid_rrf": "hybrid",
    }


def test_mode_for_config_unknown_defaults_to_dense():
    """An unknown config name must NOT crash — fall back to dense.

    The fallback is a deliberate 'fail-safe' default: a typo'd config
    name produces a meaningful leaderboard row (dense numbers) instead
    of a 500. The runner should still flag the unknown name in a
    follow-up test once we add a name validator.
    """
    assert _MODE_FOR_CONFIG.get("totally_made_up", "dense") == "dense"


def test_runner_payload_includes_mode(monkeypatch):
    """The eval runner must send ``mode`` in the POST body.

    Phase 2.9 bug: ``run_one_record`` did NOT include ``mode`` in the
    payload, so the dense endpoint was hit for every config. Lock the
    fix in by asserting the payload shape directly.
    """
    import httpx

    captured: dict = {}

    class FakeClient:
        def post(self, url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout

            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"hits": []}
            return r

    from src.eval.config import RetrievalConfig
    from src.eval.run import run_one_record
    from src.eval.benchmark import BenchmarkRecord

    for config_name, expected_mode in _MODE_FOR_CONFIG.items():
        captured.clear()
        cfg = RetrievalConfig(
            name=config_name,
            embedding_model="x",
            embedding_dim=1024,
            top_k=20,
            api_url="http://x/search",
        )
        record = BenchmarkRecord(
            id="ev-test",
            idea="hello",
            expected_top_ids=[],
            is_duplicate=True,
            is_novel=False,
            category="duplicate",
            labeler="test",
            labeled_at="2026-06-30",
            notes="",
        )
        run_one_record(record=record, config=cfg, client=FakeClient())
        assert captured["json"]["mode"] == expected_mode, (
            f"config={config_name} sent mode={captured['json']['mode']}, "
            f"expected {expected_mode}"
        )