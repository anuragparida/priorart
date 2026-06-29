"""Tests for the MLflow tracking wrapper (Phase 2.4).

Coverage
--------
1. ``resolve_tracking_uri`` honours the explicit argument > env var > default precedence.
2. ``is_tracking_server_reachable`` returns True for ``file:`` and ``databricks:``
   schemes (no auto-fallback) and respects the timeout on unreachable HTTP.
3. ``fallback_file_uri`` returns a fresh tmp path each call.
4. ``params_from_summary`` / ``metrics_from_summary`` produce the canonical
   4-param / 5-metric dict the spec calls for.
5. ``log_run`` lands a run with all 5 spec metrics and the prompt artifact
   when the tracking server is reachable (offline file-store probe when not).

We test the offline (file-store) flavour of (5) here so the suite is CI-safe;
the integration-with-the-live-server path is exercised by the ``make eval``
acceptance gate (not by this pytest).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest


# Test isolation: every test gets its own tmp tracking dir so concurrent
# mlruns files don't stomp on each other.
@pytest.fixture
def clean_mlflow_env(monkeypatch, tmp_path):
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{mlflow_dir.as_posix()}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    # Force a fresh process-level resolution each test.
    yield mlflow_dir


# ---------------------------------------------------------------------------
# URI resolution
# ---------------------------------------------------------------------------


class TestResolveTrackingUri:
    def test_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-host:15000")
        from src.eval.mlflow_logger import resolve_tracking_uri
        assert resolve_tracking_uri("http://explicit-host:5000") == "http://explicit-host:5000"

    def test_env_var_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-host:15000")
        from src.eval.mlflow_logger import resolve_tracking_uri
        assert resolve_tracking_uri() == "http://env-host:15000"

    def test_default_fallback(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        from src.eval.mlflow_logger import resolve_tracking_uri, DEFAULT_TRACKING_URI
        assert resolve_tracking_uri() == DEFAULT_TRACKING_URI
        # The default MUST be 15000 not 5000 — Honcho collision.
        assert DEFAULT_TRACKING_URI.endswith(":15000")


class TestIsTrackingServerReachable:
    def test_file_uri_always_considered_reachable(self):
        # ``file:`` URIs don't go through the network — the probe
        # short-circuits to True so we don't trigger an auto-fallback
        # the operator didn't ask for.
        from src.eval.mlflow_logger import is_tracking_server_reachable
        assert is_tracking_server_reachable("file:./mlruns") is True
        assert is_tracking_server_reachable("file:/tmp/foo") is True

    def test_databricks_uri_always_reachable(self):
        from src.eval.mlflow_logger import is_tracking_server_reachable
        assert is_tracking_server_reachable("databricks://profile") is True

    def test_unreachable_http_returns_false_quickly(self):
        # 2501 / 9 (the discard port) is the canonical unreachable
        # port used by IANA — it's reserved and never listens.
        from src.eval.mlflow_logger import is_tracking_server_reachable
        t0 = time.time()
        result = is_tracking_server_reachable(
            "http://127.0.0.1:9", timeout=0.5
        )
        elapsed = time.time() - t0
        assert result is False
        assert elapsed < 1.0  # the timeout was honoured


class TestFallbackFileUri:
    def test_unique_per_call(self):
        from src.eval.mlflow_logger import fallback_file_uri
        a = fallback_file_uri()
        time.sleep(0.005)  # ensure timestamp granularity
        b = fallback_file_uri()
        assert a != b
        assert a.startswith("file:")
        assert b.startswith("file:")


class TestArtifactLocationWritable:
    def test_writable_user_dir_returns_true(self, tmp_path):
        from src.eval.mlflow_logger import _is_artifact_location_writable
        writable = f"file://{tmp_path.as_posix()}"
        assert _is_artifact_location_writable(writable) is True

    def test_unwritable_root_returns_false(self):
        from src.eval.mlflow_logger import _is_artifact_location_writable
        # /tmp-fake/ never exists and ``od`` can't create it.
        unwritable = "file:///proc/cant-write-here"
        assert _is_artifact_location_writable(unwritable) is False

    def test_non_file_uri_returns_true(self):
        from src.eval.mlflow_logger import _is_artifact_location_writable
        for uri in (
            "s3://bucket/key",
            "gs://bucket/key",
            "http://localhost:15000/api/...",
            "databricks://profile",
        ):
            assert _is_artifact_location_writable(uri) is True


# ---------------------------------------------------------------------------
# Param / metric shape
# ---------------------------------------------------------------------------


class TestParamsAndMetricsShape:
    def test_params_contain_all_spec_keys(self):
        from src.eval.mlflow_logger import params_from_summary
        params = params_from_summary(
            config_name="dense_bge_m3",
            embedding_model="BAAI/bge-m3",
            threshold=0.65,
            benchmark_name="labeled_v100.jsonl",
            corpus_count=5949,
            corpus_snapshot_date="2026-06-08",
            prompt_template_version="compare-v1",
            api_url="http://localhost:18001/search",
            top_k=20,
        )
        # Spec: "embedding_model", "threshold", "prompt_template_version",
        # "corpus_snapshot_date" must be present.
        for required in (
            "embedding_model",
            "threshold",
            "prompt_template_version",
            "corpus_snapshot_date",
        ):
            assert required in params, f"missing required param: {required}"
        # All values are strings (MLflow's contract for log_param).
        for k, v in params.items():
            assert isinstance(v, str), f"{k!r} must be str, got {type(v).__name__}"
        # corpus_count only included when positive.
        assert "corpus_count" in params

    def test_params_omit_corpus_count_when_zero(self):
        from src.eval.mlflow_logger import params_from_summary
        params = params_from_summary(
            config_name="x",
            embedding_model="x",
            threshold=0.5,
            benchmark_name="x",
            corpus_count=0,
            corpus_snapshot_date="unknown",
            prompt_template_version="x",
            api_url="x",
            top_k=0,
        )
        assert "corpus_count" not in params

    def test_metrics_contain_all_spec_keys(self):
        from src.eval.mlflow_logger import metrics_from_summary
        summary = {
            "best_mrr": 0.559,
            "best_ndcg_at_10": 0.500,
            "best_precision_at_5": 0.40,
            "best_recall_at_10": 0.60,
            "best_fpr_on_novel": 0.80,
        }
        metrics = metrics_from_summary(summary, best_threshold=0.65)
        for required in (
            "mrr",
            "ndcg_at_10",
            "precision_at_5",
            "recall_at_10",
            "fpr_on_novel",
            "best_threshold",
        ):
            assert required in metrics, f"missing required metric: {required}"
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k!r} must be float, got {type(v).__name__}"


# ---------------------------------------------------------------------------
# end-to-end log_run against the offline file-store
# ---------------------------------------------------------------------------


class TestLogRunOffline:
    """The file-store flavour is the CI-safe path: no server required."""

    def test_log_run_lands_with_all_spec_fields(self, clean_mlflow_env):
        import mlflow
        from mlflow.tracking import MlflowClient

        from src.eval.mlflow_logger import (
            log_run,
            params_from_summary,
            metrics_from_summary,
        )

        params = params_from_summary(
            config_name="dense_bge_m3",
            embedding_model="BAAI/bge-m3",
            threshold=0.65,
            benchmark_name="labeled_v100.jsonl",
            corpus_count=100,
            corpus_snapshot_date="2026-06-08",
            prompt_template_version="compare-v1",
            api_url="http://localhost:18001/search",
            top_k=20,
        )
        metrics = metrics_from_summary(
            {
                "best_mrr": 0.5,
                "best_ndcg_at_10": 0.4,
                "best_precision_at_5": 0.3,
                "best_recall_at_10": 0.6,
                "best_fpr_on_novel": 0.15,
            },
            best_threshold=0.65,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = log_run(
                experiment_name="test-phase-2-4",
                params=params,
                metrics=metrics,
                artifacts={},
                prompt_template_text=(
                    "# SYSTEM_PROMPT (compare-v1)\n\n"
                    "you are a research analyst\n"
                ),
                run_name="unit-test-run",
                tags={"phase": "2.4", "kind": "unit-test"},
            )

        # ``file:`` URIs are always considered "reachable" by
        # ``is_tracking_server_reachable`` (no auto-fallback to a
        # different storage), so ``fallback_used`` stays False.
        # The run lands in the file-store nonetheless, which is the
        # whole point of the offline flavour.
        assert record.tracking_uri_effective.startswith("file:")
        assert record.fallback_used is False

        # Inspect via a *fresh* MlflowClient that talks to the
        # file-store root we wrote to. MLflow 3.x's ``search_runs``
        # takes ``experiment_ids=`` (a list of strings), not the
        # legacy ``experiment_names=`` kwarg.
        client = MlflowClient()
        exp = client.get_experiment_by_name("test-phase-2-4")
        assert exp is not None, "experiment not created by log_run"
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=10,
        )
        assert len(runs) >= 1, "log_run did not land any runs"

        run_id = runs[0].info.run_id
        run = client.get_run(run_id)

        # MLflow 3.x returns ``run.data.{params,metrics,tags}`` as
        # plain ``dict`` objects. Older 2.x versions returned a list
        # of entity objects with ``.key`` / ``.value`` attributes.
        # We normalise to a dict either way.
        def _as_dict(value):
            if isinstance(value, dict):
                return value
            try:
                return {entry.key: entry.value for entry in value}
            except AttributeError:
                # MLflow 3.x may also return a list of ``(key,value)``
                # tuples on some code paths.
                try:
                    return dict(value)
                except Exception:
                    return {}

        params_seen = _as_dict(run.data.params)
        metrics_seen = _as_dict(run.data.metrics)
        tags_seen = _as_dict(run.data.tags)

        # All spec-required params must show up.
        for k in (
            "embedding_model",
            "threshold",
            "prompt_template_version",
            "corpus_snapshot_date",
        ):
            assert params_seen.get(k) == params[k], f"missing/wrong param: {k}"

        # All spec-required metrics must show up.
        for k in (
            "mrr",
            "ndcg_at_10",
            "precision_at_5",
            "recall_at_10",
            "fpr_on_novel",
        ):
            assert k in metrics_seen, f"missing metric: {k}"

        # Custom tag set was applied.
        assert tags_seen.get("phase") == "2.4"
        assert tags_seen.get("kind") == "unit-test"

        # Prompt template was logged as an ARTIFACT, not a param.
        # Verify by listing the artifacts dir; the prompt_template.txt
        # file must be there.
        artifact_path = Path(runs[0].info.artifact_uri.replace("file://", ""))
        assert (artifact_path / "prompt_template.txt").exists(), (
            "prompt template MUST be logged as an artifact (file), "
            "not as a param — PHASE-2.md pitfall"
        )


# ---------------------------------------------------------------------------
# Selector helpers (run.py integration seam)
# ---------------------------------------------------------------------------


class TestCorpusSnapshotDateHelper:
    def test_reads_latest_snapshot(self, tmp_path):
        # The helper lives in src.eval.run (it's the run-side glue,
        # not the logger-side primitives).
        from src.eval.run import corpus_snapshot_date_from_snapshots_dir as reader
        # Empty dir → unknown.
        assert reader(tmp_path) == "unknown"
        # Add two snapshots.
        (tmp_path / "yc_2025-01-01.jsonl").write_text("[]")
        (tmp_path / "yc_2026-06-08.jsonl").write_text("[]")
        (tmp_path / "yc_2025-12-31.manifest.json").write_text("{}")
        # Latest is 2026-06-08.
        assert reader(tmp_path) == "2026-06-08"

    def test_ignores_non_snapshot_files(self, tmp_path):
        from src.eval.run import corpus_snapshot_date_from_snapshots_dir as reader
        (tmp_path / "yc_2025-12-31.jsonl").write_text("[]")
        (tmp_path / "not_a_snapshot.jsonl").write_text("[]")
        (tmp_path / "README.md").write_text("")
        assert reader(tmp_path) == "2025-12-31"

    def test_handles_missing_dir(self, tmp_path):
        from src.eval.run import corpus_snapshot_date_from_snapshots_dir as reader
        assert reader(tmp_path / "does_not_exist") == "unknown"


class TestPromptTemplateVersion:
    def test_prompt_version_constant_exists(self):
        from src.llm.prompts.compare import PROMPT_TEMPLATE_VERSION
        assert isinstance(PROMPT_TEMPLATE_VERSION, str)
        # The format is ``compare-vN``. Tolerate any suffix.
        assert PROMPT_TEMPLATE_VERSION.startswith("compare-v")


# ---------------------------------------------------------------------------
# Integration with src.eval.run — log_eval_run_to_mlflow no-op path
# ---------------------------------------------------------------------------


class TestLogEvalRunNoMlflow:
    """When ``--no-mlflow`` is passed, ``log_eval_run_to_mlflow`` is a
    short-circuit that returns ``None`` and never touches the network.

    This is the cheapest path in the production flow — when the
    tracking server is down and the operator only wants the leaderboard
    row, ``--no-mlflow`` saves the per_record.csv write and the
    per-record tempfile.
    """

    def test_no_mlflow_flag_returns_none(self, clean_mlflow_env):
        from src.eval.benchmark import Benchmark
        from src.eval.config import RetrievalConfig
        from src.eval.run import (
            PerRecordResult,
            log_eval_run_to_mlflow,
        )

        # Build a tiny summary that satisfies the runner contract.
        summary = {
            "rows": [
                {
                    "config": "dense_bge_m3",
                    "benchmark": "labeled_v100.jsonl",
                    "corpus_count": 100,
                    "embedding_model": "BAAI/bge-m3",
                    "threshold": 0.65,
                    "mrr": 0.5,
                    "ndcg_at_10": 0.4,
                    "precision_at_5": 0.3,
                    "recall_at_10": 0.6,
                    "fpr_on_novel": 0.15,
                    "records_total": 1,
                    "records_novel": 1,
                    "records_duplicate": 0,
                    "records_skipped": 0,
                    "search_errors": 0,
                    "selected_threshold": True,
                    "notes": "",
                }
            ],
            "best_threshold": 0.65,
            "best_mrr": 0.5,
            "best_ndcg_at_10": 0.4,
            "best_precision_at_5": 0.3,
            "best_recall_at_10": 0.6,
            "best_fpr_on_novel": 0.15,
            "fpr_cap": 0.15,
            "records_total": 1,
            "records_novel": 1,
            "records_duplicate": 0,
            "elapsed_seconds": 0.1,
            "search_errors": 0,
            "config": "dense_bge_m3",
            "benchmark": "labeled_v100.jsonl",
            "per_record": [],
        }
        cfg = RetrievalConfig(
            name="dense_bge_m3",
            embedding_model="BAAI/bge-m3",
            embedding_dim=1024,
            top_k=20,
            api_url="http://localhost:18001/search",
            notes="",
        )
        bench = Benchmark(records=(), path=Path("evals/labeled_v100.jsonl"))
        result = log_eval_run_to_mlflow(
            summary,
            config=cfg,
            benchmark=bench,
            output_csv=Path("results/leaderboard.csv"),
            per_record=(),
            no_mlflow=True,
        )
        assert result is None

