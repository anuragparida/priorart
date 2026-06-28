"""Tests for the eval harness (Phase 1.6).

Coverage
--------
- ``src.eval.metrics``: worked examples for each of the five
  formulas (MRR, nDCG@10, P@5, R@10, FPR-on-novel). Each formula
  has 2-3 small hand-checkable cases so the math is auditable.
- ``src.eval.benchmark``: load + parse + per-record field check
  on a tiny synthetic JSONL.
- ``src.eval.config``: round-trip a YAML config.
- ``src.eval.run``: smoke-test the per-record / per-threshold /
  CSV-writer pipeline using a fake ``/search`` HTTP target. We
  don't hit the real API here (the live acceptance test is
  against the running ``uvicorn`` instance — see the completion
  summary for ``t_b94045be``).

Why these tests are offline
---------------------------
The live acceptance criteria are checked by running
``python -m eval.run`` against the running API. The pytest suite
is for *correctness* of the formulas and the wiring — no real
network, no real model. A reader auditing the metrics can read
the tests and the formulas side by side and trust both.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import pytest
import yaml

from src.eval.benchmark import Benchmark, BenchmarkLoadError, BenchmarkRecord, load_benchmark
from src.eval.config import RetrievalConfig
from src.eval.metrics import (
    DEFAULT_THRESHOLD_SWEEP,
    fpr_on_novel_record,
    ndcg_at_k,
    pick_best_threshold,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Metric tests — worked examples
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    """MRR reciprocal-rank predicate."""

    def test_hit_at_rank_1(self):
        assert reciprocal_rank([10, 20, 30], [10]) == 1.0

    def test_hit_at_rank_3(self):
        assert reciprocal_rank([10, 20, 30, 40], [30]) == pytest.approx(1.0 / 3.0)

    def test_hit_at_rank_5(self):
        assert reciprocal_rank([10, 20, 30, 40, 50], [50]) == pytest.approx(1.0 / 5.0)

    def test_no_hit(self):
        assert reciprocal_rank([10, 20, 30], [99]) == 0.0

    def test_empty_ranked(self):
        assert reciprocal_rank([], [10]) == 0.0

    def test_empty_expected(self):
        # No expected → no reciprocal rank (no relevant to find)
        assert reciprocal_rank([10, 20], []) == 0.0

    def test_first_match_wins(self):
        # If the first match in the ranked list is at rank 2, the
        # reciprocal rank is 0.5, even if another expected id
        # appears at rank 1 of a different sub-list. We only
        # honour the FIRST occurrence.
        assert reciprocal_rank([10, 20, 30, 40], [40, 20]) == 0.5

    def test_dedup_of_expected(self):
        # Listing the same expected id twice still gives the same
        # reciprocal rank — the metric doesn't double-count.
        assert reciprocal_rank([10, 20, 30], [20, 20, 20]) == 0.5


class TestNDCG:
    """nDCG@10 formula (binary relevance)."""

    def test_perfect_ranking(self):
        # All relevant ids at the top → nDCG = 1.0
        ranked = [10, 20, 30]
        expected = [10, 20, 30]
        assert ndcg_at_k(ranked, expected, k=10) == pytest.approx(1.0)

    def test_no_relevant(self):
        # None of the ranked are relevant → nDCG = 0.0
        ranked = [10, 20, 30]
        expected = [99, 98]
        assert ndcg_at_k(ranked, expected, k=10) == 0.0

    def test_partial_with_log_discount(self):
        # Relevant at ranks 2 and 3, expected = 2 ids.
        # DCG = 1/log2(2+1) + 1/log2(3+1) = 1/log2(3) + 1/log2(4)
        # IDCG = same formula with [10, 20] (best possible)
        ranked = [10, 20, 30, 40]  # pretend 20 and 30 are relevant
        expected = [20, 30]
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(ranked, expected, k=4) == pytest.approx(dcg / idcg)

    def test_k_truncation(self):
        # Even if expected has 5 ids, nDCG@3 only counts the top 3.
        ranked = [10, 20, 30, 40, 50]
        expected = [10, 20, 30, 40, 50]
        assert ndcg_at_k(ranked, expected, k=3) == pytest.approx(1.0)
        # At k=5 it's still 1.0 because all 5 are relevant.
        assert ndcg_at_k(ranked, expected, k=5) == pytest.approx(1.0)

    def test_empty_ranked(self):
        assert ndcg_at_k([], [10, 20], k=10) == 0.0

    def test_empty_expected(self):
        assert ndcg_at_k([10, 20], [], k=10) == 0.0


class TestPrecisionAtK:
    """precision@K = fraction of top-K that are relevant."""

    def test_half_relevant(self):
        assert precision_at_k([10, 20, 30, 40], [20, 40], k=4) == 0.5

    def test_all_relevant(self):
        assert precision_at_k([10, 20, 30], [10, 20, 30], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([10, 20, 30], [99], k=3) == 0.0

    def test_k_caps_top(self):
        # Only the first 3 ranked are considered.
        ranked = [10, 20, 30, 99, 98]
        assert precision_at_k(ranked, [99], k=3) == 0.0  # 99 not in top 3
        assert precision_at_k(ranked, [99], k=5) == 0.2  # 99 is in top 5

    def test_k_zero(self):
        assert precision_at_k([10, 20], [10], k=0) == 0.0


class TestRecallAtK:
    """recall@K = fraction of expected that appear in top-K."""

    def test_half_relevant(self):
        # 2 of 4 expected are in top-4 → recall = 0.5
        assert recall_at_k([10, 20, 30, 40], [20, 40, 99, 98], k=4) == 0.5

    def test_all_relevant(self):
        assert recall_at_k([10, 20, 30], [10, 20, 30], k=3) == 1.0

    def test_none_relevant(self):
        assert recall_at_k([10, 20, 30], [99, 98], k=3) == 0.0

    def test_k_caps(self):
        # recall@2 only counts expected ids that appear in top 2.
        ranked = [10, 20, 30, 40]
        # expected = {10, 30, 40}. Top-2 contains 10 only.
        assert recall_at_k(ranked, [10, 30, 40], k=2) == pytest.approx(1.0 / 3.0)


class TestFprOnNovel:
    """FPR-on-novel predicate: novel record with top-1 above threshold."""

    def test_novel_with_high_top1_is_fp(self):
        assert fpr_on_novel_record(is_novel=True, top1_score=0.7, threshold=0.65) == 1.0

    def test_novel_with_low_top1_is_not_fp(self):
        assert fpr_on_novel_record(is_novel=True, top1_score=0.5, threshold=0.65) == 0.0

    def test_novel_with_no_hits_is_not_fp(self):
        # No top-1 hit → can't be a false positive (nothing claimed).
        assert fpr_on_novel_record(is_novel=True, top1_score=None, threshold=0.65) == 0.0

    def test_duplicate_never_counts(self):
        # Even if a duplicate record has a top-1 above the
        # threshold, FPR-on-novel is 0.0 (the metric only counts
        # novel records).
        assert fpr_on_novel_record(is_novel=False, top1_score=0.99, threshold=0.5) == 0.0

    def test_boundary_above_threshold(self):
        # top1 == threshold counts as a positive (>=).
        assert fpr_on_novel_record(is_novel=True, top1_score=0.65, threshold=0.65) == 1.0
        assert fpr_on_novel_record(is_novel=True, top1_score=0.6499, threshold=0.65) == 0.0


class TestPickBestThreshold:
    """``pick_best_threshold`` chooses MRR-max subject to FPR cap."""

    def test_picks_max_mrr_under_cap(self):
        mrr = {0.5: 0.7, 0.6: 0.8, 0.7: 0.9, 0.8: 0.5}
        fpr = {0.5: 0.30, 0.6: 0.20, 0.7: 0.10, 0.8: 0.05}
        # Under FPR cap 0.15: 0.7 (0.10) and 0.8 (0.05) qualify.
        # Highest MRR among those = 0.9 at threshold 0.7.
        assert pick_best_threshold(
            threshold_sweep=[0.5, 0.6, 0.7, 0.8],
            mrr_by_threshold=mrr,
            fpr_by_threshold=fpr,
            fpr_cap=0.15,
        ) == 0.7

    def test_lowest_fpr_when_no_threshold_meets_cap(self):
        mrr = {0.5: 0.7, 0.6: 0.8, 0.7: 0.9, 0.8: 0.95}
        fpr = {0.5: 0.50, 0.6: 0.40, 0.7: 0.30, 0.8: 0.20}
        # Nothing meets 0.15 cap → pick the one with the lowest FPR.
        assert pick_best_threshold(
            threshold_sweep=[0.5, 0.6, 0.7, 0.8],
            mrr_by_threshold=mrr,
            fpr_by_threshold=fpr,
            fpr_cap=0.15,
        ) == 0.8

    def test_ties_break_to_lower_threshold(self):
        # Two thresholds tie on MRR; prefer the more permissive (lower).
        mrr = {0.5: 0.8, 0.7: 0.8}
        fpr = {0.5: 0.10, 0.7: 0.10}
        assert pick_best_threshold(
            threshold_sweep=[0.5, 0.7],
            mrr_by_threshold=mrr,
            fpr_by_threshold=fpr,
            fpr_cap=0.15,
        ) == 0.5

    def test_empty_sweep_raises(self):
        with pytest.raises(ValueError):
            pick_best_threshold(
                threshold_sweep=[],
                mrr_by_threshold={},
                fpr_by_threshold={},
                fpr_cap=0.15,
            )


# ---------------------------------------------------------------------------
# Benchmark loader tests
# ---------------------------------------------------------------------------


def _write_jsonl(tmp: Path, records: Sequence[dict]) -> Path:
    p = tmp / "bench.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _sample_record(idx: int = 1) -> dict:
    return {
        "id": f"ev-{idx:03d}",
        "idea": f"sample idea {idx}",
        "source": "yc",
        "category": "duplicate",
        "expected_top_ids": [10],
        "is_duplicate": True,
        "labeler": "anurag",
        "labeled_at": "2026-06-28T12:30:00Z",
        "notes": "",
    }


class TestBenchmarkLoader:
    def test_load_minimal_record(self, tmp_path):
        p = _write_jsonl(tmp_path, [_sample_record(1)])
        bench = load_benchmark(p)
        assert len(bench) == 1
        assert bench.records[0].id == "ev-001"
        assert bench.records[0].idea == "sample idea 1"
        assert bench.records[0].is_novel is False  # is_duplicate=True → is_novel=False
        assert bench.records[0].is_duplicate is True

    def test_novel_record_has_empty_expected(self, tmp_path):
        rec = _sample_record(1)
        rec["is_duplicate"] = False
        rec["category"] = "novel"
        rec["expected_top_ids"] = []
        p = _write_jsonl(tmp_path, [rec])
        bench = load_benchmark(p)
        assert bench.records[0].is_novel is True
        assert bench.records[0].expected_top_ids == ()

    def test_blank_lines_are_skipped(self, tmp_path):
        p = tmp_path / "b.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(_sample_record(1)) + "\n")
            f.write("\n")  # blank line
            f.write("   \n")  # whitespace-only
            f.write(json.dumps(_sample_record(2)) + "\n")
        bench = load_benchmark(p)
        assert len(bench) == 2

    def test_skips_invalid_lines_by_default(self, tmp_path, capsys):
        p = tmp_path / "b.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(_sample_record(1)) + "\n")
            f.write("not-valid-json\n")
            f.write(json.dumps(_sample_record(2)) + "\n")
        bench = load_benchmark(p)
        assert len(bench) == 2
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()

    def test_missing_required_key_raises_when_not_skipping(self, tmp_path):
        bad = {"id": "ev-001", "idea": "x"}  # missing expected_top_ids etc.
        p = _write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError):
            load_benchmark(p, skip_invalid=False)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(BenchmarkLoadError):
            load_benchmark(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(BenchmarkLoadError):
            load_benchmark(p)

    def test_novel_and_duplicate_helpers(self, tmp_path):
        records = [
            {**_sample_record(1), "is_duplicate": True, "category": "duplicate"},
            {**_sample_record(2), "is_duplicate": False, "category": "novel"},
            {**_sample_record(3), "is_duplicate": False, "category": "adversarial_paraphrase"},
        ]
        p = _write_jsonl(tmp_path, records)
        bench = load_benchmark(p)
        assert len(bench.duplicate_records()) == 1
        assert len(bench.novel_records()) == 2


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "name: dense_bge_m3\n"
            "embedding_model: BAAI/bge-m3\n"
            "embedding_dim: 1024\n"
            "top_k: 20\n"
            "api_url: http://localhost:18001/search\n"
            "notes: smoke test\n"
        )
        cfg = RetrievalConfig.from_yaml(p)
        assert cfg.name == "dense_bge_m3"
        assert cfg.embedding_model == "BAAI/bge-m3"
        assert cfg.embedding_dim == 1024
        assert cfg.top_k == 20
        assert cfg.api_url == "http://localhost:18001/search"
        assert cfg.notes == "smoke test"

    def test_missing_required_key_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("name: foo\n")  # missing the others
        with pytest.raises(ValueError, match="missing required keys"):
            RetrievalConfig.from_yaml(p)

    def test_unknown_keys_are_ignored(self, tmp_path):
        # Forward-compat: typos in the YAML don't crash the loader.
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "name: foo\n"
            "embedding_model: BAAI/bge-m3\n"
            "embedding_dim: 1024\n"
            "top_k: 20\n"
            "api_url: http://localhost:18001/search\n"
            "weird_extra: ignore me\n"
        )
        cfg = RetrievalConfig.from_yaml(p)
        assert cfg.name == "foo"

    def test_wrong_type_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "name: foo\n"
            "embedding_model: BAAI/bge-m3\n"
            "embedding_dim: not_an_int\n"
            "top_k: 20\n"
            "api_url: http://localhost:18001/search\n"
        )
        with pytest.raises(ValueError, match="embedding_dim must be int"):
            RetrievalConfig.from_yaml(p)

    def test_empty_yaml_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("")
        with pytest.raises(ValueError, match="must be a mapping"):
            RetrievalConfig.from_yaml(p)


# ---------------------------------------------------------------------------
# CSV writer tests
# ---------------------------------------------------------------------------


class TestCsvWriter:
    def test_writes_header_on_first_run(self, tmp_path):
        from src.eval.run import _CSV_COLUMNS, write_csv

        out = tmp_path / "lb.csv"
        write_csv(out, [{"config": "dense_bge_m3", "threshold": 0.65, "mrr": 0.7}], append=False)
        with open(out) as f:
            lines = f.read().strip().split("\n")
        assert lines[0] == ",".join(_CSV_COLUMNS)
        assert len(lines) == 2

    def test_append_does_not_rewrite_header(self, tmp_path):
        from src.eval.run import write_csv

        out = tmp_path / "lb.csv"
        write_csv(
            out,
            [{"config": "dense_bge_m3", "benchmark": "lb.jsonl", "threshold": 0.65, "mrr": 0.7}],
            append=False,
        )
        write_csv(
            out,
            [{"config": "dense_bge_m3", "benchmark": "lb.jsonl", "threshold": 0.7, "mrr": 0.8}],
            append=True,
        )
        with open(out) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert lines[1].split(",")[0] == "dense_bge_m3"
        assert lines[1].split(",")[1] == "lb.jsonl"
        assert lines[2].split(",")[1] == "lb.jsonl"
        # Threshold values appear in the threshold column (5th, index 4).
        assert lines[1].split(",")[4] == "0.65"
        assert lines[2].split(",")[4] == "0.7"


# ---------------------------------------------------------------------------
# Markdown summary test
# ---------------------------------------------------------------------------


class TestMarkdownSummary:
    def test_marks_selected_row_in_bold(self):
        from src.eval.run import _format_markdown_table

        rows = [
            {
                "threshold": 0.65,
                "mrr": 0.70,
                "ndcg_at_10": 0.60,
                "precision_at_5": 0.50,
                "recall_at_10": 0.40,
                "fpr_on_novel": 0.10,
                "selected_threshold": True,
            },
            {
                "threshold": 0.7,
                "mrr": 0.50,
                "ndcg_at_10": 0.40,
                "precision_at_5": 0.30,
                "recall_at_10": 0.20,
                "fpr_on_novel": 0.05,
                "selected_threshold": False,
            },
        ]
        md = _format_markdown_table(
            rows,
            config_name="dense_bge_m3",
            benchmark_path=Path("evals/labeled_v100.jsonl"),
            best_threshold=0.65,
        )
        # The selected row's values must be wrapped in **bold**.
        assert "**0.65**" in md
        assert "**0.700**" in md
        # The non-selected row must NOT be bolded.
        assert "**0.7**" not in md


# ---------------------------------------------------------------------------
# Default threshold sweep
# ---------------------------------------------------------------------------


class TestDefaultSweep:
    def test_sweep_has_seven_values(self):
        assert len(DEFAULT_THRESHOLD_SWEEP) == 7

    def test_sweep_spacing(self):
        assert DEFAULT_THRESHOLD_SWEEP == [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]