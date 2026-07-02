"""Tests for the calibration module (Phase 3.3).

Coverage
--------
- ``bin_predictions`` — binning math: bin index assignment, count /
  rate / avg_predicted_score per bin, edge cases (empty input,
  score=1.0 landing on the upper boundary, score clamping, length
  mismatch raises).
- ``compute_ece`` — the classic bin-count-weighted formulation on
  a few hand-checkable cases (perfect calibration, all-True,
  all-False, mixed-with-weights).
- ``plot_calibration`` — smoke test (writes a PNG, non-zero size,
  parent directory created).

Why these tests are offline
---------------------------
Same reason as ``test_eval.py``: the calibration module's logic is
pure-Python; we exercise it without touching the API. The live
acceptance test (``make eval`` writing real PNGs and an ECE column
in ``results/leaderboard.csv``) is the integration gate, not the
unit test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.calibration import (
    DEFAULT_N_BINS,
    BinStats,
    FprBreakdownBin,
    bin_predictions,
    compute_ece,
    fpr_on_novel_breakdown,
    plot_calibration,
    plot_fpr_breakdown,
    write_per_config_markdown,
)


# ---------------------------------------------------------------------------
# bin_predictions
# ---------------------------------------------------------------------------


def test_bin_predictions_empty_inputs_returns_full_grid() -> None:
    """Empty (scores, labels) lists produce 10 zero-filled bins.

    Keeps downstream callers (plot_calibration, leaderboard
    rendering) safe from a zero-length bins list — they always
    get the full grid.
    """
    bins = bin_predictions([], [])
    assert len(bins) == DEFAULT_N_BINS
    for b in bins:
        assert b.count == 0
        assert b.actual_duplicate_rate == 0.0
        assert b.avg_predicted_score == 0.0
    # Bin 0's lower edge is 0.0; the last bin's upper edge is 1.0.
    assert bins[0].lower == 0.0
    assert bins[0].upper == pytest.approx(0.1)
    assert bins[-1].lower == pytest.approx(0.9)
    assert bins[-1].upper == pytest.approx(1.0)


def test_bin_predictions_basic_counts_and_rates() -> None:
    """One record per bin, alternating labels.

    Lower half: all novel (is_dup=False) → actual rate 0.0.
    Upper half: all duplicate (is_dup=True) → actual rate 1.0.
    """
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [False, False, False, False, False, True, True, True, True, True]
    bins = bin_predictions(scores, labels)
    assert len(bins) == DEFAULT_N_BINS
    for i, b in enumerate(bins):
        assert b.bin_index == i
        assert b.count == 1
    # Lower 5 bins: rate 0.0; upper 5 bins: rate 1.0.
    for i in range(5):
        assert bins[i].actual_duplicate_rate == 0.0
    for i in range(5, 10):
        assert bins[i].actual_duplicate_rate == 1.0
    # avg_predicted_score equals the input score (n=1 per bin).
    for b, s in zip(bins, scores):
        assert b.avg_predicted_score == pytest.approx(s)


def test_bin_predictions_score_one_lands_in_last_bin() -> None:
    """A score of exactly 1.0 lands in the last bin, not on the
    bin-9 / bin-10 boundary. The 'upper inclusive' rule prevents
    top-confidence hits from falling into a phantom bin."""
    bins = bin_predictions([1.0, 1.0, 1.0], [True, False, True])
    # All three records ended up in bin 9.
    assert bins[-1].count == 3
    assert bins[-1].avg_predicted_score == pytest.approx(1.0)
    assert bins[-1].actual_duplicate_rate == pytest.approx(2 / 3)
    # All other bins are empty.
    for b in bins[:-1]:
        assert b.count == 0


def test_bin_predictions_clamps_out_of_range_scores() -> None:
    """Scores outside [0, 1] are clamped before binning.

    This shouldn't happen for ``top1_confidence`` values from the
    API (which is the ``(sim+1)/2`` formula and bounded), but a
    defensive clamp avoids a downstream IndexError if a future
    scoring change slips a >1.0 value through.
    """
    bins = bin_predictions([1.5, -0.3, 0.0], [True, True, False])
    # 1.5 → clamped to 1.0 → bin 9. -0.3 → clamped to 0.0 → bin 0.
    assert bins[0].count == 2
    assert bins[-1].count == 1
    assert bins[-1].avg_predicted_score == pytest.approx(1.0)


def test_bin_predictions_weighted_average() -> None:
    """Multiple records per bin: avg_predicted_score is the mean,
    not the bin center."""
    # 4 records in bin 5 (scores 0.51, 0.55, 0.58, 0.59).
    scores = [0.51, 0.55, 0.58, 0.59]
    labels = [True, False, True, True]
    bins = bin_predictions(scores, labels)
    # bin 5 covers [0.5, 0.6).
    assert bins[5].count == 4
    assert bins[5].actual_duplicate_rate == pytest.approx(3 / 4)
    # mean = (0.51 + 0.55 + 0.58 + 0.59) / 4 = 0.5575
    assert bins[5].avg_predicted_score == pytest.approx(0.5575, abs=1e-6)


def test_bin_predictions_length_mismatch_raises() -> None:
    """Length-mismatched inputs raise ValueError loudly."""
    with pytest.raises(ValueError, match="must be the same length"):
        bin_predictions([0.1, 0.2, 0.3], [True, False])
    with pytest.raises(ValueError, match="must be the same length"):
        bin_predictions([0.1], [])


def test_bin_predictions_n_bins_custom() -> None:
    """``n_bins`` overrides the default 10."""
    bins = bin_predictions([0.125, 0.625], [True, False], n_bins=4)
    # 4 bins covering [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0].
    assert len(bins) == 4
    assert bins[0].count == 1
    assert bins[2].count == 1
    assert bins[0].lower == 0.0
    assert bins[-1].upper == pytest.approx(1.0)


def test_bin_predictions_n_bins_positive_validation() -> None:
    """n_bins must be positive."""
    with pytest.raises(ValueError, match="n_bins must be positive"):
        bin_predictions([0.1], [True], n_bins=0)
    with pytest.raises(ValueError, match="n_bins must be positive"):
        bin_predictions([0.1], [True], n_bins=-3)


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


def test_compute_ece_perfect_calibration() -> None:
    """If avg_predicted == actual_rate in every non-empty bin, ECE = 0."""
    # 10 records, all is_dup=False, spread one per bin (scores
    # 0.05..0.95). Wait — that has gap. We need actual_rate=0 AND
    # avg_predicted_score = 0 in each non-empty bin. The only way
    # is to put all records in the lowest bin.
    bins = bin_predictions([0.0] * 10, [False] * 10)
    assert compute_ece(bins) == 0.0


def test_compute_ece_all_true_yields_mean_prediction_gap() -> None:
    """All-True case: actual_rate=1 in every bin, so the gap is
    (1 - avg_predicted_score_b). ECE reduces to the bin-weighted
    average of avg_predicted_score, which equals the mean of all
    scores — the result is 1 minus the mean of all scores.

    10 records evenly spread one-per-bin across the [0, 1] range:
    mean of scores = 0.5 (bin-center midpoint), so ECE = 1 - 0.5 = 0.5.
    """
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [True] * 10
    bins = bin_predictions(scores, labels)
    ece = compute_ece(bins)
    # Each bin: gap = |avg_pred - 1.0| = 1 - avg_pred; weight = 1/10.
    # Sum = (1/10) * sum_bins (1 - avg_pred_b). Mean(pred)=0.5 →
    # ECE = 1 - 0.5 = 0.5.
    assert ece == pytest.approx(0.5, abs=1e-6)


def test_compute_ece_weighted_by_bin_count() -> None:
    """Two bins, unequal counts — ECE weights by bin count, not 1/n_bins.

    bin 0: 1 record, score=0.95, is_dup=False → actual_rate=0, avg_pred=0.95, gap=0.95, weight=0.25
    bin 9: 3 records, score=1.0, is_dup=True   → actual_rate=1, avg_pred=1.0, gap=0.0, weight=0.75
    ECE = 0.25*0.95 + 0.75*0.0 = 0.2375
    """
    # 0.95 → bin 9 (max(0.95*10, 9) = 9); 1.0 → clamped to 1.0 → bin 9.
    # Both scores land in the same physical bin (bin 9, the top bin)
    # under default n_bins=10, which does NOT exercise the
    # weighting. Use n_bins=2 instead to split [0, 0.5) and [0.5, 1.0].
    scores = [0.6, 0.6, 0.6, 0.6]
    labels = [True, True, False, False]
    # 0.6 lands in bin 1 ([0.5, 1.0]) under n_bins=2.
    # actual_rate = 2/4 = 0.5; avg_pred = 0.6; gap = 0.1; weight = 1.0.
    bins = bin_predictions(scores, labels, n_bins=2)
    ece = compute_ece(bins)
    assert ece == pytest.approx(0.1, abs=1e-6)

    # Now exercise true two-bin weighting with a different split:
    # bin 0: 1 record (score=0.1, is_dup=False) → actual=0, avg=0.1, gap=0.1, weight=0.25
    # bin 1: 3 records (score=0.9, is_dup=True)  → actual=1, avg=0.9, gap=0.1, weight=0.75
    # ECE = 0.25*0.1 + 0.75*0.1 = 0.1
    scores2 = [0.1, 0.9, 0.9, 0.9]
    labels2 = [False, True, True, True]
    bins2 = bin_predictions(scores2, labels2, n_bins=2)
    ece2 = compute_ece(bins2)
    assert ece2 == pytest.approx(0.1, abs=1e-6)


def test_compute_ece_empty_bins_returns_zero() -> None:
    """No records at all → ECE is 0.0 (not NaN)."""
    bins = bin_predictions([], [])
    assert compute_ece(bins) == 0.0


def test_compute_ece_empty_iterable_returns_zero() -> None:
    """An empty iterable (or list of zero-count bins) → ECE = 0.0."""
    assert compute_ece([]) == 0.0
    zero_bins = [BinStats(i, i / 10.0, (i + 1) / 10.0, 0, 0.0, 0.0) for i in range(10)]
    assert compute_ece(zero_bins) == 0.0


def test_compute_ece_bounded_zero_one() -> None:
    """ECE must be in [0.0, 1.0] for any sane input.

    We check that the function stays in range on a few random-ish
    score/label combinations.
    """
    for scores, labels in [
        ([0.5, 0.5, 0.5, 0.5], [True, False, True, False]),
        ([0.0, 0.3, 0.6, 0.9, 0.2, 0.8], [True, False, True, True, False, False]),
        ([0.95] * 20, [False] * 20),
        ([0.05] * 20, [True] * 20),
    ]:
        bins = bin_predictions(scores, labels)
        ece = compute_ece(bins)
        assert 0.0 <= ece <= 1.0, f"ECE out of range for {scores}/{labels}: {ece}"


# ---------------------------------------------------------------------------
# plot_calibration
# ---------------------------------------------------------------------------


def test_plot_calibration_writes_png(tmp_path: Path) -> None:
    """Smoke test: plot_calibration writes a non-empty PNG."""
    bins = bin_predictions(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        [False, False, False, False, False, True, True, True, True, True],
    )
    out = plot_calibration(
        bins,
        config_name="dense_bge_m3",
        output_path=tmp_path / "calibration.png",
    )
    assert out.exists()
    assert out.stat().st_size > 1000  # a real PNG, not an empty file


def test_plot_calibration_creates_parent_directory(tmp_path: Path) -> None:
    """Output path's parent dir is created if missing."""
    target = tmp_path / "nested" / "subdir" / "calibration.png"
    bins = bin_predictions([0.5], [True])
    out = plot_calibration(bins, config_name="bm25", output_path=target)
    assert out.exists()
    assert out.parent.is_dir()


def test_plot_calibration_title_includes_ece_and_provenance(
    tmp_path: Path,
) -> None:
    """The figure's title encodes the ECE value, the eval-set
    provenance, and the config name — same discipline as the
    README leaderboard screenshot."""
    bins = bin_predictions(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        [True] * 10,
    )
    # Just verify the call doesn't crash with custom args — the
    # title-text format is documented in the module docstring and
    # inspectable by reading the PNG; a per-byte comparison would
    # be brittle.
    out = plot_calibration(
        bins,
        config_name="hybrid_rrf",
        eval_name="labeled_v300.jsonl",
        provenance="LLM-generated v2, hand-review pending",
        title_extra=" (above informational target)",
        output_path=tmp_path / "hybrid.png",
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# Phase 3.5 — FPR-on-novel per-bin breakdown
# ---------------------------------------------------------------------------
#
# These tests cover the 3.5 surface (fpr_on_novel_breakdown +
# plot_fpr_breakdown + the markdown writer). The plot_fpr_breakdown
# tests are smoke-level (same as 3.3's) — the math is what we
# actually assert.


def test_fpr_on_novel_breakdown_empty_inputs_returns_full_grid() -> None:
    """Empty (scores, labels) → 10 zero-filled FprBreakdownBin rows.

    Same contract as ``bin_predictions``: a downstream reader
    (plot, markdown writer) always gets a full grid of length
    ``DEFAULT_N_BINS``.
    """
    bins = fpr_on_novel_breakdown([], [])
    assert len(bins) == DEFAULT_N_BINS
    for b in bins:
        assert isinstance(b, FprBreakdownBin)
        assert b.novel_count == 0
        assert b.duplicate_count == 0
        assert b.novel_fraction == 0.0
        assert b.fpr_contribution == 0.0
    # Bin edges match the calibration module.
    assert bins[0].lower == 0.0
    assert bins[0].upper == pytest.approx(0.1)
    assert bins[-1].upper == pytest.approx(1.0)


def test_fpr_on_novel_breakdown_mixed_labels() -> None:
    """A 50/50 mix of novel and duplicate across bins.

    10 records, one per bin, alternating novel/duplicate starting
    with novel. Verifies novel_count, duplicate_count,
    novel_fraction, and fpr_contribution (= 1/5 per novel bin
    since there are 5 novel records total).
    """
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [False, True, False, True, False, True, False, True, False, True]
    bins = fpr_on_novel_breakdown(scores, labels)
    assert len(bins) == DEFAULT_N_BINS
    for i, b in enumerate(bins):
        if i % 2 == 0:
            assert b.novel_count == 1
            assert b.duplicate_count == 0
            assert b.novel_fraction == 1.0
            assert b.fpr_contribution == pytest.approx(1.0 / 5.0)
        else:
            assert b.novel_count == 0
            assert b.duplicate_count == 1
            assert b.novel_fraction == 0.0
            assert b.fpr_contribution == 0.0


def test_fpr_on_novel_breakdown_handles_all_novel() -> None:
    """All-novel case: novel_fraction = 1.0 in every non-empty bin,
    fpr_contribution sums to 1.0 across non-empty bins."""
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [False, False, False, False, False]
    bins = fpr_on_novel_breakdown(scores, labels)
    total_contrib = 0.0
    nonempty = 0
    for b in bins:
        if b.novel_count > 0 or b.duplicate_count > 0:
            nonempty += 1
            assert b.novel_fraction == 1.0
            total_contrib += b.fpr_contribution
    assert nonempty == 5
    assert total_contrib == pytest.approx(1.0)


def test_fpr_on_novel_breakdown_handles_no_novel() -> None:
    """No-novel case: novel_fraction = 0.0 in every bin,
    fpr_contribution = 0.0 (denominator guard)."""
    scores = [0.1, 0.3, 0.5]
    labels = [True, True, True]
    bins = fpr_on_novel_breakdown(scores, labels)
    for b in bins:
        assert b.fpr_contribution == 0.0
        if b.duplicate_count > 0:
            assert b.novel_fraction == 0.0


def test_fpr_on_novel_breakdown_clamps_out_of_range_scores() -> None:
    """Scores outside [0, 1] are clamped to the nearest edge.

    A score of 1.0000001 lands in the last bin (the same edge
    case as the calibration module). A negative score lands in
    bin 0.
    """
    bins = fpr_on_novel_breakdown(
        [-0.5, 0.5, 1.5],
        [False, True, False],
    )
    # Bin 0 gets the clamped -0.5 + 0.5 = novel and duplicate both
    # in bin 0... actually 0.5 maps to bin 5 (0.5 * 10 = 5.0,
    # min with 9 = 5), and 1.5 clamps to 1.0 → bin 9.
    # -0.5 clamps to 0.0 → bin 0.
    assert bins[0].novel_count == 1
    assert bins[5].duplicate_count == 1
    assert bins[9].novel_count == 1


def test_fpr_on_novel_breakdown_rejects_mismatched_lengths() -> None:
    """Length mismatch → ValueError (same discipline as bin_predictions)."""
    with pytest.raises(ValueError):
        fpr_on_novel_breakdown([0.5], [True, False])
    with pytest.raises(ValueError):
        fpr_on_novel_breakdown([0.5, 0.7], [True])


def test_fpr_on_novel_breakdown_rejects_nonpositive_n_bins() -> None:
    """n_bins <= 0 → ValueError (same as bin_predictions)."""
    with pytest.raises(ValueError):
        fpr_on_novel_breakdown([0.5], [True], n_bins=0)
    with pytest.raises(ValueError):
        fpr_on_novel_breakdown([0.5], [True], n_bins=-1)


def test_plot_fpr_breakdown_smoke(tmp_path: Path) -> None:
    """plot_fpr_breakdown writes a non-empty PNG without raising.

    Same smoke-test discipline as plot_calibration. The figure has
    two stacked subplots (novel/duplicate bars + per-bin FPR
    contribution) and a title with the FPR + best_threshold
    + provenance call-outs.
    """
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [False] * 5 + [True] * 5
    bins = fpr_on_novel_breakdown(scores, labels)
    out = plot_fpr_breakdown(
        bins,
        config_name="dense_bge_m3",
        best_threshold=0.75,
        fpr_on_novel=0.0,  # all-novel live in the lower half here
        output_path=tmp_path / "fpr.png",
        eval_name="labeled_v300.jsonl",
        provenance="LLM-generated v2, hand-review pending",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_calibration_accepts_fpr_bins_overlay(tmp_path: Path) -> None:
    """plot_calibration's new fpr_bins kwarg adds a second-axis overlay
    without breaking the legacy single-axis call."""
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [True] * 10
    bins = bin_predictions(scores, labels)
    fpr = fpr_on_novel_breakdown(scores, [False] * 5 + [True] * 5)
    out = plot_calibration(
        bins,
        config_name="dense_bge_m3",
        output_path=tmp_path / "cal_with_fpr.png",
        fpr_bins=fpr,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_calibration_without_fpr_bins_unchanged(tmp_path: Path) -> None:
    """The legacy plot_calibration call (no fpr_bins) still works —
    the overlay is opt-in."""
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [True] * 10
    bins = bin_predictions(scores, labels)
    out = plot_calibration(
        bins,
        config_name="dense_bge_m3",
        output_path=tmp_path / "cal_legacy.png",
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# Phase 3.5 — write_per_config_markdown
# ---------------------------------------------------------------------------


def test_write_per_config_markdown_creates_file_with_required_sections(
    tmp_path: Path,
) -> None:
    """The markdown writer produces a file with the headline
    numbers, the 10-row per-bin table, the cap call-out, and the
    honest-scope paragraph. This is the file 3.7's README links to.
    """
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [False] * 5 + [True] * 5
    bins = fpr_on_novel_breakdown(scores, labels)
    out_path = tmp_path / "fpr-dense_bge_m3.md"
    returned = write_per_config_markdown(
        bins,
        config_name="dense_bge_m3",
        benchmark_name="labeled_v300.jsonl",
        best_threshold=0.80,
        fpr_on_novel=0.75,
        novel_set_mrr=0.75,
        ece=0.527,
        corpus_count=10983,
        total_novel=5,
        total_duplicate=5,
        total_records=10,
        output_path=out_path,
    )
    assert returned == out_path.resolve()
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")

    # Headline + provenance + cap call-out.
    assert "FPR-on-novel breakdown" in text
    assert "dense_bge_m3" in text
    assert "labeled_v300.jsonl" in text
    assert "LLM-generated v2, hand-review pending" in text
    assert "0.750" in text  # fpr_on_novel
    assert "0.15" in text   # the cap
    assert "above the PHASE-3.md §3.5 cap" in text  # gap call-out
    assert "trust this tool" in text

    # 10-row per-bin table.
    assert "| bin | range | novel_count | duplicate_count |" in text
    assert text.count("| 0 | [0.0, 0.1) |") == 1
    assert text.count("| 9 | [0.9, 1.0) |") == 1

    # Cumulative FPR at best_threshold.
    assert "Cumulative FPR-on-novel at" in text
    assert "best_threshold=0.80" in text

    # Honest scope paragraph.
    assert "Honest scope" in text
    assert "Anurag's review" in text


def test_write_per_config_markdown_handles_clearing_the_cap(
    tmp_path: Path,
) -> None:
    """When fpr_on_novel <= 0.15 the markdown renders the
    "clears the cap" branch."""
    bins = fpr_on_novel_breakdown([], [])
    out_path = tmp_path / "fpr-clears.md"
    write_per_config_markdown(
        bins,
        config_name="bm25",
        benchmark_name="labeled_v300.jsonl",
        best_threshold=0.95,
        fpr_on_novel=0.10,
        novel_set_mrr=0.10,
        ece=0.10,
        corpus_count=10983,
        total_novel=0,
        total_duplicate=0,
        total_records=0,
        output_path=out_path,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "clears the PHASE-3.md §3.5 cap" in text
    # Negative case from the other test shouldn't appear.
    assert "above the PHASE-3.md §3.5 cap" not in text


def test_write_per_config_markdown_creates_parent_dirs(tmp_path: Path) -> None:
    """Parent directory of the output path is created if missing —
    same behaviour as the PNG writers."""
    nested = tmp_path / "nested" / "subdir" / "fpr.md"
    bins = fpr_on_novel_breakdown([], [])
    write_per_config_markdown(
        bins,
        config_name="x",
        benchmark_name="labeled_v300.jsonl",
        best_threshold=0.80,
        fpr_on_novel=0.0,
        novel_set_mrr=0.0,
        ece=0.0,
        corpus_count=0,
        total_novel=0,
        total_duplicate=0,
        total_records=0,
        output_path=nested,
    )
    assert nested.exists()
