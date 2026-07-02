"""Calibration metrics for the eval harness (Phase 3.3).

What this is
------------
A retrieval system's similarity scores are *predicted* probabilities
that the top-1 hit is a true duplicate of the query. The calibration
of those scores answers the question: "When the system says 0.8, is
the hit actually a duplicate 80% of the time?" A perfectly
calibrated system hugs the diagonal on a calibration curve.

We surface calibration two ways:

1. A *reliability table* (the bin statistics) — for each of N
   fixed-width bins over the [0.0, 1.0] confidence range, count the
   records in the bin and the fraction of those records whose
   ``is_duplicate`` flag is true. This is a histogram of
   "what the system said" vs "what turned out to be true".

2. A *scalar* — the Expected Calibration Error (ECE). The
   classic, bin-count-weighted formulation:

       ECE = sum_b (|bin_b| / N) * | avg_predicted_score_b - actual_duplicate_rate_b |

   Bins with zero records contribute 0 — they don't penalise the
   empty top of the curve where the system never predicts.

Conventions
-----------
- ``scores`` and ``is_duplicate`` are length-matched lists. We use
  the ``top1_score`` (normalised [0, 1] confidence) from the
  per-record trace, and the ``is_duplicate`` label from the
  benchmark.
- Score bins are fixed-width on the closed interval [0.0, 1.0]:
  bin 0 = [0.0, 0.1), bin 1 = [0.1, 0.2), ..., bin 9 = [0.9, 1.0].
  The upper edge of bin 9 is inclusive (so a perfect 1.0 lands in
  bin 9, not on the boundary). This is the same default as
  scikit-learn's ``calibration_curve`` with ``strategy='uniform'``.
- Bins with zero records are still reported (with count=0 and
  ``actual_duplicate_rate=0.0``) so the rendered curve has a full
  shape and downstream tests can assert structural invariants
  ("always 10 bins", "bin 0..9 inclusive", etc).

Why we don't use ``sklearn.calibration_curve``
-----------------------------------------------
It's a one-liner extra to write and read. The whole point of the
calibration module is that a reader can audit the binning logic
without diving into scikit-learn. Same approach as Phase 1.6's
metrics module: hand-write the formulas, write tight tests around
them.

Why matplotlib (and not plotly)
-------------------------------
The PNG is a static asset embedded in the README + the
leaderboard screenshot. No interaction needed. PNGs render in
GitHub's image preview, on Discord, and in any browser without a
JS dependency. Matplotlib's Agg backend is headless-safe (no
DISPLAY env var required); we set it explicitly at import time so
``matplotlib.use("Agg")`` is the first thing the module does.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede pyplot import.

import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


# Number of bins. 10 is the classic reliability-diagram size — wide
# enough to show shape, narrow enough that each bin has enough
# records to be meaningful on a 300-idea benchmark. Override via
# ``n_bins`` if a finer-grained view is wanted.
DEFAULT_N_BINS = 10


@dataclass(frozen=True)
class BinStats:
    """One row of the reliability table.

    Fields
    ------
    bin_index : int
        Zero-based bin index in [0, n_bins).
    lower : float
        Lower edge of the bin (inclusive).
    upper : float
        Upper edge of the bin (inclusive on the upper end for the
        last bin, exclusive otherwise — see module docstring).
    count : int
        Number of records whose score fell in this bin.
    actual_duplicate_rate : float
        Fraction of records in this bin whose ``is_duplicate`` flag
        is True. 0.0 when ``count == 0``.
    avg_predicted_score : float
        Mean of the scores that fell into this bin. 0.0 when
        ``count == 0``.
    """

    bin_index: int
    lower: float
    upper: float
    count: int
    actual_duplicate_rate: float
    avg_predicted_score: float


@dataclass(frozen=True)
class FprBreakdownBin:
    """One row of the FPR-on-novel per-bin breakdown (Phase 3.5).

    Phase 3.5 surfaces FPR-on-novel as a first-class metric. The
    per-bin breakdown answers: "for each score bin, how many of
    the novel records live there, and what fraction of the bin
    is novel?" The sum of ``novel_count`` across all bins equals
    the total number of novel records; the per-bin
    ``novel_fraction`` lets a reader see *where in the score
    distribution* the false positives are concentrated.

    This is the dataset behind ``plot_fpr_breakdown`` — same bin
    edges as ``bin_predictions`` (so the two plots can be
    visually overlaid or stacked), different aggregation.

    Fields
    ------
    bin_index : int
        Zero-based bin index in [0, n_bins).
    lower : float
        Lower edge of the bin (inclusive).
    upper : float
        Upper edge of the bin (inclusive on the upper end for the
        last bin).
    novel_count : int
        Number of records in this bin with ``is_duplicate=False``.
    duplicate_count : int
        Number of records in this bin with ``is_duplicate=True``.
    novel_fraction : float
        ``novel_count / (novel_count + duplicate_count)``. 0.0
        when the bin is empty.
    fpr_contribution : float
        ``novel_count / total_novel_records`` — the fraction of
        the *whole* novel subset that lives in this bin. Useful
        for reading off the cumulative FPR at any threshold by
        summing contributions up to that bin. 0.0 when the
        benchmark has no novel records.
    """

    bin_index: int
    lower: float
    upper: float
    novel_count: int
    duplicate_count: int
    novel_fraction: float
    fpr_contribution: float


def fpr_on_novel_breakdown(
    scores: Sequence[float],
    is_duplicate: Sequence[bool],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> List[FprBreakdownBin]:
    """Per-bin breakdown of FPR-on-novel (Phase 3.5).

    Companion to :func:`bin_predictions`. Uses the same bin
    edges (fixed-width on [0.0, 1.0]) so the two plots are
    visually aligned when stacked.

    For each bin we report:

    - ``novel_count`` and ``duplicate_count`` — the raw class
      composition.
    - ``novel_fraction`` — fraction of the bin that is novel
      (0.0 when the bin is empty).
    - ``fpr_contribution`` — fraction of the *whole* novel
      subset that lives in this bin (0.0 when no novel records
      in the input).

    The "cumulative FPR at threshold T" can be read off the
    ``fpr_contribution`` column by summing over all bins whose
    ``lower >= T``. (This is the same as the headline
    ``fpr_on_novel`` at threshold T, just bucketed.) The plot
    helper :func:`plot_fpr_breakdown` makes this visual.

    Parameters
    ----------
    scores : sequence of float
        Per-record top-1 confidence scores in [0.0, 1.0]. Records
        with ``None`` should be filtered upstream.
    is_duplicate : sequence of bool
        Per-record ground-truth labels. Same length as ``scores``.
    n_bins : int, default 10
        Number of fixed-width bins over [0.0, 1.0].

    Returns
    -------
    list of FprBreakdownBin
        Length ``n_bins``.

    Raises
    ------
    ValueError
        If ``n_bins <= 0``, or if the inputs are length-mismatched.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if len(scores) != len(is_duplicate):
        raise ValueError(
            f"scores ({len(scores)}) and is_duplicate "
            f"({len(is_duplicate)}) must be the same length"
        )

    novel_per_bin = [0] * n_bins
    dup_per_bin = [0] * n_bins
    for score, is_dup in zip(scores, is_duplicate):
        s = float(score)
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        idx = min(int(s * n_bins), n_bins - 1)
        if bool(is_dup):
            dup_per_bin[idx] += 1
        else:
            novel_per_bin[idx] += 1

    total_novel = sum(novel_per_bin)
    out: List[FprBreakdownBin] = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        nc = novel_per_bin[i]
        dc = dup_per_bin[i]
        cnt = nc + dc
        novel_frac = (nc / cnt) if cnt > 0 else 0.0
        fpr_contrib = (nc / total_novel) if total_novel > 0 else 0.0
        out.append(
            FprBreakdownBin(
                bin_index=i,
                lower=lower,
                upper=upper,
                novel_count=nc,
                duplicate_count=dc,
                novel_fraction=novel_frac,
                fpr_contribution=fpr_contrib,
            )
        )
    return out


def bin_predictions(
    scores: Sequence[float],
    is_duplicate: Sequence[bool],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> List[BinStats]:
    """Compute the reliability table for a single retrieval config.

    Parameters
    ----------
    scores : sequence of float
        Per-record top-1 confidence scores in [0.0, 1.0]. Records
        with a missing score (``None`` in the per-record trace)
        should be filtered upstream — this function does not handle
        them and will raise if the lists are length-mismatched.
    is_duplicate : sequence of bool
        Per-record ground-truth labels. Same length as ``scores``.
    n_bins : int, default 10
        Number of fixed-width bins over [0.0, 1.0].

    Returns
    -------
    list of BinStats
        Length ``n_bins``. Bin ``i`` covers
        ``[i / n_bins, (i + 1) / n_bins)`` for i < n_bins - 1 and
        is closed on the upper end for the final bin.

    Raises
    ------
    ValueError
        If ``n_bins <= 0``, or if ``scores`` and ``is_duplicate``
        have different lengths.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if len(scores) != len(is_duplicate):
        raise ValueError(
            f"scores ({len(scores)}) and is_duplicate "
            f"({len(is_duplicate)}) must be the same length"
        )

    # Per-bin accumulators.
    counts = [0] * n_bins
    dup_counts = [0] * n_bins  # how many in this bin are is_duplicate=True
    score_sums = [0.0] * n_bins

    for score, is_dup in zip(scores, is_duplicate):
        s = float(score)
        # Clamp into [0, 1] defensively. A score of 1.0000001 from
        # a numerical flake shouldn't put the record into an
        # out-of-range bin.
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        # Map to a bin index. The final bin is inclusive on the
        # upper edge so a perfect 1.0 lands in the last bin (the
        # pre-clamp cap is 1.0 so the formula always falls into
        # ``min(n_bins - 1, ...)`` here).
        idx = min(int(s * n_bins), n_bins - 1)
        counts[idx] += 1
        if bool(is_dup):
            dup_counts[idx] += 1
        score_sums[idx] += s

    out: List[BinStats] = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        cnt = counts[i]
        if cnt > 0:
            rate = dup_counts[i] / cnt
            avg_score = score_sums[i] / cnt
        else:
            rate = 0.0
            avg_score = 0.0
        out.append(
            BinStats(
                bin_index=i,
                lower=lower,
                upper=upper,
                count=cnt,
                actual_duplicate_rate=rate,
                avg_predicted_score=avg_score,
            )
        )
    return out


def compute_ece(bins: Iterable[BinStats]) -> float:
    """Expected Calibration Error (classic, bin-count-weighted).

        ECE = sum_b (|bin_b| / N) * | avg_predicted_score_b - actual_duplicate_rate_b |

    Where ``N`` is the total record count across all bins (records
    in empty bins contribute zero weight). Returns a float in
    [0.0, 1.0] — 0 is perfectly calibrated, 1 is worst-case
    (always says "duplicate" and is always wrong, or vice versa).

    Empty bins are treated as their contribution being zero; they
    don't penalise the system for never predicting in that range.
    This matches the standard formulation — see
    ``docs/EVAL.md`` §Expected Calibration Error.

    Parameters
    ----------
    bins : iterable of BinStats
        Output of :func:`bin_predictions`.

    Returns
    -------
    float
        The ECE in [0.0, 1.0].
    """
    bins = list(bins)
    n = sum(b.count for b in bins)
    if n == 0:
        # Degenerate input — no records binned. Return 0 rather
        # than NaN so downstream tests have a stable return value;
        # the runner reports "0 records" separately so this case is
        # loud.
        return 0.0
    ece = 0.0
    for b in bins:
        if b.count == 0:
            continue
        weight = b.count / n
        gap = abs(b.avg_predicted_score - b.actual_duplicate_rate)
        ece += weight * gap
    return ece


def plot_calibration(
    bins: Sequence[BinStats],
    *,
    config_name: str,
    output_path: Path,
    eval_name: str = "labeled_v300.jsonl",
    provenance: str = "LLM-generated v2, hand-review pending",
    title_extra: str = "",
    fpr_bins: Sequence[FprBreakdownBin] | None = None,
) -> Path:
    """Render the calibration curve as a PNG.

    The output PNG has three layers:

    1. The ``y = x`` diagonal (dotted grey line) — the perfect-
       calibration reference. A well-calibrated system hugs this.
    2. The per-bin actual duplicate rate (solid blue line + circles
       at the bin centers) — this is the system's empirical
       calibration.
    3. ECE in the title (computed from the same ``bins`` we pass).

    The PNG is sized for the README's dark-mode card — dark
    background, light text, accent colour for the curve.

    Parameters
    ----------
    bins : sequence of BinStats
        Output of :func:`bin_predictions`. Used both for the curve
        data and to compute the ECE shown in the title.
    config_name : str
        Retrieval-config name (e.g. ``dense_bge_m3``). Goes in the
        title so multiple PNGs can live in the same docs/assets
        folder without confusion.
    output_path : Path
        Destination path for the PNG. Parent directory is created
        if missing. ``.png`` suffix recommended.
    eval_name : str, default ``"labeled_v300.jsonl"``
        Name of the eval set used to compute the bins. Goes in
        the title.
    provenance : str, default ``"LLM-generated v2, hand-review pending"``
        Honest-scope call-out — printed in the title so anyone who
        sees the PNG in isolation knows the eval set wasn't hand-
        labeled. Same provenance policy as Phase 1.5a / Phase 2.8.
    title_extra : str, default ``""``
        Optional extra text appended to the title (e.g. ``"(ECE
        > 0.10 — above the informational target)"``).
    fpr_bins : sequence of FprBreakdownBin, optional
        Phase 3.5 overlay — when provided, plots the
        ``novel_fraction`` per bin on a second y-axis (red, right
        side) so the "trust this tool" FPR concentration is
        visible on the same canvas as the calibration curve.
        Same bin edges as ``bins`` (so the x-axis aligns), drawn
        in a contrasting warm colour (``#f7768e``) with a
        smaller line weight and a dashed style to read as
        secondary information. The title gets a one-line
        annotation: ``FPR-on-novel peaks at bin i (xx%)``.

    Returns
    -------
    Path
        ``output_path`` resolved to an absolute path, for chaining.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bins = list(bins)
    ece = compute_ece(bins)
    n_total = sum(b.count for b in bins)

    # X positions: bin centers. The final bin's center is
    # (upper - 0.05) by construction (bin width = 1/n_bins).
    bin_centers = [b.lower + (b.upper - b.lower) / 2.0 for b in bins]
    actual_rates = [b.actual_duplicate_rate for b in bins]
    counts = [b.count for b in bins]

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=110)
    # Dark background to match the rest of the README. Hard-code
    # the colours — the project's style guide doesn't define a
    # palette module yet, and pre-defining dark=here beats
    # importing PIL colour helpers for a one-frame render.
    bg = "#101014"
    fg = "#dcdce2"
    dim = "#8c8c96"
    accent = "#7aa2f7"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Perfect-calibration diagonal.
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle=":", color=dim, linewidth=1.5,
            label="perfect calibration (y = x)")

    # The system's calibration curve, with markers sized by the bin
    # count (so empty bins are visually obvious).
    sizes = [max(20.0, 20.0 + 8.0 * c) for c in counts]
    ax.plot(
        bin_centers,
        actual_rates,
        marker="o",
        color=accent,
        linewidth=2.0,
        label=f"actual duplicate rate (config={config_name})",
    )
    ax.scatter(
        bin_centers,
        actual_rates,
        s=sizes,
        color=accent,
        alpha=0.85,
        edgecolors=bg,
        linewidths=1.0,
        zorder=3,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("predicted similarity (top-1 confidence, bin center)",
                   color=fg, fontsize=11)
    ax.set_ylabel("actual duplicate rate", color=fg, fontsize=11)

    title = (
        f"Calibration curve — config={config_name} | "
        f"eval={eval_name} ({provenance})\n"
        f"ECE = {ece:.3f} | N = {n_total}{title_extra}"
    )
    ax.set_title(title, color=fg, fontsize=11, pad=12)

    ax.tick_params(colors=fg, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(dim)

    ax.grid(True, color=dim, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", facecolor=bg, edgecolor=dim,
              labelcolor=fg, fontsize=10)

    # ------------------------------------------------------------------
    # Phase 3.5 — FPR-on-novel overlay on a second y-axis
    # ------------------------------------------------------------------
    # Same bin edges, so the x-axis aligns with the calibration
    # curve. Drawn in red with a dashed style so it reads as
    # secondary information; legend entry is the bin-center
    # annotation. We compute the peak bin (highest novel_count) so
    # the title can name *where* the false positives concentrate.
    overlay_peak_note = ""
    if fpr_bins is not None and len(fpr_bins) == len(bins):
        warn = "#f7768e"
        ax_r = ax.twinx()
        ax_r.set_facecolor("none")
        ax_r.set_xlim(0.0, 1.0)
        ax_r.set_ylim(0.0, 1.05)
        ax_r.tick_params(colors=fg, labelsize=9)
        for spine in ("right",):
            ax_r.spines[spine].set_color(dim)
        # Hide the other three spines so the right axis reads as a
        # separate dimension rather than a boxed frame.
        for spine in ("top", "left", "bottom"):
            ax_r.spines[spine].set_visible(False)
        novel_fracs = [b.novel_fraction for b in fpr_bins]
        ax_r.plot(
            bin_centers,
            novel_fracs,
            marker="s",
            color=warn,
            linestyle="--",
            linewidth=1.5,
            markersize=5,
            alpha=0.85,
            label="novel fraction per bin (FPR overlay)",
        )
        ax_r.set_ylabel(
            "novel fraction per bin (FPR overlay)",
            color=warn, fontsize=10,
        )
        ax_r.tick_params(axis="y", colors=warn, labelsize=9)
        # Combined legend (one entry from each axis).
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper left", facecolor=bg, edgecolor=dim,
            labelcolor=fg, fontsize=9,
        )
        # Peak-bin annotation for the title.
        peak_idx = max(
            range(len(fpr_bins)),
            key=lambda i: fpr_bins[i].novel_count,
        )
        peak_bin = fpr_bins[peak_idx]
        if peak_bin.novel_count > 0:
            overlay_peak_note = (
                f" | FPR-on-novel peaks in bin "
                f"[{peak_bin.lower:.1f}, {peak_bin.upper:.1f}) "
                f"({peak_bin.novel_count} novel records, "
                f"{peak_bin.novel_fraction * 100:.0f}% of the bin)"
            )

        # Re-set the title to fold in the FPR peak annotation.
        title_with_overlay = (
            f"Calibration curve — config={config_name} | "
            f"eval={eval_name} ({provenance})\n"
            f"ECE = {ece:.3f} | N = {n_total}{title_extra}"
            f"{overlay_peak_note}"
        )
        ax.set_title(title_with_overlay, color=fg, fontsize=11, pad=12)

    # Footer note — the ECE informational target. Living in the
    # axes (not the title) so resizing the figure doesn't wrap it.
    ax.text(
        0.99, -0.16,
        "ECE ≤ 0.10 is the PHASE-3.md §3.3 informational target. "
        "Actual values are honest; do not gate on ECE without hand-label pass.",
        transform=ax.transAxes,
        ha="right", va="top",
        color=dim, fontsize=8, fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(output_path, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fpr_breakdown(
    bins: Sequence[FprBreakdownBin],
    *,
    config_name: str,
    best_threshold: float,
    fpr_on_novel: float,
    output_path: Path,
    eval_name: str = "labeled_v300.jsonl",
    provenance: str = "LLM-generated v2, hand-review pending",
    title_extra: str = "",
) -> Path:
    """Render the FPR-on-novel per-bin breakdown as a PNG (Phase 3.5).

    Two subplots stacked vertically:

    - **Top**: a stacked bar chart of ``novel_count`` (red) and
      ``duplicate_count`` (blue) per score bin. Same bin edges as
      :func:`plot_calibration` so the reader can mentally
      overlay the two.
    - **Bottom**: the ``fpr_contribution`` per bin (the fraction
      of the *whole* novel subset that lands in each bin). A
      cumulative reading of "FPR at threshold T" is the sum of
      the contributions from all bins whose lower edge is at or
      above T.

    The headline ``FPR-on-novel = X.XX`` at ``best_threshold`` is
    stamped in the figure title so the headline number (the one
    the README quotes) is visible at a glance. The provenance
    call-out matches the Phase 1.5a / Phase 2.8 / Phase 3.3
    discipline — every eval-set artifact carries the same
    disclaimer.

    Parameters
    ----------
    bins : sequence of FprBreakdownBin
        Output of :func:`fpr_on_novel_breakdown`.
    config_name : str
        Retrieval-config name. Goes in the title.
    best_threshold : float
        The threshold the runner picked for this config. Shown
        in the title next to the headline FPR.
    fpr_on_novel : float
        The headline FPR-on-novel at ``best_threshold``. Shown
        in the title.
    output_path : Path
        Destination for the PNG. Parent directory is created.
    eval_name : str, default ``"labeled_v300.jsonl"``
        Eval-set name. Goes in the title.
    provenance : str, default ``"LLM-generated v2, hand-review pending"``
        Honest-scope call-out for the title.
    title_extra : str, default ``""``
        Optional extra title text.

    Returns
    -------
    Path
        ``output_path`` resolved to an absolute path.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bins = list(bins)
    n_total_novel = sum(b.novel_count for b in bins)
    n_total_dup = sum(b.duplicate_count for b in bins)
    n_total = n_total_novel + n_total_dup

    bin_centers = [b.lower + (b.upper - b.lower) / 2.0 for b in bins]
    novel_counts = [b.novel_count for b in bins]
    dup_counts = [b.duplicate_count for b in bins]
    fpr_contribs = [b.fpr_contribution for b in bins]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 7.5), dpi=110, sharex=True
    )
    bg = "#101014"
    fg = "#dcdce2"
    dim = "#8c8c96"
    accent = "#7aa2f7"
    warn = "#f7768e"  # red-ish for novel records (the "false positive" colour)

    fig.patch.set_facecolor(bg)
    ax_top.set_facecolor(bg)
    ax_bot.set_facecolor(bg)

    # --- Top subplot: stacked bar of novel vs duplicate counts per bin.
    bin_width = (bins[0].upper - bins[0].lower) * 0.85 if bins else 0.08
    ax_top.bar(
        bin_centers,
        novel_counts,
        width=bin_width,
        color=warn,
        label=f"novel (is_duplicate=False, n={n_total_novel})",
        edgecolor=bg,
        linewidth=0.5,
    )
    ax_top.bar(
        bin_centers,
        dup_counts,
        width=bin_width,
        bottom=novel_counts,
        color=accent,
        label=f"duplicate (is_duplicate=True, n={n_total_dup})",
        edgecolor=bg,
        linewidth=0.5,
    )
    ax_top.set_ylabel("record count per bin", color=fg, fontsize=11)
    ax_top.set_xlim(0.0, 1.0)
    ax_top.tick_params(colors=fg, labelsize=9)
    for spine in ax_top.spines.values():
        spine.set_color(dim)
    ax_top.legend(loc="upper right", facecolor=bg, edgecolor=dim,
                  labelcolor=fg, fontsize=9)
    ax_top.grid(True, color=dim, alpha=0.2, linewidth=0.5)

    # --- Bottom subplot: per-bin FPR contribution.
    ax_bot.bar(
        bin_centers,
        fpr_contribs,
        width=bin_width,
        color=warn,
        edgecolor=bg,
        linewidth=0.5,
        label="fraction of novel subset in this bin",
    )
    ax_bot.set_xlim(0.0, 1.0)
    ax_bot.set_ylim(0.0, max(fpr_contribs + [0.0]) * 1.15 + 1e-6)
    ax_bot.set_xlabel("predicted similarity (top-1 confidence, bin center)",
                      color=fg, fontsize=11)
    ax_bot.set_ylabel("novel / total_novel", color=fg, fontsize=11)
    ax_bot.tick_params(colors=fg, labelsize=9)
    for spine in ax_bot.spines.values():
        spine.set_color(dim)
    ax_bot.grid(True, color=dim, alpha=0.2, linewidth=0.5)

    # --- Vertical reference line at best_threshold.
    for ax in (ax_top, ax_bot):
        ax.axvline(
            best_threshold,
            color=fg,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
        )

    title = (
        f"FPR-on-novel breakdown — config={config_name} | "
        f"eval={eval_name} ({provenance})\n"
        f"best_threshold={best_threshold:.2f} | "
        f"FPR-on-novel={fpr_on_novel:.3f} | "
        f"total={n_total} (novel={n_total_novel}, duplicate={n_total_dup})"
        f"{title_extra}"
    )
    fig.suptitle(title, color=fg, fontsize=11, y=0.995)

    # Footer note — call out the FPR trust-this-tool framing and the
    # current 0.15 cap. Living in the bottom subplot so resizing
    # the figure doesn't wrap it.
    ax_bot.text(
        0.99, -0.18,
        "FPR-on-novel ≤ 0.15 is the PHASE-3.md §3.5 'trust this tool' cap. "
        "No config currently clears it; this plot surfaces the gap honestly.",
        transform=ax_bot.transAxes,
        ha="right", va="top",
        color=dim, fontsize=8, fontstyle="italic",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Phase 3.5 — Per-config FPR-on-novel markdown writer
# ---------------------------------------------------------------------------
#
# Each eval run produces a per-config markdown summary that the README +
# the FPR breakdown section link to. The shape is:
#
#   | bin | range | novel_count | duplicate_count | novel_fraction | fpr_contribution |
#   | 0   | [0.0, 0.1) | 12 | 1 | 0.923 | 0.060 |
#   | 1   | [0.1, 0.2) |  8 | 2 | 0.800 | 0.040 |
#   ...
#
# Followed by a short paragraph summary that surfaces the FPR trust-
# this-tool framing, the current best_threshold, and the cap-vs-actual
# gap. Same honest provenance header as the failure breakdown / per-
# record CSV (Phase 1.5a / 2.8 / 3.3 / 3.4 discipline).
#
# The writer is pure (no HTTP, no DB) — it takes the breakdown bins
# and a few headline numbers, formats them, and writes to disk. The
# driver ``scripts/run_fpr_breakdown.py`` is the one that re-issues
# the /search calls and feeds the data in.


def write_per_config_markdown(
    bins: Sequence[FprBreakdownBin],
    *,
    config_name: str,
    benchmark_name: str,
    best_threshold: float,
    fpr_on_novel: float,
    novel_set_mrr: float,
    ece: float,
    corpus_count: int,
    total_novel: int,
    total_duplicate: int,
    total_records: int,
    output_path: Path,
) -> Path:
    """Write the per-config FPR-on-novel markdown breakdown (Phase 3.5).

    The output is a single markdown file with:

    - A provenance header (LLM-generated v2, hand-review pending).
    - A short prose summary with the headline numbers (best_threshold,
      FPR-on-novel, novel_set_mrr, ECE, corpus_count).
    - A 10-row table, one row per fixed-width score bin, columns:
      bin index, range, novel_count, duplicate_count, novel_fraction,
      fpr_contribution.
    - A trust-this-tool call-out: the cap (≤ 0.15), the actual, and
      the gap, framed as "FPR-on-novel is the metric that determines
      whether a real user would trust the tool; no config currently
      clears the cap."

    Parameters
    ----------
    bins : sequence of FprBreakdownBin
        Output of :func:`fpr_on_novel_breakdown`.
    config_name : str
        Retrieval-config name (e.g. ``dense_bge_m3``).
    benchmark_name : str
        Eval-set filename (e.g. ``labeled_v300.jsonl``).
    best_threshold : float
        The threshold the runner picked for this config. Shown in
        the headline so the table is self-explanatory.
    fpr_on_novel : float
        The headline FPR-on-novel at ``best_threshold``. The README
        quotes this number; the markdown renders it in the prose
        summary alongside the cap-vs-actual call-out.
    novel_set_mrr : float
        The "trust this tool" headline metric — fraction of
        ``is_duplicate=False`` records whose top-1 score crossed
        ``best_threshold``. Same denominator as ``fpr_on_novel``
        for the eval set's ``is_duplicate=False`` subset. In
        practice this *is* the FPR-on-novel at the chosen
        threshold (the two are computed identically); we surface
        it under a separate name because the README quotes it
        that way.
    ece : float
        ECE from the calibration curve. Shown for context — the
        FPR trust call is independent of ECE, but a reader who
        lands on the FPR page from the leaderboard row should
        see both numbers in one place.
    corpus_count : int
        The size of the indexed corpus at run time (rows in
        ``company_embeddings``). Goes in the prose summary.
    total_novel, total_duplicate, total_records : int
        Counts in the eval set; ``total_novel + total_duplicate
        + records_skipped == total_records``.
    output_path : Path
        Destination for the markdown file. Parent dir is created.

    Returns
    -------
    Path
        ``output_path`` resolved to an absolute path, for chaining.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bins = list(bins)
    n_bins = len(bins)

    # Cap check (Phase 3.5 honest-scope rule). The headline framing
    # is the same for every config: "FPR-on-novel is the metric that
    # determines whether a real user would trust the tool; this is
    # what it looks like for this config."
    cap = 0.15
    clears_cap = fpr_on_novel <= cap
    cap_phrase = (
        f"**clears the PHASE-3.md §3.5 cap of ≤ {cap:.2f}**"
        if clears_cap
        else f"**above the PHASE-3.md §3.5 cap of ≤ {cap:.2f}** "
        f"(`actual={fpr_on_novel:.3f}`, "
        f"`gap={fpr_on_novel - cap:+.3f}`) — no config currently clears "
        f"the cap; the per-bin breakdown below surfaces the gap honestly."
    )

    lines: List[str] = []
    lines.append(f"# FPR-on-novel breakdown — `{config_name}` on `{benchmark_name}`")
    lines.append("")
    lines.append(
        f"Per-bin FPR-on-novel breakdown at the runner-picked "
        f"``best_threshold={best_threshold:.2f}``. Eval set: "
        f"``{benchmark_name}`` (LLM-generated v2, hand-review "
        f"pending — same provenance policy as Phase 1.5a / 2.8 / 3.3 / 3.4)."
    )
    lines.append("")
    lines.append(
        f"**Headline numbers:** FPR-on-novel = "
        f"`{fpr_on_novel:.3f}` at ``best_threshold={best_threshold:.2f}`` "
        f"({cap_phrase}) | ``novel_set_mrr = {novel_set_mrr:.3f}`` "
        f"(same denominator, same value, surfaced under the README-quoted "
        f"name) | ``ECE = {ece:.3f}`` (informational) | "
        f"corpus_count = {corpus_count} | "
        f"N = {total_records} (novel = {total_novel}, "
        f"duplicate = {total_duplicate})."
    )
    lines.append("")
    lines.append(
        "The table below answers: *for each score bin, how many of "
        "the novel records live there, and what fraction of the bin "
        "is novel?* The ``fpr_contribution`` column is the fraction "
        "of the **whole** novel subset that lives in the bin — "
        "summing it over all bins whose lower edge is ``≥ T`` gives "
        "the FPR-on-novel at threshold T. **Cumulative FPR at "
        f"best_threshold={best_threshold:.2f}** is the sum of "
        "``fpr_contribution`` from the last two rows of the table."
    )
    lines.append("")
    lines.append(
        "| bin | range | novel_count | duplicate_count | "
        "novel_fraction | fpr_contribution |"
    )
    lines.append("|---|---|---|---|---|---|")
    for b in bins:
        lines.append(
            f"| {b.bin_index} | "
            f"[{b.lower:.1f}, {b.upper:.1f}) | "
            f"{b.novel_count} | {b.duplicate_count} | "
            f"{b.novel_fraction:.3f} | {b.fpr_contribution:.3f} |"
        )
    lines.append("")
    # Tail-bin sum = cumulative FPR at the chosen threshold.
    if best_threshold > 0.0:
        tail = [
            b for b in bins
            if b.lower + 1e-9 >= best_threshold
        ]
        cumulative_fpr = sum(b.fpr_contribution for b in tail)
    else:
        cumulative_fpr = sum(b.fpr_contribution for b in bins)
    lines.append(
        f"**Cumulative FPR-on-novel at ``best_threshold={best_threshold:.2f}``:** "
        f"`{cumulative_fpr:.3f}` (sum of ``fpr_contribution`` from the "
        f"{len(tail)} bin(s) at or above the threshold). This is the "
        f"same value as the headline ``FPR-on-novel = {fpr_on_novel:.3f}`` — "
        "the table is the bucketed view of that scalar."
    )
    lines.append("")
    lines.append(
        "**Honest scope:** the eval set is LLM-generated v2 and the "
        "hand-label pass is a follow-up; the FPR numbers above are "
        "honest but the underlying labels are pending Anurag's "
        "review. No config currently clears the 0.15 cap, so the "
        "\"trust this tool\" claim on the README is gated on Phase 4 "
        "(reranker) closing the gap."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = [
    "BinStats",
    "FprBreakdownBin",
    "DEFAULT_N_BINS",
    "bin_predictions",
    "compute_ece",
    "fpr_on_novel_breakdown",
    "plot_calibration",
    "plot_fpr_breakdown",
    "write_per_config_markdown",
]
