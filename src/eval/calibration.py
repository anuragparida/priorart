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


__all__ = [
    "BinStats",
    "DEFAULT_N_BINS",
    "bin_predictions",
    "compute_ece",
    "plot_calibration",
]
