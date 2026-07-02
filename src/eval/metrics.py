"""Retrieval-quality metrics for the eval harness.

Five metrics, each with a precise, tested implementation
(see ``tests/test_eval.py`` for the worked examples):

1. ``reciprocal_rank(ranked_ids, expected_ids)`` — 1/rank of first
   expected id in the ranked list, or 0.0 if none found. MRR is the
   mean of this across the benchmark.

2. ``ndcg_at_k(ranked_ids, expected_ids, k)`` — normalised DCG at K,
   where each relevant position contributes ``1 / log2(i + 1)`` (the
   standard formulation). IDCG is the same formula with the
   expected ids ordered by their best possible rank.

3. ``precision_at_k(ranked_ids, expected_ids, k)`` — fraction of the
   top-K results that are in ``expected_ids``.

4. ``recall_at_k(ranked_ids, expected_ids, k)`` — fraction of the
   expected ids that appear anywhere in the top-K results.

5. ``fpr_on_novel(ranked_ids, expected_ids, threshold)`` — among
   ``is_duplicate=False`` records, fraction whose top-1 cosine
   similarity is above ``threshold``. Phase 1 target: ≤ 0.15 at the
   production threshold.

Conventions
-----------
- ``ranked_ids`` is the list of company ids returned by the
  retrieval, in ranked order (best first).
- ``expected_ids`` is the set (or list, treated as a set) of
  company ids the label says are the right answers.
- All metrics are bounded in [0, 1] except ``FPR-on-novel`` which is
  also in [0, 1].
- Empty ``ranked_ids`` → reciprocal_rank = 0, nDCG = 0, P@K = 0,
  R@K = 0. FPR-on-novel is 0 by definition (no top-1 hit means no
  false positive).
- The expected_ids are deduplicated internally — a label that lists
  the same id twice still counts as one relevant result.

Why we don't use ``sklearn.metrics``
------------------------------------
These five formulas are 30 lines of Python total. Pulling in
``sklearn`` for them would add a multi-MB install for no real win
and would obscure what the metrics actually compute. The whole
point of the eval harness is that the reader can audit the math.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def _to_set(xs: Iterable[int]) -> Set[int]:
    """Convert any iterable of ints to a set for membership tests."""
    return {int(x) for x in xs}


def reciprocal_rank(ranked_ids: Sequence[int], expected_ids: Iterable[int]) -> float:
    """Return 1 / rank of the first relevant id, or 0.0 if none found.

    Rank is 1-indexed (the first item is at rank 1, so a hit at
    position 0 → 1.0; at position 4 → 0.2). The function returns
    0.0 if no expected id appears in ``ranked_ids``.

    Example::

        >>> reciprocal_rank([10, 20, 30, 40], [30, 99])
        0.5
        >>> reciprocal_rank([10, 20, 30, 40], [99])
        0.0
        >>> reciprocal_rank([10, 20, 30, 40], [])
        0.0
    """
    expected = _to_set(expected_ids)
    if not expected:
        return 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if int(rid) in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[int],
    expected_ids: Iterable[int],
    k: int = 10,
) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Each relevant position contributes ``1 / log2(rank + 1)`` to the
    DCG. The IDCG is the same formula applied to a hypothetical
    ranking where all relevant ids are packed at the top in their
    best positions. ``ndcg = dcg / idcg`` (0.0 if ``idcg == 0``,
    i.e. there were no relevant ids).

    We use the binary-relevance variant: a result is either
    relevant (in ``expected_ids``) or not. Multi-grade relevance is
    out of scope for Phase 1 — the eval labels are a set of "right
    answers" per query, not a graded list.

    Example::

        >>> ndcg_at_k([10, 20, 30, 40], [20, 30], k=4)  # 2 hits at rank 2,3
        0.6131...
    """
    expected = _to_set(expected_ids)
    if not expected:
        return 0.0

    # Truncate to top-k. The runner fetches top_k hits from the API;
    # the metric is computed over the first k of them.
    top = list(ranked_ids[:k])

    # DCG over the actual ranking
    dcg = 0.0
    for i, rid in enumerate(top, start=1):
        if int(rid) in expected:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG: how many relevant ids CAN fit in the top-k, packed at
    # the top. If expected has more than k, we only count the first
    # k — the metric is "how well did we rank the best k expected".
    n_rel_in_k = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel_in_k + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    ranked_ids: Sequence[int],
    expected_ids: Iterable[int],
    k: int = 5,
) -> float:
    """Fraction of the top-K results that are in ``expected_ids``.

    Example::

        >>> precision_at_k([10, 20, 30, 40, 50], [20, 40, 99], k=5)
        0.4
    """
    expected = _to_set(expected_ids)
    if k <= 0:
        return 0.0
    top = list(ranked_ids[:k])
    if not top:
        return 0.0
    hits = sum(1 for rid in top if int(rid) in expected)
    return hits / k


def recall_at_k(
    ranked_ids: Sequence[int],
    expected_ids: Iterable[int],
    k: int = 10,
) -> float:
    """Fraction of ``expected_ids`` that appear in the top-K results.

    Example::

        >>> recall_at_k([10, 20, 30, 40], [20, 40, 99], k=4)
        0.6666...
    """
    expected = _to_set(expected_ids)
    if not expected:
        return 0.0
    top = list(ranked_ids[:k])
    if not top:
        return 0.0
    hits = sum(1 for eid in expected if int(eid) in {int(r) for r in top})
    return hits / len(expected)


def fpr_on_novel_record(
    *,
    is_novel: bool,
    top1_score: float | None,
    threshold: float,
) -> float:
    """The actual FPR-on-novel predicate (per-record).

    Returns 1.0 iff this record is novel AND its top-1 similarity
    (normalised to [0, 1]) is above the threshold. Otherwise 0.0.

    The runner computes the aggregate as the mean over the
    novel-only subset of the benchmark.

    Parameters
    ----------
    is_novel : bool
        True for records labeled ``is_duplicate=False``.
    top1_score : float or None
        The normalised similarity of the top-1 hit, in [0, 1], or
        ``None`` if the retrieval returned no hits at all.
    threshold : float
        The cutoff in [0, 1]. The runner sweeps
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] per the spec.
    """
    if not is_novel:
        return 0.0
    if top1_score is None:
        return 0.0
    return 1.0 if float(top1_score) >= float(threshold) else 0.0


def novel_set_positive_rate(
    *,
    is_novel: bool,
    top1_score: float | None,
    best_threshold: float,
) -> float:
    """Per-record contribution to the "novel-set MRR" / trust-this-tool metric.

    This is the headline scalar that PHASE-3.md §3.5 calls
    ``novel_set_mrr``: the fraction of ``is_duplicate=False``
    records whose top-1 score crossed the *production* threshold
    (``best_threshold`` — the threshold that maximises MRR subject
    to FPR ≤ 0.15, with a best-effort fallback if no threshold
    clears the cap).

    The math is identical to :func:`fpr_on_novel_record` — both
    reduce to "1.0 if is_novel AND top1_score ≥ threshold else 0.0"
    at the per-record level. The runner exposes both because they
    are surfaced in different places:

    - ``fpr_on_novel`` is the *sweep* metric (per-threshold
      per-row in the leaderboard CSV) — used for the threshold
      picker and the per-threshold dashboard view.
    - ``novel_set_mrr`` is the *config-level* headline — the
      single value the README quotes, computed once per config at
      the chosen best threshold, surfaced on every leaderboard row
      so the cell is always visible to the reader.

    Parameters
    ----------
    is_novel : bool
        True for records labeled ``is_duplicate=False``.
    top1_score : float or None
        The normalised similarity of the top-1 hit, in [0, 1].
    best_threshold : float
        The production threshold (the one the runner picked for
        this config). NOT a per-row sweep value — this is a fixed
        per-config scalar.

    Returns
    -------
    float
        1.0 if the record is novel and above ``best_threshold``,
        0.0 otherwise (including the ``top1_score is None`` and
        ``not is_novel`` cases).
    """
    return fpr_on_novel_record(
        is_novel=is_novel,
        top1_score=top1_score,
        threshold=best_threshold,
    )


#: The default cosine-threshold sweep for Phase 1 (per docs/PHASE-1.md §1.6
#: and docs/EVAL.md "Threshold sweep"). 7 cutoffs, 0.05 spacing.
DEFAULT_THRESHOLD_SWEEP: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def pick_best_threshold(
    *,
    threshold_sweep: Sequence[float],
    mrr_by_threshold: dict[float, float],
    fpr_by_threshold: dict[float, float],
    fpr_cap: float = 0.15,
) -> float:
    """Pick the threshold that maximises MRR subject to FPR ≤ fpr_cap.

    Returns the threshold on the sweep with the highest MRR, breaking
    ties by preferring the *lower* threshold (more permissive — we
    keep the system useful even when MRR is tied). If no threshold
    meets the FPR cap, returns the threshold with the *lowest* FPR.

    Parameters
    ----------
    threshold_sweep : sequence of float
        The cutoffs to consider (Phase 1: 0.5 .. 0.8).
    mrr_by_threshold : dict[float, float]
        MRR at each threshold on the sweep.
    fpr_by_threshold : dict[float, float]
        FPR-on-novel at each threshold on the sweep.
    fpr_cap : float
        Maximum acceptable FPR (default 0.15 per Phase 1 target).
    """
    if not threshold_sweep:
        raise ValueError("threshold_sweep is empty")
    eligible = [t for t in threshold_sweep if fpr_by_threshold.get(t, 1.0) <= fpr_cap]
    if not eligible:
        # No threshold meets the cap — pick the one with the lowest FPR
        # (so the leaderboard still has a "best effort" pick) and call
        # it out in the runner's summary.
        return min(threshold_sweep, key=lambda t: fpr_by_threshold.get(t, 1.0))
    # Among eligible, highest MRR wins; tie -> lower threshold.
    return max(
        eligible,
        key=lambda t: (mrr_by_threshold.get(t, 0.0), -t),
    )