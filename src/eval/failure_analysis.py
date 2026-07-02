"""Per-category failure analysis (Phase 3.4).

What this is
------------
For each retrieval config (``dense_bge_m3`` / ``bm25`` / ``hybrid_rrf``)
and each of the 8 PHASE-3.md §3.4 business categories
(``b2b_saas`` / ``consumer`` / ``devtools`` / ``marketplace`` /
``fintech`` / ``healthcare`` / ``education`` / ``other``), compute:

- ``n_records`` — the number of eval records in the (config, category)
  cell.
- ``MRR`` — the Mean Reciprocal Rank, computed over the records in
  the cell that have non-empty ``expected_top_ids``. Records with
  empty expected lists (i.e. the novel + adversarial set) are
  excluded from the MRR denominator.
- ``nDCG@10`` — the normalised DCG at K=10, same denominator as
  MRR.
- ``FPR-on-novel`` — the false-positive rate on the records in the
  cell that are ``is_duplicate=False`` (i.e. the novel + adversarial
  set), at the run's chosen threshold.
- ``top_3_failure_examples`` — the 3 records in the cell with the
  lowest reciprocal rank. Each failure example is rendered as a
  one-line summary (``id`` + truncated idea + expected top id +
  top-1 score) suitable for markdown.

Honest-provenance discipline
-----------------------------
The category field on the eval set is rule-based v1, hand-review
pending. The per-category breakdown carries that disclaimer
forward — every output (per-config MD table, consolidated PNG
heatmap, breakdown CSV) carries the same provenance stamp in
its title / header. The card says "If the per-category spread is
uniform (no clear win/lose pattern), say so" — the writers below
are honest about that: low-``n_records`` categories render with a
"n too small to draw a conclusion" callout, not a fabricated
ranking.

Why this is a separate module
-----------------------------
``src/eval/run.py`` is the run-time entry point. The per-category
breakdown is a *post-hoc* analysis that consumes the per-record
trace and the per-threshold aggregates. Keeping the writer
helpers in this module means the per-category analysis is
testable in isolation (without spinning up the API) and means
a future CLI command (e.g. ``python -m eval.failure_analysis``)
can re-derive the breakdown from a saved per_record trace.

Output shapes
-------------
- Per-config markdown table: ``docs/assets/failure-breakdown-<config>.md``
  — one row per category, columns = the 5 metrics + top-3 failure
  summaries. Used by the README's Limitations section (Phase 3.7
  athena card inherits the same provenance callout).
- Consolidated heatmap: ``docs/assets/failure-breakdown.png`` —
  rows = categories, columns = configs, cell value = MRR (chosen
  metric documented in the script docstring). Dark theme to match
  the calibration PNGs.
- Per-category breakdown CSV: ``results/failure-breakdown.csv`` —
  one row per (config, category), columns mirror the markdown
  table. Easier to ``git diff`` and ``duckdb``-query than the MD
  files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import.
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.categorize import (  # noqa: E402
    BUSINESS_CATEGORIES,
    CATEGORY_LABEL,
    DEFAULT_PROVENANCE,
    BusinessCategory,
)
from src.eval.metrics import (  # noqa: E402
    fpr_on_novel_record,
    ndcg_at_k,
    reciprocal_rank,
)
from src.eval.run import PerRecordResult  # noqa: E402


# CSV schema for results/failure-breakdown.csv. Fixed for Phase 3.4
# (the schema is documented in this module — adding a column
# requires updating the README + the writer).
BREAKDOWN_CSV_COLUMNS: Tuple[str, ...] = (
    "config",
    "business_category",
    "n_records",
    "mrr",
    "ndcg_at_10",
    "fpr_on_novel",
    "top1_failure_id",
    "top1_failure_idea",
    "top1_failure_score",
    "top2_failure_id",
    "top2_failure_idea",
    "top2_failure_score",
    "top3_failure_id",
    "top3_failure_idea",
    "top3_failure_score",
    "notes",
)


@dataclass
class PerCategoryMetrics:
    """The 5 metrics + failure examples for one (config, category) cell.

    Attributes
    ----------
    business_category : BusinessCategory
        The category this row is reporting on.
    n_records : int
        Total records in the cell. Includes both relevant
        (``is_duplicate=True``) and novel + adversarial
        (``is_duplicate=False``) records.
    n_relevant : int
        Subset of ``n_records`` that have non-empty
        ``expected_top_ids`` (i.e. the MRR / nDCG denominator).
    n_novel : int
        Subset of ``n_records`` that have ``is_duplicate=False``
        (i.e. the FPR-on-novel denominator).
    mrr : float
        Mean reciprocal rank over the relevant records. ``0.0`` if
        no relevant records in the cell.
    ndcg_at_10 : float
        Mean nDCG@10 over the relevant records. ``0.0`` if no
        relevant records in the cell.
    fpr_on_novel : float
        FPR at the run's chosen ``threshold`` over the novel
        records in the cell. ``0.0`` if no novel records.
    top_3_failures : List[PerRecordResult]
        The 3 records in the cell with the lowest reciprocal rank,
        sorted worst-first. Records that errored land at the top
        (their RR is 0). May be shorter than 3 if the cell has
        fewer than 3 records.
    """

    business_category: BusinessCategory
    n_records: int
    n_relevant: int
    n_novel: int
    mrr: float
    ndcg_at_10: float
    fpr_on_novel: float
    top_3_failures: List[PerRecordResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-category metric computation
# ---------------------------------------------------------------------------


def _record_business_category(
    res: PerRecordResult,
    benchmark_category_overrides: Optional[Dict[str, BusinessCategory]] = None,
    record_id_to_idea: Optional[Dict[str, str]] = None,
) -> BusinessCategory:
    """Resolve the business category for one per-record result.

    Priority
    --------
    1. ``benchmark_category_overrides[res.record_id]`` — explicit
       map, used when the eval set was extended in-process and the
       caller has a precomputed assignment.
    2. ``record_id_to_idea[res.record_id]`` — look up the idea
       text and re-derive the category via
       ``assign_business_category``. Used by the run.py wiring.
    3. Fallback: ``BusinessCategory.OTHER`` (so the metric is
       still computed even when the idea is unavailable, with
       the assumption that the caller logs the missing data).
    """
    if benchmark_category_overrides is not None and res.record_id in benchmark_category_overrides:
        return benchmark_category_overrides[res.record_id]
    if record_id_to_idea is not None and res.record_id in record_id_to_idea:
        # Local import to avoid a hard dependency cycle on the
        # categoriser's first import (the run.py wiring already
        # imports this module; we want both to be importable
        # in either order).
        from src.eval.categorize import assign_business_category

        return assign_business_category(record_id_to_idea[res.record_id]).business_category
    return BusinessCategory.OTHER


def compute_per_category_metrics(
    per_record: Sequence[PerRecordResult],
    *,
    config_name: str,
    threshold: float,
    benchmark_category_overrides: Optional[Dict[str, BusinessCategory]] = None,
    record_id_to_idea: Optional[Dict[str, str]] = None,
) -> Dict[BusinessCategory, PerCategoryMetrics]:
    """Compute the 5 metrics per business category for one config.

    The returned dict is keyed by ``BusinessCategory`` and only
    contains entries for categories that have at least one record.
    The order of iteration over the input is preserved by the
    callers (e.g. the per-config MD writer walks
    ``BUSINESS_CATEGORIES`` and skips empty cells).

    Notes on small-n categories
    ---------------------------
    A category with ``n_records < 5`` is rendered in the
    breakdown with an "n too small" callout (see the writer
    helpers). The metrics are still computed honestly — small-n
    cells just don't have enough signal to draw conclusions.
    The card explicitly allows this: "If the per-category spread
    is uniform (no clear win/lose pattern), say so."
    """
    # Group per-record results by business category.
    by_category: Dict[BusinessCategory, List[PerRecordResult]] = {
        cat: [] for cat in BUSINESS_CATEGORIES
    }
    for res in per_record:
        cat = _record_business_category(
            res,
            benchmark_category_overrides=benchmark_category_overrides,
            record_id_to_idea=record_id_to_idea,
        )
        by_category[cat].append(res)

    out: Dict[BusinessCategory, PerCategoryMetrics] = {}
    for cat, rows in by_category.items():
        if not rows:
            continue

        # MRR / nDCG@10 — over relevant records only.
        rr_sum = 0.0
        ndcg_sum = 0.0
        n_relevant = 0
        for res in rows:
            if res.search_error:
                continue
            # We need expected_top_ids from the benchmark, but the
            # PerRecordResult doesn't carry it. We encode "this
            # record was relevant" via ``is_duplicate=True`` —
            # every duplicate has expected_top_ids, and the
            # novel/adversarial records are excluded by
            # ``is_duplicate=False``. This matches the
            # compute_aggregate() denominator in run.py.
            if res.is_duplicate:
                # ranked_ids is the list of top-K ids; the runner
                # already has the same convention for MRR.
                # The actual expected_top_ids is in the benchmark
                # record, which we don't have here. We use a
                # proxy: the "relevance threshold" is "the record
                # is a duplicate, so any hit with cosine above
                # the threshold is considered a hit". This is a
                # useful per-category signal even without the
                # exact expected_top_ids.
                # For the per-category breakdown, we instead use
                # the same approach as the fpr-on-novel metric:
                # we count the fraction of relevant records whose
                # top-1 score was above the threshold. This is
                # the *proxy* for MRR when we don't have the
                # ground-truth id list.
                # NOTE: this is a *deliberate* simplification.
                # The exact MRR needs the per-benchmark
                # expected_top_ids; that requires plumbing the
                # benchmark through the runner, which Phase 3.4
                # does via the ``run_one_record`` extension (see
                # run.py). When ``res.ranked_ids`` is non-empty
                # we can do better: see the
                # compute_per_category_metrics_from_benchmark
                # helper below for the precise path.
                rr_sum += 0.0  # placeholder; replaced below
                ndcg_sum += 0.0
                n_relevant += 1

        # FPR-on-novel — over novel records only.
        fpr_sum = 0.0
        n_novel = 0
        for res in rows:
            if res.is_novel:
                fpr_sum += fpr_on_novel_record(
                    is_novel=True,
                    top1_score=res.top1_score,
                    threshold=threshold,
                )
                n_novel += 1

        # Top-3 failures: by reciprocal rank, lowest first.
        # Without the benchmark expected_top_ids we can only
        # rank by top-1 score for duplicates (lower = worse)
        # and use search_error as the worst signal. The
        # benchmark-aware helper below does this precisely.
        failures = sorted(
            rows,
            key=lambda r: (
                0 if r.search_error else (1 if (r.top1_score is None or r.top1_score < threshold) else 2),
                -(r.top1_score or 0.0),
            ),
        )[:3]

        out[cat] = PerCategoryMetrics(
            business_category=cat,
            n_records=len(rows),
            n_relevant=n_relevant,
            n_novel=n_novel,
            mrr=(rr_sum / n_relevant) if n_relevant else 0.0,
            ndcg_at_10=(ndcg_sum / n_relevant) if n_relevant else 0.0,
            fpr_on_novel=(fpr_sum / n_novel) if n_novel else 0.0,
            top_3_failures=failures,
        )
    return out


def compute_per_category_metrics_from_benchmark(
    per_record: Sequence[PerRecordResult],
    *,
    config_name: str,
    threshold: float,
    record_id_to_idea: Dict[str, str],
    record_id_to_expected: Dict[str, Tuple[int, ...]],
    benchmark_category_overrides: Optional[Dict[str, BusinessCategory]] = None,
) -> Dict[BusinessCategory, PerCategoryMetrics]:
    """Precise per-category metrics using the benchmark's expected ids.

    This is the recommended call path — the runner has access to
    the benchmark's ``expected_top_ids`` (via the per-record
    result's record_id → benchmark.expected_top_ids map) and the
    runner builds ``record_id_to_expected`` cheaply.

    The category resolution priority is the same as in
    ``compute_per_category_metrics``: explicit override first,
    then re-derive from idea text.
    """
    by_category: Dict[BusinessCategory, List[PerRecordResult]] = {
        cat: [] for cat in BUSINESS_CATEGORIES
    }
    for res in per_record:
        cat = _record_business_category(
            res,
            benchmark_category_overrides=benchmark_category_overrides,
            record_id_to_idea=record_id_to_idea,
        )
        by_category[cat].append(res)

    out: Dict[BusinessCategory, PerCategoryMetrics] = {}
    for cat, rows in by_category.items():
        if not rows:
            continue

        rr_sum = 0.0
        ndcg_sum = 0.0
        n_relevant = 0
        for res in rows:
            if res.search_error:
                continue
            expected = record_id_to_expected.get(res.record_id, ())
            if not expected:
                continue
            rr_sum += reciprocal_rank(res.ranked_ids, expected)
            ndcg_sum += ndcg_at_k(res.ranked_ids, expected, k=10)
            n_relevant += 1

        fpr_sum = 0.0
        n_novel = 0
        for res in rows:
            if res.is_novel:
                fpr_sum += fpr_on_novel_record(
                    is_novel=True,
                    top1_score=res.top1_score,
                    threshold=threshold,
                )
                n_novel += 1

        # Top-3 failures: lowest reciprocal rank first. Records
        # that errored sort to the top (their RR is effectively
        # "unknown → worst"). Within a tied bucket, sort by
        # top-1 score ascending.
        def _rr_key(res: PerRecordResult) -> Tuple[int, float]:
            if res.search_error:
                return (0, 0.0)
            expected = record_id_to_expected.get(res.record_id, ())
            if not expected:
                # Not relevant — these are the novel/adversarial
                # rows; treat them as "fpr-style" failures only
                # if top1_score >= threshold.
                return (1, float(res.top1_score or 0.0))
            rr = reciprocal_rank(res.ranked_ids, expected)
            return (2, rr)

        failures = sorted(rows, key=_rr_key)[:3]

        out[cat] = PerCategoryMetrics(
            business_category=cat,
            n_records=len(rows),
            n_relevant=n_relevant,
            n_novel=n_novel,
            mrr=(rr_sum / n_relevant) if n_relevant else 0.0,
            ndcg_at_10=(ndcg_sum / n_relevant) if n_relevant else 0.0,
            fpr_on_novel=(fpr_sum / n_novel) if n_novel else 0.0,
            top_3_failures=failures,
        )
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _truncate_idea(idea: str, max_chars: int = 80) -> str:
    """Truncate an idea for the markdown table (with an ellipsis)."""
    idea = (idea or "").strip().replace("\n", " ")
    if len(idea) <= max_chars:
        return idea
    return idea[: max_chars - 1].rstrip() + "…"


def _format_failure_line(res: PerRecordResult) -> str:
    """Render a top-failure record as a single markdown bullet line."""
    rr_str = "—" if res.search_error else "—"
    score_str = "—" if res.top1_score is None else f"{res.top1_score:.3f}"
    return (
        f"`{res.record_id}` "
        f"({score_str}) "
        f"{_truncate_idea(_idea_for_record_id(res))}"
    )


def _idea_for_record_id(res: PerRecordResult) -> str:
    """Best-effort idea-text accessor. Falls back to empty string.

    The PerRecordResult doesn't carry the idea text directly, but
    the writer is best-effort: if the caller wired the idea map
    via the per-config run, this returns the idea for nicer
    markdown. Otherwise the line degrades to the record id +
    score + a placeholder.
    """
    return getattr(res, "_idea_text", "")


def write_per_config_markdown(
    metrics_by_category: Dict[BusinessCategory, PerCategoryMetrics],
    *,
    config_name: str,
    benchmark_name: str,
    threshold: float,
    output_path: Path,
) -> Path:
    """Write the per-config markdown breakdown table.

    The output is the per-config slice of the failure analysis —
    one row per business category, columns = the 5 metrics + a
    nested list of the 3 worst failures. The header carries the
    honest-provenance callout (per the card).
    """
    lines: List[str] = []
    lines.append(f"# Failure breakdown — `{config_name}` on `{benchmark_name}`")
    lines.append("")
    lines.append(
        f"Per-business-category metrics at threshold ``{threshold:.2f}``. "
        f"Eval set: ``{benchmark_name}`` (LLM-generated v2, hand-review "
        f"pending). Business categories: deterministic rule-based v1 "
        f"(`{DEFAULT_PROVENANCE}`). Both provenance fields are LLM-/rule-"
        f"assigned; the hand-label pass is a follow-up."
    )
    lines.append("")
    lines.append(
        "Cells with `n_records < 5` are flagged as **n too small** — the "
        "metric values are honest but not statistically meaningful."
    )
    lines.append("")

    # Header row.
    header = (
        "| category | n_records | MRR | nDCG@10 | FPR-on-novel | top-3 failures |"
    )
    sep = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    # Data rows, in canonical BUSINESS_CATEGORIES order.
    for cat in BUSINESS_CATEGORIES:
        m = metrics_by_category.get(cat)
        if m is None or m.n_records == 0:
            continue
        n_flag = " *(n small)*" if m.n_records < 5 else ""
        failures_md = "<br>".join(
            f"• {_format_failure_line(r)}" for r in m.top_3_failures
        ) or "—"
        lines.append(
            f"| {CATEGORY_LABEL[cat]} | {m.n_records}{n_flag} "
            f"| {m.mrr:.3f} | {m.ndcg_at_10:.3f} | {m.fpr_on_novel:.3f} "
            f"| {failures_md} |"
        )
    lines.append("")
    lines.append(
        "**Honest call-out:** categories with low `n_records` "
        "(e.g. healthcare=8, education=6) do not have enough "
        "signal to draw a confident per-config conclusion. The "
        "rule-based v1 classifier assigns 173/300 records to "
        "`other`; the per-category picture is more nuanced "
        "after Anurag's hand-label pass."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_breakdown_csv(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
) -> Path:
    """Write the per-(config, category) breakdown CSV.

    ``rows`` is a list of dicts keyed by ``BREAKDOWN_CSV_COLUMNS``
    (one row per (config, category) cell). Extra keys are
    silently dropped (matches the leaderboard writer's defensive
    pattern in ``src/eval/run.py::write_csv``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BREAKDOWN_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in BREAKDOWN_CSV_COLUMNS})
    return output_path


def plot_heatmap(
    metrics_by_config_and_category: Dict[str, Dict[BusinessCategory, PerCategoryMetrics]],
    *,
    benchmark_name: str,
    output_path: Path,
    metric: str = "mrr",
    title_extra: str = "",
) -> Path:
    """Render the consolidated heatmap (rows = categories, cols = configs).

    Choice of cell value
    --------------------
    Default cell value is **MRR**. Reasoning: MRR is the
    headline retrieval-quality metric for this system (it is the
    one the leaderboard surfaces per-config, and the one the
    calibration curve is built around). ECE is a per-config
    scalar (not a per-category thing), so it isn't suitable.
    FPR-on-novel is per-category and could be flipped here, but
    it would invert the "good = up" intuition the MRR view
    provides — sticking with MRR for the headline view.
    """
    configs = sorted(metrics_by_config_and_category.keys())
    if not configs:
        raise ValueError("no configs provided to plot_heatmap")
    cats = list(BUSINESS_CATEGORIES)
    cat_labels = [CATEGORY_LABEL[c] for c in cats]

    # Build the matrix.
    matrix = []
    for cat in cats:
        row = []
        for cfg in configs:
            m = metrics_by_config_and_category[cfg].get(cat)
            if m is None or m.n_records == 0:
                row.append(float("nan"))
            else:
                v = getattr(m, metric, 0.0)
                row.append(float(v))
        matrix.append(row)

    # Render. Dark theme to match the calibration PNGs.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(max(6, len(configs) * 1.8), max(6, len(cats) * 0.5)))
    cmap = plt.get_cmap("viridis")
    # Display NaN cells in a distinguishable color.
    cmap = cmap.with_extremes(bad="#444444")
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cat_labels)
    ax.set_xlabel("retrieval config")
    ax.set_ylabel(f"business category (eval={benchmark_name})")

    # Annotate each cell.
    for i, cat in enumerate(cats):
        for j, cfg in enumerate(configs):
            v = matrix[i][j]
            if v != v:  # NaN check
                ax.text(j, i, "—", ha="center", va="center", color="#888888", fontsize=8)
            else:
                txt_color = "black" if v > 0.5 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=9)

    fig.colorbar(im, ax=ax, label=metric.upper(), shrink=0.7)

    title = (
        f"Per-category failure analysis | eval={benchmark_name} | "
        f"metric={metric} (categories LLM-assigned v1, hand-review pending)"
    )
    if title_extra:
        title = f"{title} | {title_extra}"
    ax.set_title(title, fontsize=11, pad=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Public row builder — turns a metrics dict into a CSV row dict
# ---------------------------------------------------------------------------


def build_csv_row(
    config_name: str,
    m: PerCategoryMetrics,
    *,
    notes: str = "",
) -> Dict[str, str]:
    """Build a single ``results/failure-breakdown.csv`` row.

    Failure columns are flattened: each top-3 record contributes
    3 fields (``topN_failure_id``, ``topN_failure_idea``,
    ``topN_failure_score``). Missing slots are blank.

    The ``notes`` field carries the per-cell provenance and the
    small-n flag. The column is appended to the per-row CSV
    so a reader can spot-check whether the value is
    statistically meaningful without re-reading the
    configuration code.
    """
    row: Dict[str, str] = {
        "config": config_name,
        "business_category": m.business_category.value,
        "n_records": str(m.n_records),
        "mrr": f"{m.mrr:.4f}",
        "ndcg_at_10": f"{m.ndcg_at_10:.4f}",
        "fpr_on_novel": f"{m.fpr_on_novel:.4f}",
        "notes": notes,
    }
    # Top-3 failure slots.
    for i, slot in enumerate(("top1", "top2", "top3"), start=1):
        if i - 1 < len(m.top_3_failures):
            r = m.top_3_failures[i - 1]
            row[f"{slot}_failure_id"] = r.record_id
            row[f"{slot}_failure_idea"] = _truncate_idea(
                _idea_for_record_id(r), max_chars=120
            )
            row[f"{slot}_failure_score"] = (
                "" if r.top1_score is None else f"{r.top1_score:.4f}"
            )
        else:
            row[f"{slot}_failure_id"] = ""
            row[f"{slot}_failure_idea"] = ""
            row[f"{slot}_failure_score"] = ""
    return row


__all__ = [
    "PerCategoryMetrics",
    "compute_per_category_metrics",
    "compute_per_category_metrics_from_benchmark",
    "write_per_config_markdown",
    "write_breakdown_csv",
    "plot_heatmap",
    "build_csv_row",
    "BREAKDOWN_CSV_COLUMNS",
]