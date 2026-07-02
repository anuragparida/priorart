#!/usr/bin/env python3
"""Phase 3.4 — per-category failure-analysis driver.

What this is
------------
Runs the per-category failure analysis across all three
retrieval configs (``dense_bge_m3`` / ``bm25`` / ``hybrid_rrf``)
on a labeled benchmark, and writes:

- One ``docs/assets/failure-breakdown-<config>.md`` per config
  (per-config markdown table).
- One consolidated ``docs/assets/failure-breakdown.png`` heatmap
  (rows = categories, columns = configs, cell = MRR).
- One ``results/failure-breakdown.csv`` (rows = (config, category)).

The driver is the entry point for Phase 3.4. It reuses the same
HTTP-based eval loop as ``python -m eval.run`` (so the per-record
trace is the same data the leaderboard CSV is built on), then
calls the per-category metrics + writers from
``src/eval/failure_analysis``.

Why a separate driver (not in ``eval.run``)
-------------------------------------------
``eval.run`` is the run-time entry point used by `make eval`
and the 3.2 config-change sensor. Adding the per-category
breakdown to its main loop would bloat that path with optional
args; the failure analysis is a *post-hoc* step that needs the
per-record trace (which is already persisted to DuckDB by
``run_eval``). This driver re-issues the /search calls and
derives the breakdown in the same process — the cost is
~3 * 300 /search round-trips per (config, threshold) which is
acceptable for a 300-record benchmark (sub-minute on a local
API).

Why not just post-process the DuckDB per_record table
-----------------------------------------------------
The /search results on the live API are *ground truth* for the
eval run; re-using the live results keeps the breakdown in sync
with whatever the leaderboard is showing. The post-hoc path
would be a follow-up if the corpus / thresholds change in
between.

Usage
-----

    uv run python scripts/run_failure_breakdown.py
    uv run python scripts/run_failure_breakdown.py --benchmark evals/labeled_v100.jsonl
    uv run python scripts/run_failure_breakdown.py --threshold 0.65
    uv run python scripts/run_failure_breakdown.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402

from src.eval.benchmark import Benchmark, load_benchmark  # noqa: E402
from src.eval.config import RetrievalConfig  # noqa: E402
from src.eval.failure_analysis import (  # noqa: E402
    build_csv_row,
    compute_per_category_metrics_from_benchmark,
    plot_heatmap,
    write_breakdown_csv,
    write_per_config_markdown,
)
from src.eval.run import (  # noqa: E402
    DEFAULT_THRESHOLD_SWEEP,
    PerRecordResult,
    _MODE_FOR_CONFIG,
    pick_best_threshold,
)
from src.eval.categorize import (  # noqa: E402
    BUSINESS_CATEGORIES,
    CATEGORY_LABEL,
    DEFAULT_PROVENANCE,
    BusinessCategory,
)


# Map the same mode discriminator that run.py uses so the
# per-category driver issues /search calls against the right
# retrieval mode. Mirrors the table in src/eval/run.py.


def _load_configs(configs_dir: Path) -> List[RetrievalConfig]:
    """Load every retrieval config from the configs/ directory.

    Returns a stable-order list (sorted by filename). Phase 2.9
    shipped three configs: ``bm25.yaml``, ``dense_bge_m3.yaml``,
    ``hybrid_rrf.yaml``. We skip hidden files and non-yaml
    artefacts so the driver is robust against future additions.
    """
    cfgs: List[RetrievalConfig] = []
    for path in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = RetrievalConfig.from_yaml(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path.name}: {exc}", file=sys.stderr)
            continue
        cfgs.append(cfg)
    return cfgs


def _per_record_for_config(
    cfg: RetrievalConfig,
    bench: Benchmark,
    *,
    threshold: float,
    fpr_cap: float,
) -> Tuple[List[PerRecordResult], float]:
    """Run the per-record /search loop for one config, return the trace.

    Returns ``(per_record, best_threshold)`` where ``best_threshold``
    is the per-config threshold the runner would have chosen
    (MRR-max under FPR-on-novel cap). The same value is used for
    the FPR-on-novel column in the per-category breakdown.
    """
    mode = _MODE_FOR_CONFIG.get(cfg.name, "dense")
    per_record: List[PerRecordResult] = []
    started = time.time()
    # Small inter-request delay to keep the API's Postgres pool
    # from saturating on the dense / hybrid paths (bm25 doesn't
    # hit Postgres, so it runs fast). 50ms × 300 = ~15s overhead
    # per config, well within the 2h budget. The previous run
    # (commit af92704) saw "too many clients" errors when 300
    # /search calls landed inside a few seconds; this delay
    # throttles the burst enough to let the pool recycle.
    inter_request_delay = 0.05
    with httpx.Client() as client:
        for rec in bench.records:
            payload = {"query": rec.idea, "top_k": cfg.top_k, "mode": mode}
            body: Optional[dict] = None
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    r = client.post(cfg.api_url, json=payload, timeout=30.0)
                    r.raise_for_status()
                    body = r.json()
                    last_exc = None
                    break
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    last_exc = exc
                    # 503 = "corpus not reachable" (the API hit
                    # the Postgres pool ceiling). Back off and
                    # retry — by attempt 3 the pool has usually
                    # recycled enough connections.
                    time.sleep(0.5 * (attempt + 1))
            if body is None:
                per_record.append(
                    PerRecordResult(
                        record_id=rec.id,
                        category=rec.category,
                        is_duplicate=rec.is_duplicate,
                        is_novel=rec.is_novel,
                        search_error=f"{type(last_exc).__name__}: {last_exc}",
                    )
                )
                time.sleep(inter_request_delay)
                continue
            hits = body.get("hits", []) or []
            ranked_ids = tuple(int(h["id"]) for h in hits if "id" in h)
            ranked_scores = tuple(
                float(h.get("confidence", h.get("similarity", 0.0)))
                for h in hits
            )
            top1 = ranked_scores[0] if ranked_scores else None
            per_record.append(
                PerRecordResult(
                    record_id=rec.id,
                    category=rec.category,
                    is_duplicate=rec.is_duplicate,
                    is_novel=rec.is_novel,
                    ranked_ids=ranked_ids,
                    ranked_scores=ranked_scores,
                    top1_score=top1,
                    search_error=None,
                )
            )
            time.sleep(inter_request_delay)
    elapsed = time.time() - started

    # Best threshold by MRR-max under FPR cap (mirrors run.py).
    # We compute MRR / FPR per threshold here so the chosen
    # threshold matches the leaderboard exactly. For a 300-record
    # benchmark this is ~7 thresholds * 300 records = cheap.
    from src.eval.metrics import (  # local: avoid a top-level cycle
        fpr_on_novel_record,
        reciprocal_rank,
    )

    best = threshold  # default to user override
    best_mrr = -1.0
    for thr in DEFAULT_THRESHOLD_SWEEP:
        rr_sum = 0.0
        n_relevant = 0
        fpr_sum = 0.0
        n_novel = 0
        by_id = {r.id: r for r in bench.records}
        for res in per_record:
            rec = by_id.get(res.record_id)
            if rec is None or res.search_error:
                continue
            if rec.expected_top_ids:
                rr_sum += reciprocal_rank(res.ranked_ids, rec.expected_top_ids)
                n_relevant += 1
            if rec.is_novel:
                fpr_sum += fpr_on_novel_record(
                    is_novel=True, top1_score=res.top1_score, threshold=thr
                )
                n_novel += 1
        if not n_relevant:
            continue
        mrr = rr_sum / n_relevant
        fpr = (fpr_sum / n_novel) if n_novel else 0.0
        if fpr <= fpr_cap and mrr > best_mrr:
            best_mrr = mrr
            best = thr
    return per_record, best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-b", "--benchmark",
        type=Path,
        default=_REPO_ROOT / "evals" / "labeled_v300.jsonl",
        help="Path to the labeled benchmark JSONL.",
    )
    p.add_argument(
        "-c", "--configs-dir",
        type=Path,
        default=_REPO_ROOT / "configs",
        help="Path to the retrieval-config YAML directory.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the threshold used for FPR-on-novel and the "
            "per-category breakdown. If unset, the per-config "
            "best threshold (MRR-max under FPR cap) is used."
        ),
    )
    p.add_argument(
        "--fpr-cap",
        type=float,
        default=0.15,
        help="FPR-on-novel cap used for best-threshold selection.",
    )
    p.add_argument(
        "--docs-assets-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "assets",
        help="Output directory for the per-config MD + the heatmap PNG.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Output directory for results/failure-breakdown.csv.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the breakdown but don't write any files (just print).",
    )
    args = p.parse_args()

    bench: Benchmark = load_benchmark(args.benchmark)
    cfgs: List[RetrievalConfig] = _load_configs(args.configs_dir)
    if not cfgs:
        print(f"error: no configs found in {args.configs_dir}", file=sys.stderr)
        return 1
    print(
        f"[failure-breakdown] benchmark={bench.path.name} "
        f"records={len(bench)} configs={[c.name for c in cfgs]}"
    )

    # Pre-compute the per-record → idea / expected maps once
    # (the per-category driver re-derives the business category
    # from the idea text; expected_top_ids is needed for the
    # precise MRR computation).
    record_id_to_idea: Dict[str, str] = {r.id: r.idea for r in bench.records}
    record_id_to_expected: Dict[str, Tuple[int, ...]] = {
        r.id: tuple(r.expected_top_ids) for r in bench.records
    }

    csv_rows: List[Dict[str, str]] = []
    metrics_by_config: Dict[str, Dict[BusinessCategory, "object"]] = {}
    for cfg in cfgs:
        print(f"[failure-breakdown] running {cfg.name}…")
        per_record, best_thr = _per_record_for_config(
            cfg, bench, threshold=args.threshold or 0.65, fpr_cap=args.fpr_cap
        )
        if args.threshold is not None:
            best_thr = args.threshold
        n_err = sum(1 for r in per_record if r.search_error)
        print(
            f"[failure-breakdown]   best_threshold={best_thr:.2f} "
            f"per_record={len(per_record)} errors={n_err}"
        )

        # Wire the idea text onto each per-record result so the
        # writers can render nice failure-example lines.
        for r in per_record:
            r._idea_text = record_id_to_idea.get(r.record_id, "")

        # Per-category metrics (precise, benchmark-aware).
        from src.eval.failure_analysis import PerCategoryMetrics  # noqa
        metrics = compute_per_category_metrics_from_benchmark(
            per_record,
            config_name=cfg.name,
            threshold=best_thr,
            record_id_to_idea=record_id_to_idea,
            record_id_to_expected=record_id_to_expected,
        )
        metrics_by_config[cfg.name] = metrics

        # Per-config markdown.
        md_path = args.docs_assets_dir / f"failure-breakdown-{cfg.name}.md"
        if not args.dry_run:
            write_per_config_markdown(
                metrics,
                config_name=cfg.name,
                benchmark_name=bench.path.name,
                threshold=best_thr,
                output_path=md_path,
            )
            print(f"[failure-breakdown]   wrote {md_path}")

        # CSV rows for this config.
        for cat in BUSINESS_CATEGORIES:
            m = metrics.get(cat)
            if m is None or m.n_records == 0:
                continue
            n_flag = " (n small)" if m.n_records < 5 else ""
            note = (
                f"deterministic-rule-based-v1; "
                f"n={m.n_records} (n_relevant={m.n_relevant}, n_novel={m.n_novel}){n_flag}"
            )
            csv_rows.append(build_csv_row(cfg.name, m, notes=note))

    # Consolidated heatmap.
    if not args.dry_run:
        heatmap_path = args.docs_assets_dir / "failure-breakdown.png"
        plot_heatmap(
            metrics_by_config,
            benchmark_name=bench.path.name,
            output_path=heatmap_path,
            metric="mrr",
            title_extra=f"categories={DEFAULT_PROVENANCE}",
        )
        print(f"[failure-breakdown] wrote {heatmap_path}")

        # Per-category breakdown CSV.
        csv_path = args.results_dir / "failure-breakdown.csv"
        write_breakdown_csv(csv_rows, csv_path)
        print(f"[failure-breakdown] wrote {csv_path}")

    # Print a compact summary.
    print()
    print("[failure-breakdown] per-(config, category) MRR summary:")
    print("  " + "config".ljust(14) + " | " + " | ".join(c.value.ljust(12) for c in BUSINESS_CATEGORIES))
    for cfg in cfgs:
        m = metrics_by_config[cfg.name]
        cells = []
        for cat in BUSINESS_CATEGORIES:
            v = m.get(cat)
            if v is None or v.n_records == 0:
                cells.append("—".ljust(12))
            else:
                cells.append(f"{v.mrr:.3f} (n={v.n_records})".ljust(12))
        print("  " + cfg.name.ljust(14) + " | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())