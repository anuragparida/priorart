"""Eval harness — runs a benchmark against a retrieval config.

What this is
------------
``python -m eval.run --benchmark evals/labeled_v100.jsonl \
                        --config configs/dense_bge_m3.yaml \
                        --output results/leaderboard.csv``

Pipeline
--------
1. Load the retrieval config (a YAML file in ``configs/``).
2. Load the benchmark (a JSONL file in ``evals/``).
3. For each record: POST ``/search`` with the idea text, capture
   the ranked list of hits and their similarities.
4. For each cosine threshold in the sweep
   ``[0.50, 0.55, ..., 0.80]``: compute MRR, nDCG@10, precision@5,
   recall@10, FPR-on-novel over the entire benchmark.
5. Pick the threshold that maximises MRR subject to
   ``FPR-on-novel <= 0.15`` (the Phase 1 acceptance cap).
6. Write one CSV row per (config, threshold) to the output file
   (append mode if it exists). Also append the same rows to a
   DuckDB database (``results/eval.duckdb`` by default) so they're
   queryable across runs.
7. Print a Markdown summary table to stdout (and also write a
   ``.md`` next to the CSV) so a reader can paste the leaderboard
   into the README.

Why HTTP and not direct DB access
---------------------------------
The eval harness measures the *system* end-to-end — including
embedding latency, pgvector query plan, the FastAPI serialization
layer, and the post-fetch dedup logic. Calling the DB directly
would skip half of that and give numbers that don't reflect what
the UI sees. Phase 2 adds comparison-quality metrics that DO call
the LLM directly (since the /ideas/analyze path is the only way

Why ``--offline`` mode exists (Phase 3.6.2, card t_68dd7a03)
-----------------------------------------------------------
The eval-regression workflow's HTTP path needs a live API +
bge-m3 (for query embedding) — both unavailable on cold-cache
CI runs. The ``--offline`` flag swaps the live ``/search`` call
for an in-process backend (``src.eval.offline_search``) that
uses precomputed query embeddings from
``data/cache/eval_query_embeddings.npz`` and runs the same SQL
the API runs. Same numbers as the live API (verified row-by-row
in ``tests/test_phase36_workflow.py::test_offline_*``), no bge-m3
download, no model loading. See that file for the contract.

Why a DuckDB store alongside the CSV
------------------------------------
The CSV is the human-readable artifact (one row per (config,
threshold), easy to ``git diff`` in PRs). DuckDB is the
queryable, durable store (it survives a CSV rewrite, supports
``SELECT * FROM leaderboard WHERE config='dense_bge_m3'`` without
parsing the CSV, and is a single file you can commit alongside
the CSV). Both are written in the same run.

Failure modes that should *not* silently drop records
-----------------------------------------------------
- Network errors to /search → record marked as ``search_error`` in
  the per-record trace, the run continues.
- HTTP non-2xx → same.
- Empty hits (corpus not indexed) → metrics degrade gracefully
  (MRR=0, FPR=0 — there's nothing to false-positive on).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import httpx
import logging
import numpy as np
import typer
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

from src.config import EVALS_DIR, RESULTS_DIR, SNAPSHOTS_DIR
from src.eval.benchmark import Benchmark, BenchmarkRecord, load_benchmark
from src.eval.config import RetrievalConfig
from src.eval.offline_search import (  # noqa: E402  (Phase 3.6.2, card t_68dd7a03)
    _load_companies_for_bm25,
    offline_bm25,
    offline_dense,
    offline_hybrid,
)
from src.eval.metrics import (
    DEFAULT_THRESHOLD_SWEEP,
    fpr_on_novel_record,
    ndcg_at_k,
    novel_set_positive_rate,
    pick_best_threshold,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.eval.mlflow_logger import (
    RunRecord,
    log_run,
    metrics_from_summary,
    params_from_summary,
)
from src.eval.calibration import (
    bin_predictions as bin_predictions_calibration,
    compute_ece as compute_ece_metric,
    fpr_on_novel_breakdown as fpr_breakdown,
    plot_calibration as plot_calibration_curve,
    plot_fpr_breakdown as plot_fpr_breakdown_curve,
)


# CSV schema — fixed for v1 (Phase 1.6) with v2 additions (Phase 2.9,
# Phase 3.3). If you add a column, downstream readers (the dashboard
# in 1.11, the README leaderboard screenshot in 1.10, the
# ``render_leaderboard*.py`` scripts, the calibration PNG in 3.3)
# need to be updated too. The schema is documented in docs/EVAL.md
# as the "Leaderboard CSV schema".
#
# Phase 3.3: added ``ece`` (Expected Calibration Error) at the
# chosen threshold. The ECE is independent of the threshold — it's
# a property of the retrieval-config's score distribution and the
# eval-set's label distribution, not the threshold sweep. We still
# write one ``ece`` value per row so the CSV is uniform (the
# downstream reader doesn't need a separate row key).
#
# Phase 3.5: added ``novel_set_mrr`` (the per-config headline
# "fraction of novel records above best_threshold" — the metric
# the README quotes as the "trust this tool" number). It's a
# per-config constant (the same value on every row of a given
# config block in the leaderboard) — the runner computes it once
# after the threshold picker lands and patches it onto every
# row before the CSV / DuckDB write.
_CSV_COLUMNS: Tuple[str, ...] = (
    "config",
    "benchmark",
    "corpus_count",
    "embedding_model",
    "threshold",
    "mrr",
    "ndcg_at_10",
    "precision_at_5",
    "recall_at_10",
    "fpr_on_novel",
    "ece",
    "novel_set_mrr",
    "records_total",
    "records_novel",
    "records_duplicate",
    "records_skipped",
    "search_errors",
    "selected_threshold",
    "notes",
)


# ---------------------------------------------------------------------------
# Per-record runner
# ---------------------------------------------------------------------------


@dataclass
class PerRecordResult:
    """The outcome of running one benchmark record through /search."""

    record_id: str
    category: str
    is_duplicate: bool
    is_novel: bool
    ranked_ids: Tuple[int, ...] = field(default_factory=tuple)
    ranked_scores: Tuple[float, ...] = field(default_factory=tuple)
    top1_score: Optional[float] = None  # normalised confidence in [0, 1]
    search_error: Optional[str] = None


# Map retrieval config names to the ``mode`` discriminator that
# ``POST /search`` expects. Phase 2.9 added ``bm25`` and ``hybrid``
# modes to the API (see ``src/api/search.py::SearchRequest.mode``).
# Without this mapping every config would hit the dense endpoint
# and the leaderboard rows would all be identical — a silent
# correctness bug that ships fake BM25 / Hybrid numbers.
_MODE_FOR_CONFIG: Dict[str, str] = {
    "dense_bge_m3": "dense",
    "bm25": "bm25",
    "hybrid_rrf": "hybrid",
}


def run_one_record(
    record: BenchmarkRecord,
    *,
    config: RetrievalConfig,
    client: httpx.Client,
) -> PerRecordResult:
    """POST the record's idea to /search and capture the ranked hits.

    The contract: /search returns ``{"hits": [{"id", "similarity",
    "confidence", ...}, ...]}``. We use ``confidence`` (the
    normalised [0, 1] value, ``(sim+1)/2``) for the FPR threshold,
    which keeps the threshold sweep readable (0.65 = "65%
    confidence") and matches the Phase 1.4 contract.

    The retrieval ``mode`` is selected from the config name
    (``_MODE_FOR_CONFIG``). Unknown config names default to
    ``dense`` so a typo doesn't silently break the leaderboard —
    the row is still written but the mode comment flags it.

    On error: the record is marked ``search_error`` and the run
    continues. The runner reports the count of errored records at
    the end so a flapping API doesn't silently produce fake
    numbers.
    """
    mode = _MODE_FOR_CONFIG.get(config.name, "dense")
    payload = {"query": record.idea, "top_k": config.top_k, "mode": mode}
    try:
        r = client.post(config.api_url, json=payload, timeout=30.0)
        r.raise_for_status()
        body = r.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return PerRecordResult(
            record_id=record.id,
            category=record.category,
            is_duplicate=record.is_duplicate,
            is_novel=record.is_novel,
            search_error=f"{type(exc).__name__}: {exc}",
        )

    hits = body.get("hits", []) or []
    ranked_ids = tuple(int(h["id"]) for h in hits if "id" in h)
    ranked_scores = tuple(
        float(h.get("confidence", h.get("similarity", 0.0))) for h in hits
    )
    top1 = ranked_scores[0] if ranked_scores else None

    return PerRecordResult(
        record_id=record.id,
        category=record.category,
        is_duplicate=record.is_duplicate,
        is_novel=record.is_novel,
        ranked_ids=ranked_ids,
        ranked_scores=ranked_scores,
        top1_score=top1,
        search_error=None,
    )


# ---------------------------------------------------------------------------
# Offline path (Phase 3.6.2, card t_68dd7a03)
# ---------------------------------------------------------------------------
#
# Mirrors ``run_one_record`` but talks to the in-process
# ``src.eval.offline_search`` backend instead of /search over HTTP.
# The offline path needs:
#
#   1. The SQLAlchemy engine pointing at the CI DB (same as the API).
#   2. The precomputed query embedding for the current record, looked
#      up by record.id in ``data/cache/eval_query_embeddings.npz``.
#      The record's ``idea`` text is also passed for the BM25 / hybrid
#      modes (which don't use the embedding).
#
# Why a class instead of a function
# ---------------------------------
# The BM25 index is ~10K rows × a few hundred tokens — building it
# once per OfflineSearcher instance (== once per run_eval call) keeps
# the BM25 path fast across the 300-record benchmark. The class also
# caches the query embedding lookup, which is a hot-path dict read.
class OfflineSearcher:
    """In-process search backend for the eval runner.

    Loads the precomputed query embedding cache lazily on first use.
    Builds a fresh BM25 index on first use. Caches the corpus row
    list across calls within a single instance so the BM25 build
    runs once per ``run_eval`` invocation, not once per record.

    Mirrors the API's /search response shape:
    ``[{"id", "name", "description", "similarity", "confidence"}, ...]``
    """

    def __init__(self, query_embeddings_npz: Path) -> None:
        self._npz_path = Path(query_embeddings_npz)
        if not self._npz_path.exists():
            raise FileNotFoundError(
                f"offline: query embeddings .npz not found at {self._npz_path}. "
                "Run `python scripts/build_eval_query_embeddings.py` first."
            )
        # Lazy-loaded state — populated on first search() call.
        self._record_id_to_vec: Optional[Dict[str, np.ndarray]] = None
        self._engine: Optional[Engine] = None
        self._corpus_count: Optional[int] = None
        # BM25 corpus cache (tokenised once per OfflineSearcher, not
        # once per record). The hybrid mode fires both dense + bm25
        # per record — without this cache, the BM25 build is the
        # hot path and the eval takes ~2x as long. The cache is
        # None until the first bm25 / hybrid call.
        self._corpus_cache: Optional[Tuple[List[int], List[List[str]], Dict[int, Tuple[str, str]]]] = None

    def _load_corpus_for_bm25(
        self,
    ) -> Tuple[List[int], List[List[str]], Dict[int, Tuple[str, str]]]:
        """Load the BM25 tokenised corpus once per OfflineSearcher."""
        if self._corpus_cache is None:
            self._corpus_cache = _load_companies_for_bm25(self._get_engine())
        return self._corpus_cache

    @property
    def npz_path(self) -> Path:
        return self._npz_path

    def _load_query_embeddings(self) -> Dict[str, np.ndarray]:
        """Load the .npz into a {record_id: vector} dict (lazy)."""
        if self._record_id_to_vec is not None:
            return self._record_id_to_vec
        data = np.load(self._npz_path, allow_pickle=True)
        ids = list(data["record_id"])
        vecs = data["embeddings"]
        # The .npz is written by ``scripts/build_eval_query_embeddings.py``
        # in the same order as the benchmark JSONL. We index by record id
        # so the runner doesn't have to care about order.
        self._record_id_to_vec = {
            str(rid): np.asarray(v, dtype=np.float32)
            for rid, v in zip(ids, vecs)
        }
        return self._record_id_to_vec

    def _get_engine(self) -> Engine:
        if self._engine is None:
            from src.data.db import get_engine  # local import — keeps the module import-order clean
            self._engine = get_engine()
        return self._engine

    def corpus_count(self) -> int:
        """Return the ``company_embeddings`` row count, cached."""
        if self._corpus_count is None:
            engine = self._get_engine()
            with engine.connect() as conn:
                self._corpus_count = int(
                    conn.execute(text("SELECT count(*) FROM company_embeddings")).scalar_one()
                )
        return self._corpus_count

    def search(
        self,
        record: "BenchmarkRecord",
        *,
        mode: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Run one search and return the API-shaped hit list.

        The ``mode`` argument is the canonical name (``dense`` /
        ``bm25`` / ``hybrid``). Unknown modes raise ValueError —
        callers in the live path (the API) catch this; the offline
        path surfaces it as a ``search_error`` on the result.
        """
        engine = self._get_engine()
        if mode == "dense":
            vec = self._load_query_embeddings().get(record.id)
            if vec is None:
                # Record not in the precomputed cache — log + return
                # empty so the eval still produces a row (with
                # search_error set upstream if the runner wants).
                logger.warning(
                    "offline: record_id=%s has no precomputed query embedding; "
                    "the eval will see an empty hit list. Re-run "
                    "`scripts/build_eval_query_embeddings.py` to refresh the cache.",
                    record.id,
                )
                return []
            hits = offline_dense(engine, vec, top_k=top_k)
        elif mode == "bm25":
            corpus_cache = self._load_corpus_for_bm25()
            hits = offline_bm25(
                engine, record.idea, top_k=top_k, _corpus_cache=corpus_cache
            )
        elif mode == "hybrid":
            vec = self._load_query_embeddings().get(record.id)
            if vec is None:
                # Hybrid needs the dense path — same warning as dense.
                logger.warning(
                    "offline: record_id=%s has no precomputed query embedding; "
                    "hybrid will fall back to bm25-only.",
                    record.id,
                )
                corpus_cache = self._load_corpus_for_bm25()
                hits = offline_bm25(
                    engine, record.idea, top_k=top_k, _corpus_cache=corpus_cache
                )
            else:
                corpus_cache = self._load_corpus_for_bm25()
                hits = offline_hybrid(
                    engine,
                    vec,
                    record.idea,
                    top_k=top_k,
                    _corpus_cache=corpus_cache,
                )
        else:
            raise ValueError(f"unknown offline search mode: {mode!r}")
        return [h.to_dict() for h in hits]


def run_one_record_offline(
    record: "BenchmarkRecord",
    *,
    config: RetrievalConfig,
    searcher: OfflineSearcher,
) -> PerRecordResult:
    """Offline-mode equivalent of ``run_one_record``.

    Uses the precomputed query embedding for the dense + hybrid
    modes; uses the record's plain ``idea`` text for the bm25 mode.
    Same return shape as ``run_one_record`` so the rest of the
    eval pipeline (threshold sweep, CSV writing, calibration) is
    untouched.
    """
    mode = _MODE_FOR_CONFIG.get(config.name, "dense")
    try:
        hits = searcher.search(record, mode=mode, top_k=config.top_k)
    except (ValueError, RuntimeError) as exc:
        return PerRecordResult(
            record_id=record.id,
            category=record.category,
            is_duplicate=record.is_duplicate,
            is_novel=record.is_novel,
            search_error=f"{type(exc).__name__}: {exc}",
        )
    ranked_ids = tuple(int(h["id"]) for h in hits if "id" in h)
    ranked_scores = tuple(
        float(h.get("confidence", h.get("similarity", 0.0))) for h in hits
    )
    top1 = ranked_scores[0] if ranked_scores else None
    return PerRecordResult(
        record_id=record.id,
        category=record.category,
        is_duplicate=record.is_duplicate,
        is_novel=record.is_novel,
        ranked_ids=ranked_ids,
        ranked_scores=ranked_scores,
        top1_score=top1,
        search_error=None,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    """Per-threshold aggregate metrics for one benchmark run."""

    threshold: float
    mrr: float
    ndcg_at_10: float
    precision_at_5: float
    recall_at_10: float
    fpr_on_novel: float
    # Per-config headline (Phase 3.5). Computed once per run at
    # ``best_threshold`` and patched onto every threshold row of
    # the leaderboard so the value is visible in every dashboard
    # cell. Distinct from ``fpr_on_novel`` (which is a sweep
    # metric) — see ``novel_set_positive_rate`` for the
    # distinction.
    novel_set_mrr: float = 0.0


def compute_aggregate(
    results: Sequence[PerRecordResult],
    benchmark: Benchmark,
    *,
    threshold: float,
    best_threshold: Optional[float] = None,
) -> AggregateMetrics:
    """Compute MRR / nDCG@10 / P@5 / R@10 / FPR-on-novel over the run.

    Records that errored are skipped (they have empty ``ranked_ids``,
    so their contribution to MRR/nDCG/P/R is 0.0 by construction — the
    runner reports the error count separately so this isn't silent).

    The MRR / nDCG / P@R denominators are the *relevant* records
    only — those with a non-empty ``expected_top_ids``. The novel
    and adversarial records (which have empty expected lists) are
    excluded from the MRR denominator and only contribute to
    FPR-on-novel. This is standard practice for retrieval eval
    harnesses: queries with no relevant answer have no reciprocal
    rank by construction, and including them in the denominator
    would deflate MRR based on a labeler's choice of how many
    negative examples to write.

    FPR-on-novel, by contrast, is computed over the novel subset
    only — that's the metric that measures the "false alarm" rate
    and there is no useful value to it on a relevant-query record.

    The ``novel_set_mrr`` field is the per-config headline
    (Phase 3.5). It is computed at ``best_threshold`` (NOT at
    ``threshold`` — the sweep value). When ``best_threshold`` is
    not provided (e.g. in the all-records-errored degenerate path)
    it falls back to 0.0. The runner patches this value onto
    every threshold row of the leaderboard so the headline number
    is always visible in the dashboard.
    """
    by_id: Dict[str, BenchmarkRecord] = {r.id: r for r in benchmark.records}

    rr_sum = 0.0
    ndcg_sum = 0.0
    p5_sum = 0.0
    r10_sum = 0.0
    n_relevant = 0  # count of records that contributed to MRR/nDCG/P/R

    fpr_sum = 0.0
    n_novel = 0  # count of novel records that contributed to FPR

    # novel_set_mrr accumulator: sum of 1.0 over novel records
    # whose top1_score is above ``best_threshold``. We compute
    # this in the same pass so the runner only walks the results
    # list once.
    novel_set_sum = 0.0
    if best_threshold is not None:
        for res in results:
            if res.search_error:
                continue
            novel_set_sum += novel_set_positive_rate(
                is_novel=res.is_novel,
                top1_score=res.top1_score,
                best_threshold=best_threshold,
            )

    for res in results:
        rec = by_id.get(res.record_id)
        if rec is None:
            continue
        if res.search_error:
            # Errored record — skip from MRR/nDCG/P/R; still count
            # for the error tally. Skipped here so the aggregates
            # stay clean.
            continue

        # Only records with at least one expected id contribute
        # to MRR / nDCG / P / R. Novel + adversarial records are
        # scored via FPR-on-novel below.
        if rec.expected_top_ids:
            rr_sum += reciprocal_rank(res.ranked_ids, rec.expected_top_ids)
            ndcg_sum += ndcg_at_k(res.ranked_ids, rec.expected_top_ids, k=10)
            p5_sum += precision_at_k(res.ranked_ids, rec.expected_top_ids, k=5)
            r10_sum += recall_at_k(res.ranked_ids, rec.expected_top_ids, k=10)
            n_relevant += 1

        if res.is_novel:
            fpr_sum += fpr_on_novel_record(
                is_novel=True,
                top1_score=res.top1_score,
                threshold=threshold,
            )
            n_novel += 1

    if n_relevant == 0:
        # All relevant records errored — return zeros. The runner
        # reports this loudly downstream.
        return AggregateMetrics(
            threshold=threshold,
            mrr=0.0,
            ndcg_at_10=0.0,
            precision_at_5=0.0,
            recall_at_10=0.0,
            fpr_on_novel=0.0,
            novel_set_mrr=0.0,
        )

    # novel_set_mrr is the headline FPR-on-novel at best_threshold
    # (Phase 3.5). Same denominator as fpr_on_novel — the novel
    # subset. We re-derive it here from the accumulator to keep
    # the code path straightforward (the sweep pass doesn't
    # recompute it, we just used the best_threshold pass above).
    novel_set_mrr = (novel_set_sum / n_novel) if n_novel else 0.0

    return AggregateMetrics(
        threshold=threshold,
        mrr=rr_sum / n_relevant,
        ndcg_at_10=ndcg_sum / n_relevant,
        precision_at_5=p5_sum / n_relevant,
        recall_at_10=r10_sum / n_relevant,
        fpr_on_novel=(fpr_sum / n_novel) if n_novel else 0.0,
        novel_set_mrr=novel_set_mrr,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    append: bool = True,
) -> None:
    """Write the per-threshold rows to a leaderboard CSV.

    If ``append=True`` (the default), existing rows are preserved
    — re-running the eval appends new rows for the same config /
    benchmark rather than overwriting. This keeps a per-config
    history in the CSV (the dashboard in 1.11 reads it).

    Schema is fixed at the top of this module — never reorder
    columns without updating ``docs/EVAL.md``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if not file_exists or mode == "w":
            writer.writeheader()
        for row in rows:
            # Defensive: only write known columns; extras get dropped
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})


def write_duckdb(
    db_path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    config: RetrievalConfig,
    benchmark: Benchmark,
    per_record: Sequence[PerRecordResult],
) -> None:
    """Persist the run to DuckDB for queryable history.

    Two tables:

    - ``leaderboard`` — one row per (config, threshold), the same
      shape as the CSV. Cumulative across runs.
    - ``per_record`` — one row per benchmark record (with its
      ranked top-K ids + scores as JSON), keyed to the latest run.
      Replaced (not appended) per run so the table stays small.

    We use ``CREATE TABLE IF NOT EXISTS`` + ``INSERT`` (no MERGE
    for v1 — Phase 1 has a single config and a single run, the
    history grows by appends).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        # Leaderboard table — append every row from this run.
        # ``corpus_count`` is stored as VARCHAR (it can be empty
        # when the live API isn't reachable during a run; Phase
        # 1.11's dashboard can join against the corpus table to
        # fill it in).
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS leaderboard (
                config VARCHAR,
                benchmark VARCHAR,
                corpus_count VARCHAR,
                embedding_model VARCHAR,
                threshold DOUBLE,
                mrr DOUBLE,
                ndcg_at_10 DOUBLE,
                precision_at_5 DOUBLE,
                recall_at_10 DOUBLE,
                fpr_on_novel DOUBLE,
                ece DOUBLE,
                novel_set_mrr DOUBLE,
                records_total BIGINT,
                records_novel BIGINT,
                records_duplicate BIGINT,
                records_skipped BIGINT,
                search_errors BIGINT,
                selected_threshold BOOLEAN,
                notes VARCHAR,
                run_at TIMESTAMP DEFAULT current_timestamp
            )
            """
        )
        # Forward-compatible column add for older ``eval.duckdb``
        # files that pre-date Phase 3.3. ``ALTER TABLE … ADD
        # COLUMN`` is idempotent only if we guard on the column
        # existing — DuckDB does not have ``ADD COLUMN IF NOT
        # EXISTS`` prior to 0.10, so we check the schema first.
        existing_cols = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'leaderboard'"
            ).fetchall()
        }
        if "ece" not in existing_cols:
            con.execute(
                "ALTER TABLE leaderboard ADD COLUMN ece DOUBLE"
            )
        # Phase 3.5: same forward-compat dance for ``novel_set_mrr``.
        if "novel_set_mrr" not in existing_cols:
            con.execute(
                "ALTER TABLE leaderboard ADD COLUMN novel_set_mrr DOUBLE"
            )
        if rows:
            # DuckDB can ingest a list of dicts via ``executemany``
            # but the column order needs to match the table. We use
            # a parameterised INSERT and the column names in
            # ``_CSV_COLUMNS`` (minus ``run_at``, which is the
            # default).
            insert_cols = [c for c in _CSV_COLUMNS if c != "run_at"]
            cols = ", ".join(insert_cols)
            placeholders = ", ".join("?" for _ in insert_cols)
            insert_sql = f"INSERT INTO leaderboard ({cols}) VALUES ({placeholders})"
            tuples = [tuple(row.get(c) for c in insert_cols) for row in rows]
            con.executemany(insert_sql, tuples)

        # Per-record trace — replaced each run.
        con.execute("DROP TABLE IF EXISTS per_record")
        con.execute(
            """
            CREATE TABLE per_record (
                config VARCHAR,
                benchmark VARCHAR,
                record_id VARCHAR,
                category VARCHAR,
                is_duplicate BOOLEAN,
                is_novel BOOLEAN,
                top1_score DOUBLE,
                search_error VARCHAR,
                ranked_ids JSON,
                ranked_scores JSON
            )
            """
        )
        by_id = {r.id: r for r in benchmark.records}
        if per_record:
            insert_sql = (
                "INSERT INTO per_record (config, benchmark, record_id, category, "
                "is_duplicate, is_novel, top1_score, search_error, ranked_ids, "
                "ranked_scores) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            tuples = [
                (
                    config.name,
                    str(benchmark.path),
                    res.record_id,
                    res.category,
                    res.is_duplicate,
                    res.is_novel,
                    res.top1_score,
                    res.search_error,
                    json.dumps(list(res.ranked_ids)),
                    json.dumps(list(res.ranked_scores)),
                )
                for res in per_record
            ]
            con.executemany(insert_sql, tuples)
    finally:
        con.close()


def _format_markdown_table(
    rows: Sequence[Dict[str, Any]],
    *,
    config_name: str,
    benchmark_path: Path,
    best_threshold: float,
) -> str:
    """Format the per-threshold rows as a Markdown table for the README.

    The dashboard in 1.11 and the README's leaderboard screenshot
    in 1.10 both paste this verbatim, so the format is fixed.

    Phase 3.3 added the ``ECE`` column (the bin-count-weighted
    Expected Calibration Error) — same value for every row in the
    table because ECE is a run-level metric, not a per-threshold
    metric.

    Phase 3.5 added the ``novel_set_mrr`` column — the per-config
    headline FPR-on-novel at the chosen ``best_threshold``. Same
    value on every row of a given config block (it's a config-level
    scalar, not a per-threshold sweep value). This is the
    "trust this tool" number the README quotes.
    """
    lines: List[str] = []
    lines.append(f"# Eval leaderboard — `{config_name}` on `{benchmark_path.name}`")
    lines.append("")
    lines.append(
        "Metrics are computed at each cosine threshold on the sweep "
        "[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]. The "
        "`selected` row is the threshold that maximises MRR subject "
        "to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap). "
        "**ECE** is run-level (independent of the threshold sweep). "
        "PHASE-3.md §3.3 target: ECE ≤ 0.10 (informational). "
        "**novel_set_mrr** is the per-config headline FPR-on-novel at "
        "the chosen best_threshold (PHASE-3.md §3.5 'trust this tool' "
        "metric) — same value on every row of this config block. "
        "Eval set: `labeled_v300.jsonl` (LLM-generated v2, "
        "hand-review pending)."
    )
    lines.append("")
    header = (
        "| threshold | MRR | nDCG@10 | precision@5 | recall@10 | "
        "FPR-on-novel | ECE | novel_set_mrr | selected |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        is_sel = bool(row.get("selected_threshold"))
        marker = "**" if is_sel else ""
        thr = row.get("threshold", "")
        ece_cell = row.get("ece", "")
        try:
            ece_str = f"{float(ece_cell):.3f}"
        except (TypeError, ValueError):
            ece_str = str(ece_cell) if ece_cell != "" else "—"
        # novel_set_mrr is the same value on every row of a config
        # block. Format with 3 decimals to match the FPR column.
        try:
            nsm_str = f"{float(row.get('novel_set_mrr', 0)):.3f}"
        except (TypeError, ValueError):
            nsm_str = "—"
        lines.append(
            f"| {marker}{thr}{marker} "
            f"| {marker}{float(row.get('mrr', 0)):.3f}{marker} "
            f"| {marker}{float(row.get('ndcg_at_10', 0)):.3f}{marker} "
            f"| {marker}{float(row.get('precision_at_5', 0)):.3f}{marker} "
            f"| {marker}{float(row.get('recall_at_10', 0)):.3f}{marker} "
            f"| {marker}{float(row.get('fpr_on_novel', 0)):.3f}{marker} "
            f"| {marker}{ece_str}{marker} "
            f"| {marker}{nsm_str}{marker} "
            f"| {marker}{'YES' if is_sel else ''}{marker} |"
        )
    lines.append("")
    lines.append(f"Best threshold (MRR-max under FPR ≤ 0.15): **{best_threshold}**")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_eval(
    config: RetrievalConfig,
    benchmark: Benchmark,
    *,
    output_csv: Path,
    db_path: Optional[Path] = None,
    threshold_sweep: Sequence[float] = DEFAULT_THRESHOLD_SWEEP,
    fpr_cap: float = 0.15,
    offline: bool = False,
    precomputed_query_embeddings: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full eval and return the run summary.

    Steps (see module docstring for the why):
    1. For each record: POST ``/search`` (HTTP) **or** call the
       in-process offline backend (when ``offline=True``). The
       offline path is what the CI regression workflow uses —
       it doesn't need a live API or a bge-m3 download.
    2. For each threshold: compute the 5 aggregate metrics.
    3. Pick the best threshold (MRR-max under FPR cap).
    4. Write the per-threshold rows to the CSV (append).
    5. Write the same rows + per-record trace to DuckDB.
    6. Return a summary dict for the CLI to print.
    """
    if offline and not precomputed_query_embeddings:
        raise ValueError(
            "offline=True requires --precomputed-query-embeddings pointing at "
            "data/cache/eval_query_embeddings.npz (or a snapshot thereof). "
            "Run `python scripts/build_eval_query_embeddings.py` to produce it."
        )
    started = time.time()
    per_record: List[PerRecordResult] = []
    if offline:
        searcher = OfflineSearcher(Path(precomputed_query_embeddings))
        for rec in benchmark.records:
            res = run_one_record_offline(rec, config=config, searcher=searcher)
            per_record.append(res)
    else:
        with httpx.Client() as client:
            for rec in benchmark.records:
                res = run_one_record(rec, config=config, client=client)
                per_record.append(res)

    search_errors = sum(1 for r in per_record if r.search_error)
    n_novel = sum(1 for r in per_record if r.is_novel)
    n_dup = sum(1 for r in per_record if r.is_duplicate)

    # Per-threshold aggregates (Phase 3.5: we need to know ``best``
    # before we can compute ``novel_set_mrr`` — the headline FPR
    # at the picked best_threshold). Two-pass approach: first pass
    # is the threshold sweep (novel_set_mrr=0 by default since
    # best_threshold isn't known yet); second pass is a single
    # compute_aggregate call at best_threshold to extract the
    # headline value, which we then patch onto every row.
    aggregates: Dict[float, AggregateMetrics] = {}
    for thr in threshold_sweep:
        aggregates[thr] = compute_aggregate(per_record, benchmark, threshold=thr)

    mrr_by_t = {t: a.mrr for t, a in aggregates.items()}
    fpr_by_t = {t: a.fpr_on_novel for t, a in aggregates.items()}
    best = pick_best_threshold(
        threshold_sweep=threshold_sweep,
        mrr_by_threshold=mrr_by_t,
        fpr_by_threshold=fpr_by_t,
        fpr_cap=fpr_cap,
    )

    # Phase 3.5 — novel_set_mrr headline (the per-config FPR at
    # best_threshold). Recompute the aggregate at best_threshold
    # so we get the populated value, then read it off.
    headline_agg = compute_aggregate(
        per_record, benchmark, threshold=best, best_threshold=best
    )
    novel_set_mrr_value = headline_agg.novel_set_mrr

    # Build the CSV rows. The "selected_threshold" boolean is True
    # only for the best row, so the CSV reader can pick out the
    # production threshold without re-running the picker.
    # ``corpus_count`` is empty here (the runner doesn't fetch it
    # itself — the live API does that and Phase 1.11's dashboard
    # reads the corpus size from the ``companies`` table). The
    # CSV column stays for schema stability.
    #
    # The ``ece`` value is patched in below after the calibration
    # pass — until that runs we don't know it.
    rows: List[Dict[str, Any]] = []  # noqa: F841
    for thr, agg in aggregates.items():
        rows.append(
            {
                "config": config.name,
                "benchmark": benchmark.path.name,
                "corpus_count": "",
                "embedding_model": config.embedding_model,
                "threshold": thr,
                "mrr": agg.mrr,
                "ndcg_at_10": agg.ndcg_at_10,
                "precision_at_5": agg.precision_at_5,
                "recall_at_10": agg.recall_at_10,
                "fpr_on_novel": agg.fpr_on_novel,
                "ece": "",  # filled below from the calibration pass
                "novel_set_mrr": novel_set_mrr_value,  # Phase 3.5 headline
                "records_total": len(benchmark),
                "records_novel": n_novel,
                "records_duplicate": n_dup,
                "records_skipped": 0,
                "search_errors": search_errors,
                "selected_threshold": (thr == best),
                "notes": config.notes,
            }
        )

    # Fill in the corpus count from the live API before writing.
    # If the API is unreachable we leave the value blank — Phase
    # 1.11's dashboard queries the corpus size from the DB
    # directly, so a missing value is recoverable.
    if offline:
        # Offline path: read the corpus count from the DB directly
        # (the OfflineSearcher has it cached, but we recompute so
        # the call is self-contained). The offline path always
        # knows the corpus count because the .npz was just loaded.
        try:
            cc = searcher.corpus_count()
            for row in rows:
                row["corpus_count"] = cc
        except Exception as exc:  # noqa: BLE001 — count is informational
            logger.warning("offline: failed to read corpus_count: %s", exc)
    else:
        try:
            with httpx.Client() as probe:
                r = probe.get(
                    config.api_url.replace("/search", "/healthz"),
                    timeout=5.0,
                )
                if r.status_code == 200:
                    body = r.json()
                    cc = body.get("corpus_count")
                    if isinstance(cc, int):
                        for row in rows:
                            row["corpus_count"] = cc
        except (httpx.HTTPError, json.JSONDecodeError):
            pass

    # ------------------------------------------------------------------
    # Phase 3.3 — Calibration curve + ECE
    # ------------------------------------------------------------------
    # Compute the reliability table BEFORE writing the leaderboard
    # CSV / DuckDB rows so the ``ece`` column lands populated (not as
    # an empty string) in both sinks. ECE is a property of
    # (retrieval-config, eval-set) — it doesn't depend on the
    # threshold sweep — but we still write one ``ece`` value per row
    # so the CSV is uniform and downstream readers don't need a
    # separate key.
    cal_scores: List[float] = []
    cal_labels: List[bool] = []
    for res in per_record:
        if res.top1_score is None:
            continue
        cal_scores.append(float(res.top1_score))
        # The benchmark's is_duplicate flag is what we want here
        # (matches the ECE definition: "fraction of records in this
        # bin where is_duplicate=True"). ``PerRecordResult`` already
        # carries it forward from the benchmark.
        cal_labels.append(bool(res.is_duplicate))
    bins = bin_predictions_calibration(cal_scores, cal_labels)
    ece_value = compute_ece_metric(bins)
    # Patch every row, then proceed to the writes with the value
    # already on every dict. The writers (``write_csv`` /
    # ``write_duckdb``) treat the dict as the source of truth and
    # read each cell via ``row.get(k, "")``.
    for row in rows:
        row["ece"] = ece_value

    # Phase 3.3 honest-scope discipline (c8aa1fb): the labeled_v300
    # set is LLM-generated, awaiting Anurag's hand-label pass. Stamp
    # the provenance on the first row's ``notes`` cell so the
    # "provenance row in the CSV header" requirement (PHASE-3.md
    # §3.3) survives any downstream grep / pandas read without
    # breaking the column schema.
    _PROVENANCE_NOTES = (
        "provenance=llm-generated-v2-pending-anurag-hand-review "
        "(eval=labeled_v300.jsonl; ECE = LLM-generated, hand-label pending)"
    )
    if rows:
        first = rows[0]
        existing = str(first.get("notes", "") or "").strip()
        first["notes"] = (
            f"{existing} {_PROVENANCE_NOTES}" if existing else _PROVENANCE_NOTES
        )

    typer.echo(
        f"[eval] ece={ece_value:.4f} bins={len(bins)} "
        f"records_in_calibration={len(cal_scores)} "
        f"({len(cal_scores)}/{len(per_record)} non-errored)"
    )

    write_csv(output_csv, rows, append=True)
    if db_path is not None:
        write_duckdb(
            db_path,
            rows,
            config=config,
            benchmark=benchmark,
            per_record=per_record,
        )

    elapsed = time.time() - started
    best_agg = aggregates[best]

    # ----------------------------------------------------------------
    # Phase 3.5 — FPR-on-novel per-bin breakdown (computed once,
    # used by both PNGs).
    # ----------------------------------------------------------------
    # Same data (per-record top1_score + is_duplicate) but a
    # different aggregation. The FPR breakdown is the "trust this
    # tool" framing: per-bin counts of novel (red) vs duplicate
    # (blue) records, plus the per-bin FPR contribution so the
    # reader can read off "FPR at any threshold" by summing the
    # contribution bars. The 3.3 calibration PNG now also takes
    # this as an overlay on a second y-axis.
    fpr_bins = fpr_breakdown(cal_scores, cal_labels)
    docs_assets_dir = EVALS_DIR.parent / "docs" / "assets"

    # ----------------------------------------------------------------
    # Phase 3.3 — Calibration curve + ECE (with 3.5 FPR overlay)
    # ----------------------------------------------------------------
    # Write the calibration PNG. We do it at every run, not just on
    # the "selected" row — the curve is the same regardless of
    # threshold, and a per-config PNG that can be diffed across
    # runs is more useful than a guarded-once write.
    cal_png_path = (
        docs_assets_dir / f"calibration-{config.name}.png"
    )
    # Honest provenance stamp matches the Phase 1.5a /
    # Phase 2.8 discipline (c8aa1fb / 5c1c8fa): the eval set is
    # LLM-generated and the call-out lives in BOTH the title and
    # the per-record CSV header so the artifact stays honest
    # wherever it surfaces.
    title_extra = ""
    if ece_value > 0.10:
        title_extra = (
            f" (ECE > 0.10 — above the PHASE-3.md §3.3 "
            f"informational target; recorded verbatim)"
        )
    plot_calibration_curve(
        bins,
        config_name=config.name,
        eval_name=benchmark.path.name,
        provenance="LLM-generated v2, hand-review pending",
        output_path=cal_png_path,
        title_extra=title_extra,
        fpr_bins=fpr_bins,  # Phase 3.5 overlay (FPR-on-novel)
    )

    # ----------------------------------------------------------------
    # Phase 3.5 — FPR-on-novel per-bin breakdown (companion PNG)
    # ----------------------------------------------------------------
    # Dedicated 2-subplot PNG (stacked bars + per-bin FPR
    # contribution). Lives next to the calibration PNG so the two
    # are visually comparable.
    fpr_png_path = (
        docs_assets_dir / f"fpr-on-novel-breakdown-{config.name}.png"
    )
    fpr_title_extra = ""
    if best_agg.fpr_on_novel > 0.15:
        fpr_title_extra = (
            f" (FPR > 0.15 — above the PHASE-3.md §3.5 "
            f"'trust this tool' cap; recorded verbatim)"
        )
    plot_fpr_breakdown_curve(
        fpr_bins,
        config_name=config.name,
        best_threshold=best,
        fpr_on_novel=best_agg.fpr_on_novel,
        eval_name=benchmark.path.name,
        provenance="LLM-generated v2, hand-review pending",
        output_path=fpr_png_path,
        title_extra=fpr_title_extra,
    )

    # ----------------------------------------------------------------
    # Phase 3.5 — Per-config FPR-on-novel markdown summary
    # ----------------------------------------------------------------
    # One short table per config, lives next to the failure
    # breakdown markdown. The "trust this tool" framing lives in
    # the cap_phrase call-out: the headline FPR vs the 0.15 cap,
    # and the cumulative FPR at best_threshold as a cross-check.
    from src.eval.calibration import write_per_config_markdown
    md_path = (
        docs_assets_dir / f"fpr-on-novel-breakdown-{config.name}.md"
    )
    # corpus_count is only available once the leaderboard rows
    # are written; we re-derive the int here so the markdown
    # carries the same value as the per-record CSV header.
    corpus_count_for_md = 0
    for row in rows:
        cc = row.get("corpus_count", "")
        if isinstance(cc, int):
            corpus_count_for_md = cc
            break
        if cc and str(cc).isdigit():
            corpus_count_for_md = int(cc)
            break
    write_per_config_markdown(
        fpr_bins,
        config_name=config.name,
        benchmark_name=benchmark.path.name,
        best_threshold=best,
        fpr_on_novel=best_agg.fpr_on_novel,
        novel_set_mrr=novel_set_mrr_value,
        ece=ece_value,
        corpus_count=corpus_count_for_md,
        total_novel=len(benchmark.novel_records()),
        total_duplicate=len(benchmark.duplicate_records()),
        total_records=len(benchmark),
        output_path=md_path,
    )

    return {
        "config": config.name,
        "benchmark": benchmark.path.name,
        "best_threshold": best,
        "best_mrr": best_agg.mrr,
        "best_ndcg_at_10": best_agg.ndcg_at_10,
        "best_precision_at_5": best_agg.precision_at_5,
        "best_recall_at_10": best_agg.recall_at_10,
        "best_fpr_on_novel": best_agg.fpr_on_novel,
        "novel_set_mrr": novel_set_mrr_value,
        "rows": rows,
        "per_record": per_record,
        "elapsed_seconds": elapsed,
        "search_errors": search_errors,
        "records_total": len(benchmark),
        "records_novel": n_novel,
        "records_duplicate": n_dup,
        "fpr_cap": fpr_cap,
        "ece": ece_value,
        "calibration_bins": bins,
        "calibration_png": str(cal_png_path),
        "fpr_breakdown_bins": fpr_bins,
        "fpr_breakdown_png": str(fpr_png_path),
    }


# ---------------------------------------------------------------------------
# Phase 2.4 — MLflow tracker integration
# ---------------------------------------------------------------------------
#
# This block plugs the eval runner into MLflow so every `make eval` /
# `python -m eval.run` invocation is traceable end-to-end:
#
# - ``corpus_snapshot_date_from_snapshots_dir`` reads the latest
#   ``yc_<date>.jsonl`` filename from ``data/snapshots/`` to get the
#   snapshot date used for the run params.
# - ``write_per_record_csv`` writes the per-record trace to disk so
#   MLflow has a real artifact to upload (the spec calls for the
#   per-record CSV as an artifact).
# - ``log_eval_run_to_mlflow`` is the single MLflow entrypoint used
#   by the CLI. It groups params / metrics / artifacts / prompt
#   template into one MLflow run, with a clean fallback when the
#   tracking server is unreachable.
#
# Keeping these helpers in this module (instead of calling MLflow
# directly from the CLI) means the runner can be called from a
# Jupyter notebook / unit test / web admin endpoint later without
# re-implementing the wiring.


_PER_RECORD_CSV_COLUMNS: Tuple[str, ...] = (
    "record_id",
    "category",
    "is_duplicate",
    "is_novel",
    "top1_score",
    "ranked_ids",
    "ranked_scores",
    "search_error",
)


def corpus_snapshot_date_from_snapshots_dir(
    snapshots_dir: Optional[Path] = None,
) -> str:
    """Return the date of the latest snapshot in ``data/snapshots``.

    Reads the ``yc_<YYYY-MM-DD>.jsonl`` filename pattern. Falls back
    to ``"unknown"`` when no snapshot is on disk (so the eval still
    runs in a freshly-cloned fresh-build CI step).
    """
    sd = snapshots_dir or SNAPSHOTS_DIR
    if not sd.exists():
        return "unknown"
    dates: List[str] = []
    for p in sd.iterdir():
        if not p.is_file():
            continue
        # yc_<YYYY-MM-DD>.jsonl
        stem = p.stem  # "yc_2026-06-08"
        if not stem.startswith("yc_"):
            continue
        date_part = stem[len("yc_"):]
        if len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-":
            dates.append(date_part)
    return max(dates) if dates else "unknown"


def write_per_record_csv(
    path: Path,
    per_record: Sequence[PerRecordResult],
    *,
    config_name: str,
    benchmark_name: str,
) -> Path:
    """Write the per-record trace to disk so MLflow can pick it up.

    The spec calls for "the per-record CSV" as an MLflow artifact.
    The runner already loads each record through ``/search`` and
    holds the result in memory; this helper gives the result a
    durable on-disk form for the tracking side. The CSV is intended
    for MLflow / debugging — the *authoritative* per-record trace
    lives in DuckDB (column-for-column identical).

    Returns ``path`` so callers can chain ``log_artifact`` calls.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PER_RECORD_CSV_COLUMNS)
        writer.writeheader()
        for res in per_record:
            writer.writerow(
                {
                    "record_id": res.record_id,
                    "category": res.category,
                    "is_duplicate": res.is_duplicate,
                    "is_novel": res.is_novel,
                    "top1_score": res.top1_score if res.top1_score is not None else "",
                    "ranked_ids": ",".join(str(i) for i in res.ranked_ids),
                    "ranked_scores": ",".join(
                        f"{s:.4f}" for s in res.ranked_scores
                    ),
                    "search_error": res.search_error or "",
                }
            )
    return path


def log_eval_run_to_mlflow(
    summary: Dict[str, Any],
    *,
    config: RetrievalConfig,
    benchmark: Benchmark,
    output_csv: Path,
    per_record: Sequence[PerRecordResult],
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    no_mlflow: bool = False,
) -> Optional[RunRecord]:
    """Wire the eval run into MLflow.

    Steps (only when ``no_mlflow`` is False):

    1. Derive ``corpus_snapshot_date`` from ``data/snapshots/``.
    2. Read the prompt template version from
       ``src/llm/prompts/compare.PROMPT_TEMPLATE_VERSION``.
    3. Write the per-record CSV next to the leaderboard CSV.
    4. Call ``log_run`` with params / metrics / artifacts /
       prompt text. Returns the ``RunRecord`` for the CLI to print.
    """
    if no_mlflow:
        return None

    # Importing here to keep the module import-order clean (the
    # mlflow SDK is heavy and we don't want it to fire when the
    # module is just being collected).
    try:
        from src.llm.prompts.compare import (
            PROMPT_TEMPLATE_VERSION,
            SYSTEM_PROMPT,
            _USER_PROMPT_TEMPLATE,
        )
    except ImportError:
        # The prompt module is in the same tree; if it can't be
        # imported that's a real bug, so fall back to a placeholder
        # rather than crashing the eval.
        PROMPT_TEMPLATE_VERSION = "compare-v1"
        SYSTEM_PROMPT = ""
        _USER_PROMPT_TEMPLATE = ""

    corpus_date = corpus_snapshot_date_from_snapshots_dir()
    corpus_count = (
        summary["rows"][0].get("corpus_count", "")
        if summary["rows"] else ""
    )
    if not isinstance(corpus_count, int):
        corpus_count_int = 0
    else:
        corpus_count_int = corpus_count

    # Write the per-record CSV next to the leaderboard CSV so
    # the artifact upload keeps a stable path on subsequent runs.
    per_record_path = (
        output_csv.parent / f"per_record.{config.name}.{benchmark.path.stem}.csv"
    )
    write_per_record_csv(
        per_record_path,
        per_record,
        config_name=config.name,
        benchmark_name=benchmark.path.name,
    )

    # Write a one-row leaderboard slice capturing *this* run's
    # best threshold — used as a second MLflow artifact (the spec
    # calls for "the leaderboard CSV row" alongside the per-record
    # CSV).
    leaderboard_slice_path = (
        output_csv.parent
        / f"leaderboard_row.{config.name}.{benchmark.path.stem}.csv"
    )
    with leaderboard_slice_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(row)

    # Build the param dict via the canonical helper so we can't
    # drift from the schema documented in mlflow_logger.
    best_threshold = summary["best_threshold"]
    rows_first = summary["rows"][0]
    params = params_from_summary(
        config_name=config.name,
        embedding_model=rows_first.get("embedding_model", config.embedding_model),
        threshold=best_threshold,
        benchmark_name=benchmark.path.name,
        corpus_count=corpus_count_int,
        corpus_snapshot_date=corpus_date,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        api_url=config.api_url,
        top_k=int(getattr(config, "top_k", 0) or 0),
    )
    # Run name carries the config + benchmark + corpus-date so the
    # MLflow UI list is readable at a glance.
    run_name = (
        f"{config.name} | {benchmark.path.stem} | corpus={corpus_date} | "
        f"t={best_threshold:.2f} | run={int(time.time())}"
    )

    # Compose the prompt template artifact body — system + user.
    # We log it as TEXT (artifact), NOT as a param, per the Phase 2.4
    # pitfall rule.
    prompt_text = (
        f"# SYSTEM_PROMPT ({PROMPT_TEMPLATE_VERSION})\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"# _USER_PROMPT_TEMPLATE\n\n"
        f"{_USER_PROMPT_TEMPLATE}\n"
    )

    metrics = metrics_from_summary(summary, best_threshold=best_threshold)

    return log_run(
        experiment_name=experiment_name or "phase-2-baseline",
        params=params,
        metrics=metrics,
        artifacts={
            "leaderboard_row.csv": leaderboard_slice_path,
            "per_record.csv": per_record_path,
            **(
                {"leaderboard_full.csv": output_csv}
                if output_csv.exists()
                else {}
            ),
        },
        prompt_template_text=prompt_text,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags={
            "phase": "2.4",
            "card": "t_bc2a06cc",
            "config": config.name,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    help=(
        "Eval harness (Phase 1.6). Run a labeled benchmark against a "
        "retrieval config and write a per-threshold leaderboard CSV "
        "(plus a Markdown summary + a DuckDB queryable store)."
    ),
)


def _resolve_path(p: Path, base: Path) -> Path:
    """Resolve a path against a base dir if it's relative."""
    return p if p.is_absolute() else (base / p)


@app.command()
def main(
    benchmark: Path = typer.Option(
        ...,
        "--benchmark",
        "-b",
        help="Path to the labeled benchmark JSONL (e.g. evals/labeled_v100.jsonl).",
        exists=False,  # we check ourselves to give a better error
    ),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the retrieval config YAML (e.g. configs/dense_bge_m3.yaml).",
        exists=False,
    ),
    output: Path = typer.Option(
        RESULTS_DIR / "leaderboard.csv",
        "--output",
        "-o",
        help="Path to the leaderboard CSV (append mode).",
    ),
    db: Optional[Path] = typer.Option(
        RESULTS_DIR / "eval.duckdb",
        "--db",
        help="Path to the DuckDB store. Pass an explicit path or 'none' to disable.",
    ),
    fpr_cap: float = typer.Option(
        0.15,
        "--fpr-cap",
        help="Maximum acceptable FPR-on-novel when picking the best threshold.",
        min=0.0,
        max=1.0,
    ),
    markdown_out: Optional[Path] = typer.Option(
        None,
        "--markdown-out",
        "-m",
        help="Optional path to write the Markdown summary table to.",
    ),
    threshold: Optional[str] = typer.Option(
        None,
        "--threshold",
        help=(
            "Override the threshold sweep (comma-separated, e.g. "
            "'0.6,0.7,0.8'). Default: docs/PHASE-1.md §1.6 sweep."
        ),
    ),
    # ------------------------------------------------------------------
    # Phase 2.4 — MLflow tracker integration
    # ------------------------------------------------------------------
    # Three flags; together they replicate the spec's
    # `python -m eval.run --experiment-name "phase-2-baseline" \
    #                       --config configs/dense_bge_m3.yaml`
    # invocation, with explicit opt-outs and a tracking-URI override
    # for offline / file-based runs.
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        "-x",
        help=(
            "MLflow experiment name to log this run under. Default: "
            "'phase-2-baseline'. The experiment is created on first use."
        ),
    ),
    mlflow_tracking_uri: Optional[str] = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help=(
            "Override the MLflow tracking URI. Default: the "
            "MLFLOW_TRACKING_URI env var, falling back to "
            "http://localhost:15000. Use 'file:./mlruns' for fully-"
            "offline runs (no server required)."
        ),
    ),
    no_mlflow: bool = typer.Option(
        False,
        "--no-mlflow",
        help=(
            "Skip MLflow logging for this run. The CSV / DuckDB "
            "outputs are still written; only the MLflow call is "
            "bypassed. Useful when the tracker server is down and "
            "the operator only wants the leaderboard row."
        ),
    ),
    # ------------------------------------------------------------------
    # Phase 3.6.2 — Offline mode (card t_68dd7a03)
    # ------------------------------------------------------------------
    # The offline mode is what the CI regression workflow uses. It
    # bypasses the live ``/search`` HTTP path (which needs a live
    # API + bge-m3) and runs the same SQL ANN / BM25 / hybrid
    # paths in-process using precomputed query embeddings. The
    # ``--precomputed-query-embeddings`` path defaults to the
    # committed cache ``data/cache/eval_query_embeddings.npz`` —
    # the file the build script ``build_eval_query_embeddings.py``
    # writes. If you change the benchmark or the model version,
    # re-run that script and the offline mode picks up the new
    # cache automatically.
    offline: bool = typer.Option(
        False,
        "--offline",
        help=(
            "Run the eval in-process using precomputed query "
            "embeddings (no /search HTTP, no bge-m3 download). "
            "Requires --precomputed-query-embeddings. The "
            "eval-regression workflow uses this mode on cold-cache "
            "CI runs (see .github/workflows/eval-regression.yml)."
        ),
    ),
    precomputed_query_embeddings: Optional[Path] = typer.Option(
        None,
        "--precomputed-query-embeddings",
        help=(
            "Path to the eval query embeddings .npz (default: "
            "data/cache/eval_query_embeddings.npz when --offline "
            "is set). Required when --offline is set."
        ),
    ),
) -> None:
    """Run the eval harness end-to-end."""
    # Resolve paths (defaults from src.config point at the repo).
    benchmark_path = _resolve_path(benchmark, Path.cwd())
    if not benchmark_path.exists():
        # Fall back to repo-rooted resolution (CWD-relative vs
        # repo-rooted matters when running from `make eval`).
        repo_relative = _resolve_path(benchmark, EVALS_DIR.parent)
        if repo_relative.exists():
            benchmark_path = repo_relative
    if not benchmark_path.exists():
        typer.echo(f"benchmark file not found: {benchmark}", err=True)
        raise typer.Exit(code=1)

    config_path = _resolve_path(config, Path.cwd())
    if not config_path.exists():
        repo_relative = _resolve_path(config, EVALS_DIR.parent)
        if repo_relative.exists():
            config_path = repo_relative
    if not config_path.exists():
        typer.echo(f"config file not found: {config}", err=True)
        raise typer.Exit(code=1)

    output_path = _resolve_path(output, EVALS_DIR.parent)
    db_path_value = _resolve_path(db, EVALS_DIR.parent) if db is not None else None

    # Threshold sweep override (comma-separated string → floats).
    sweep: List[float] = list(DEFAULT_THRESHOLD_SWEEP)
    if threshold:
        try:
            sweep = [float(x.strip()) for x in threshold.split(",") if x.strip()]
        except ValueError as exc:
            typer.echo(f"invalid --threshold: {exc}", err=True)
            raise typer.Exit(code=1)

    cfg = RetrievalConfig.from_yaml(config_path)
    bench = load_benchmark(benchmark_path)

    typer.echo(
        f"[eval] config={cfg.name} benchmark={bench.path.name} "
        f"records={len(bench)} novel={len(bench.novel_records())} "
        f"thresholds={sweep}"
    )

    # Resolve the offline default: when --offline is set and the
    # caller didn't pass --precomputed-query-embeddings, fall back
    # to the committed cache (the file the build script writes).
    offline_embeddings: Optional[Path] = None
    if offline:
        offline_embeddings = (
            precomputed_query_embeddings
            if precomputed_query_embeddings is not None
            else Path("data/cache/eval_query_embeddings.npz")
        )

    summary = run_eval(
        cfg,
        bench,
        output_csv=output_path,
        db_path=db_path_value,
        threshold_sweep=sweep,
        fpr_cap=fpr_cap,
        offline=offline,
        precomputed_query_embeddings=offline_embeddings,
    )

    # Markdown summary.
    md = _format_markdown_table(
        summary["rows"],
        config_name=cfg.name,
        benchmark_path=bench.path,
        best_threshold=summary["best_threshold"],
    )
    if markdown_out is not None:
        md_path = _resolve_path(markdown_out, EVALS_DIR.parent)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        typer.echo(f"[eval] wrote Markdown summary to {md_path}")

    # Stdout summary.
    typer.echo("")
    typer.echo(md)
    typer.echo(
        f"[eval] done in {summary['elapsed_seconds']:.1f}s. "
        f"search_errors={summary['search_errors']} "
        f"fpr_cap={summary['fpr_cap']:.2f} "
        f"best_threshold={summary['best_threshold']} "
        f"(MRR={summary['best_mrr']:.3f}, "
        f"FPR-on-novel={summary['best_fpr_on_novel']:.3f})"
    )

    # ------------------------------------------------------------------
    # Phase 2.4 — MLflow logging (params / metrics / artifacts / prompt)
    # ------------------------------------------------------------------
    mlflow_record = log_eval_run_to_mlflow(
        summary,
        config=cfg,
        benchmark=bench,
        output_csv=output_path,
        per_record=summary["per_record"],
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        no_mlflow=no_mlflow,
    )
    if mlflow_record is not None:
        typer.echo(
            f"[mlflow] experiment='{experiment_name or 'phase-2-baseline'}' "
            f"run_id={mlflow_record.run_id} "
            f"tracking_uri={mlflow_record.tracking_uri_effective} "
            f"fallback={mlflow_record.fallback_used}"
        )

    # Exit non-zero if no threshold met the FPR cap — the runner
    # still wrote the leaderboard, but the caller should know.
    if summary["best_fpr_on_novel"] > fpr_cap:
        typer.echo(
            f"[eval] WARNING: no threshold on the sweep met the "
            f"FPR cap of {fpr_cap:.2f}; best-effort threshold "
            f"{summary['best_threshold']} has FPR="
            f"{summary['best_fpr_on_novel']:.3f}",
            err=True,
        )
        # Don't fail the run — the leaderboard is the artifact and
        # the operator should be able to inspect it. Phase 1.11's
        # dashboard will display the warning.


if __name__ == "__main__":
    app()