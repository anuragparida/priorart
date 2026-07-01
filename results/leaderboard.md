# PriorArt Eval Leaderboard

Generated from `results/leaderboard.csv`. Rows are grouped by (config, benchmark) and sorted by threshold. The `selected` row is the threshold that maximises MRR subject to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap).

Eval set provenance: `evals/labeled_v300.jsonl` is **LLM-generated v2 with honest provenance** (`labeler=ai-assisted-claude-minimax-m3`, `provenance=llm-generated-v2-pending-anurag-hand-review`) per the Phase 1.5a fix (commit c8aa1fb). MRR targets in PHASE-2.md §Definition-of-done are INFORMATIONAL until the hand-label pass lands.

<!-- ECE computed against LLM-generated v300; hand-label pending -->
<!-- ECE ≤ 0.10 is the PHASE-3.md §3.3 *informational* target; recorded verbatim below. -->

Cohere rerank is opt-in only (AGENTS.md + PHASE-2.md §Pitfalls) and NOT a default Phase 2 config — not present in this table.

## `dense_bge_m3` on `labeled_v300.jsonl` — corpus=10983 (BAAI/bge-m3)

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | ECE | selected |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.5 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.5 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.55 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.55 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.55 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.6 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.6 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.6 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.65 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.65 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.65 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.7 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.7 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.7 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 1.0 | 0.527 |  |
| 0.75 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 0.995 | 0.527 |  |
| 0.75 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 0.995 | 0.527 |  |
| 0.75 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 0.995 | 0.527 |  |
| 0.8 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 0.79 | 0.527 | **YES** |
| 0.8 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 0.79 | 0.527 | **YES** |
| 0.8 | 0.5666666666666668 | 0.5752371901428583 | 0.1199999999999999 | 0.6 | 0.79 | 0.527 | **YES** |

Best threshold (MRR-max under FPR ≤ 0.15): **0.8** — MRR=0.5666666666666668, FPR-on-novel=0.79, ECE=0.527

_Notes: Baseline Phase 1 config. Dense bge-m3 only, no reranker._

## `bm25` on `labeled_v300.jsonl` — corpus=10983 (rank_bm25)

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | ECE | selected |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 | **YES** |
| 0.55 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 |  |
| 0.6 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 |  |
| 0.65 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 |  |
| 0.7 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 |  |
| 0.75 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 |  |
| 0.8 | 0.3921907993966818 | 0.41028163062722056 | 0.09 | 0.47 | 1.0 | 0.603 |  |

Best threshold (MRR-max under FPR ≤ 0.15): **0.5** — MRR=0.3921907993966818, FPR-on-novel=1.0, ECE=0.603

_Notes: Phase 2.9 BM25 config — lexical retrieval (rank_bm25.BM25Okapi) over companies.name + companies.description. k1=1.5, b=0.75 (literature defaults). Tiny English stopword set; lowercased + split on non-alphanumeric runs. The corpus index is built lazily on first call and cached at module level; ingest_

## `hybrid_rrf` on `labeled_v300.jsonl` — corpus=10983 (BAAI/bge-m3 + rank_bm25)

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | ECE | selected |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 1.0 | 0.506 |  |
| 0.55 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 1.0 | 0.506 |  |
| 0.6 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 1.0 | 0.506 |  |
| 0.65 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 1.0 | 0.506 |  |
| 0.7 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 1.0 | 0.506 |  |
| 0.75 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 0.98 | 0.506 |  |
| 0.8 | 0.4579285714285715 | 0.4818750215641461 | 0.10599999999999994 | 0.56 | 0.63 | 0.506 | **YES** |

Best threshold (MRR-max under FPR ≤ 0.15): **0.8** — MRR=0.4579285714285715, FPR-on-novel=0.63, ECE=0.506

_Notes: Phase 2.9 Hybrid RRF config — Reciprocal Rank Fusion of dense (bge-m3 pgvector ANN) + BM25 (rank_bm25 lexical) over the same companies table. k=60 (Cormack et al. 2009 default). Dense confidence is inherited on the fused hit so the threshold sweep is comparable across modes. Both retrievers over-fet_
