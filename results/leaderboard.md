# Eval leaderboard — `hybrid_rrf` on `labeled_v300.jsonl`

Metrics are computed at each cosine threshold on the sweep [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]. The `selected` row is the threshold that maximises MRR subject to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap). **ECE** is run-level (independent of the threshold sweep). PHASE-3.md §3.3 target: ECE ≤ 0.10 (informational). **novel_set_mrr** is the per-config headline FPR-on-novel at the chosen best_threshold (PHASE-3.md §3.5 'trust this tool' metric) — same value on every row of this config block. Eval set: `labeled_v300.jsonl` (LLM-generated v2, hand-review pending).

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | ECE | novel_set_mrr | selected |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.458 | 0.482 | 0.106 | 0.560 | 1.000 | 0.506 | 0.630 |  |
| 0.55 | 0.458 | 0.482 | 0.106 | 0.560 | 1.000 | 0.506 | 0.630 |  |
| 0.6 | 0.458 | 0.482 | 0.106 | 0.560 | 1.000 | 0.506 | 0.630 |  |
| 0.65 | 0.458 | 0.482 | 0.106 | 0.560 | 1.000 | 0.506 | 0.630 |  |
| 0.7 | 0.458 | 0.482 | 0.106 | 0.560 | 1.000 | 0.506 | 0.630 |  |
| 0.75 | 0.458 | 0.482 | 0.106 | 0.560 | 0.980 | 0.506 | 0.630 |  |
| **0.8** | **0.458** | **0.482** | **0.106** | **0.560** | **0.630** | **0.506** | **0.630** | **YES** |

Best threshold (MRR-max under FPR ≤ 0.15): **0.8**
