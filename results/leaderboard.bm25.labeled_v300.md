# Eval leaderboard — `bm25` on `labeled_v300.jsonl`

Metrics are computed at each cosine threshold on the sweep [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]. The `selected` row is the threshold that maximises MRR subject to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap).

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | selected |
|---|---|---|---|---|---|---|
| **0.5** | **0.392** | **0.410** | **0.090** | **0.470** | **1.000** | **YES** |
| 0.55 | 0.392 | 0.410 | 0.090 | 0.470 | 1.000 |  |
| 0.6 | 0.392 | 0.410 | 0.090 | 0.470 | 1.000 |  |
| 0.65 | 0.392 | 0.410 | 0.090 | 0.470 | 1.000 |  |
| 0.7 | 0.392 | 0.410 | 0.090 | 0.470 | 1.000 |  |
| 0.75 | 0.392 | 0.410 | 0.090 | 0.470 | 1.000 |  |
| 0.8 | 0.392 | 0.410 | 0.090 | 0.470 | 1.000 |  |

Best threshold (MRR-max under FPR ≤ 0.15): **0.5**
