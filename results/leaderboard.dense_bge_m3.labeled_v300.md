# Eval leaderboard — `dense_bge_m3` on `labeled_v300.jsonl`

Metrics are computed at each cosine threshold on the sweep [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]. The `selected` row is the threshold that maximises MRR subject to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap).

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | selected |
|---|---|---|---|---|---|---|
| 0.5 | 0.567 | 0.575 | 0.120 | 0.600 | 1.000 |  |
| 0.55 | 0.567 | 0.575 | 0.120 | 0.600 | 1.000 |  |
| 0.6 | 0.567 | 0.575 | 0.120 | 0.600 | 1.000 |  |
| 0.65 | 0.567 | 0.575 | 0.120 | 0.600 | 1.000 |  |
| 0.7 | 0.567 | 0.575 | 0.120 | 0.600 | 1.000 |  |
| 0.75 | 0.567 | 0.575 | 0.120 | 0.600 | 0.995 |  |
| **0.8** | **0.567** | **0.575** | **0.120** | **0.600** | **0.790** | **YES** |

Best threshold (MRR-max under FPR ≤ 0.15): **0.8**
