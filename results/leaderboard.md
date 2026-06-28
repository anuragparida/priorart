# Eval leaderboard — `dense_bge_m3` on `labeled_v100.jsonl`

Metrics are computed at each cosine threshold on the sweep [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]. The `selected` row is the threshold that maximises MRR subject to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap).

| threshold | MRR | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | selected |
|---|---|---|---|---|---|---|
| 0.5 | 0.559 | 0.611 | 0.135 | 0.800 | 1.000 |  |
| 0.55 | 0.559 | 0.611 | 0.135 | 0.800 | 1.000 |  |
| 0.6 | 0.559 | 0.611 | 0.135 | 0.800 | 1.000 |  |
| 0.65 | 0.559 | 0.611 | 0.135 | 0.800 | 1.000 |  |
| 0.7 | 0.559 | 0.611 | 0.135 | 0.800 | 1.000 |  |
| 0.75 | 0.559 | 0.611 | 0.135 | 0.800 | 1.000 |  |
| **0.8** | **0.559** | **0.611** | **0.135** | **0.800** | **0.800** | **YES** |

Best threshold (MRR-max under FPR ≤ 0.15): **0.8**
