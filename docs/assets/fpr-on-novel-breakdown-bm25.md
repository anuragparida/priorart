# FPR-on-novel breakdown — `bm25` on `labeled_v300.jsonl`

Per-bin FPR-on-novel breakdown at the runner-picked ``best_threshold=0.50``. Eval set: ``labeled_v300.jsonl`` (LLM-generated v2, hand-review pending — same provenance policy as Phase 1.5a / 2.8 / 3.3 / 3.4).

**Headline numbers:** FPR-on-novel = `1.000` at ``best_threshold=0.50`` (**above the PHASE-3.md §3.5 cap of ≤ 0.15** (`actual=1.000`, `gap=+0.850`) — no config currently clears the cap; the per-bin breakdown below surfaces the gap honestly.) | ``novel_set_mrr = 1.000`` (same denominator, same value, surfaced under the README-quoted name) | ``ECE = 0.603`` (informational) | corpus_count = 10983 | N = 300 (novel = 200, duplicate = 100).

The table below answers: *for each score bin, how many of the novel records live there, and what fraction of the bin is novel?* The ``fpr_contribution`` column is the fraction of the **whole** novel subset that lives in the bin — summing it over all bins whose lower edge is ``≥ T`` gives the FPR-on-novel at threshold T. **Cumulative FPR at best_threshold=0.50** is the sum of ``fpr_contribution`` from the last two rows of the table.

| bin | range | novel_count | duplicate_count | novel_fraction | fpr_contribution |
|---|---|---|---|---|---|
| 0 | [0.0, 0.1) | 0 | 0 | 0.000 | 0.000 |
| 1 | [0.1, 0.2) | 0 | 0 | 0.000 | 0.000 |
| 2 | [0.2, 0.3) | 0 | 0 | 0.000 | 0.000 |
| 3 | [0.3, 0.4) | 0 | 0 | 0.000 | 0.000 |
| 4 | [0.4, 0.5) | 0 | 0 | 0.000 | 0.000 |
| 5 | [0.5, 0.6) | 0 | 0 | 0.000 | 0.000 |
| 6 | [0.6, 0.7) | 0 | 0 | 0.000 | 0.000 |
| 7 | [0.7, 0.8) | 0 | 0 | 0.000 | 0.000 |
| 8 | [0.8, 0.9) | 1 | 0 | 1.000 | 0.005 |
| 9 | [0.9, 1.0) | 199 | 100 | 0.666 | 0.995 |

**Cumulative FPR-on-novel at ``best_threshold=0.50``:** `1.000` (sum of ``fpr_contribution`` from the 5 bin(s) at or above the threshold). This is the same value as the headline ``FPR-on-novel = 1.000`` — the table is the bucketed view of that scalar.

**Honest scope:** the eval set is LLM-generated v2 and the hand-label pass is a follow-up; the FPR numbers above are honest but the underlying labels are pending Anurag's review. No config currently clears the 0.15 cap, so the "trust this tool" claim on the README is gated on Phase 4 (reranker) closing the gap.
