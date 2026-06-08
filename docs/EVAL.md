# Eval Harness — Deep Dive

> The eval harness is the single most important part of this project. The whole CV claim depends on it. If the eval is shoddy, the project is a wrapper script. If the eval is rigorous, the project is a defensible artifact. **Do not write the eval set after the system** — write it first, then build against it. Otherwise you rationalize the benchmark to match the system's outputs and lose the entire thing.

---

## What we're measuring

The system has two distinct things to evaluate, and they should be scored **separately** because a high score on one does not imply a high score on the other.

1. **Retrieval quality** — "given a new idea, does the corpus-and-embedding pipeline return the right similar launches at the right ranks?"
2. **Comparison quality** — "given the top-3 retrieved launches, does the LLM produce a useful structured comparison?"

Phase 1 focuses on retrieval. Phase 2 adds comparison. Phase 3 adds the calibration + per-category breakdown on top of both.

---

## The benchmark: `evals/labeled_v300.jsonl`

### Construction policy

**300 hand-labeled triples**, balanced 100/100/100:

- **100 known-duplicate** (the "is there something like this?" positive set):
  - 40 drawn from YC directory: pick 20 active companies, write 2 paraphrasings of each as a "new idea." Label `is_duplicate=true`, `expected_top_ids=[company_id]`.
  - 30 from Product Hunt top launches.
  - 30 from HN "Show HN" top posts.
  - Sources are tracked in the record's `source` field for per-source breakdown.

- **100 known-novel** (the "this is genuinely new" negative set):
  - The single most important category. The system must not overclaim duplicates.
  - Examples: "an AI tool for composing Persian poetry," "a subscription service for vintage typewriters," "a forum for left-handed electricians in Berlin," "a tool for ranking the best public fountains in Rome."
  - These are ideas with no plausible match in YC + Product Hunt + HN.
  - Label `is_duplicate=false`, `expected_top_ids=[]`.

- **100 adversarial** (the "this looks like a duplicate but isn't quite" stress set):
  - 30 **paraphrase-with-pivot** — "Uber for X" where X is novel but the pattern is not.
  - 30 **market-overlap-different-tech** — "Stripe for crypto" (Stripe exists, crypto payments exist, the intersection is novel).
  - 20 **same-tech-different-domain** — "GitHub for legal contracts" (GitHub exists, the legal-contracts domain is not the same).
  - 20 **temporal-evolution** — "Slack" in 2024 vs. "Slack" in 2014 (the company exists but the market has moved on; the system should treat it as adjacent, not duplicate).

### Record schema

```json
{
  "id": "ev-001",
  "idea": "AI-powered contract review for SMB law firms",
  "source": "yc",
  "category": "duplicate" | "novel" | "adversarial_paraphrase" | "adversarial_market_overlap" | "adversarial_same_tech_diff_domain" | "adversarial_temporal",
  "expected_top_ids": [123, 456],
  "is_duplicate": true,
  "labeler": "anurag",
  "labeled_at": "2026-06-09T14:00:00Z",
  "notes": "Paraphrasing of Ironclad (YC S18)"
}
```

### Labeling rules

- **No LLM labeling. Ever.** Hand-label every triple. Spend the 3+ hours.
- A record is `is_duplicate=true` if a founder pitching this idea to a YC partner would be told "this already exists" with at least one specific reference.
- A record is `is_duplicate=false` if the closest YC + Product Hunt + HN launches are *adjacent* (same market, different angle) but not direct competitors.
- A record is `category=adversarial_*` if the system *should* be uncertain — the right answer is "low confidence, not a duplicate, but worth a closer look."
- All labels committed with `labeler=anurag` and a `labeled_at` timestamp. If you ever relabel a record (e.g. you realize a "novel" was actually a paraphrase of a real company), bump the `labeled_at` and add a `notes` line explaining the change. **Never delete a record.** History matters.

### `evals/labeled_v300.README.md`

A 1-page doc explaining the policy:
- Why 300 (enough to claim statistical signal, small enough to hand-label).
- Why the 100/100/100 split (we care about the false-positive rate on novel ideas more than the recall on duplicates).
- How the adversarial categories were chosen (the failure modes we expect the system to have).
- Per-category expected behavior (e.g. "on `adversarial_paraphrase`, the system should return a low-confidence near-duplicate verdict, not a clean 'no match'").

---

## Metrics

### Phase 1 (retrieval)

#### MRR (Mean Reciprocal Rank)
Average over the benchmark of `1/rank_of_first_relevant_result`. If the first relevant result is at rank 1, MRR=1.0. If at rank 5, MRR=0.2. If never found, MRR=0.

**For our purposes:** rank is the position in the ANN-search result list. The "first relevant result" is the first company in the result list whose ID is in `expected_top_ids`.

#### nDCG@10
Normalized Discounted Cumulative Gain at K=10. Rewards getting the right results at the top, with logarithmic position discount.

`nDCG@10 = DCG@10 / IDCG@10` where `DCG@K = sum_{i=1}^{K} (rel_i / log2(i+1))` and `rel_i = 1` if the i-th result is in `expected_top_ids`, else 0.

#### precision@5, recall@10
- **precision@5** = fraction of top-5 results that are in `expected_top_ids`.
- **recall@10** = fraction of `expected_top_ids` that appear in the top-10 results.

#### FPR-on-novel (the most important metric)
Among the 100 `is_duplicate=false` records, what fraction had a top-1 result with cosine similarity > threshold. **The lower, the better.** A high FPR-on-novel means the system is crying wolf — claiming "duplicate" for genuinely novel ideas — and no real user will trust the tool.

Phase 1 target: FPR-on-novel ≤ 0.15 at the production threshold (0.65 for bge-m3 cosine).

#### Threshold sweep
Run the eval at cosine thresholds `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]`. Report all 5 metrics at each threshold. The leaderboard shows the threshold that maximizes MRR subject to FPR-on-novel ≤ 0.15.

### Phase 2 (additions)

#### Comparison quality (held-out subset)
A separate held-out set of 30 records where the labeled expected output is the `CompetitorVerdict` (similarity_axes, key_differences, likely_failure_modes). For each, run the full pipeline, capture the LLM output, and score:

- **Schema validity** (Pydantic validation) — 1.0 if parses cleanly, 0.0 otherwise.
- **Cosine similarity** of the LLM-generated `similarity_axes` vector to the labeled `similarity_axes` vector (embed both with bge-m3, take cosine).
- **Key-difference recall** — fraction of the labeled `key_differences` that appear in the LLM output (lexical overlap after normalization).
- **LLM-as-judge disagreement rate** — for 5 hand-picked examples, have a *second* LLM call judge whether the LLM output matches the labeled output. Report disagreement between the deterministic scorer and the LLM judge. Phase 3 adds this formally; Phase 2 does it as a spot-check.

### Phase 3 (additions)

#### Calibration curve
For each similarity bin (0.0–0.1, 0.1–0.2, ..., 0.9–1.0), plot the actual duplicate rate (fraction of records in the bin where `is_duplicate=true`). A well-calibrated system hugs the diagonal.

**Expected Calibration Error (ECE)** = `sum_b |bin_count_b / N| * |bin_predicted_duplicate_rate_b - bin_actual_duplicate_rate_b|`. Target: ECE ≤ 0.10 on the dense config.

#### Per-category MRR / FPR
Per category (duplicate / novel / adversarial_paraphrase / adversarial_market_overlap / adversarial_same_tech_diff_domain / adversarial_temporal), per config: MRR and FPR-on-novel. This is the "where does the system fail" breakdown.

#### Leaderboard CSV schema
```csv
config,benchmark,corpus_snapshot,embedding_model,threshold,mrr,nDCG@10,precision@5,recall@10,fpr_on_novel,ece,comparison_schema_valid,comparison_axes_cosine,llm_judge_agreement,notes
dense_bge_m3,labeled_v300,yc_2026-06-09_ph_2026-06-09_hn_2026-06-09,bge-m3-v0.1,0.65,0.78,0.71,0.62,0.55,0.08,0.07,1.00,0.84,0.93,baseline
bm25,labeled_v300,...,bm25,0.65,0.55,0.48,0.42,0.40,0.18,0.22,1.00,0.71,0.81,sparse baseline
hybrid_rrf,labeled_v300,...,bge-m3+bm25,0.65,0.82,0.74,0.65,0.58,0.07,0.06,1.00,0.85,0.94,production
```

---

## The eval runner

`src/eval/runner.py` exposes:

```python
def run(config: RetrievalConfig, benchmark: Path, output: Path) -> LeaderboardRow:
    """Run the full eval for one config. Returns one LeaderboardRow."""
```

CLI: `python -m src.eval.run --config configs/dense_bge_m3.yaml --benchmark evals/labeled_v300.jsonl --output results/leaderboard.csv --mlflow-experiment "phase-3-baseline"`.

Behaviors:
- Loads the corpus from the configured snapshot dates (fails loudly if a snapshot is missing).
- For each record in the benchmark: run the full pipeline (embed → ANN search → top-K), record the result.
- Compute all metrics. Write the row to the CSV (append mode if the file exists).
- Log the run to MLflow with params + metrics.
- Plot the calibration curve if `output` is a directory.
- Print a Markdown summary table to stdout.

**Make targets:**
- `make eval` — full benchmark, all configs.
- `make eval-quick` — 30-record smoke subset, ~2 min, used in CI for fast feedback.
- `make eval-config CONFIG=configs/foo.yaml` — single config, full benchmark.

---

## The regression suite

`make eval` runs in GitHub Actions on:
- Every PR that touches `configs/`, `evals/`, `src/embedding/`, `src/llm/`, `models.yaml`, or `pyproject.toml`.
- Every push to `main`.
- Nightly at 03:00 UTC.

**Build-fail thresholds (production config = `hybrid_rrf`):**
- MRR < 0.7 → fail.
- FPR-on-novel > 0.15 → fail.
- ECE > 0.12 → fail.
- Comparison schema validity < 0.95 → fail.

The Action posts the leaderboard diff as a PR comment using `gh-actions-remark` or a custom action that diffs the new CSV against the committed one.

---

## What "good" looks like at the end of Phase 3

```csv
config,mrr,nDCG@10,precision@5,recall@10,fpr_on_novel,ece,comparison_schema_valid
hybrid_rrf,0.82,0.74,0.65,0.58,0.07,0.06,1.00
dense_bge_m3,0.78,0.71,0.62,0.55,0.08,0.07,1.00
bm25,0.55,0.48,0.42,0.40,0.18,0.22,1.00
```

And the README can claim, honestly:

> The hybrid config (dense bge-m3 + BM25, fused with RRF) achieves MRR=0.82 on a hand-labeled 300-idea benchmark, with a 7% false-positive rate on the novel-idea set. Calibration error is 6%. The eval harness runs as a regression suite on every config change; the build fails if MRR drops below 0.7 or FPR-on-novel exceeds 15%. Numbers are reproducible from the committed benchmark and the committed corpus snapshot.

That's the line. That's the artifact. That's the differentiator from every "AI startup validator" wrapper out there.

---

## Anti-patterns to refuse

- **LLM-generated labels.** "We used GPT-4 to generate 1000 training triples." No. The labels rationalize to match the system. Hand-label.
- **Single-metric optimization.** "We optimized for MRR." The MRR can be 0.99 by being aggressive about flagging duplicates — but then the FPR-on-novel is 0.5 and the tool is unusable. Always report the full set.
- **Composite scores.** "We have a single 0–100 'quality' score." Hide the failure modes. Always show MRR, nDCG, precision, recall, FPR, ECE, schema-validity separately.
- **Opaque thresholds.** "We tuned the threshold to 0.6543." Show the threshold sweep. The reader should be able to pick a different threshold and reproduce your numbers.
- **Closed-corpus benchmarks.** "We evaluated on a private dataset of 10K ideas." No. The whole point is reproducibility. Commit the corpus, commit the labels, let anyone reproduce.
- **No reproducibility story.** "We ran the eval once and got these numbers." Show the Makefile, show the commands, show the Docker Compose. A reader should be able to run `make eval` and get the same numbers.
