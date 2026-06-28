# Methodology

> How PriorArt is evaluated. What each metric means, how the labeled
> benchmark was constructed, and what the threshold-sweep leaderboard
> is telling you.

If you only have two minutes, read the [Leaderboard at a glance](#leaderboard-at-a-glance)
section. If you want to reproduce the numbers, read
[How to reproduce](#how-to-reproduce). Everything else is the long
form.

---

## What we measure, and why both halves separately

PriorArt has two distinct things it can do wrong, and they need to
be scored independently. A high score on one does not imply a high
score on the other.

1. **Retrieval quality** — given a new idea, does the corpus + embedding pipeline return the right similar launches at the right ranks?
2. **Comparison quality** — given the top-K retrieved launches, does the LLM produce a useful structured comparison?

Phase 1 ships the eval harness for retrieval only. Comparison quality
gets a separate harness in Phase 2 (LLM-as-judge against a hand-built
reference comparison per benchmark record). For Phase 1 the LLM call
exists in `/ideas/analyze` but is not scored.

The retrieval eval is what the README's leaderboard screenshot shows.

---

## The benchmark: `evals/labeled_v100.jsonl`

100 hand-labeled triples, balanced three ways:

| Category                              | Count | `is_duplicate` | Role |
|---------------------------------------|------:|:--------------:|------|
| `duplicate`                           |    40 | `true`         | Positive set — the system should retrieve the anchor at rank 1. |
| `novel`                               |    30 | `false`        | Negative set — the system should *not* claim a duplicate. The most important category. |
| `adversarial_paraphrase`              |    10 | `false`        | "Uber for X" where X is novel. Low confidence is the right answer. |
| `adversarial_market_overlap`          |    10 | `false`        | Adjacent market, not a direct competitor. |
| `adversarial_same_tech_diff_domain`   |    10 | `false`        | Same tech, different vertical. |
| **Total**                             | **100** |             | |

The 30 `adversarial_*` records are intentionally `is_duplicate=false`.
The correct system behavior on these is **low confidence + human
review**, not "duplicate!" — see the [labeling rules](#labeling-rules).

### Construction method

**Anchors (40 records).** Twenty confirmed YC companies × two
hand-written paraphrasings each. The 20 anchors were verified by
hitting the live API on `localhost:18001` with the YC description and
confirming the canonical company returns at rank 1 with cosine
similarity ≥ 0.80. The full anchor list is in
`scripts/build_labeled_v100.py` as the `ANCHORS` tuple.

**Novel (30 records).** Long-tail ideas with no plausible YC match —
Persian poetry composition, vintage typewriter rentals, public-fountain
rankers, etc. Picked because they would be the *embarrassing* wins if
the system cried wolf on them. This category exists to drive FPR
down, not MRR up.

**Adversarial (30 records).** 10 in each of the three
`adversarial_*` categories. These are the records where the right
answer is "I'm not sure, look closer," not "this is a duplicate" and
not "this is novel."

### Record schema

```json
{
  "id": "ev-001",
  "idea": "AI-powered contract review for SMB law firms",
  "source": "yc",
  "category": "duplicate",
  "expected_top_ids": [123, 456],
  "is_duplicate": true,
  "labeler": "anurag",
  "labeled_at": "2026-06-09T14:00:00Z",
  "notes": "Paraphrasing of Ironclad (YC S18)"
}
```

`expected_top_ids` is forward-compatible — for the Phase 3 expansion to
300 records, some records will carry multiple `expected_top_ids` for
cases where more than one company is a valid match.

### Labeling rules

- **Hand-labeling only. No LLM labeling, ever.** If you ever catch
  yourself reaching for an LLM to "pre-label" the set, stop. The eval
  set is the artifact. If the LLM pre-labels and you correct, the
  corrections are biased toward what the LLM already produced.
- A record is `is_duplicate=true` if a YC partner, hearing this idea
  in a 10-minute office-hours slot, would say "this already exists"
  with at least one specific reference.
- A record is `is_duplicate=false` if the closest YC + Product Hunt
  + HN launches are *adjacent* (same market, different angle) but not
  direct competitors.
- A record is `category=adversarial_*` if the system should be
  uncertain — the right answer is "low confidence, not a duplicate,
  but worth a closer look."
- All labels committed with `labeler=anurag` and a `labeled_at`
  timestamp. If you relabel a record, bump `labeled_at` and add a
  `notes` line explaining the change. **Never delete a record.**
  History matters.

The full policy lives in [`evals/labeled_v100.README.md`](../evals/labeled_v100.README.md).

---

## Metrics

All metrics are computed over the 100-record benchmark unless stated
otherwise. The eval runner sweeps cosine thresholds in
`[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]` and reports every metric
at every threshold.

### MRR — Mean Reciprocal Rank

Average over the benchmark of `1 / rank_of_first_relevant_result`.

- Rank 1 → MRR contribution 1.0
- Rank 5 → MRR contribution 0.2
- Never found → MRR contribution 0

**For PriorArt:** the "first relevant result" is the first company in
the ANN-search result list whose ID is in `expected_top_ids`. MRR is
computed over the duplicate and adversarial records only — the 30
novel records have empty `expected_top_ids` and contribute 0 by
construction, which would otherwise dilute the average.

### nDCG@10

Normalized Discounted Cumulative Gain at K=10. Rewards getting the
right results at the top, with logarithmic position discount.

```
DCG@K   = sum_{i=1}^{K}  (rel_i / log2(i+1))
nDCG@K  = DCG@K / IDCG@K
```

with `rel_i = 1` if the i-th result is in `expected_top_ids`, else 0.

### precision@5

Fraction of the top-5 results that are in `expected_top_ids`. Rewards
the front of the ranking specifically — useful because users see the
top-5, not the top-10.

### recall@10

Fraction of `expected_top_ids` that appear in the top-10 results.
For Phase 1 this is almost always 1.0 (each duplicate record has
exactly one `expected_top_id` and 10 ranks is plenty to find it) —
it's a sanity check, not the headline metric.

### FPR-on-novel — the headline metric

Among the records with `is_duplicate=false`, what fraction had a
top-1 result with cosine similarity **above the swept threshold**.

**Lower is better.** A high FPR-on-novel means the system is crying
wolf ("duplicate!" for genuinely novel ideas), and no real user will
trust the tool. The whole point of the eval harness is that this
number is visible.

**Phase 1 acceptance cap:** FPR-on-novel ≤ 0.15 at the production
threshold.

The current `dense_bge_m3` run on the 100-record benchmark clears
MRR (0.559 ≥ 0.50) but does not clear FPR-on-novel (0.80 at the
best-effort threshold 0.80 vs. the 0.15 cap). The Phase 2 reranker /
hybrid lever is the path to closing that gap — see
[Limitations](#limitations) below.

### Threshold sweep

A cosine threshold is the cut-off above which a top-1 hit counts as
"this looks like a duplicate." Picking the threshold is itself a
decision and there's no universal right answer — it trades recall on
duplicates against FPR on novel ideas. The eval runner reports all
five metrics at every threshold in
`[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]` and marks the threshold
that maximizes MRR subject to FPR-on-novel ≤ 0.15.

If no threshold clears the FPR cap, the leaderboard picks the
threshold that minimizes FPR subject to MRR ≥ 0.5 (Phase 1 acceptance
floor) and flags the FPR-cap miss in the [run report](#run-report).

### Selected threshold

The runner picks one threshold per run and writes it as the
`selected_threshold=True` row in `results/leaderboard.csv`. All other
rows are `selected_threshold=False`.

---

## Leaderboard at a glance

The current Phase 1 numbers on `labeled_v100.jsonl` (snapshot at
`docs/assets/leaderboard-v1.png`, exact values in
[`results/leaderboard.csv`](../results/leaderboard.csv)):

| threshold | MRR    | nDCG@10 | precision@5 | recall@10 | FPR-on-novel | selected |
|----------:|-------:|--------:|------------:|----------:|-------------:|:--------:|
|      0.50 |  0.559 |   0.611 |       0.135 |     0.800 |        1.000 |          |
|      0.55 |  0.559 |   0.611 |       0.135 |     0.800 |        1.000 |          |
|      0.60 |  0.559 |   0.611 |       0.135 |     0.800 |        1.000 |          |
|      0.65 |  0.559 |   0.611 |       0.135 |     0.800 |        1.000 |          |
|      0.70 |  0.559 |   0.611 |       0.135 |     0.800 |        1.000 |          |
|      0.75 |  0.559 |   0.611 |       0.135 |     0.800 |        1.000 |          |
|  **0.80** |  **0.559** | **0.611** | **0.135** | **0.800** | **0.800** | **YES** |

Best threshold: **0.80** (MRR-max under FPR cap; FPR cap itself was
not met on this sweep — see [Limitations](#limitations)).

---

## Run report

`results/leaderboard.csv` is the durable artifact. `results/leaderboard.md`
is the human-readable summary. `results/eval.duckdb` is a queryable
DuckDB file with one row per (record, threshold) so you can ask "what
did the system return for record `ev-037` at threshold 0.7?" without
re-running the eval.

The CSV is appended (not replaced) on every `make eval` so the
leaderboard has history. The DuckDB is overwritten — it's the
latest-run view, not the historical one.

---

## How to reproduce

```bash
git clone <repo-url> priorart && cd priorart
uv sync
docker compose up -d              # postgres + pgvector
make eval                         # runs the eval against the live API
```

Expected output: a fresh `results/leaderboard.csv` with the same
numbers to ±0.001 MRR (the eval is deterministic — same config, same
benchmark, same embedding model).

To add a new retrieval config, see
[OPERATIONS.md § Adding a retrieval config](OPERATIONS.md#adding-a-retrieval-config).

---

## Limitations

Be honest about what this harness does and doesn't measure.

1. **One embedding model in Phase 1.** Phase 2 will add `bm25`,
   `hybrid_rrf`, and `cohere_rerank` configs as siblings of
   `configs/dense_bge_m3.yaml`. The leaderboard in this doc is the
   dense-bge-m3 row only.
2. **No LLM comparison eval in Phase 1.** The structured LLM call in
   `/ideas/analyze` is unevaluated — there's no labeled reference
   comparison to score against. Phase 2 ships the LLM-as-judge harness.
3. **FPR-on-novel cap not met.** Dense `bge-m3` on a 5,990-company
   corpus has only a 0.08 gap between duplicate (avg top-1 0.888)
   and novel (avg top-1 0.809). No cosine threshold on
   `[0.50, 0.80]` clears the FPR ≤ 0.15 cap. The Phase 2 reranker /
   hybrid lever is the path to closing that gap.
4. **100 records is small.** A 95% CI on MRR at p ≈ 0.7 over 100
   records is roughly ±0.05. Per-category breakdowns on the 100-record
   set are not statistically meaningful — that's what the Phase 3 300-
   record expansion is for. Don't quote per-category numbers from
   Phase 1; quote the headline MRR + FPR-on-novel only.
5. **Public corpus only.** Internal accelerator tooling (YC, a16z,
   Antler, Techstars) sees the real production version of this
   problem. PriorArt sees the public slice. The eval harness
   measures PriorArt; it doesn't measure the accelerator's version.

---

## Where this goes next

Phase 2 widens the leaderboard:

- New retrieval configs (`bm25`, `hybrid_rrf`, `cohere_rerank`). Each
  one is a new row.
- LLM-comparison harness, graded against a hand-built reference
  comparison per benchmark record.
- Langfuse trace → eval-result linkage so a regression in the
  leaderboard points back to the prompt or model change that caused it.

Phase 3 widens the benchmark:

- 300 records instead of 100 (per-category breakdowns become
  statistically meaningful).
- GitHub Actions regression on every config change — leaderboard
  diff posted as a PR comment.
- Calibration curve on the dense-bge-m3 sweep, not just the per-threshold
  scalar numbers.