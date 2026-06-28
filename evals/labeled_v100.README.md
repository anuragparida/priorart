# `evals/labeled_v100.jsonl` — Phase 1 labeled benchmark

> **The eval set is the artifact.** It is committed alongside the code so anyone can reproduce the leaderboard numbers by running `make eval`. If you change the labels, document the change here. If you change the metric definitions, document it in `docs/EVAL.md`. The two are the source of truth.

## Why 100 ideas (not 300)

The Phase 1 plan in `docs/PHASE-1.md` §1.5 calls for a hand-labeled 100-idea benchmark; the 300-idea version in `docs/EVAL.md` is the Phase 3 target (300 is large enough to claim statistical signal on per-category breakdowns, but takes ~9 hours of focused labeling). For Phase 1 we ship 100 because:

- 100 is enough to claim MRR ≥ 0.5 is a real signal, not noise — the 95% CI on a 100-record MRR estimate at p ≈ 0.7 is roughly ±0.05.
- Phase 1's acceptance test is a single number (MRR ≥ 0.5) plus the threshold sweep. 100 records give a stable-enough per-threshold comparison.
- Hand-labeling 100 records takes ~2 hours, which fits inside the Phase 1 weekend budget. The 200 more for Phase 3 can be added later without breaking the existing 100.

The label schema is forward-compatible with the 300-record extension — a record's fields don't change. Phase 3 just adds more `novel` and `adversarial_*` records.

## Composition: 40 / 30 / 30

100 records, balanced three ways:

| Category                          | Count | `is_duplicate` | Why                                                                                              |
|-----------------------------------|------:|:--------------:|--------------------------------------------------------------------------------------------------|
| `duplicate`                       |    40 | `true`         | 20 confirmed YC anchors × 2 hand-written paraphrasings.                                          |
| `novel`                           |    30 | `false`        | Long-tail ideas with no plausible YC match. The single most important category — drives FPR.    |
| `adversarial_paraphrase`          |    10 | `false`        | "Uber for X" where X is novel. The system should return low confidence, not call it a duplicate.|
| `adversarial_market_overlap`      |    10 | `false`        | Adjacent market, not a direct competitor (Stripe for crypto, etc.).                              |
| `adversarial_same_tech_diff_domain` |   10 | `false`        | Same tech stack, different vertical (GitHub for legal contracts, etc.).                          |
| **Total**                         | **100** |              |                                                                                                  |

The 30 adversarial records are intentionally `is_duplicate=false` — the *correct* system behaviour on these is to return low confidence and let a human decide, not to scream "duplicate!" and waste the user's time.

## Construction method

### Anchors

The 20 anchors are confirmed top-1 direct-description matches against the live priorart API on `localhost:18001` (port 18000 is squatted by the local clausecraft stack — see `t_fcc690b4` completion notes). The anchor list is in `scripts/build_labeled_v100.py` as the `ANCHORS` tuple — each entry is `(id, name, yc_description)`. All 20 have been verified by POSTing the YC description to `/search` and checking that the canonical company returns at rank 1 with cosine similarity ≥ 0.80.

The 30 novel and 30 adversarial records are hand-curated by the project owner (Anurag) and committed on first authoring.

### Paraphrases

Each anchor gets two paraphrases — short, founder-pitch-style ideas that capture the *essence* of the anchor's product without using its name. The two paraphrases intentionally use different phrasings so the system has to do real semantic matching (not keyword matching). Example for `Draftwise` (YC id 1660):

```
- "AI tool that helps lawyers draft and redline contracts faster"
- "Negotiation copilot for legal teams working on commercial agreements"
```

Both should retrieve Draftwise in the top-1 with high similarity. A system that only matches on exact keywords will miss both — that's the point.

### Anti-patterns avoided

- **No LLM-generated labels.** Every paraphrase is hand-written by Anurag. An LLM that generates labels will rationalise them to match the system's outputs (or vice versa) and the benchmark becomes meaningless.
- **No composite scores.** Every record carries a single `is_duplicate` boolean; the metrics are computed per-record and aggregated by the runner. There is no human-graded "quality" column.
- **No opaque thresholds.** The cosine threshold is recorded in the leaderboard CSV (`threshold` column) and the full sweep is in the runner's output. A reader can pick any threshold on the sweep and reproduce the FPR/MRR.
- **No closed-corpus benchmarks.** The YC public directory is the corpus; the snapshot is committed at `data/snapshots/yc_<date>.jsonl`. Re-running the eval against a different snapshot is allowed (and expected for the regression suite in Phase 1.11).

## Record schema

One JSON object per line. See `src/eval/benchmark.py::BenchmarkRecord` for the typed schema; see `docs/EVAL.md` "Record schema" for the field-level documentation. The full set of required fields:

```json
{
  "id": "ev-001",
  "idea": "AI-powered contract review for SMB law firms",
  "source": "yc",
  "category": "duplicate",
  "expected_top_ids": [1660],
  "is_duplicate": true,
  "labeler": "anurag",
  "labeled_at": "2026-06-28T12:30:00Z",
  "notes": "Hand-written paraphrase of Draftwise (YC id=1660)"
}
```

`source` is `yc` for the paraphrase records (the source of the anchor) and `synthetic` for the novel / adversarial records. `notes` is free-form and is the place to capture labeler reasoning.

## Re-running / re-labeling

If a record is re-labeled (e.g. you realise a "novel" was actually a paraphrase of a real company), bump the `labeled_at` timestamp and add a `notes` line explaining the change. **Never delete a record.** History matters — the eval harness reads every line in the file and aggregates over all of them.

To regenerate `evals/labeled_v100.jsonl` from the source-of-truth Python data structures:

```
uv run python scripts/build_labeled_v100.py
```

The script is the canonical encoding. The JSONL is the serialised artifact.

## Change log

- **2026-06-28** — Initial 100 records authored (`labeled_at=2026-06-28T12:30:00Z`). 20 confirmed anchors × 2 paraphrases + 30 novel + 30 adversarial. Labels committed by Anurag.