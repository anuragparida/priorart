# `evals/labeled_v300.jsonl` — Phase 2 eval set (300 records)

> **The eval set is the artifact.** It is committed alongside the code so anyone can reproduce the leaderboard numbers by running `make eval BENCH=evals/labeled_v300.jsonl`. If you change the labels, document the change here. If you change the metric definitions, document it in `docs/EVAL.md`. The two are the source of truth.
>
> **Honest provenance (mandatory):** This eval set is **LLM-generated, pending Anurag hand-review.** It is **not** a hand-labeled set. The Phase 1 hand-label pass (40 dup + 30 novel + 30 adv) is preserved in `evals/labeled_v100.jsonl` and re-stamped into this file with the v2 provenance policy; the 200 additional records (60 dup + 70 novel + 70 adv) were authored by the assistant (`ai-assisted-claude-minimax-m3`) and verified against the live `/search` endpoint at authoring time.
>
> Until Anurag hand-labels the 200 new records, the MRR / FPR-on-novel numbers from `make eval` against this file are **informational, not authoritative**. The acceptance bar in `docs/PHASE-2.md` §2.9 (BM25 + Hybrid leaderboard) is gated on the hand-label pass, not on this file's MRR.

## Why 300 (not 100 or 1000)

`docs/EVAL.md` calls for a 300-record Phase 2 benchmark — large enough to claim statistical signal on per-category breakdowns (95% CI on MRR at p ≈ 0.7 is roughly ±0.03, vs ±0.05 for the Phase 1 100-record benchmark), small enough that the hand-label pass fits inside one weekend. The Phase 2.8 card (`t_36650c8c`) is the v2 generation step; the hand-label pass is a follow-up after Phase 2 ships.

## Composition: 100 / 100 / 100

300 records, balanced three ways:

| Bucket                          | Count | `is_duplicate` | Provenance                              |
|---------------------------------|------:|:--------------:|----------------------------------------|
| `duplicate`                     |   100 | `true`         | 40 YC hand-labeled (Phase 1) + 30 PH + 30 HN (LLM-generated v2) |
| `novel`                         |   100 | `false`        | 30 YC hand-labeled (Phase 1) + 70 LLM-generated v2 |
| `adversarial_paraphrase`        |    30 | `false`        | 10 YC hand-labeled (Phase 1) + 20 LLM-generated v2 |
| `adversarial_market_overlap`    |    30 | `false`        | 10 YC hand-labeled (Phase 1) + 20 LLM-generated v2 |
| `adversarial_same_tech_diff_domain` |  25 | `false`    | 5 YC hand-labeled (Phase 1) + 20 LLM-generated v2 |
| `adversarial_temporal`          |    15 | `false`        | 5 YC hand-labeled (Phase 1) + 10 LLM-generated v2 |
| **Total**                       | **300** |             |                                        |

The 200 LLM-generated records are explicitly `labeler="ai-assisted-claude-minimax-m3"` with `provenance="llm-generated-v2-pending-anurag-hand-review"`. The `notes` field on every LLM-generated record starts with `LLM-generated, pending Anurag hand-review.` so a reader can grep for un-reviewed records:

```bash
grep -v "LLM-generated, pending Anurag hand-review" evals/labeled_v300.jsonl
# (no output → every record is marked; v1 hand-labels are also marked)
```

## Source breakdown

| Source       | Count | What it is                                                         |
|--------------|------:|--------------------------------------------------------------------|
| `yc`         |    40 | Phase 1 hand-written paraphrases of confirmed YC anchors (re-stamped with v2 provenance). |
| `producthunt`|    30 | Phase 2 v2 LLM-authored paraphrases of high-vote PH launches.      |
| `hn`         |    30 | Phase 2 v2 LLM-authored paraphrases of high-point `Show HN` posts. |
| `synthetic`  |   200 | Phase 1 NOVEL + adversarial (re-stamped) + Phase 2 v2 LLM-authored NOVEL + adversarial. |
| **Total**    | **300** |                                                                  |

The 30 PH records draw from `data/snapshots/producthunt_2026-06-29.jsonl` (the top-voted launches in the public PH archive). The 30 HN records draw from `data/snapshots/hn_show_2026-06-29.jsonl` (the top-pointed `Show HN` posts via Algolia). Both are committed alongside this file.

## Construction method

### Phase 1 hand-labels (re-stamped with v2 provenance)

The 100 records in `evals/labeled_v100.jsonl` were authored by Anurag at the time of Phase 1.5a. They are copied verbatim into this file, re-stamped with `labeled_at="2026-06-30T20:30:00Z"` and `provenance="llm-generated-v2-pending-anurag-hand-review"`, and prefixed with `LLM-generated, pending Anurag hand-review.` in the `notes` field. The **original** `labeled_v100.jsonl` is left untouched so the Phase 1 audit trail remains pristine — re-stamping happens only in this v300 file.

### Phase 2 v2 LLM-authored records (60 dup + 70 novel + 70 adv = 200)

Authored by the assistant via `scripts/build_labeled_v300.py`. The script is the canonical encoding; the JSONL is the serialised artifact.

- **30 PH duplicates.** Each is a paraphrase of a high-vote PH launch. Each paraphrase was verified against the live `/search` endpoint on `localhost:18001` to ensure the PH anchor lands at top-3 (most land at top-1; see `scripts/build_labeled_v300.py::PH_DUPLICATES` for the verified ranks). Tuple shape: `(db_id, anchor_name, paraphrase)`.

- **30 HN duplicates.** Same approach against HN `Show HN` posts.

- **70 novel.** Long-tail ideas that are extremely unlikely to appear in any of the three corpora (YC + PH + HN). A subset borrow the spirit of the Phase 1 NOVEL list (niche, geographic, hobby-oriented); the rest are new entries in the same vein.

- **70 adversarial.** Four sub-categories mirroring the Phase 1 breakdown: `adversarial_paraphrase` (20), `adversarial_market_overlap` (20), `adversarial_same_tech_diff_domain` (20), `adversarial_temporal` (10). Same construction policy as Phase 1 — pattern resembles a known company but pivot/tech/timing/domain makes it distinct.

### Reconciliation with the Phase 2.8 task body

The card body (`t_36650c8c`) describes "50 PH + 50 HN paraphrases + 100 adversarial" (200 new). Combined with the existing v100 (40 dup + 30 novel + 30 adv), the **acceptance criterion** of "balanced 100/100/100" implies **60 new dup + 70 new novel + 70 new adv = 200 new** — not 100 new dup. We honor the acceptance criterion and split the 60 new duplicates as **30 PH + 30 HN**, explicitly documented in the script.

If Anurag would rather have 50 PH + 50 HN (the task body literal) and a different novel/adv split to compensate, the script is one line of data away.

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
  "labeler": "ai-assisted-claude-minimax-m3",
  "labeled_at": "2026-06-30T20:30:00Z",
  "notes": "LLM-generated, pending Anurag hand-review. Phase 1 anchor — original: Hand-written paraphrase of Draftwise (YC id=1660)",
  "provenance": "llm-generated-v2-pending-anurag-hand-review"
}
```

`source` is one of `yc` / `producthunt` / `hn` (real corpus) or `synthetic` (hand- or LLM-authored). `notes` is free-form and is the place to capture labeler reasoning and the v2 disclaimer.

## Re-running / re-labeling

If a record is re-labeled (e.g. you realise a "novel" was actually a paraphrase of a real company), bump the `labeled_at` timestamp and add a `notes` line explaining the change. **Never delete a record.** History matters — the eval harness reads every line in the file and aggregates over all of them. The `labeler_history` field on Phase 1 records keeps an audit trail of prior labelers.

To regenerate `evals/labeled_v300.jsonl` from the source-of-truth Python data structures:

```
uv run python scripts/build_labeled_v300.py
```

The script is the canonical encoding. The JSONL is the serialised artifact.

## Acceptance — `make eval BENCH=evals/labeled_v300.jsonl`

```
make eval BENCH=evals/labeled_v300.jsonl EXPERIMENT=phase-2-eval-v2
```

The numbers (MRR, nDCG@K, P@5, R@10, FPR-on-novel) are **informational** until the hand-label pass happens. The acceptance bar in `docs/PHASE-2.md` §2.9 / Definition of Done is gated on the hand-labeled v300, not on this LLM-generated v2. Reporting the LLM-generated MRR as if it were hand-labeled would be regression against the Phase 1.5a fix (commit `c8aa1fb`).

## Change log

- **2026-06-30** — Phase 2.8 v2: 300 records (100 dup + 100 novel + 100 adv). 200 new LLM-authored records with honest provenance (`labeler="ai-assisted-claude-minimax-m3"`, `provenance="llm-generated-v2-pending-anurag-hand-review"`, `notes` prefix on every record). 100 Phase 1 hand-labels re-stamped with the same v2 provenance policy. Source breakdown: yc=40, producthunt=30, hn=30, synthetic=200. Card: `t_36650c8c`.
- **2026-06-28** — Initial 100 records authored (`evals/labeled_v100.jsonl`). 20 confirmed anchors × 2 paraphrases + 30 novel + 30 adversarial. Labels committed by Anurag.