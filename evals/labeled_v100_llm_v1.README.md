# `labeled_v100_llm_v1.jsonl` — AI-assisted v1 of the priorart eval set

> **This is AI-assisted v1, not the final hand-labeled set.**
> It exists so the eval harness (Phase 1.6) and the downstream Phase 1 chain
> can ship while Anurag hand-corrects in batches. The protocol below makes
> that correction tractable without breaking the eval.

---

## Why AI-labeling was used

The original Phase 1.5 card (`t_5860410c`) required Anurag to hand-label 100
triples. It has been blocked for **20+ days** with no progress, which gates
Phase 1.6 (eval harness) and every downstream card. Per Anurag's explicit
instruction (2026-06-28):

> "do it with AI, it's fine. I'm not gonna touch it... just do it and get the
> tickets moving."

This file is the AI-actionable replacement. When Anurag finishes hand-correction,
rename the file from `labeled_v100_llm_v1.jsonl` to `labeled_v100.jsonl` — the eval
harness (Phase 1.6) should read whichever exists.

---

## Composition (40 / 30 / 30)

| Category | Count | Source | Rationale |
|---|---|---|---|
| `duplicate` | 40 | 20 active YC companies × 2 paraphrasings | Positive set. Drives MRR / nDCG / precision / recall. |
| `novel` | 30 | Hand-picked long-tail, 29 distinct domains | Negative set. Drives FPR-on-novel — the most important metric. |
| `adversarial_paraphrase` | 10 | "Uber for X" with novel X | Tests whether the system overclaims on the obvious pattern. |
| `adversarial_market_overlap` | 10 | Real market, novel tech twist | Tests whether the system conflates adjacent markets. |
| `adversarial_same_tech_diff_domain` | 5 | Version-control / canvas pattern in a new domain | Tests whether the system anchors on tech, not domain. |
| `adversarial_temporal` | 5 | 2026 rebuild of a 2014-era pattern | Tests whether the system treats stale patterns as fresh. |
| **Total** | **100** | | |

### Duplicate set construction

20 active YC companies chosen from `data/snapshots/yc_2026-06-08.jsonl`
(5949 companies, 4100 active). Selection criteria:

- `status == "Active"`
- Description length between 25 and 200 chars (rich enough to paraphrase)
- Spread across verticals: AI devtools, fintech, healthcare, food, biotech,
  manufacturing, creative, energy, recruitment, banking, IoT.

Each company has a **deterministic integer ID** = its line number in the
snapshot. `expected_top_ids` references that line number.

### Novel set construction

29 distinct domains, intentionally weird (Persian poetry, vintage typewriters,
left-handed electricians, public fountains in Rome, beehive acoustics, snail-mail
pen pals, sourdough starter exchange, etc.) — the same shape as the spec
examples. No plausible match in YC + Product Hunt + HN. Used to measure
false-positive rate on truly novel ideas.

### Adversarial set construction

The point of adversarial records is "plausible-looking duplicate at first
glance, actually distinct." Categories are weighted toward the failure modes
the system is expected to have:

- `adversarial_paraphrase` (10): the "X for Y" pattern is over-learned by LLMs.
  System should return low-confidence near-duplicate, not a clean "no match."
- `adversarial_market_overlap` (10): real market + novel tech. Should be
  flagged as adjacent, not duplicate.
- `adversarial_same_tech_diff_domain` (5): same tech (version control /
  canvas), different domain. Should not over-index on tech.
- `adversarial_temporal` (5): 2026 rebuild of a 2014-era pattern. Should
  recognize the market has moved.

---

## Record schema

```json
{
  "id": "ev-001",
  "idea": "AI-powered contract review for SMB law firms",
  "source": "yc" | "synthetic",
  "category": "duplicate" | "novel" | "adversarial_paraphrase" |
              "adversarial_market_overlap" | "adversarial_same_tech_diff_domain" |
              "adversarial_temporal",
  "expected_top_ids": [53],
  "is_duplicate": true,
  "labeler": "ai-assisted-claude-minimax-m3",
  "labeled_at": "2026-06-28T12:30:00Z",
  "notes": "LLM-generated, pending Anurag hand-review. ..."
}
```

### Critical transparency rules

Every record in this file:

1. Has `labeler = "ai-assisted-claude-minimax-m3"` (the model used to generate it).
2. Has `notes` starting with `"LLM-generated, pending Anurag hand-review."`.

This makes provenance obvious in `results/leaderboard.csv` and protects the
eval's integrity: any reviewer can immediately see which records are
AI-generated vs. Anurag-approved.

---

## Hand-correction protocol

Anurag's batch hand-correction workflow:

1. Open `evals/labeled_v100_llm_v1.jsonl` in an editor.
2. For any record he disagrees with, edit **in place**:
   - Flip `is_duplicate` if needed.
   - Fix `expected_top_ids` if a different company is the right answer.
   - Rewrite `category` if it was miscategorized.
   - Bump `labeled_at` to the current timestamp (ISO 8601, UTC).
   - Add a `human_review` block:

     ```json
     "human_review": {
       "reviewed_at": "2026-06-30T10:15:00Z",
       "verdict": "confirmed" | "corrected" | "rejected",
       "notes": "AI was right; the new Company X at line 6789 is the closest match."
     }
     ```

3. **Never delete a record.** History matters — the LLM judgment is part of
   the artifact even when Anurag corrects it.

After hand-correction is done, rename:

```bash
mv evals/labeled_v100_llm_v1.jsonl evals/labeled_v100.jsonl
```

The eval harness (Phase 1.6) should read whichever file exists.

---

## Known caveats

These records are AI-labeled and have not been hand-reviewed. Until Anurag
corrects them, expect:

- **False positives in the duplicate set.** Some paraphrasings may be too
  close to adjacent companies (not the listed one). Marked with
  `LOW-CONFIDENCE` in the `notes` field where the worker flagged them.
- **False negatives in the novel set.** A few "novel" ideas may actually
  match a real YC company that wasn't in the worker's hand-picked sample.
  Particularly vulnerable: long-tail domains that happen to overlap with
  active YC companies in adjacent verticals.
- **Adversarial records named in natural English.** The adversarial ideas
  avoid naming specific YC companies (no "Stripe for X" since Stripe is a
  real YC company), but they share English vocabulary (e.g. "atlas",
  "canvas"). These are not duplicates of the YC companies with those names —
  the verticals differ. Re-verify if you suspect a name match.

---

## Provenance

- Generated: 2026-06-28
- Generator: hermes_perseus (MiniMax-M3), as the labeling pass
- Snapshot: `data/snapshots/yc_2026-06-08.jsonl` (5949 YC companies)
- Card: `t_2d613dd4` (Phase 1.5a — AI-assisted golden eval set v1)