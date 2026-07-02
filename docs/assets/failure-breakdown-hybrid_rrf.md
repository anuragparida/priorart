# Failure breakdown — `hybrid_rrf` on `labeled_v300.jsonl`

Per-business-category metrics at threshold ``0.65``. Eval set: ``labeled_v300.jsonl`` (LLM-generated v2, hand-review pending). Business categories: deterministic rule-based v1 (`deterministic-rule-based-v1-pending-anurag-hand-review`). Both provenance fields are LLM-/rule-assigned; the hand-label pass is a follow-up.

Cells with `n_records < 5` are flagged as **n too small** — the metric values are honest but not statistically meaningful.

| category | n_records | MRR | nDCG@10 | FPR-on-novel | top-3 failures |
|---|---|---|---|---|---|
| B2B SaaS | 11 | 0.222 | 0.222 | 1.000 | • `ev-084` (0.826) Fleet management software for autonomous sidewalk delivery robots on hospital c…<br>• `ev-261` (0.828) Shopify for B2B wholesale order management at craft breweries<br>• `ev-001` (0.814) Spreadsheet copilot that writes formulas and pulls data into your sheets on dem… |
| Consumer | 18 | 0.500 | 0.500 | 1.000 | • `ev-224` (0.757) Subscription box for rare heirloom Andean tuber varieties from Peruvian smallho…<br>• `ev-178` (0.758) Subscription box for artisanal goat cheese from the Pyrenees<br>• `ev-204` (0.762) Subscription box for cold-pressed camelina oil from Nordic smallholders |
| Devtools | 15 | 0.570 | 0.602 | 1.000 | • `ev-232` (0.812) Uber for on-demand horse logging services in the Carpathians<br>• `ev-093` (0.816) Version-control platform for restaurant recipes, with allergen-aware merges and…<br>• `ev-091` (0.820) Version-control platform for biotech wet-lab protocols, with branching for fail… |
| Marketplace | 46 | 0.000 | 0.000 | 1.000 | • `ev-064` (0.762) Directory of Wizard-of-Oz-themed escape rooms with difficulty ratings and group…<br>• `ev-226` (0.767) Booking platform for kayak expeditions through remote Aleutian island kelp fore…<br>• `ev-069` (0.778) On-demand locksmith service for venues with stuck or vandalized public pianos |
| Fintech | 23 | 0.200 | 0.200 | 1.000 | • `ev-100` (0.794) Music streaming service that pays artists per-second and publishes listener dat…<br>• `ev-291` (0.794) Music streaming service that pays artists per-second and publishes listener dat…<br>• `ev-097` (0.800) Subscription rental marketplace for designer evening wear with peer-to-peer dam… |
| Healthcare | 9 | 0.000 | 0.000 | 1.000 | • `ev-096` (0.809) Topic-organized group chat for hospital clinical teams with HIPAA-grade audit t…<br>• `ev-266` (0.814) Zoom for telehealth group therapy sessions with billing<br>• `ev-280` (0.819) GitHub for pharmaceutical-formulation versioning with regulatory hooks |
| Education | 6 | 0.000 | 0.000 | 1.000 | • `ev-173` (0.789) Online course teaching classical Ottoman Turkish calligraphy<br>• `ev-060` (0.810) Citizen-science app for reporting bird-window collisions to help architects des…<br>• `ev-098` (0.871) Cohort-based online school for working professionals with live workshops and pr… |
| Other | 172 | 0.524 | 0.555 | 1.000 | • `ev-227` (0.737) Tool for ranking the best spots to hear Aurora Borealis in northern Norway<br>• `ev-248` (0.741) Taskrabbit for one-off hand-forged Damascus steel knife commissions<br>• `ev-244` (0.746) Taskrabbit for restoring vintage Polaroid SX-70 cameras |

**Honest call-out:** categories with low `n_records` (e.g. healthcare=8, education=6) do not have enough signal to draw a confident per-config conclusion. The rule-based v1 classifier assigns 173/300 records to `other`; the per-category picture is more nuanced after Anurag's hand-label pass.
