# Failure breakdown — `dense_bge_m3` on `labeled_v300.jsonl`

Per-business-category metrics at threshold ``0.65``. Eval set: ``labeled_v300.jsonl`` (LLM-generated v2, hand-review pending). Business categories: deterministic rule-based v1 (`deterministic-rule-based-v1-pending-anurag-hand-review`). Both provenance fields are LLM-/rule-assigned; the hand-label pass is a follow-up.

Cells with `n_records < 5` are flagged as **n too small** — the metric values are honest but not statistically meaningful.

| category | n_records | MRR | nDCG@10 | FPR-on-novel | top-3 failures |
|---|---|---|---|---|---|
| B2B SaaS | 11 | 0.222 | 0.222 | 1.000 | • `ev-261` (0.833) Shopify for B2B wholesale order management at craft breweries<br>• `ev-084` (0.847) Fleet management software for autonomous sidewalk delivery robots on hospital c…<br>• `ev-001` (0.844) Spreadsheet copilot that writes formulas and pulls data into your sheets on dem… |
| Consumer | 18 | 0.500 | 0.500 | 1.000 | • `ev-224` (0.763) Subscription box for rare heirloom Andean tuber varieties from Peruvian smallho…<br>• `ev-221` (0.770) Service that connects retired Tibetan monastery woodcarvers with modern designe…<br>• `ev-178` (0.775) Subscription box for artisanal goat cheese from the Pyrenees |
| Devtools | 15 | 0.800 | 0.800 | 1.000 | • `ev-232` (0.812) Uber for on-demand horse logging services in the Carpathians<br>• `ev-091` (0.823) Version-control platform for biotech wet-lab protocols, with branching for fail…<br>• `ev-092` (0.845) Version-control platform for screenplays, with side-by-side diff and alternate-… |
| Marketplace | 46 | 0.000 | 0.000 | 1.000 | • `ev-064` (0.762) Directory of Wizard-of-Oz-themed escape rooms with difficulty ratings and group…<br>• `ev-195` (0.797) Booking platform for hot-air balloon rides over the Cappadocia fairy chimneys a…<br>• `ev-203` (0.797) Booking platform for traditional Korean hanok village overnight stays |
| Fintech | 23 | 0.200 | 0.200 | 1.000 | • `ev-067` (0.817) Subscription lending library of adaptive toys for children with disabilities<br>• `ev-260` (0.825) Stripe for split-tender payments at restaurants with multiple concepts<br>• `ev-252` (0.830) Stripe for marketplace seller payouts with embedded KYC |
| Healthcare | 9 | 0.000 | 0.000 | 1.000 | • `ev-096` (0.812) Topic-organized group chat for hospital clinical teams with HIPAA-grade audit t…<br>• `ev-266` (0.817) Zoom for telehealth group therapy sessions with billing<br>• `ev-262` (0.826) Calendly for veterinary clinic appointment scheduling |
| Education | 6 | 0.000 | 0.000 | 1.000 | • `ev-173` (0.808) Online course teaching classical Ottoman Turkish calligraphy<br>• `ev-060` (0.810) Citizen-science app for reporting bird-window collisions to help architects des…<br>• `ev-098` (0.871) Cohort-based online school for working professionals with live workshops and pr… |
| Other | 172 | 0.652 | 0.665 | 1.000 | • `ev-248` (0.747) Taskrabbit for one-off hand-forged Damascus steel knife commissions<br>• `ev-227` (0.761) Tool for ranking the best spots to hear Aurora Borealis in northern Norway<br>• `ev-201` (0.766) Subscription for hand-tied fishing flies tied by retired Scottish ghillies |

**Honest call-out:** categories with low `n_records` (e.g. healthcare=8, education=6) do not have enough signal to draw a confident per-config conclusion. The rule-based v1 classifier assigns 173/300 records to `other`; the per-category picture is more nuanced after Anurag's hand-label pass.
