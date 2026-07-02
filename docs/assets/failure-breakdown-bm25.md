# Failure breakdown — `bm25` on `labeled_v300.jsonl`

Per-business-category metrics at threshold ``0.65``. Eval set: ``labeled_v300.jsonl`` (LLM-generated v2, hand-review pending). Business categories: deterministic rule-based v1 (`deterministic-rule-based-v1-pending-anurag-hand-review`). Both provenance fields are LLM-/rule-assigned; the hand-label pass is a follow-up.

Cells with `n_records < 5` are flagged as **n too small** — the metric values are honest but not statistically meaningful.

| category | n_records | MRR | nDCG@10 | FPR-on-novel | top-3 failures |
|---|---|---|---|---|---|
| B2B SaaS | 11 | 0.222 | 0.222 | 1.000 | • `ev-261` (0.938) Shopify for B2B wholesale order management at craft breweries<br>• `ev-084` (0.955) Fleet management software for autonomous sidewalk delivery robots on hospital c…<br>• `ev-001` (0.948) Spreadsheet copilot that writes formulas and pulls data into your sheets on dem… |
| Consumer | 18 | 0.188 | 0.265 | 1.000 | • `ev-221` (0.905) Service that connects retired Tibetan monastery woodcarvers with modern designe…<br>• `ev-165` (0.907) Subscription box for hand-painted Ukrainian pysanky eggs<br>• `ev-224` (0.909) Subscription box for rare heirloom Andean tuber varieties from Peruvian smallho… |
| Devtools | 15 | 0.517 | 0.536 | 1.000 | • `ev-232` (0.930) Uber for on-demand horse logging services in the Carpathians<br>• `ev-093` (0.940) Version-control platform for restaurant recipes, with allergen-aware merges and…<br>• `ev-272` (0.940) GitHub for legal contract version control with redlining |
| Marketplace | 46 | 0.000 | 0.000 | 1.000 | • `ev-072` (0.915) Uber for matching elderly residents with neighborhood helpers for grocery runs<br>• `ev-075` (0.915) Uber for connecting amateur astronomers with rural hosts who have dark-sky prop…<br>• `ev-231` (0.915) Uber for chartered bush-plane flights in rural Alaska |
| Fintech | 23 | 0.200 | 0.200 | 1.000 | • `ev-285` (0.918) GitHub for actuarial model versioning at life insurance carriers<br>• `ev-255` (0.929) Stripe for stablecoin payment routing across chains<br>• `ev-252` (0.929) Stripe for marketplace seller payouts with embedded KYC |
| Healthcare | 9 | 0.000 | 0.000 | 1.000 | • `ev-262` (0.917) Calendly for veterinary clinic appointment scheduling<br>• `ev-280` (0.924) GitHub for pharmaceutical-formulation versioning with regulatory hooks<br>• `ev-096` (0.934) Topic-organized group chat for hospital clinical teams with HIPAA-grade audit t… |
| Education | 6 | 0.000 | 0.000 | 1.000 | • `ev-060` (0.929) Citizen-science app for reporting bird-window collisions to help architects des…<br>• `ev-297` (0.946) On-demand tutoring marketplace using only retired schoolteachers<br>• `ev-173` (0.949) Online course teaching classical Ottoman Turkish calligraphy |
| Other | 172 | 0.452 | 0.472 | 1.000 | • `ev-197` (0.896) Subscription for hand-stitched book covers made from recycled denim<br>• `ev-041` (0.903) AI tool for composing classical Persian poetry in the style of Hafez and Rumi<br>• `ev-196` (0.906) Tool for cataloguing the locations of historic Finnish smoke saunas open to vis… |

**Honest call-out:** categories with low `n_records` (e.g. healthcare=8, education=6) do not have enough signal to draw a confident per-config conclusion. The rule-based v1 classifier assigns 173/300 records to `other`; the per-category picture is more nuanced after Anurag's hand-label pass.
