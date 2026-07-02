# Phase 4 — Real market-scope estimation (Weekend 4)

> **Tier:** Should-be. **Goal:** Replace the LLM-hand-waving `market_scope_signal` (Phase 1.7) with a corpus-grounded, evidence-backed signal that returns a *quantitative* layer when the data is strong enough, and an honest "weak — directional only" call when it isn't. No paid TAM/SEMrush/Crunchbase dependency. Built on the SearXNG + Firecrawl stack that's already wired for `web_fallback_if_empty`.

---

## Why this phase exists

Phase 1.7 shipped `market_scope` as a Pydantic enum (`wide_open / crowded_but_growing / saturated / niche_but_real`) populated by a single Claude call inside the structured-comparison LLM. The README's Limitations section (1st of 6 items) calls it out: **"Market-scope is a stub."** Phase 3.11 confirmed Anurag wants a real implementation in Phase 4.

The "real" shape is bounded by three operational facts:

1. **No paid TAM/SEMrush/Crunchbase/PitchBook/CB Insights keys in this environment.** SEMrush-style quantitative estimates (search volume, keyword difficulty, exact TAM numbers) require either a paid subscription or a curated dataset neither of which the user has. Any Phase 4 design that secretly depends on those is going to ship a thinner stub than Phase 3.
2. **The local 10,983-company corpus is itself a market-scope signal.** Density of similar launches, recency, business-category breakdown, and cosine-saturation are *real* numbers we already compute — they're just buried inside the retrieval path.
3. **SearXNG + Firecrawl are already wired.** Phase 2.10's `web_fallback_if_empty` activity uses them to scrape the open web for novel ideas. The same path can provide a low-volume, evidence-backed layer for the "wide_open" / "niche_but_real" directions where the local corpus is genuinely thin.

**Phase 4 = quantitative envelope grounded in our own corpus + targeted web augmentation, never a paid API, never a fabricated TAM number.** When the data is too thin to be quantitative, say so — `quantitative: null` and the rationale names why.

---

## What ships at the end of Phase 4

- [ ] `MarketScopeQuant` Pydantic model — same 4-direction envelope, plus a `quantitative: Optional[MarketScopeQuant]` field with corpus-derived counts and a low-volume web-augmented search-volume proxy.
- [ ] `market_scope_signal` activity replaced (was a pass-through in Phase 2.1) with a dedicated, Pydantic-validated computation that combines: (a) corpus density from `company_embeddings`, (b) business-category distribution from `companies.business_category`, (c) recency from `companies.launched_at`, (d) an optional SearXNG-augmented search-volume signal for sparse directions.
- [ ] `supporting_evidence` for the market-scope claim is real: a list of `{source: "corpus" | "web", url?: str, company_id?: int, snippet?: str}` entries — not LLM-invented URLs.
- [ ] Confidence surfaced in the API: `confidence: Literal["directional", "evidence_backed", "quantitative"]`. `directional` replaces the current stub behavior; `evidence_backed` means ≥1 corpus source + ≥1 web source; `quantitative` means the `MarketScopeQuant` layer is populated.
- [ ] Eval harness: a `market_scope_accuracy` metric on a 50-idea labeled benchmark where Anurag (or an LLM acting with explicit hand-review provenance) classifies the ground-truth direction. Reported alongside MRR / nDCG / ECE in the leaderboard CSV.
- [ ] README's Limitations §1 ("Market-scope is a stub") becomes honest: "Market-scope is **corpus-grounded and evidence-backed**, not a SEMrush replacement. We surface 3 confidence levels; for ideas outside our 11K-company corpus, the signal degrades to `directional`."
- [ ] CI regression: `market_scope_accuracy` regression cap hard-coded as a constant in `scripts/ci/eval_gate.py`, per the type-level-guardrail rule from Phase 3.
- [ ] The `market_scope_signal` Temporal activity is the only "additional LLM call" point in the workflow — Anthropic Claude Sonnet 4.5 is used for the direction-label synthesis over the corpus stats + web snippets, and the call is wrapped in Langfuse (per the Phase 2.3 pattern).

**Definition of done:** The frontend shows the new confidence badge. The leaderboard has `market_scope_accuracy` alongside MRR / nDCG / ECE. `make eval` runs the new metric. The README's first Limitations bullet reflects the new shape honestly. CI fails if `market_scope_accuracy` regresses below the hard-coded floor.

---

## Detailed task breakdown

Each task is sized to ~2–3 hours. Total Phase 4 budget: ~14 hours — narrower than Phase 3 because the platform is already in place.

### 4.1 — `MarketScopeQuant` Pydantic model + envelope extension (1.5h)

- [ ] In `src/llm/schemas.py`, add:
  - `MarketScopeQuant(BaseModel)` with fields: `competitor_count: int`, `recent_3y_count: int`, `category_distribution: dict[str, int]`, `search_volume_proxy: Optional[int] = None` (the SearXNG-derived proxy; null when the corpus is dense enough to not need it), `saturation_index: float = Field(ge=0, le=1)`, `growth_rate: Optional[float] = None`.
  - Extend `MarketScope` envelope to carry: `quantitative: Optional[MarketScopeQuant]`, `confidence: Literal["directional", "evidence_backed", "quantitative"]`, `evidence: list[MarketScopeEvidence]` (where `MarketScopeEvidence` has `source: Literal["corpus", "web"]`, `url: Optional[str]`, `company_id: Optional[int]`, `snippet: Optional[str]`, `as_of: datetime`).
- [ ] Update `IdeaVerdict.market_scope` to the new envelope. **Backwards-compat:** keep `market_scope: MarketScope` (the old enum) accessible via `market_scope.direction` so the existing frontend `marketScope.ts` doesn't break — the frontend will just learn the new `confidence` badge.
- [ ] **Verify:** `pytest` passes; the OpenAPI schema for `/ideas/analyze` regenerates with the new fields; the existing 1.7 / 1.8 / 1.11 test fixtures still pass (they reference `market_scope` as a string — read both the string and the structured object).

### 4.2 — `market_scope_signal` activity: corpus density computation (2.5h)

- [ ] Replace the Phase 2.1 pass-through activity in `src/workflow/activities.py` with a real implementation that takes the `top_k_payload` (already returned by `ann_search`) and queries `companies` for the broader neighborhood (top 200 by cosine, the corpus-relative region the user is asking about).
- [ ] Compute, in one SQL round-trip per metric:
  - `competitor_count` = COUNT(*) of `companies` in the neighborhood.
  - `recent_3y_count` = COUNT(*) WHERE `launched_at > now() - interval '3 years'`.
  - `category_distribution` = histogram of `business_category` over the neighborhood, top 8 categories + `other` bucket.
  - `saturation_index` = `competitor_count / 200` clamped to [0, 1]. (200 is the "dense" cap; ≤200 means uncapped, >200 = saturated.)
  - `growth_rate` = `recent_3y_count / max(competitor_count, 1)` — fraction of launches in the last 3 years.
- [ ] Direction rules (deterministic, not LLM):
  - `saturated` if `competitor_count ≥ 100` AND `growth_rate < 0.25`.
  - `crowded_but_growing` if `competitor_count ≥ 50` AND `growth_rate ≥ 0.25`.
  - `wide_open` if `competitor_count < 10` AND the neighborhood is empty enough that a SearXNG call would be useful (skip if no web fallback available).
  - `niche_but_real` if `10 ≤ competitor_count < 50`.
  - Else: fall through to the LLM synthesis step.
- [ ] When the deterministic rules fire, set `confidence="quantitative"`, fill `MarketScopeQuant` from the computed numbers, leave `evidence` as a list of `{source: "corpus", company_id: int}` entries (up to 5, top cosine).
- [ ] **Verify:** unit test with a fixture of 5 fake neighborhoods (dense B2B SaaS, sparse devtool, novel AI legal review, crowded fintech, niche healthcare). Direction + confidence are correct. SQL queries are batched in a single CTE, not N+1.

### 4.3 — SearXNG + Firecrawl augmentation for sparse directions (2.5h)

- [ ] New module `src/workflow/market_scope_web.py` — analogous to `src/workflow/web_fallback.py` but for the "wide_open" / sparse-niche case.
- [ ] One SearXNG query: `"{idea} startup competitors"` against the self-hosted instance via Firecrawl's `/v2/search` endpoint (the same path Phase 2.10 uses).
- [ ] Firecrawl scrape the top 3 non-duplicate results, return `[{url, title, snippet, as_of}]`.
- [ ] **Cost / abuse guard:** the web augmentation only fires when:
  - The deterministic rules in 4.2 land on `wide_open` or `niche_but_real` with `competitor_count < 10`, AND
  - The Temporal activity is configured to allow external HTTP (default: allowed; CI's `eval-regression` workflow MUST set the offline mode flag, matching the Phase 3.6.2 pattern).
- [ ] When web augmentation fires: append the top 3 results to `evidence` as `{source: "web", url, snippet, as_of}`. Set `confidence="evidence_backed"`. Leave `quantitative` partially populated (`search_volume_proxy` is the count of distinct domains in the top-3 results, e.g. 3 = niche-but-real, 10+ = wide-open-with-known-players).
- [ ] **Verify:** unit test with a stubbed Firecrawl client returning 3 fake search results. `evidence` list contains 3 web entries. `confidence == "evidence_backed"`. `search_volume_proxy == 3`.

### 4.4 — LLM direction synthesis for the ambiguous middle (2h)

- [ ] Most analyses will land in the ambiguous middle (the deterministic rules in 4.2 don't fire — 4.2's "else" branch). For those, run a small Claude Sonnet 4.5 call (via `instructor`, per the Phase 1.7 pattern) that takes the corpus stats from 4.2 + the top-3 web snippets from 4.3 and emits a `MarketScope` direction + a 1–2 sentence rationale that explicitly cites the corpus stats and the top-1 web snippet.
- [ ] **Prompt contract:** the rationale MUST reference at least one of `competitor_count`, `recent_3y_count`, or a web snippet URL. If the LLM emits a rationale with no citations, the activity retries once (instructor handles schema validation) and then falls back to the deterministic rules' best guess with `confidence="directional"`.
- [ ] **Langfuse:** wrap the call with `langfuse_context` per Phase 2.3. Log `market_scope_synthesis` as the span name, `idea_hash`, `competitor_count`, `recent_3y_count`, `web_scrape_count`, `final_direction`, `final_confidence` as metadata.
- [ ] **Cost guard:** this call is a *separate* LLM call, not a re-run of the structured-comparison call. Budget: ≤500 input tokens, ≤200 output tokens per invocation. Document in `models.yaml` and add a regression assertion in `tests/test_market_scope_synthesis.py` that the prompt length is bounded.
- [ ] **Verify:** unit test with a fixture of 3 ambiguous ideas. LLM call is mocked. Rationale references corpus stats. Langfuse wrapper is called with the right metadata.

### 4.5 — `market_scope_accuracy` eval-harness metric (3h)

- [ ] **Build the labeled benchmark first** — this is the most important rule from `docs/EVAL.md` ("do not write the eval set after the system").
- [ ] 50-idea benchmark in `evals/market_scope_v1.jsonl` with fields `{idea, expected_direction, expected_confidence_at_least, labeler, labeler_provenance, notes}`. The benchmark is hand-rolled: 10 ideas per direction × 5 directions, drawn from the public YC + Product Hunt corpus. 10 are obvious (saturated B2B CRM, wide-open "AI for underwater basket weaving"); 20 are mid-difficulty; 20 are adversarial (sleeper-niche ideas that look wide_open but aren't).
- [ ] **Provenance:** `labeler=anurag-hand-reviewed` if Anurag hand-labels them; otherwise `labeler=ai-assisted-claude-minimax-m3` with `provenance=llm-generated-v1-pending-anurag-hand-review`. **Do NOT stamp as hand-reviewed if they aren't.** This is the same rule Phase 1.5/2.8 had to learn the hard way.
- [ ] Metric: `market_scope_accuracy` = fraction of records where the predicted `direction` matches `expected_direction`. Per-direction breakdown (don't just report the average — saturated should be 95%+, wide_open can be lower because the signal is genuinely weaker there).
- [ ] If `confidence` is below `expected_confidence_at_least`, the record counts as a *soft* miss — surface in the leaderboard as `accuracy_with_confidence_floor` separately.
- [ ] Output: `results/leaderboard.csv` gains `market_scope_accuracy` and `market_scope_accuracy_with_confidence_floor` columns. Per-direction breakdown to `docs/assets/market-scope-accuracy.md` (one row per direction, like the per-category failure analysis from Phase 3.4).
- [ ] **Hard-coded regression cap** in `scripts/ci/eval_gate.py`:
  - `MARKET_SCOPE_ACCURACY_FLOOR: float = 0.60` — the floor for the overall accuracy.
  - `MARKET_SCOPE_ACCURACY_PER_DIRECTION_FLOOR: float = 0.40` — the floor for *any single direction* (catches "the system is great at 'saturated' but misses everything else").
  - Encode as module-level constants (per the Phase 3 type-level-guardrail rule), not config values. Deviation rationale documented in the module docstring.
- [ ] **CI integration:** update `.github/workflows/eval-regression.yml` to run `make eval` and have the gate step check the new `market_scope_accuracy` columns. The gate step's `conclusion` output covers the new floor (Phase 3.6.1's fix).
- [ ] **Verify:** `make eval` reports the new metric. `python scripts/ci/eval_gate.py --csv results/leaderboard.csv` exits 0 on the committed leaderboard (with the metric populated) and 1 on a synthetic leaderboard where `market_scope_accuracy` is below the floor. actionlint passes on the updated workflow YAML.

### 4.6 — Frontend confidence badge + README Limitations update (2h)

- [ ] Update `src/frontend/src/components/IdeaAnalyzer.tsx` (or the relevant component — TBD by 4.6's kickoff) to show a small badge next to the market-scope direction:
  - `directional` — gray, "directional"
  - `evidence_backed` — blue, "evidence-backed" + tooltip with the top-3 web sources
  - `quantitative` — green, "quantitative" + tooltip with the corpus stats
- [ ] The existing color coding (the four directions) stays. The new badge is additive, on top of the direction chip.
- [ ] Update `src/frontend/src/lib/marketScope.ts` to read the new envelope shape.
- [ ] **README.md Limitations §1 rewrite:**
  - Old: "**Market-scope is a stub.** The structured `market_scope_signal` (wide-open / crowded-but-growing / saturated / niche-but-real) exists in the Pydantic `IdeaVerdict` schema and is emitted by the `market_scope_signal` Temporal activity, but it is **not** benchmarked against ground-truth verdicts and is labeled as *directional* in the API response."
  - New: "**Market-scope is corpus-grounded, not a SEMrush replacement.** We compute density / recency / category distribution from our 11K-company corpus and add a SearXNG-augmented evidence layer for sparse directions. The API returns a 3-level `confidence` field — `directional` (the old stub behavior, when the corpus is thin), `evidence_backed` (corpus + ≥1 web source), `quantitative` (corpus stats + 4-direction envelope fully populated). We do not estimate search volume or TAM; for that, use SEMrush."
- [ ] Update `docs/ARCHITECTURE.md` to reflect the new `market_scope_signal` activity (was a pass-through, now a 4-step pipeline: corpus density → web augmentation → LLM synthesis → envelope assembly).
- [ ] **Update the architecture diagram** (`docs/assets/architecture.svg`): the existing `market_scope` box currently says "Qwen 2.5 32B" — change to "market_scope_signal: corpus density → SearXNG → Claude Sonnet 4.5" with a 4-step inline label.
- [ ] **Verify:** frontend builds (`pnpm build`), the badge renders on a real `/ideas/analyze` response (verify against `localhost:18001` if running, otherwise via the test harness), the README's Limitations §1 reads the new text, the architecture diagram still renders cleanly in the README.

---

## What is NOT in Phase 4 (intentionally out of scope)

- **No paid TAM / SEMrush / Crunchbase / PitchBook / CB Insights integration.** No API keys, no subscription cost. The corpus is the only TAM proxy; the web augmentation is a low-volume evidence layer, not a paid estimate.
- **No search-volume numbers.** We do not estimate monthly searches. `search_volume_proxy` is a domain-count from SearXNG, not a keyword-volume number.
- **No real-time market data.** The market-scope signal is a function of the corpus + the web fallback at the time of the `/ideas/analyze` call. It does not subscribe to feeds, track trends over time, or surface historical direction changes (those would be Phase 5 work if ever).
- **No eval set larger than 50.** The benchmark is a 50-idea hand-rolled set, not the 300-idea retrieval benchmark. Market-scope is a harder labeling problem (subjective, requires ground-truth research) — 50 well-labeled beats 300 noisy ones.
- **No change to the structured-comparison LLM call.** The `llm_compare_topk` call is unchanged. The market-scope signal is now a *separate* activity that consumes the same `top_k_payload` and adds its own pipeline. Token-cost impact: ≤700 additional input tokens per `/ideas/analyze` call (4.2 SQL is local, 4.3 web is 3 scraped pages, 4.4 LLM is ≤500 tokens).
- **No eval-driven re-tuning of the deterministic direction rules.** The 4 thresholds in 4.2 (10/50/100/200) are starting points, calibrated by hand. The eval set in 4.5 measures whether the *combined* system hits the floor — not whether each threshold is optimal. Tuning the thresholds is a Phase 5 concern if the floor is missed.
- **No removal of the old `market_scope_signal` pass-through fallback.** The old behavior is preserved as the `directional` confidence level. The new pipeline degrades gracefully to it.

---

## Hard rules (apply to every 4.x child card)

- **No paid API integration.** Every 4.x card that touches the web MUST go through SearXNG + Firecrawl. If a 4.x card finds itself wanting a SEMrush-style number, that's a "stop and decompose" moment — surface to Apollo, don't paper over it.
- **No LLM-invented URLs in `evidence`.** Every `evidence` entry with `source: "web"` MUST come from a real SearXNG + Firecrawl scrape, with the URL the scrape actually returned. The LLM synthesis step in 4.4 is allowed to *cite* an `evidence` URL, never to *invent* one.
- **Honest provenance on the labeled benchmark.** If the 50 ideas in 4.5 are LLM-generated, the file's `labeler` is `ai-assisted-claude-minimax-m3` and `provenance` is `llm-generated-v1-pending-anurag-hand-review`. Same rule the project has held since Phase 1.5.
- **Hard-coded regression caps.** `MARKET_SCOPE_ACCURACY_FLOOR` and `MARKET_SCOPE_ACCURACY_PER_DIRECTION_FLOOR` are module-level constants in `scripts/ci/eval_gate.py`, never config values. Per the Phase 3 type-level-guardrail rule.
- **No breaking changes to the `IdeaVerdict` schema.** Add fields with sensible defaults; preserve `market_scope.direction` as the string accessor the existing frontend code uses.
- **The Phase 3 hard rule on `corpus_count` / Sources of Truth still applies.** The new `MarketScopeQuant` reads from the same `companies` and `company_embeddings` tables. If 4.2's SQL needs a new column or index, surface it to Apollo before adding it — Phase 3 has been disciplined about not changing the table schema mid-phase.
- **Langfuse wrapping for any new LLM call.** The synthesis call in 4.4 follows the Phase 2.3 pattern (`langfuse_context` with metadata, scoring field). No unwrapped LLM calls.
- **GitHub Actions regression self-contained.** The CI must continue to be offline-capable. The 4.3 web-augmentation activity MUST check the offline-mode flag (set by the eval-regression workflow env) and short-circuit to `directional` confidence when the flag is set. The eval harness itself never calls out to the web.
- **No notify-subscribe** to Anurag's chat surfaces (per the standing rule on the framework auto-subscribe cap).

---

## Order of operations (dependency DAG)

```
4.1 (Pydantic envelope)
  ↓
4.2 (corpus density activity)   4.3 (SearXNG/Firecrawl web)
  ↓                                ↓
  └──── 4.4 (LLM synthesis) ◀─────┘
              ↓
           4.5 (eval-harness metric + benchmark + gate + CI)
              ↓
           4.6 (frontend badge + README + arch diagram)

Phase 4 review (4.7, Helena, PASS/FAIL, ~1.5h)
```

- **4.2 and 4.3 are independent** — they can be developed in parallel (different modules, different test fixtures).
- **4.4 depends on both 4.2 and 4.3** (it consumes both their outputs).
- **4.5 depends on 4.2 + 4.3 + 4.4** (it benchmarks the *combined* system).
- **4.6 depends on 4.1 + 4.4** (it renders the new envelope and confidence).
- **4.7 review depends on all of 4.1-4.6.**

---

## Workspace

`/home/ody/workspace/priorart`. Same dir as Phases 1–3. The `dir` workspace is the standard project workspace — `git status` should be clean at start, every 4.x card commits with the standard `Phase 4.N (card t_xxx): <what changed>` format that Phase 3 established.

---

## Phase 4 review (4.7, ~1.5h)

Helena, same shape as Phase 3.12. Severity-tagged findings. PASS/FAIL verdict. Specific things to verify:

- **critical:** No paid API keys in `models.yaml`, no SEMrush / Crunchbase / CB Insights / PitchBook env vars. The web path is *only* SearXNG + Firecrawl.
- **critical:** The eval benchmark `evals/market_scope_v1.jsonl` has honest provenance. If `labeler=ai-assisted-claude-minimax-m3`, no README claim of hand-review.
- **critical:** The CI regression workflow's offline-mode flag short-circuits the 4.3 web augmentation. `make eval` in CI does not call out to the network.
- **critical:** The new `market_scope_accuracy` regression caps are hard-coded constants in `scripts/ci/eval_gate.py`, not config values.
- **critical:** `evidence` entries with `source: "web"` are real SearXNG + Firecrawl URLs. Grep the test suite for a fixture URL pattern that proves the activity takes the scrape output, not a string the LLM wrote.
- **high:** The 4.2 SQL is a single CTE, not N+1 queries per direction metric.
- **high:** The 4.4 LLM synthesis call's prompt is bounded (≤500 input tokens, ≤200 output) and asserts citations in the rationale.
- **high:** README Limitations §1 rewrite is in place, with the new 3-level confidence vocabulary.
- **high:** Architecture diagram (`docs/assets/architecture.svg`) is updated and dark-themed, accurate.
- **medium:** The frontend badge renders on a real response. The frontend build passes.
- **medium:** Langfuse traces for the synthesis call have the documented metadata fields.
- **low:** `make smoke` and `pytest` pass with the new code.

PASS: no critical or high findings. FAIL: any critical or high finding. The `market_scope_accuracy` baseline measurement on the 50-idea benchmark is the headline number for the Phase 4 PR description — flag it prominently in the verdict so Anurag sees it in the review summary.

---

## Provenance

This phase spec was written by Apollo (`t_61d1c753`) on 2026-07-02 after the Phase 3 review (t_59692f85) passed. The doc-side sketch of "TAM lookup / SEMrush / manual-research / Brave Search fall-back" from t_2f56bfa4 was the input; this phase replaces it with the corpus + SearXNG/Firecrawl shape that's actually buildable in this environment without paid API keys. The "no SEMrush, no LLM-invented URLs, hard-coded caps, honest provenance" hard rules are the same ones the project held to in Phases 1–3 — they don't get relaxed in Phase 4 just because the new feature is more subjective.

**Anurag signoff gate:** before 4.1 starts, this PHASE-4.md needs a "go" from Anurag in a comment on `t_61d1c753` or via a direct message. Standing permission from 2026-06-28 covers the *option-choice* (Phase 4 deferred vs Phase 3.5 vs stub-forever); it does NOT cover the *build-shape* (corpus + SearXNG vs TAM lookup). If Anurag disagrees with the corpus+SearXNG shape and wants a paid-API integration instead, this spec changes before any child card is created.
