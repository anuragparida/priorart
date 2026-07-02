# Startup-Idea Deduplication & Competitor Research Service ("PriorArt")

> Paste a startup idea. Get back a ranked list of similar past YC / Product Hunt / Hacker News launches with Pydantic-validated structured comparisons, a market-scope signal, and a reproducible evaluation harness benchmarking retrieval quality against a labeled public-corpus benchmark. Wraps a multi-step Temporal workflow (per-idea pipeline with retry + web-search fallback) on top of a Dagster-managed corpus ingestion + nightly re-embedding data platform. The whole system is built like a production ML/AI platform, not a wrapper script.

## Why it's signal

This is the **direct, public-safe evolution of the Mercedes-Benz thesis** (Apr–Oct 2025: LLM-based vector search, structured JSON outputs, PG vector, similarity metrics, retrieval@K). The thesis was internally scoped; this project is the same engineering, pointed at a public problem, with a public corpus and a reproducible benchmark behind it. Anyone reading the CV line sees the lineage instantly.

Three layers of signal stack:

1. **The thesis story.** Structured JSON outputs + vector similarity + retrieval@K + Pydantic validation. All in one project, in a production-shaped system. This is the Mercedes work, public.
2. **The MLOps story.** Temporal (per-idea workflow with retry / fallback / human-in-the-loop) + Dagster (corpus ingestion + nightly re-embedding batch data platform) + MLflow (experiment tracking) + Langfuse (LLM observability) + pgvector (the actual store). This is the platform-engineer story.
3. **The eval story.** A labeled 300-idea benchmark drawn from public YC / Product Hunt archives, with MRR / nDCG@K / precision@K / recall@K / a calibration curve, running as a regression suite on every config change. This is the "I know how to know if it's working" story.

Strong signal for: senior IC / staff-plus AI platform, retrieval/ML systems, applied-LLM engineering, and any role where the first system-design question is "how do you measure it?" The product itself is also a real daily-driver utility for any founder, PM, or accelerator — useful, not just demonstrative.

## Realistic timeline

**Phase 1 (Weekend 1) — Must-be:** FastAPI + pgvector + bge-m3 ingestion, a 100-idea labeled benchmark, MRR / nDCG@K / precision@K metrics, the multi-step LLM-comparison pipeline with Pydantic-validated structured output, a minimal shadcn/ui dark-mode UI, README with leaderboard screenshot. **Ship-able by Sunday night.**

**Phase 2 (Weekend 2) — Should-be:** Temporal workflow for the per-idea pipeline (retry, fallback to web search, human-in-the-loop on low-confidence verdicts), Langfuse tracing on all LLM calls, MLflow for embedding-model / threshold / prompt-template A/B experiments, corpus expansion to 300-idea benchmark, 3 retrieval configs (BM25 / dense / hybrid) compared in the leaderboard. **This is the "production-grade" weekend.**

**Phase 3 (Weekend 3) — Can-be:** Dagster for corpus ingestion + nightly re-embedding jobs (Temporal handles per-idea; Dagster handles batch data), calibration curve + false-positive-rate metrics, GitHub Actions regression suite that posts the leaderboard diff on every PR, demo asciinema recording, polished README. **The cherry.**

## Tech stack (specific, no generic terms)

- **Backend:** `uv` + Python 3.12 + FastAPI + SQLAlchemy 2.x + Pydantic v2.
- **Database:** Postgres 16 + pgvector (in Docker). One schema, three tables: `companies` (metadata), `company_embeddings` (vectors, joined by id), `eval_runs` (leaderboard history).
- **Embeddings:** `BAAI/bge-m3` (multilingual — relevant for any German/European expansion). Local via sentence-transformers, no API cost. Alternative: `text-embedding-3-small` if you want to skip local model load.
- **Vector search:** pgvector with HNSW index. Reciprocal Rank Fusion for hybrid (BM25 + dense). Cohere rerank as a 4th config.
- **Orchestration (per-idea workflow):** **Temporal.io**. Workflows model the multi-step pipeline (embed → ANN search → LLM-compare → market-scope → assemble) with retry on transient LLM failures, fallback to web search when the corpus returns no match, and a human-in-the-loop signal channel for low-confidence verdicts.
- **Orchestration (batch data platform):** **Dagster**. Assets model the corpus ingestion pipeline (scrape → clean → dedup → embed → load), plus the nightly re-embedding job and the eval-harness regression runs. Sensible schedules (`@daily`, `@weekly`) and a sensor that fires the regression suite on every config-file change.
- **Experiment tracking:** **MLflow** self-hosted (single Docker container, SQLite backend, ~5 min setup). Tracks embedding-model versions, threshold sweeps, prompt-template A/B tests.
- **LLM observability:** **Langfuse** (you already have it self-hosted on 13000/13001). Trace every LLM call with embedding latency, ANN search latency, top-K IDs, verdict, total latency, token cost.
- **LLM:** Anthropic Claude Sonnet 4.5 for the structured-comparison call (good at long, nuanced comparisons). MiniMax-M3 or a local Qwen 2.5 32B for the cheap / fast calls (e.g. market-scope classification).
- **Constrained generation:** Pydantic v2 + instructor (or outlines) for the structured LLM output. Deterministic validation, exact violation reasons.
- **Frontend:** `pnpm` + Vite + TypeScript + React 18 + Tailwind + shadcn/ui. Dark mode by default. One page: idea input → ranked competitors + structured verdicts + market-scope signal + evidence links. Honest empty states.
- **Web search fallback:** Brave Search API (generous free tier) or SerpAPI (free quota). Self-hosted SearXNG via the `self-hosted-firecrawl-hermes` skill is the local-first alternative. Scraped pages go through Firecrawl (already self-hosted) for clean markdown extraction.
- **Eval harness storage:** DuckDB single file (zero infra, queryable, easy to commit snapshots of the leaderboard).
- **CI:** GitHub Actions for the regression suite only (per-config change → run eval → post leaderboard diff as PR comment). Public repo, so Actions are fine.
- **Dependency mgmt:** `uv` for backend, `pnpm` for frontend, Docker Compose for the platform.

## Architecture (high level)

The system has **three Temporal workflows**, **four Dagster assets**, **one eval-harness regression runner**, and **one observability surface** (Langfuse).

```
┌─────────────────────────────────────────────────────────────┐
│                        USER / API                            │
│  POST /ideas/analyze {"idea": "..."}  →  returns verdict     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│   TEMPORAL WORKFLOW: "IdeaAnalysisWorkflow"                  │
│   ─ embed_idea()                                            │
│   ─ ann_search()         ←── pgvector HNSW                  │
│   ─ llm_compare_topk()   ←── Pydantic-validated output      │
│   ─ market_scope_signal()                                   │
│   ─ web_fallback_if_empty()                                 │
│   ─ assemble_verdict()                                      │
│   (every step: retry, Langfuse-traced, MLflow-logged)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│   POSTGRES + PGVECTOR (snapshot of public corpus)            │
│   companies · company_embeddings · eval_runs                 │
└─────────────────────────┬───────────────────────────────────┘
                          │  (refreshed by Dagster, nightly)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│   DAGSTER: data platform                                    │
│   asset: yc_directory        (scrape + clean + load)         │
│   asset: product_hunt_archive  (scrape + dedup + load)       │
│   asset: hn_show_posts       (HN Algolia API, paginate)     │
│   asset: company_embeddings   (chunk + embed + write HNSW)  │
│   sensor: config_changed     → trigger eval-harness job     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│   EVAL HARNESS REGRESSION (runs on config change + nightly) │
│   - 300 labeled idea→{duplicate, top-K, expected_ids}       │
│   - metrics: MRR, nDCG@10, precision@K, recall@K,           │
│              calibration_curve, FPR-on-novel                 │
│   - output: CSV + static HTML leaderboard, committed        │
│   - fails build if MRR drops below threshold                 │
└─────────────────────────────────────────────────────────────┘
```

## Landscape — what already exists in this space

Honest, one-line lay of the land. The full "no public tool does all of this" argument holds up; the closest competitors each miss on a specific axis.

| Tool | What it does well | What it doesn't do |
|---|---|---|
| **Siftt / IdeasGPT / ValidatorAI / Sprintbase** | "Validate your idea" UX. Fast onboarding. | Thin LLM-wrapper + Google search. No curated corpus, no retrieval quality measured, no structured comparison, no eval harness. Vibes, not evidence. |
| **Crunchbase Pro / Pitchbook / CB Insights** | Real market intel, investor-grade. Paywalled. | $10K+/yr, investor-facing, not "drop your idea in." The opposite of a demo. |
| **SEMrush / Ahrefs / SimilarWeb** | Real traffic + SEO estimates. | Paywalled at the level you need. Free tier is throttled. Data exists; access doesn't. |
| **Glimpse / Exploding Topics / SparkToro** | Trend detection, niche discovery. | Adjacent to market-scope signal, not duplicate detection. |
| **Internal YC / a16z / Antler / Techstars tooling** | This is the real production version. Processes 10K+ apps/cycle. | Locked behind NDAs. No public version, no eval harness, no paper. |
| **Gornall, Huang et al. (academic — startup success, "P(roduct) Market Fit")** | Real empirical work on what makes startups succeed. | No productionized system. The data is here; the tool isn't. |
| **Startup Graveyard / Autopsy.io** | Public failure postmortems. | Static, not queryable, not embedded into a comparison engine. |
| **DeepEval / RAGAS / TruLens (eval libraries)** | The standard retrieval-eval libraries. | Generic — they don't have a labeled *idea-dedup* benchmark. You'd be using them as a layer, not a replacement for your own harness. |
| **This project** | End-to-end: idea → vector dedup against a public corpus → structured LLM comparison → market-scope signal → reproducible eval harness → MLOps platform. | A demo, not a SaaS. Self-hosted, single-tenant, public-data only. |

The "no public tool does all of this" claim survives: the wrappers miss the corpus + eval, the corpora miss the comparison + market-scope, the eval libraries miss the domain-specific benchmark, the academic work misses the system.

## Phase 1 — Must-be (Weekend 1, ship by Sunday night)

**Backend:**
- `uv` project with FastAPI + SQLAlchemy 2 + Pydantic v2.
- Postgres + pgvector in Docker Compose. One schema with three tables: `companies`, `company_embeddings` (vector(1024) for bge-m3, HNSW index), `eval_runs`.
- YC public directory scraper: ~5K companies, name + 1-paragraph description + tags + batch + status. Output: `data/snapshots/yc_<date>.jsonl` with a manifest.
- Embedding pipeline: chunk by sentence (not arbitrary token windows), embed with `bge-m3`, store in `company_embeddings`.
- ANN search endpoint: `POST /search {"query": "...", "top_k": 20}` returns ranked IDs + similarity scores.
- LLM comparison endpoint: `POST /ideas/analyze {"idea": "..."}` orchestrates embed → ANN search → structured LLM compare (top-3 only, for cost) → market-scope stub → assemble verdict. Pydantic-validated output: `IdeaVerdict { top_competitors: list[CompetitorVerdict], market_scope: MarketScope, supporting_evidence: list[URL] }`.
- Eval harness: 100 hand-labeled `{"idea", "expected_top_ids", "is_duplicate"}` triples in `evals/labeled_v1.jsonl`. Runner computes MRR, nDCG@10, precision@5, recall@10, FPR-on-novel. Output: `results/leaderboard.csv`.

**Frontend:**
- One page: textarea for idea → submit → list of top competitors with structured verdicts (similarity axes, key differences, likely failure modes) + market-scope badge + evidence links.
- Dark mode, shadcn/ui, no over-design.
- Honest empty state when the corpus returns nothing.

**Repo hygiene:**
- `docker compose up` brings up the whole stack. Health check on `/healthz` returns 200.
- README with: 1-paragraph pitch, the leaderboard screenshot, quickstart, methodology section, the landscape table above.
- `data/snapshots/yc_<date>.jsonl` committed.
- `evals/labeled_v1.jsonl` committed.
- `results/leaderboard.csv` committed (regenerated by CI).
- Apache 2.0.

**Phase 1 ships when:** `uv sync && docker compose up && python -m eval.run && pnpm dev` produces a working idea-lookup at `localhost:15173` (frontend) / `localhost:18000` (API), MRR ≥ 0.5 on the 100-idea benchmark, and the README's leaderboard screenshot is real (not mocked).

## Phase 2 — Should-be (Weekend 2, the "production-grade" weekend)

**Temporal workflow:**
- Refactor `/ideas/analyze` to a Temporal workflow (`IdeaAnalysisWorkflow`). Each step is an activity with retry policy (exponential backoff, max 3 retries on transient LLM failures, dead-letter on schema-violation).
- Add a `web_fallback_if_empty` activity: if top-K returns nothing above the threshold, run a Brave/SearXNG search, scrape the top-3 results via Firecrawl, embed them, re-run ANN search. Self-hosted SearXNG is the local-first option.
- Add a `human_review` signal channel: low-confidence verdicts (cosine similarity in 0.55–0.70 band, or LLM self-reported confidence < 0.7) get parked in a Temporal "waiting for signal" state. A simple admin endpoint can resume the workflow with a corrected verdict.
- Add a `compare_topk` activity that runs the structured LLM comparison in parallel for top-3 (cost-controlled).

**MLflow:**
- Self-host MLflow (Docker, SQLite backend).
- Log every embedding-model version, threshold value, prompt-template version as a run.
- A `make mlflow-compare` target that produces a sweep table (which embedding model / threshold combo gives the best MRR on the benchmark).

**Langfuse:**
- Already self-hosted. Wrap the LLM call in a `langfuse_context` with the idea as input, the verdict as output, embedding latency + ANN search latency as metadata.
- The Langfuse dashboard becomes the "production observability" screenshot in the README.

**Corpus expansion:**
- Product Hunt archive scraper: 5K most-upvoted launches over the last 3 years. Add as `data/snapshots/producthunt_<date>.jsonl`.
- Hacker News "Show HN" posts: pull via HN Algolia API, paginate, filter for posts with >50 points + an external link. Add as `data/snapshots/hn_show_<date>.jsonl`.
- Expand eval set to 300 labeled triples, balanced across duplicate / similar-but-different / novel / adversarial-paraphrase.

**Retrieval configs:**
- Add BM25 (rank_bm25) and Hybrid (RRF) configs alongside the dense one. The leaderboard now compares 3 configurations.
- Optional: Cohere rerank as a 4th config.

**Phase 2 ships when:** Temporal is running (`temporal server start-dev`), the workflow can be observed end-to-end in the Temporal UI, the Langfuse dashboard shows the trace, the MLflow UI shows the sweep, and the leaderboard CSV compares 3 retrieval configs on the 300-idea benchmark.

## Phase 3 — Can-be (Weekend 3, the cherry)

**Dagster:**
- Migrate the corpus ingestion scripts to Dagster assets: `yc_directory`, `product_hunt_archive`, `hn_show_posts`, `company_embeddings` (re-embeds on snapshot change).
- Add a `nightly_re_embed` schedule.
- Add a sensor that watches the `configs/` directory and fires the eval-harness regression job on any change.
- Dagster UI on port 13002 (or wherever it lands in your port scheme).

**Eval harness maturity:**
- **Calibration curve:** bin predictions by similarity score, plot actual duplicate-rate per bin. A well-calibrated system has the curve hugging the diagonal.
- **FPR-on-novel:** explicitly measure how often the system marks a genuinely-novel idea as a duplicate. The most important metric; the one that determines whether the tool is useful or annoying.
- **Per-category failure analysis:** which kinds of ideas (B2B SaaS, consumer, devtools, etc.) does the system fail on? Surface the failure modes.

**CI / regression suite:**
- GitHub Actions: on every PR that touches `configs/`, `evals/`, `src/embedding/`, or `src/llm/`, run the eval harness and post the leaderboard diff as a PR comment.
- Build fails if MRR drops below 0.7 or FPR-on-novel exceeds 0.15.

**Polish:**
- `asciinema` demo of a full flow: paste idea → see verdict → click into top competitor → see structured comparison.
- `docs/METHODOLOGY.md` explaining each metric, the benchmark construction, the label policy, the calibration target.
- `docs/LANDSCAPE.md` (the table above, expanded).
- `CONTRIBUTING.md` for adding a new retrieval config or a new data source.
- GitHub topics: `ai`, `rag`, `retrieval`, `evaluation`, `mlops`, `pgvector`, `startups`, `competitor-research`, `temporal`, `dagster`.

**Phase 3 ships when:** Dagster UI shows the assets + schedules + the config-change sensor firing the regression. The leaderboard HTML has the calibration curve and the per-category failure breakdown. The Actions regression is green. The asciinema demo is linked in the README.

## Pitfalls

- **Do not use a private corpus.** YC public directory is OK (it's public). Crunchbase's free tier is OK for 200 lookups/month in dev, but not as the primary source. Never use HDI / Mercedes data. The whole point of the project is reproducibility and the public-claim on the CV.
- **The eval set is the hardest and most important part.** Spend disproportionate time on it. 300 hand-labeled triples — 100 known-duplicates across paraphrasings, 100 known-novel (long-tail ideas with no close YC match), 100 adversarial (slight pivots, market overlap, similar-tech-different-domain) — beats any LLM-generated labels. **Do not label with an LLM.** You'll rationalize the labels to match the system's outputs and lose the entire benchmark.
- **Do not trust a single metric.** Always report MRR + nDCG@K + precision@K + recall@K + FPR-on-novel + calibration. The leaderboard should show all of them, not a composite. The FPR-on-novel is the one that determines if a real user would trust the tool.
- **Do not conflate "retrieval quality" with "comparison quality."** Score them separately. A high-MRR retriever can still produce bad LLM comparisons. Run the structured-comparison call on the labeled set and score it against human-written expected verdicts (a smaller, ~30-pair held-out set for this).
- **Do not skip latency / cost.** In a demo, it's tempting to call the LLM 20 times per request. The Temporal workflow will show you the latency for free; log it. Track token cost per request. Cap the LLM call to top-3 by default.
- **Do not use Cohere rerank as the default config.** It's an API call. The default must run on local embeddings + local models, with rerank as an opt-in config. Otherwise the project is broken the moment the API key expires.
- **Do not build a real market-scope estimator (in Phase 1).** SEMrush / Ahrefs / SimilarWeb are paywalled. Google Trends is free but noisy. Build the *stub* (classify as `wide_open` / `crowded_but_growing` / `saturated` / `niche_but_real` from top-K density + LLM judgment) and label it as a directional signal. The honest version requires a paid API; the stub is the right call. Note in the README "future work: integrate SEMrush / SimilarWeb when budget allows." **Phase 4 (see `docs/PHASE-4.md`) replaces this stub with a corpus-grounded + SearXNG-augmented implementation that stays free of paid APIs and is benchmarked against a 50-idea labeled set.**
- **Do not auto-subscribe Temporal events to all your Discord channels.** Wire the eval-harness-regression-failed event to your Telegram (user-personal escalation), and Temporal-workflow-stalled events to the same. Keep worker notifications on the relevant agent's own channel.
- **Do not write the eval harness after building the system.** Write the eval set first, build the system against it, iterate. Otherwise you rationalize the benchmark to match the system.

## Definition of done

A new contributor can `git clone`, `uv sync`, `docker compose up`, drop an updated corpus snapshot into `data/snapshots/`, run `make eval`, and reproduce the leaderboard CSV + HTML to within ±0.01 MRR. The Temporal UI shows the per-idea workflow with traces. The Langfuse UI shows the LLM calls. The MLflow UI shows the experiment sweep. The Dagster UI shows the assets + the config-change sensor firing the regression. The README explains when to pick which retrieval config. The eval harness fails the build when MRR drops below 0.7. Numbers are reproducible from the committed config. Failure analysis points at specific ideas where each config falls down.

## Scope discipline (intentionally not built)

- **No hosted service.** Self-hosted Docker Compose. CLI + the local FastAPI + the local Vite UI.
- **No multi-tenant.** One operator, one corpus snapshot, one leaderboard.
- **No fine-tuning.** The point is to measure retrieval, not adapt models.
- **No real-time ingestion.** The corpus is a snapshot, versioned. Dagster handles the refresh schedule; per-request ingestion is out of scope.
- **No production market-scope estimator.** Stub + honest label. Future work.
- **No auth.** It's a demo. Single-tenant. Add basic-auth at the FastAPI level if you expose it.
- **No Kubernetes.** Docker Compose, single host. (You already have Honcho / Langfuse / Firecrawl on this host; this project joins them.)
- **No real-time Temporal cluster.** Use `temporal server start-dev` for local dev. Document the prod migration path (Helm chart, Postgres-backed) in `docs/OPERATIONS.md` but don't build it.

## Polish checklist

- README with: 1-paragraph pitch, the leaderboard screenshot, the landscape table, quickstart, methodology link, the Temporal/Dagster/MLflow/Langfuse architecture diagram, "how to add a retrieval config" guide, "how to add a data source" guide.
- `docs/METHODOLOGY.md` with each metric explained, the benchmark construction policy, the label policy, the calibration target.
- `docs/LANDSCAPE.md` (expanded version of the table above).
- `docs/OPERATIONS.md` with the Temporal + Dagster dev-loop walkthrough.
- 2-minute asciinema demo: paste idea → see ranked competitors → see structured verdict.
- GitHub topics: `ai`, `rag`, `retrieval`, `evaluation`, `mlops`, `pgvector`, `startups`, `competitor-research`, `temporal`, `dagster`, `pydantic`.
- Apache 2.0.
- `/examples` with one real run's full output (verdict JSON + Langfuse trace link + Temporal workflow ID).
- `data/snapshots/` with the YC + Product Hunt + HN snapshots committed (deduped + cleaned).
- `evals/labeled_v300.jsonl` committed.
- `results/leaderboard.csv` + `results/leaderboard.html` committed (regenerated by CI).
- `models.yaml` registry (which embedding model, which LLM, which prompt template) checked in.
- `configs/` directory with one YAML per retrieval configuration.
