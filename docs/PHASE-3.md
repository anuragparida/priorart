# Phase 3 — The Cherry (Weekend 3)

> **Tier:** Can-be. **Goal:** Dagster for the batch data platform. Calibration curve + FPR-on-novel as first-class metrics. GitHub Actions regression suite. Polished README. asciinema demo. This is the weekend that turns the project from "I built this in 2 weekends" into "I built a real platform with a maintenance story."

---

## What ships at the end of Phase 3

- [ ] Dagster UI on port 13002 with the corpus-ingestion assets, the nightly re-embedding schedule, and the config-change sensor that fires the eval regression.
- [ ] Calibration curve as a first-class metric in the leaderboard (per-config, plotted as a PNG via matplotlib).
- [ ] Per-category failure analysis (which kinds of ideas — B2B SaaS, consumer, devtools, etc. — does the system fail on, per config).
- [ ] GitHub Actions workflow `eval-regression.yml` that runs the eval harness on every PR touching `configs/`, `evals/`, `src/embedding/`, or `src/llm/`, and posts the leaderboard diff as a PR comment.
- [ ] Build fails if MRR drops below 0.7 or FPR-on-novel exceeds 0.15 (on the production config — `hybrid_rrf`).
- [ ] asciinema demo of a full flow (~2 min): paste idea → see ranked competitors → see structured comparison.
- [ ] Polished README: the architecture diagram, the leaderboard, the calibration curve, the failure analysis, the Temporal/Dagster/MLflow/Langfuse walkthrough, the landscape, the methodology.
- [ ] `docs/OPERATIONS.md` with the dev-loop walkthrough + the prod-migration path for Temporal (Helm chart, Postgres-backed).
- [ ] GitHub topics finalized. Repository is portfolio-ready.

**Definition of done:** Dagster UI shows the assets + schedules + the config-change sensor. The leaderboard HTML has the calibration curve and the per-category failure breakdown. The Actions regression is green. The asciinema demo is embedded in the README. A new contributor can `git clone && uv sync && docker compose up` and reproduce every number in the README.

---

## Detailed task breakdown

Each task is sized to ~2–4 hours. Total Phase 3 budget: ~16 hours, the lightest of the three phases because the platform is already in place.

### 3.1 — Dagster dev environment (1.5h)
- [ ] Add `dagster` to `pyproject.toml`. Local: `dagster dev` on port 13002 (avoid conflicts with Langfuse 13000/13001).
- [ ] Migrate the Phase 2 corpus-ingestion scripts to Dagster **assets**:
  - `yc_directory` — scrape + clean + load to staging table.
  - `product_hunt_archive` — scrape + dedup + load.
  - `hn_show_posts` — Algolia pagination + Firecrawl scrape + load.
  - `company_embeddings` — re-embed on snapshot change, write HNSW.
  - `eval_benchmark` — track the current eval-set version, surface staleness.
- [ ] Add a `@daily` schedule for the nightly re-embedding.
- [ ] **Verify:** Dagster UI materializes all assets, the schedule shows up, the lineage graph is correct.

### 3.2 — Config-change sensor (1.5h)
- [ ] Dagster sensor that watches `configs/` and `models.yaml`. On any change, fires a job that runs the eval-harness regression.
- [ ] The job runs `make eval` and posts the result to MLflow + commits the leaderboard diff.
- [ ] **Verify:** edit `configs/dense_bge_m3.yaml`, watch Dagster fire the job, see the new leaderboard row in MLflow.

### 3.3 — Calibration curve metric (2h)
- [ ] Bin predictions by similarity score (10 bins, 0.0–1.0 in 0.1 steps).
- [ ] Per bin: count, actual duplicate rate (fraction of records in the bin where `is_duplicate=true`).
- [ ] Plot the calibration curve (predicted similarity on x-axis, actual duplicate rate on y-axis). A well-calibrated system hugs the diagonal.
- [ ] Expected Calibration Error (ECE) as a single-number summary.
- [ ] Output: `docs/assets/calibration-<config>.png`, one per config.
- [ ] The leaderboard CSV gains an `ece` column.
- [ ] **Verify:** the dense config should have ECE ≤ 0.10 on the 300-idea benchmark. If it's higher, the threshold needs recalibration or the corpus is mismatched.

### 3.4 — Per-category failure analysis (2h)
- [ ] Extend the eval set with a `category` field per record (B2B SaaS, consumer, devtools, marketplace, fintech, healthcare, education, other).
- [ ] Per category, per config: MRR, nDCG@10, FPR-on-novel, top-3 failure examples.
- [ ] Output: `docs/assets/failure-breakdown-<config>.md` (one per config) + a consolidated `docs/assets/failure-breakdown.png`.
- [ ] **Verify:** surface a meaningful pattern. E.g. "the system fails on consumer social apps (MRR 0.42) but excels on devtools (MRR 0.89)." If everything is uniform, the categories need reworking.

### 3.5 — FPR-on-novel as a first-class metric (1h, mostly done in Phase 2)
- [ ] The FPR-on-novel is already computed in Phase 1's eval runner. Phase 3 work is to surface it prominently: leaderboard column, calibration-style breakdown, the README's "trust this tool" claim.
- [ ] Add a per-config "novel-set MRR" — among the 100 `is_duplicate=false` records, what fraction of top-1 results had cosine > threshold. **This is the metric that determines whether a real user would trust the tool.**
- [ ] README: "We report FPR-on-novel explicitly because overclaiming duplicates is the worst failure mode of an idea-dedup tool. Our hybrid config has FPR-on-novel = 0.08, meaning a novel idea is incorrectly flagged as a duplicate only 8% of the time."

### 3.6 — GitHub Actions regression suite (2h)
- [ ] `.github/workflows/eval-regression.yml`:
  - Triggers: PR that touches `configs/**`, `evals/**`, `src/embedding/**`, `src/llm/**`, `models.yaml`, or `pyproject.toml`. Also runs on `push` to `main` and on a nightly cron.
  - Spins up Postgres + pgvector in a service container.
  - Loads the committed corpus snapshot (no fresh scrape — committed snapshots are the reproducibility guarantee).
  - Runs `make eval` against the 300-idea benchmark.
  - Posts the leaderboard diff as a PR comment using `gh-actions-remark`.
  - **Fails the build if** MRR drops below 0.7 on `hybrid_rrf` or FPR-on-novel exceeds 0.15.
- [ ] **Verify:** open a PR that changes the prompt template, see the Action fire, see the leaderboard diff in the PR comments, see the build fail if MRR drops.

### 3.7 — Polished README (2h)
- [ ] Frontmatter: 1-paragraph pitch, the leaderboard screenshot, the architecture diagram, the calibration curve, the failure breakdown.
- [ ] Sections:
  - **What it is** — one paragraph, the CV claim.
  - **Demo** — asciinema embed.
  - **Leaderboard** — table with the 3 configs, 5 metrics, the calibration curve.
  - **Architecture** — diagram + 4-paragraph walkthrough of the Temporal/Dagster/MLflow/Langfuse split.
  - **Quickstart** — `git clone && uv sync && docker compose up && make eval` works on a fresh box.
  - **Methodology** — link to `docs/METHODOLOGY.md`.
  - **Landscape** — link to `docs/LANDSCAPE.md`.
  - **How to add a retrieval config** — guide for contributors.
  - **How to add a data source** — guide for contributors.
  - **Limitations** — the market-scope stub, the eval-set size, the public-corpus bias. Be honest.
- [ ] Badges: build status (Actions), license, GitHub topics, "made with bge-m3 + pgvector + Temporal + Dagster."

### 3.8 — asciinema demo (1h)
- [ ] Script the 2-minute flow:
  - `make eval` runs the harness (30s).
  - `pnpm dev` starts the frontend.
  - Open browser, paste "AI-powered contract review for SMB law firms."
  - Show the ranked competitors + structured verdicts.
  - Open the Temporal UI, show the workflow that just completed.
  - Open the Langfuse UI, show the LLM trace.
- [ ] Record with `asciinema rec --title "PriorArt demo" --command "bash scripts/demo.sh"` (~2 min).
- [ ] Upload to asciinema.org, embed in the README.

### 3.9 — `docs/OPERATIONS.md` (1h)
- [ ] Dev-loop walkthrough: start Temporal, start Dagster, start the API, run the eval, view in MLflow + Langfuse.
- [ ] Common failure modes: Temporal worker not registered, pgvector index missing, Langfuse key missing, MLflow backend down.
- [ ] Prod-migration path for Temporal: Helm chart, Postgres-backed (`temporal` + `temporal_visibility` databases), `temporal operator namespace create`. Documented but not built.

### 3.10 — Final smoke test + release (1h)
- [ ] Fresh-clone test: `git clone <url> && cd priorart && uv sync && docker compose up && make eval && pnpm dev` — every command works on the first try.
- [ ] All URLs in the README resolve to real artifacts.
- [ ] Tag the release: `git tag v0.1.0 -m "Phase 3 complete"` and push the tag.
- [ ] Update the GitHub repo description and topics.

---

## What is NOT in Phase 3 (intentionally out of scope)

- Production market-scope estimation. The stub stays.
- Real-time ingestion. The corpus is a snapshot, refreshed nightly via Dagster.
- Multi-tenant. Single operator, single corpus, single leaderboard.
- Auth. Self-hosted, single-user.
- Kubernetes / multi-host. Docker Compose on a single host.
- Custom-trained embedding model. We use `bge-m3` off the shelf.
- Fine-tuning. We measure, not adapt.

These are listed in `SPEC.md` under "Scope discipline" and are non-negotiable. Don't re-litigate them during build.

---

## Pitfalls (Phase 3 specific)

- **Do not skip the calibration curve.** It's the single metric that distinguishes "we tuned a threshold" from "we have a calibrated system." Skip it and the eval story is half-built.
- **Do not add Cohere rerank as a 4th config "to round it out."** It requires an API key. If the key expires, the config breaks. Either add it as a properly-gated opt-in (with a clear error if the key is missing) or don't add it. Default must be offline-runnable.
- **Do not let the Actions workflow hit external services.** The regression suite must be self-contained. No Brave Search calls, no Cohere rerank, no Anthropic API. The eval is offline — that's the reproducibility guarantee.
- **Do not let the asciinema demo exceed 2 minutes.** Anything longer is a video, not a demo. Keep it tight.
- **Do not skip the "Limitations" section in the README.** Honest limitations are credibility. The market-scope stub is a stub. The eval set is 300 hand-labeled triples, not 10K. The corpus is biased toward what launched on YC + Product Hunt + HN, which is biased toward Silicon-Valley-style startups. Say it.

---

## Verification at end of Phase 3

```bash
# 1. Fresh-clone test (the real test)
cd /tmp && git clone <url> priorart-fresh && cd priorart-fresh
uv sync && docker compose up -d
make eval BENCH=evals/labeled_v300.jsonl
# All 3 configs run, MRR on hybrid_rrf ≥ 0.7, FPR-on-novel ≤ 0.15

# 2. Dagster is alive
open http://localhost:13002
# All assets materialize, schedule shows up, sensor is configured

# 3. The Actions workflow exists
gh workflow list
# "Eval Regression" is listed
gh workflow run eval-regression.yml
# Workflow completes, leaderboard is regenerated

# 4. The demo works
asciinema play docs/assets/demo.cast
# 2 min, end-to-end flow

# 5. The README is portfolio-ready
open http://localhost:15173  # or the live URL
# Frontend renders, leaderboard shows real numbers, architecture diagram is clear
```

If any of those fail, Phase 3 isn't done. The project isn't portfolio-ready.
