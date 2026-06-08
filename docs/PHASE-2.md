# Phase 2 — The "Production-Grade" Weekend (Weekend 2)

> **Tier:** Should-be. **Goal:** Take the Phase 1 prototype and turn it into a system that reads as a senior-engineered production platform. **Temporal** orchestrates the per-idea workflow. **Langfuse** traces every LLM call. **MLflow** tracks experiments. The corpus expands to YC + Product Hunt + HN. The eval set expands to 300. The leaderboard compares 3 retrieval configs (dense / BM25 / hybrid). This is the weekend the project earns the MLOps story on the CV.

---

## What ships at the end of Phase 2

- [ ] Temporal workflow (`IdeaAnalysisWorkflow`) replacing the FastAPI handler's inline orchestration. Retry, fallback, human-in-the-loop signal channel all wired.
- [ ] Web-search fallback activity (SearXNG via your self-hosted instance, or Brave Search API as a fallback).
- [ ] Langfuse traces on every LLM call, with metadata for embedding latency, ANN search latency, top-K IDs, verdict, total latency, token cost.
- [ ] MLflow self-hosted (Docker, SQLite). Embedding-model / threshold / prompt-template A/B runs logged.
- [ ] Corpus expanded: YC + Product Hunt archive (5K top-upvoted launches) + HN "Show HN" posts (paginated via Algolia).
- [ ] Eval set expanded to 300 hand-labeled triples (100 duplicate, 100 novel, 100 adversarial).
- [ ] 3 retrieval configs in the leaderboard: dense (bge-m3), BM25, hybrid (RRF).
- [ ] The leaderboard CSV + Markdown leaderboard committed. Per-config MRR / nDCG@K / precision@K / recall@K / FPR-on-novel.
- [ ] README updated with: the Temporal/Dagster/MLflow/Langfuse architecture diagram, the new leaderboard, the expanded corpus section.

**Definition of done:** Temporal UI shows the per-idea workflow running end-to-end with traces. Langfuse dashboard shows the LLM call with metadata. MLflow UI shows the experiment sweep. The leaderboard CSV compares 3 configs on the 300-idea benchmark, all with MRR ≥ 0.6 (the dense config should hit 0.75+).

---

## Detailed task breakdown

Each task is sized to ~2–4 hours. Total Phase 2 budget: ~20 hours (slightly more than Phase 1 because Temporal + Langfuse + MLflow all have a learning-curve tax).

### 2.1 — Temporal dev environment (1.5h)
- [ ] `pip install temporalio`. Add to `pyproject.toml`.
- [ ] `temporal server start-dev` (local dev). Verify the UI at `localhost:8233`.
- [ ] `src/workflow/` skeleton: `worker.py`, `workflows.py`, `activities.py`, `shared.py` (for dataclasses / types).
- [ ] First workflow: `IdeaAnalysisWorkflow` with activities `embed_idea`, `ann_search`, `llm_compare_topk`, `market_scope_signal`, `assemble_verdict`. **Port the existing Phase 1 logic into Temporal activities verbatim** — no behavior changes in this step.
- [ ] FastAPI handler becomes a Temporal client: `POST /ideas/analyze` → `await client.start_workflow(IdeaAnalysisWorkflow.run, ...)` → returns the workflow ID + a handle.
- [ ] **Add a `GET /workflows/{id}` endpoint** that returns the workflow status (Temporal `describe_workflow`).
- [ ] **Verify end-to-end:** start a workflow, watch it run in the Temporal UI, see the LLM call land in Langfuse (added in 2.3).

### 2.2 — Retry + fallback semantics (2h)
- [ ] Activity-level retry policies: exponential backoff, max 3 attempts on transient LLM failures (5xx, rate limit), no retry on schema-violation (fail fast, surface the error).
- [ ] `web_fallback_if_empty` activity: if top-K from the corpus returns nothing above the threshold, run a SearXNG search, scrape the top-3 results via Firecrawl (your self-hosted instance), embed them, re-run ANN search. This is the "corpus is too narrow, go to the web" loop.
- [ ] Add a Temporal **signal channel** for low-confidence verdicts (cosine similarity in 0.55–0.70 band, or LLM self-reported confidence < 0.7). The workflow parks in a "waiting for human" state. A simple admin endpoint `POST /workflows/{id}/signal/review` resumes it with a corrected verdict (or "confirm as-is").
- [ ] **Verify:** trigger a workflow with a deliberately-novel idea (one of the eval-set's "novel" examples) and watch the web fallback fire in the Temporal UI.

### 2.3 — Langfuse integration (1.5h)
- [ ] You already have Langfuse on 13000/13001. Add `langfuse` to `pyproject.toml`.
- [ ] Wrap the LLM call in `langfuse_context` with: `name="idea-compare"`, `input=idea + top-3`, `output=CompetitorVerdict list`, `metadata={"embedding_latency_ms", "ann_search_latency_ms", "top_k_ids", "model_version", "prompt_template_version", "token_cost_usd"}`.
- [ ] Add Langfuse scoring: a `user_feedback` field on the trace (left empty for now; populated in Phase 3 by the UI's "thumbs up/down").
- [ ] **Verify:** every LLM call shows up in the Langfuse UI with the metadata. Take a screenshot for the README.
- [ ] **Watch out for the redaction filter.** Langfuse keys go in `.env` (not committed). Use `OR_AUTH` / `OR_TOKEN`-style env var names where the value isn't actually a secret to avoid the harness redaction trip-ups.

### 2.4 — MLflow (1.5h)
- [ ] Add `mlflow` to `pyproject.toml`. Self-host: `docker run -d -p 5000:5000 -v mlflow-data:/mlflow-data ghcr.io/mlflow/mlflow mlflow server --backend-store-uri sqlite:///mlflow-data/mlflow.db --default-artifact-root /mlflow-data/artifacts --host 0.0.0.0`. Port 5000 collides with Honcho — use 15000 instead.
- [ ] `mlflow ui` shows the experiment list at `localhost:15000`.
- [ ] Wrap the eval harness: every `eval.run` invocation logs a run with params (`embedding_model`, `threshold`, `prompt_template_version`, `corpus_snapshot_date`) and metrics (the 4 metrics + FPR-on-novel).
- [ ] `python -m eval.run --experiment-name "phase-2-baseline" --config configs/dense_bge_m3.yaml` writes a run to MLflow. Re-run with `bm25` config, get a second run. The MLflow UI compare view shows the side-by-side.
- [ ] **Verify:** MLflow UI shows ≥ 3 runs (one per config) with comparable metrics. Take a screenshot for the README.

### 2.5 — Product Hunt archive scraper (2.5h)
- [ ] Product Hunt's `/posts` pages are paginated and public. Scrape the top 5K most-upvoted launches over the last 3 years: name, tagline, description, categories, votes count, comments count, URL.
- [ ] Output: `data/snapshots/producthunt_<date>.jsonl` with manifest.
- [ ] Dedup against YC by name similarity (cosine on bge-m3 embeddings of names, dedup if ≥ 0.85). Manual review queue for the 5–10% borderline cases.
- [ ] Idempotent. Re-running on the same date produces a byte-identical file.

### 2.6 — HN "Show HN" scraper (1.5h)
- [ ] HN Algolia API: `https://hn.algolia.com/api/v1/search?query=show%20hn&tags=story&numericFilters=points>50`. Paginate.
- [ ] For each post: title, URL, points, comments, author, date, the linked external URL (that's the "launch" — what we're indexing).
- [ ] Scrape the linked external URL via Firecrawl for a 1-paragraph description. If the URL is dead, skip.
- [ ] Output: `data/snapshots/hn_show_<date>.jsonl` with manifest.

### 2.7 — Ingestion for the expanded corpus (1.5h)
- [ ] Re-run ingestion on YC + Product Hunt + HN. Three sources, one `companies` table (with a `source` column to disambiguate). HNSW index rebuilt.
- [ ] `corpus_count` in `/healthz` reflects the merged count.
- [ ] Update `models.yaml` with the snapshot dates and the source counts.

### 2.8 — Eval set v2: 300 triples (3h, do this early in the weekend)
- [ ] Expand `evals/labeled_v100.jsonl` → `evals/labeled_v300.jsonl`. Balanced 100/100/100 across duplicate / novel / adversarial.
- [ ] The new 200 triples: 50 from Product Hunt (paraphrasings of top launches), 50 from HN (paraphrasings of top "Show HN" posts), 100 mixed-source adversarial.
- [ ] **Hand-labeled. No LLM.** Same policy as Phase 1.
- [ ] Update `evals/labeled_v300.README.md` with the new label policy + per-source breakdown.

### 2.9 — BM25 + Hybrid configs (1.5h)
- [ ] `configs/bm25.yaml` — `rank_bm25` over the merged corpus, default `k1=1.5, b=0.75`.
- [ ] `configs/hybrid_rrf.yaml` — dense (bge-m3) + BM25 fused with Reciprocal Rank Fusion (k=60). The RRF implementation: query both retrievers, merge by `1/(k+rank)` score.
- [ ] Run the eval harness against all 3 configs. Commit `results/leaderboard.csv` with 3 rows.

### 2.10 — Leaderboard regeneration + UI tweak (1h)
- [ ] The frontend now shows the active config (default: hybrid) and a small badge indicating which retrieval config produced the result.
- [ ] The README's leaderboard screenshot is regenerated from `results/leaderboard.csv`.
- [ ] Add a `docs/assets/leaderboard-v2.png` (or replace `v1.png`).

### 2.11 — Architecture diagram + README update (1.5h)
- [ ] Draw the architecture diagram from `SPEC.md` (the Temporal / Dagster / MLflow / Langfuse / pgvector block) as an SVG or a clean Mermaid diagram. Save as `docs/assets/architecture.png`.
- [ ] Update README with: the diagram, the Temporal workflow walkthrough, the Langfuse + MLflow screenshots.
- [ ] `docs/ARCHITECTURE.md` — full write-up of why Temporal handles the per-idea workflow and Dagster handles the batch data platform. Include the boundary rationale.

### 2.12 — Smoke test + commit (30min)
- [ ] `make eval` runs against the 300-idea benchmark with all 3 configs.
- [ ] Temporal UI: start a workflow, watch it complete, see the trace in Langfuse, see the run in MLflow.
- [ ] Commit + push to `main`.

---

## What is NOT in Phase 2 (deferred to Phase 3)

- Dagster assets for corpus ingestion — Phase 3.
- Calibration curve + per-category failure breakdown — Phase 3.
- GitHub Actions regression suite — Phase 3.
- asciinema demo — Phase 3.
- Cohere rerank as a 4th opt-in config — Phase 3 (or skip entirely; the user can add it later).
- Polish on the UI (loading states, error toasts, etc.) — Phase 3.

---

## Pitfalls (Phase 2 specific)

- **Do not skip the Temporal signal channel.** Low-confidence verdicts parking in a "human review" state is the *production* pattern. Without it, Temporal is just a fancy async FastAPI.
- **Do not let the web fallback silently become the primary path.** The whole point of the curated YC + PH + HN corpus is that it's higher-quality than the open web. The fallback should fire < 10% of the time on the eval set. If it fires more, the corpus or the threshold needs work.
- **Do not use Langfuse's hosted SaaS.** Self-host. The point is reproducibility.
- **Do not put MLflow in the same container as the API.** Separate container, separate concerns. MLflow is a platform service, not a per-request service.
- **Do not log the LLM prompt + the full top-K descriptions to MLflow as params.** Params are for hyperparameters. Log the prompt to the run artifacts (`mlflow.log_text`), not the params.
- **Do not expand the eval set with LLM-generated labels.** Hand-label the 200 new triples. The same pitfall from Phase 1 applies twice as hard now because the labels are being compared across configs.
- **Do not add Dagster in Phase 2.** The user explicitly asked for "Temporal + Dagster both," but Dagster is Phase 3 work. Adding it now is scope creep.

---

## Verification at end of Phase 2

```bash
# 1. Temporal is up and the workflow runs
temporal server start-dev &
curl -X POST http://localhost:18000/ideas/analyze \
  -H "Content-Type: application/json" \
  -d '{"idea": "AI-powered contract review for SMB law firms"}'
# returns a workflow_id
curl http://localhost:18000/workflows/<id>   # status: "completed"

# 2. Temporal UI shows the workflow
open http://localhost:8233
# Workflow completes, all activities are visible, retries (if any) are logged

# 3. Langfuse shows the LLM call
open http://localhost:13000
# Trace "idea-compare" with full metadata

# 4. MLflow shows the experiment runs
open http://localhost:15000
# 3+ runs (one per config), compare view works

# 5. Eval harness runs clean against 300 triples
make eval BENCH=evals/labeled_v300.jsonl
head results/leaderboard.csv
# 3 rows, dense MRR ≥ 0.7, hybrid MRR ≥ 0.65, bm25 MRR ≥ 0.5

# 6. The corpus is bigger
curl http://localhost:18000/healthz
# corpus_count should be ~10-15K (YC + PH + HN)
```

If any of those fail, Phase 2 isn't done.
