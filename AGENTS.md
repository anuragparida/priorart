# AGENTS.md

> Onboarding notes for AI agents (and humans) working on **PriorArt** — a startup-idea deduplication & competitor-research service. This is a *production-grade ML/AI platform* project, not a wrapper script. Treat it that way.

## What this is

A self-hosted web service that takes a free-text startup idea as input and returns:

1. A ranked list of similar past launches from a public corpus (Y Combinator directory, Product Hunt archive, Hacker News "Show HN" posts).
2. A Pydantic-validated **structured comparison** for the top competitors (similarity axes, key differences, likely failure modes, evidence links).
3. A **market-scope signal** (wide-open / crowded-but-growing / saturated / niche-but-real) — explicitly labeled as *directional*, not a SEMrush replacement.
4. A **reproducible evaluation harness** that benchmarks retrieval quality against a labeled public-corpus benchmark (MRR, nDCG@K, precision@K, recall@K, calibration curve, FPR-on-novel).

The system is built like a production ML/AI platform: **Temporal** orchestrates the per-idea workflow (retry, fallback, human-in-the-loop), **Dagster** orchestrates the batch data platform (corpus ingestion, nightly re-embedding, eval-harness regression on config change), **pgvector** is the actual store, **MLflow** tracks experiments, **Langfuse** traces LLM calls.

**Public-spec source:** `../project_ideas/14_idea_competitor_dedup.md` (also inlined here as `SPEC.md`).
**Repository (planned):** TBD — not yet created. Stay in this folder until greenlit.
**Default branch:** `main` (no feature branches, matching your cross-project convention).

---

## Why this project exists (one-line CV claim)

> Built an end-to-end production-grade startup-idea deduplication and competitor-research service: pgvector + bge-m3 retrieval, Pydantic-validated LLM structured outputs, multi-step Temporal workflows with web-search fallback, Dagster-managed corpus ingestion, Langfuse observability, and a reproducible MLflow-tracked evaluation harness (MRR / nDCG@K / calibration) over a labeled 300-idea benchmark drawn from the public YC + Product Hunt corpus.

This is the **direct, public-safe evolution of the Mercedes-Benz thesis** (LLM-based vector search, structured JSON outputs, PG vector, similarity metrics, retrieval@K). The thesis was internally scoped; this project is the same engineering pointed at a public problem, with a public corpus and a reproducible benchmark behind it. Don't lose sight of that line — it's the whole point.

---

## Tech stack (only what matters)

### Backend & data
- **`uv`** + Python 3.12 + FastAPI + SQLAlchemy 2.x + Pydantic v2.
- **Postgres 16 + pgvector** in Docker. HNSW index, `vector(1024)` for `bge-m3`. Three tables: `companies` (metadata), `company_embeddings` (vectors), `eval_runs` (leaderboard history).
- **`BAAI/bge-m3`** for embeddings (local via sentence-transformers; multilingual for European expansion). Alternative: `text-embedding-3-small` if you want to skip local model load.
- **BM25** via `rank_bm25`. **Hybrid** via Reciprocal Rank Fusion. **Optional: Cohere rerank** as a 4th config.
- **DuckDB** (single file) for eval-harness results — queryable, easy to commit, zero infra.

### Orchestration & MLOps
- **Temporal.io** (`temporal server start-dev` for local). Models the per-idea workflow with retry / fallback / human-in-the-loop.
- **Dagster** for the batch data platform (corpus ingestion assets, nightly re-embedding schedule, config-change sensor firing the eval regression).
- **MLflow** self-hosted (Docker, SQLite backend). Tracks embedding-model versions, threshold sweeps, prompt-template A/B tests.
- **Langfuse** (you already have it on 13000/13001) — wraps every LLM call. Trace dashboard becomes the "production observability" demo.

### LLM
- **Anthropic Claude Sonnet 4.5** for the structured-comparison call (good at long, nuanced comparisons).
- **MiniMax-M3** or a local **Qwen 2.5 32B** for cheap / fast calls (market-scope classification).
- **Constrained generation:** Pydantic v2 + `instructor` (or `outlines`) for deterministic JSON output.

### Frontend
- **`pnpm`** + Vite + TypeScript + React 18 + Tailwind + shadcn/ui. **Dark mode by default.** One page: idea input → ranked competitors + structured verdicts + market-scope + evidence links.

### Web search fallback (Phase 2)
- **Brave Search API** (generous free tier) or **SerpAPI** (free quota). Local-first alternative: your self-hosted **SearXNG** via the `self-hosted-firecrawl-hermes` skill. Scraped pages go through your self-hosted **Firecrawl** for clean markdown extraction.

### CI
- **GitHub Actions for the eval-harness regression only** (per-config change → run eval → post leaderboard diff as PR comment). Public repo, so Actions are fine here. **Do not** add a deploy pipeline — the user handles deploy separately.

---

## File map (read these first)

```
priorart/
├── AGENTS.md                    ← you are here
├── README.md                    ← top-level orientation
├── SPEC.md                      ← the full project_ideas/14 spec, inlined for repo-self-containedness
└── docs/
    ├── PHASE-1.md               ← Weekend 1, ship by Sunday
    ├── PHASE-2.md               ← Weekend 2, the "production-grade" weekend
    ├── PHASE-3.md               ← Weekend 3, the cherry
    ├── EVAL.md                  ← deep-dive on the eval harness, metrics, benchmark construction
    ├── ARCHITECTURE.md          ← diagrams, data flow, Temporal/Dagster boundary rationale
    ├── LANDSCAPE.md             ← competitive-landscape table (expanded from SPEC)
    └── OPERATIONS.md            ← Temporal + Dagster dev-loop walkthrough, prod migration path
```

Once Phase 1 starts, add:

```
src/
├── api/                         ← FastAPI routes
├── workflow/                    ← Temporal workflows + activities
├── data/                        ← SQLAlchemy models, pgvector queries, embeddings
├── llm/                         ← Claude / local model wrappers, Pydantic schemas
├── eval/                        ← eval harness (regression runner + leaderboard)
└── frontend/                    ← Vite + React + shadcn/ui

data/
├── snapshots/                   ← versioned corpus snapshots (yc_<date>.jsonl, etc.)
└── eval/                        ← labeled triples

configs/                         ← one YAML per retrieval configuration
models.yaml                      ← model registry (which embedding, which LLM, which prompt)
docker-compose.yml               ← postgres + pgvector + temporal + dagster + langfuse + mlflow + api + frontend
```

---

## Critical "where to start" pointers

### 1. **Read the spec first** → `SPEC.md`
This is the source of truth. Every phase doc references back to it. If something in `docs/PHASE-*.md` contradicts `SPEC.md`, **SPEC.md wins**.

### 2. **Read the eval doc carefully** → `docs/EVAL.md`
The eval harness is the single most important part of this project. The whole CV claim depends on it. **Do not write the eval set after the system** — write it first, then build against it. Otherwise you rationalize the benchmark to match the system's outputs and lose the entire artifact.

### 3. **Read the architecture doc** → `docs/ARCHITECTURE.md`
Understand *why* Temporal and Dagster each have their job, and the boundary between them. Don't merge them "to save time" — the boundary is the signal.

### 4. **Don't start building** until Anurag signs off on the spec and phase plan.
This folder is **documentation-only as of now**. Anurag will review tomorrow and decide on the build approach. No `uv init`, no `docker compose up` until then.

---

## Phase plan (top-level)

The three phases map 1:1 to the `Must / Should / Can` tiers in the spec. Read each phase doc for the detailed task breakdown.

| Phase | Weekend | Goal | Tier |
|---|---|---|---|
| **Phase 1** | 1 | Working idea-lookup API + UI + 100-idea labeled benchmark + 4 metrics, shipped by Sunday night. | Must-be |
| **Phase 2** | 2 | Temporal workflow + Langfuse + MLflow + 300-idea benchmark + 3 retrieval configs. The "production-grade" weekend. | Should-be |
| **Phase 3** | 3 | Dagster + calibration curve + FPR-on-novel + GitHub Actions regression + polished README + asciinema. | Can-be |

**Hard rule:** Phase 1 must be done before Phase 2 starts. Phase 2 must be done before Phase 3 starts. No backsliding — don't add a Dagster asset in Phase 1 to "save time later." The phases are scope-separation for a reason.

---

## What the user cares about (so you don't accidentally regress)

Honored from `anuragparida.com/AGENTS.md` and the workspace-level `AGENTS.md`:

1. **Work on `main`, no feature branches.** (Cross-project convention.)
2. **No HDI internal data, no Mercedes internal data, no employer playbooks.** Public data only. **The whole point of this project is reproducibility and the public claim on the CV.** Hard stop.
3. **Eval before polish.** Every AI-flavored project ships with a real eval set in `evals/`. This is the differentiator for *this* project too — and the eval set is the hardest, most important part.
4. **No token / no credentials in chat.** `~/.config/gh/hosts.yml` already has `gh` CLI auth. Don't ask for tokens.
5. **Stay within the spec.** Don't add libraries, frameworks, or architectural changes without asking. The tech stack in the spec is the tech stack. If something doesn't work as written, surface the trade-off and ask — don't silently swap.
6. **Outcome-led framing.** When you write a phase doc, a task card, a status update — lead with what changed or what's unblocked, not what you did. The user is a rational-feedback, signal-dense writer; match that.
7. **Grill during planning, run autonomously after the plan is locked.** Phase planning = lots of clarifying questions. Building = high autonomy, escalate only on real blockers. (This is the workspace-level rule, repeated for emphasis.)
8. **No "Great question" / "I'd be happy to help" / "Absolutely."** Open with the answer.
9. **The project must run.** Whoever finishes a build card must `docker compose up -d` (or the equivalent) and confirm a user-facing URL returns HTTP 200, then post the URL in the card completion message. Mobile/desktop/embedded targets are exempt. This is the workspace-level hard rule.

---

## ⚠️ Things that look like bugs but aren't (and vice versa)

| You might think… | Reality |
|---|---|
| "Let me skip the eval harness in Phase 1 to ship faster" | The eval harness is the *deliverable*, not a side-effect. Without it, the project is a SaaS wrapper, not a portfolio piece. The 100-idea benchmark + 4 metrics are the Phase 1 floor. |
| "Cohere rerank should be the default config" | Rerank is an API call. The default must work offline. Add it as a 4th *opt-in* config, behind a flag. |
| "Let me build a real market-scope estimator with SEMrush data" | SEMrush / Ahrefs / SimilarWeb are paywalled. The honest version requires a paid API the user doesn't want yet. Build the stub, label it directional, note "future work: integrate SEMrush when budget allows." Don't fake it. |
| "The Temporal workflow is overkill — I'll just do async FastAPI" | Temporal is the **signal**. The CV line says "multi-step Temporal workflows with web-search fallback." If you swap to FastAPI background tasks, the MLOps story collapses. The boundary is the value. |
| "Let me label the eval set with an LLM to save time" | **Do not.** LLM-labeled eval sets rationalize to match the system. Hand-label 300 triples. Spend the time. The user explicitly called this out as a pitfall. |
| "I'll merge Temporal and Dagster to simplify" | Don't. Temporal = per-idea workflow (long-running, retry-heavy, partial-failure-tolerant). Dagster = batch data platform (asset-centric, scheduled, sensor-driven). Each has a defensible job. The boundary is the senior-engineer signal. |
| "The YC directory is too small (~5K) to be a real benchmark" | It's 5K with a known-batch-known-status schema. Quality beats volume for a labeled benchmark. The 300-idea hand-labeled eval is the *test set*; the 5K corpus is the *index*. Different things. |
| "The German-Anki project numbers were placeholder, so these are too" | The numbers you'll see in this project (MRR, nDCG, etc.) are **measured at build time, not invented**. Verify before publishing. The leaderboard CSV is regenerated by the eval harness — it's a real artifact, not a story. |
| "I should add a 'Currently' / 'Now' card to the README" | No. Don't pattern-match from the personal site. This is a project README, not a portfolio card. Different rules. |
| "The user will deploy this for me" | No. The user handles deploy separately. **Do not** add a GitHub Actions deploy pipeline. The regression-suite Action is the *only* Action in this repo. |

---

## Commands (planned — to be confirmed before Phase 1)

```bash
# Phase 1 quickstart (proposed, not yet built)
uv sync
docker compose up -d
python -m eval.run --benchmark evals/labeled_v100.jsonl --output results/leaderboard.csv
pnpm dev                              # frontend
uvicorn src.api.main:app --reload     # backend
```

```bash
# Phase 2 adds
temporal server start-dev             # Temporal dev server
docker compose up -d langfuse mlflow  # observability + experiment tracking

# Phase 3 adds
dagster dev                           # Dagster UI
```

```bash
# Eval regression
make eval              # full eval harness run
make eval-quick        # 30-idea smoke subset, ~2 min
```

---

## Git workflow

```bash
git add -A
git -c user.name="Anurag Parida" -c user.email="anuragparida37@gmail.com" commit -m "..."
git push origin main
```

(No `gh auth login` needed — the host `~/.config/gh/hosts.yml` already has the token.)

This is a public-facing portfolio repo. No `main` force-pushes, no `git reset --hard` to "undo" anything — use `git revert` instead.

### Commit cadence (per workspace `AGENTS.md`)

- **One commit per kanban card, at minimum.** Card ID (`t_xxxxxxxx`) in the
  subject when there is one. Format: `Phase N (card t_xxx): <what changed>`.
- **Commit before yielding.** Don't leave a working tree with uncommitted
  meaningful changes at handoff time.
- **Default to `main`.** No long-lived feature branches — phase branches are
  OK in flight, but merge to `main` and push at the end of each phase.

---

## What success looks like

A new contributor can `git clone`, `uv sync`, `docker compose up`, drop a fresh corpus snapshot into `data/snapshots/`, run `make eval`, and reproduce the leaderboard CSV + HTML to within ±0.01 MRR. The Temporal UI shows the per-idea workflow with traces. The Langfuse UI shows the LLM calls. The MLflow UI shows the experiment sweep. The Dagster UI shows the assets + the config-change sensor firing the regression. The README explains when to pick which retrieval config. The eval harness fails the build when MRR drops below 0.7. Numbers are reproducible from the committed config. Failure analysis points at specific ideas where each config falls down.

**The CV line above is the success metric.** If a change makes any of the words in that line less defensible, it's the wrong change.

---

*Last updated: 2026-06-08, documentation-only scaffold (no build yet). If you find something in this doc that's wrong, fix it — future agents (and future Anurag) will thank you.*
