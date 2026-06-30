# PriorArt

> Startup-idea deduplication against the public YC + Product Hunt + HN
> corpus, with a reproducible eval harness and a labeled benchmark.

![PriorArt leaderboard — Phase 2, 3 configs on labeled_v300.jsonl](docs/assets/leaderboard-v2.png)

A self-hosted web service. Paste an idea, get a ranked list of similar
past launches, a Pydantic-validated structured comparison for the top
competitors, and a market-scope signal. The retrieval is benchmarked
against a hand-labeled 300-idea benchmark drawn from the public
corpus — see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the
metric definitions and the [live leaderboard CSV](results/leaderboard.csv)
for the current numbers.

---

## The CV line

> Built an end-to-end production-grade startup-idea deduplication and
> competitor-research service: pgvector + bge-m3 retrieval, Pydantic-
> validated LLM structured outputs, multi-step Temporal workflows with
> web-search fallback, Dagster-managed corpus ingestion, Langfuse
> observability, and a reproducible MLflow-tracked evaluation harness
> (MRR / nDCG@K / calibration) over a labeled public-corpus benchmark.

This is the public-safe evolution of the Mercedes-Benz thesis
(LLM-based vector search, structured JSON outputs, PG vector,
similarity metrics, retrieval@K). The thesis was internally scoped;
this project is the same engineering pointed at a public problem,
with a public corpus and a reproducible benchmark behind it.

---

## Status

**Phase 1 ✓ shipped.** Working idea-lookup API on `localhost:18001`,
Postgres + pgvector in Docker, 5,949-company YC corpus embedded with
`BAAI/bge-m3`, 100-record labeled benchmark, eval harness computing
MRR / nDCG@10 / precision@5 / recall@10 / FPR-on-novel, Vite + React
19 + shadcn/ui dark-mode frontend on `localhost:15174`.

**Phase 2 ✓ shipped (architecture + leaderboard v2 in this card).**
Temporal `IdeaAnalysisWorkflow` orchestrates the per-idea pipeline with
retry + web-search fallback + low-confidence signal channel (Phase 2.1+2.2).
Langfuse v2 SDK wraps every LLM call with the full metadata set
(Phase 2.3). MLflow self-hosted on port 15000 tracks every eval run
with 9 params + 6 metrics + 4 artifacts (Phase 2.4). Corpus expanded
to 10,983 records (YC + Product Hunt + HN; Phase 2.5–2.7). Eval set
expanded to 300 (Phase 2.8). Three retrieval configs in the leaderboard:
dense bge-m3, BM25, hybrid RRF (Phase 2.9). Web-fallback activity
verified live (Phase 2.10). Architecture diagram + this README update
(Phase 2.11 — current card). **Phase 2 review (2.12) is the gate.**

**Phase 1 acceptance gate:** MRR ≥ 0.50 on the 100-idea labeled
benchmark. **Current:** MRR = 0.559 ✓. FPR-on-novel cap of 0.15 is
not yet met on the dense-only config — see
[`docs/METHODOLOGY.md` § Limitations](docs/METHODOLOGY.md#limitations)
for the honest read and the Phase 2 reranker / hybrid lever.

**Phase 2 acceptance gate** (per [`docs/PHASE-2.md`](docs/PHASE-2.md)
§ Definition of Done): 3 retrieval configs in the leaderboard, all
with MRR ≥ 0.6 (dense ≥ 0.75 target). **Current on labeled_v300.jsonl:**
dense=0.567, bm25=0.392, hybrid_rrf=0.458. **None clears the dense ≥ 0.75
target.** See **Limitations** below — the eval set is LLM-generated v2
and the targets are INFORMATIONAL until the hand-label pass lands.

---

## Quickstart

```bash
# 1. Clone + install
git clone <repo-url> priorart && cd priorart
uv sync

# 2. Start Postgres + pgvector
docker compose up -d

# 3. Ingest the YC snapshot + embed (one-time, ~5 min)
make scrape
make ingest

# 4. Start the API
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 18001

# 5. Start the frontend
cd src/frontend && pnpm install && pnpm dev

# 6. Reproduce the leaderboard
make eval
```

Then open `http://localhost:15174` (frontend) or hit
`http://localhost:18001/healthz` (API).

The full eval-leaderboard CSV lands at `results/leaderboard.csv` after
`make eval`. The screenshot above is rendered from that CSV by
`scripts/render_leaderboard_screenshot.py` — the numbers in the image
match the CSV to the digit.

---

## Architecture (one screen)

![PriorArt system architecture — Temporal workflow, Dagster (Phase 3, dashed), Langfuse, MLflow, pgvector](docs/assets/architecture.png)

**Phase 2 layers** Temporal (per-idea workflow), Langfuse (LLM
observability), and MLflow (experiment tracking) on top of the
existing FastAPI + pgvector core. **Phase 3** layers Dagster
(dashed amber boundary in the diagram) for batch data platform —
scheduled nightly re-embedding and the `config_change` sensor that
fires the eval regression. The boundary between Phase 1 (retrieval),
Phase 2 (workflow + observability), and Phase 3 (data platform) is
deliberate: each phase proves its abstraction before the next layer
earns its keep.

### Temporal workflow walkthrough (brief)

The `IdeaAnalysisWorkflow` (Phase 2.1) sequences six activities with
explicit retry policies and a low-confidence signal channel:

1. **`embed_idea`** — `BAAI/bge-m3` (1024-dim). Retry: 3 attempts, exp backoff.
2. **`ann_search`** — pgvector HNSW over `company_embeddings`. Top-K=20.
3. **Confidence band check** — if top-1 cosine < 0.55, the workflow
   parks waiting on the `review` signal (low-confidence human review).
4. **`llm_compare_topk`** — Claude Sonnet 4.5 via `instructor` +
   Pydantic v2. Retry: 3 attempts, **non-retryable** on schema violation.
5. **Web fallback** (Phase 2.2) — if top-1 cosine < 0.40, SearXNG →
   Firecrawl → re-embed the top-3 scrape results → re-run ANN.
6. **`market_scope_signal`** + **`assemble_verdict`** — cheap local
   call (Qwen 2.5 32B or MiniMax-M3) + pure-function assembly into
   the Pydantic `IdeaVerdict`.

Every LLM call is wrapped in a Langfuse trace (Phase 2.3) with
embedding latency, ANN latency, top-K IDs, model version, prompt
template version, and token cost as metadata. Every eval run logs a
complete MLflow run (Phase 2.4) with 9 params + 6 metrics + 4
artifacts (prompt template as `mlflow.log_text`).

**For the full Temporal + Dagster boundary rationale, the Python
walkthrough, the asset lineage, and the production migration path,
read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).**

### Phase 2 observability assets

| Layer | URL | What it shows |
|---|---|---|
| **Temporal UI** | `http://localhost:8233` | Per-idea workflow runs, activity history, retry events, signal channels |
| **Langfuse UI** | `http://localhost:13000` | LLM traces with full metadata (Phase 2.3 ships the wrapper; **literal UI screenshot pending live `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`** — see OPERATIONS.md § Failure modes) |
| **MLflow UI** | `http://localhost:15000` | Experiment list, runs with params+metrics, compare view (Phase 2.4 ships the wrapper + 3 FINISHED runs in `phase-2-baseline`; **literal compare-view screenshot pending an operator with a display / headless browser**) |

---

## The competitive landscape

Three categories of existing tools, each missing something PriorArt has.
Full table in [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md).

| Category | Examples | What they do | What they don't |
|---|---|---|---|
| AI-wrapper idea validators | Siftt, IdeasGPT, ValidatorAI, Sprintbase | Fast UX, thin LLM + Google search. | No curated corpus, no eval harness, no structured comparison. Vibes, not evidence. |
| Market-intelligence platforms | Crunchbase Pro, Pitchbook, CB Insights, SEMrush, Ahrefs | Real data, investor-grade. | Paywalled at the level you need. Investor-facing, not founder-facing. |
| Internal accelerator tooling | YC, a16z, Antler, Techstars | The real production version. | Locked behind NDAs. |
| Generic LLM eval libraries | DeepEval, RAGAS, TruLens | Industry-standard metrics. | Generic — no domain benchmark. |

**The gap:** no public tool does **idea → vector dedup against a
labeled public corpus → structured LLM comparison → market-scope
signal → reproducible eval harness**, end-to-end.

---

## Eval leaderboard (Phase 2)

![PriorArt eval leaderboard v2 — 3 retrieval configs on labeled_v300.jsonl](docs/assets/leaderboard-v2.png)

The leaderboard image above is rendered **directly from
[`results/leaderboard.csv`](results/leaderboard.csv)** by
[`scripts/render_leaderboard_v2_screenshot.py`](scripts/render_leaderboard_v2_screenshot.py).
The numbers in the image match the CSV to the digit. One row per
config — the `selected_threshold` (MRR-max per config; the runner
falls back when no threshold meets the FPR ≤ 0.15 cap).

To regenerate after `make eval`:

```bash
make screenshot-v2    # writes docs/assets/leaderboard-v2.png
```

The full per-threshold sweep lives at
[`results/leaderboard.md`](results/leaderboard.md) (one section per
config with the threshold-sweep table). The `eval.duckdb` file is the
single-file DuckDB queryable view of every run — see
[`docs/EVAL.md`](docs/EVAL.md) for the schema and the queries that
derive the leaderboard.

**Honest read on the numbers:** the v300 leaderboard shows dense
(MRR=0.567) beating hybrid_rrf (0.458) and bm25 (0.392). The
**FPR-on-novel cap of 0.15 is not cleared by any config** — see
**Limitations** below for why and what closes the gap.

---

## How to add a retrieval config

The eval runner is config-driven. Adding a new retrieval config
(BM25, hybrid RRF, Cohere rerank, etc.) is three steps:

1. **Write the config YAML.** Drop a sibling of
   [`configs/dense_bge_m3.yaml`](configs/dense_bge_m3.yaml) into
   `configs/`. The schema is documented at the top of that file.

   ```yaml
   # configs/bm25.yaml
   name: bm25
   api_url: http://localhost:18001/search?config=bm25
   top_k: 20
   notes: Sparse BM25 retrieval, no embeddings.
   ```

2. **Wire the API to honor the new config.** Add a `ConfigName` enum
   (or whatever your router uses) and branch on it in
   `POST /search`. Phase 1 ships the dense-bge-m3 path; Phase 2 adds
   `bm25`, `hybrid_rrf`, `cohere_rerank` as siblings.

3. **Re-run the eval.** `python -m eval.run --config configs/bm25.yaml`
   appends a new row to `results/leaderboard.csv`. Compare against the
   dense-bge-m3 row in `docs/METHODOLOGY.md`.

The eval runner overwrites the DuckDB on each run (latest view) but
appends to the CSV (history). This is the regression suite — when
Phase 3 lands the GitHub Actions check, MRR dropping below 0.5 on
`main` fails the build.

---

## Limitations

Be honest about what this is and what it isn't.

- **Eval set v2 is LLM-generated, not hand-labeled. MRR targets in
  PHASE-2.md §Definition-of-done are INFORMATIONAL until the
  hand-label pass lands.** Per the Phase 1.5a fix (commit c8aa1fb)
  and Phase 2.8 (card `t_36650c8c`), `evals/labeled_v300.jsonl` was
  generated by `claude-minimax-m3` with explicit provenance
  (`labeler=ai-assisted-claude-minimax-m3`,
  `provenance=llm-generated-v2-pending-anurag-hand-review`) recorded
  on every record. The Phase 1 100-record hand-labeled subset is
  the gold-standard floor; the v2 expansion is the operational
  benchmark until the hand-label pass on the 200 new triples
  completes. **Don't quote MRR numbers from this leaderboard as if
  they were on a hand-labeled set.** Phase 3 (or Anurag's hand-label
  pass) closes this gap.
- **Public corpus only.** Internal accelerator tooling sees the real
  production version of this problem. PriorArt sees the public slice.
- **10,983-company merged corpus** (YC + Product Hunt + HN). Per-source
  counts in `/healthz`: `yc=5949`, `producthunt=4000`, `hn=993`. PH
  and HN sources are Phase 2.5+2.6+2.7 work; the merge dedups on name
  cosine ≥ 0.85 with a borderline review queue at 0.75–0.85.
- **FPR-on-novel cap not met on any config.** No cosine threshold on
  `[0.50, 0.80]` clears the 0.15 cap. On `labeled_v300.jsonl`:
  dense_bge_m3 (best at threshold 0.8) gives MRR=0.567 with
  FPR=0.79; hybrid_rrf gives MRR=0.458 with FPR=0.63; bm25 gives
  MRR=0.392 with FPR=1.00. The Phase 3 calibration curve +
  per-category FPR breakdown is the path to understanding *where*
  each config fails (not just *that* it fails).
- **None of the 3 configs clears the PHASE-2.md MRR ≥ 0.6 / dense ≥ 0.75
  target.** Even setting aside the LLM-generated eval set caveat
  above, the dense config drops from 0.559 on labeled_v100 to 0.567
  on labeled_v300 — and never gets close to 0.75. Hybrid RRF
  (MRR=0.458) underperforms dense on this benchmark despite the
  cross-list coverage boost; a per-query failure analysis is the
  Phase 3 lever (the eval set is needed hand-labeled first before
  failure-mode classification is trustworthy). The right Phase 3
  work is the per-source FPR breakdown + calibration curve, not
  adding more configs.
- **No LLM-comparison eval yet.** The structured LLM call in
  `/ideas/analyze` exists and is Langfuse-traced (Phase 2.3) but
  isn't scored against ground-truth verdicts. Phase 3 ships the
  LLM-as-judge harness (deferred from Phase 2.10 by scope discipline).
- **Literal Langfuse + MLflow UI screenshots are deferred.** The
  wrappers are fully wired (Phase 2.3 + 2.4) and unit-tested, but
  producing a literal screenshot of the Langfuse UI requires
  `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` + `ANTHROPIC_API_KEY`
  in `.env` to fire a real trace; the MLflow compare-view screenshot
  requires a headless browser to navigate `localhost:15000`. Both
  are operator tasks, not code work — see
  [`docs/OPERATIONS.md` § Common failure modes](docs/OPERATIONS.md#common-failure-modes).
- **Demo, not SaaS.** Single-tenant, self-hosted, public-data only.

The full limitations list lives in
[`docs/METHODOLOGY.md` § Limitations](docs/METHODOLOGY.md#limitations).

---

## The phase plan

|| Phase | Weekend | Goal | Tier |
||---|---|---|---|
|| **Phase 1** ✓ | 1 | Working idea-lookup API + UI + 100-idea labeled benchmark + 5 metrics, shipped by Sunday night. | Must-be |
|| **Phase 2** ✓ (pending 2.12 review) | 2 | Temporal workflow + Langfuse + MLflow + 3 retrieval configs (dense / BM25 / hybrid RRF) + corpus expansion to 10,983 + eval set expansion to 300. | Should-be |
|| **Phase 3** | 3 | Dagster + calibration curve + FPR-on-novel breakdown + GitHub Actions regression + LLM-comparison eval + Dagster-managed config_change sensor + hand-label pass on v300. | Can-be |

Hard rule: Phase 1 must be done before Phase 2 starts. Phase 2 must
be done before Phase 3 starts. Each phase doc in
[`docs/PHASE-*.md`](docs/) is the detailed task breakdown.

**Honest progress on the CV line:** the words *"multi-step Temporal
workflows with web-search fallback"*, *"Dagster-managed corpus
ingestion"*, *"Langfuse observability"*, and *"reproducible MLflow-
tracked evaluation harness (MRR / nDCG@K / calibration)"* all need
to be defensible to ship this project. Phase 2 covers the first
three fully (Temporal + Langfuse + MLflow all shipped). Dagster is
Phase 3. The calibration curve is Phase 3. Until those land, the
CV claim is one phase short.

---

## Documentation

| File | What's in it |
|---|---|
| [`SPEC.md`](SPEC.md) | The full project spec — 12 sections, landscape table, phase plan, definition of done. Start here. |
| [`AGENTS.md`](AGENTS.md) | Onboarding notes for AI agents and humans. Project structure, hard rules, things that look like bugs but aren't. |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | What each metric means, how the labeled benchmark was constructed, the label policy, limitations. |
| [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md) | The competitive landscape, expanded. "No public tool does all of this." |
| [`docs/EVAL.md`](docs/EVAL.md) | Deep-dive on the eval harness. The single most important part of this project. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Why Temporal + Dagster both, the boundary between them. |
| [`docs/PHASE-1.md`](docs/PHASE-1.md) — [`PHASE-3.md`](docs/PHASE-3.md) | Per-weekend task breakdowns. |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Local dev-loop walkthrough, common failure modes, how to add a retrieval config. |

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).