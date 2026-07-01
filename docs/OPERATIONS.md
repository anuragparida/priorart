# Operations

> How to run the whole thing locally, how to debug the common failure modes, and how to migrate to production. This is the "I know how to operate what I built" document.

---

## Local dev-loop walkthrough

### One-time setup

```bash
# 1. Clone
git clone <repo-url> priorart && cd priorart

# 2. Backend deps
uv sync

# 3. Frontend deps
cd src/frontend && pnpm install && cd ../..

# 4. Start the stack
docker compose up -d
# - postgres (pgvector) on 15432
# - api on 18000
# - frontend on 15173
# (Phase 2+) temporal on 7233
# (Phase 2+) langfuse on 13000/13001
# (Phase 2+) mlflow on 15000
# (Phase 3+) dagster on 13002

# 5. Verify the stack
curl http://localhost:18000/healthz
# {"status": "ok", "db": "ok", "model": "bge-m3", "corpus_count": N}
```

### Daily loop

```bash
# Start the Temporal dev server (Phase 2+)
temporal server start-dev &

# Start the Dagster dev server (Phase 3+)
dagster dev &

# Run the eval harness
make eval

# Start the API (with hot reload)
uvicorn src.api.main:app --reload

# Start the frontend
cd src/frontend && pnpm dev
```

### End-to-end smoke test

```bash
# Health check
curl http://localhost:18000/healthz

# Search
curl -X POST http://localhost:18000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI-powered contract review for SMB law firms", "top_k": 5}'

# Full analysis
curl -X POST http://localhost:18000/ideas/analyze \
  -H "Content-Type: application/json" \
  -d '{"idea": "AI-powered contract review for SMB law firms"}'
# Returns a valid IdeaVerdict JSON
```

### Dagster dev-loop (Phase 3+)

Dagster orchestrates the **batch data platform** (corpus ingestion, nightly re-embedding, config-change regression). The per-idea request path still flows through Temporal — the boundary is documented in `docs/ARCHITECTURE.md`.

#### One-time

Dagster ships in the same `docker-compose.yml` as the rest of the stack. No extra setup beyond `docker compose up -d dagster`. The container `priorart-dagster` listens on **13002** (NOT 13000/13001, which belong to the shared Langfuse instance).

#### Daily loop

```bash
# Start Dagster (Phase 3+) — webserver + daemon + code-server in one process
make dagster-up
# First boot: 30-60s (image pull + uv sync + Dagster SDK install + code-server warm-up)
# Subsequent boots: < 30s (named volume `dagster-home` persists lineage + run history)
# Open the UI: http://localhost:13002

# Materialize every corpus asset once (cold start / new clone / recovered DB)
dagster asset materialize --select '*'
# Or via the UI: Assets → "Materialize all"

# Trigger the nightly re-embedding schedule on demand (without waiting for cron)
dagster schedule launch nightly_re_embedding
# The schedule fires nightly_re_embedding_job (excludes eval_benchmark by
# design — the eval set is a freshness signal, not a corpus node).
# Or via the UI: Runs → nightly_re_embedding_job → "Launch"

# Verify the nightly schedule is wired
# UI → Schedules → nightly_re_embedding  (cron: 30 2 * * * UTC)
```

The 5 assets (`yc_directory`, `product_hunt_archive`, `hn_show_posts`, `company_embeddings`, `eval_benchmark`) are wrapped around the existing Phase 1.2/2.5/2.6/2.7 scrapers as **subprocess materializations** — no scraper rewrite, no second source of truth. `company_embeddings` is idempotent (mtime check vs the latest corpus manifest), so the daily schedule is a no-op when no source refreshed.

#### Config-change regression loop (lands in Phase 3.2)

> **Status:** The sensor (`config_change_sensor`) lands in card **t_877e48cd (Phase 3.2, Perseus)**, currently running. The 3.1 handoff declared the sensor in `src/dagster_assets/__init__.py` as a placeholder but the live sensor + the `config_change_eval_job` it fires are not yet wired into `Definitions`. Don't document them as shipped until 3.2 lands.

Once 3.2 ships, the loop is:

```bash
# Edit a config
vim configs/dense_bge_m3.yaml        # bump threshold from 0.7 to 0.75

# Dagster UI → Sensors → config_change_sensor
# Tick within < 60s (sensor debounces 30s so a multi-file edit = one eval run, not N)

# Dagster UI → Runs → config_change_eval_job
# Run completes, MLflow run appears (Phase 2.4 already logs every eval.run invocation)

# Verify on the host
make eval BENCH=evals/labeled_v300.jsonl    # regenerated leaderboard reflects the new sweep
```

#### Common failure modes (Dagster-specific)

See the row table below for the full Dagster failure-mode inventory. The pattern is consistent: a wrong port, a missing `Definitions` registration, a TTY-bound subprocess, or a stale partition date.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `psycopg.OperationalError: connection refused` on 15432 | Postgres container not up | `docker compose ps` and `docker compose up -d postgres` |
| `extension "vector" is not available` | Wrong Postgres image (not `pgvector/pgvector:pg16`) | Update `docker-compose.yml`, rebuild |
| `bge-m3 model not found` on first run | Sentence-transformers downloading on first use | Wait 2-3 min for download; subsequent runs are cached in `~/.cache/huggingface` |
| `InstructorJSONValidationError` from LLM | Claude output didn't match Pydantic schema | Inspect the response in Langfuse; the prompt probably needs tightening. **Do not loosen the schema.** |
| Temporal workflow stuck in "Running" | Activity retry loop on a non-transient error | Check Temporal UI for the activity's error history; add the error type to `non_retryable_error_types` in the retry policy |
| Langfuse shows no traces | API keys missing or wrong env var | `echo $LANGFUSE_PUBLIC_KEY $LANGFUSE_SECRET_KEY` — both must be set. If the harness redaction filter mangled them, regenerate and paste via the single-line pattern |
| MLflow shows no runs | Tracking URI wrong | `echo $MLFLOW_TRACKING_URI` — should be `http://localhost:15000` |
| Dagster UI unreachable on 13002 | Wrong port, `dagster` service not up, or `langfuse-web` squatting on 13002 | `docker compose ps` → `make dagster-up`. Confirm `models.yaml: dagster.webserver_port` matches `DAGSTER_PORT` in the Makefile (default 13002) |
| Dagster assets not materializing on click | Asset subprocess exited non-zero, or Dagster hasn't loaded the asset definition | `make dagster-logs` for the traceback; `make dagster-restart` to re-import `src/dagster_assets/`; verify `make dagster-validate` prints `OK: 5 assets, 1 nightly job, 1 @daily schedule all wired up` |
| "Asset `company_embeddings` is missing partition" | Snapshot date changed (e.g. `data/snapshots/corpus_2026-06-29.jsonl` → `…-07-01.jsonl`) but no backfill ran | UI → Assets → company_embeddings → "Materialize" with the new partition key, or trigger a backfill from the lineage view |
| Sensor `config_change_sensor` not firing | Workspace not watching `configs/`, or sensor isn't wired into `Definitions` (Phase 3.2 not yet shipped) | Confirm with `make dagster-validate`; once 3.2 lands, the sensor's cursor lives in `$DAGSTER_HOME/sensors/` on the named volume — clear the cursor to force a re-tick |
| Materialization hangs (no progress for >5 min) | `bge-m3` CPU encoding under TTY pressure (subprocess inherits the controlling terminal) | Run from `script -qc 'dagster job launch nightly_re_embedding_job' /dev/null`, or detach the terminal (`nohup … &`); the asset subprocess must not block on stdin |
| `eval_benchmark` asset reports stale version | `models.yaml: dagster.eval_benchmark.version` wasn't bumped when `evals/labeled_v300.jsonl` was replaced | Bump the version field; re-materialize the asset; the leaderboard column should follow on the next eval run |
| Frontend shows CORS error | FastAPI not allowing the frontend origin | Add the frontend URL to the FastAPI CORS middleware allowlist |
| `make eval` runs but reports MRR=0 | Corpus not loaded | `python -m src.data.ingest --snapshot data/snapshots/yc_<date>.jsonl` |
| `make eval` runs but reports FPR=1.0 | Threshold too low | Run with `--threshold-sweep` and pick the threshold that maximizes MRR subject to FPR ≤ 0.15 |
| WebSearchFallback always firing | Threshold too high or corpus too small | Same fix — re-run the threshold sweep with the expanded corpus |

---

## How to add a new retrieval config

```bash
# 1. Copy an existing config
cp configs/dense_bge_m3.yaml configs/dense_cohere_rerank.yaml

# 2. Edit the new config
# - change the embedding model or add a rerank step
# - keep the schema (the eval runner reads it)
```

```yaml
# configs/dense_cohere_rerank.yaml
name: dense_cohere_rerank
embedding_model: bge-m3
retriever: pgvector_hnsw
top_k_retrieve: 50       # over-retrieve for rerank
reranker: cohere_rerank_v3
top_k_rerank: 10
threshold: 0.6
```

```bash
# 3. Run the eval against the new config
python -m src.eval.run --config configs/dense_cohere_rerank.yaml \
  --benchmark evals/labeled_v300.jsonl \
  --output results/leaderboard.csv

# 4. The new row appears in leaderboard.csv
# 5. The GitHub Actions regression runs against all configs in configs/
```

---

## How to add a new data source

```bash
# 1. Write a scraper
# src/data/sources/<source_name>.py implements:
#    def scrape(snapshot_date: str) -> pd.DataFrame
#    def schema() -> list[str]   # the expected columns

# 2. Add a Dagster asset
# src/data/dagster_assets.py:
#    @asset
#    def <source_name>(snapshot_date: str) -> pd.DataFrame:
#        return scrape_<source>(snapshot_date)

# 3. Add to the merge in company_embeddings
# src/data/dagster_assets.py:
#    @asset
#    def company_embeddings(<source_name>, ...):
#        merged = merge_sources([..., <source_name>])
#        ...

# 4. Materialize
dagster asset materialize --asset <source_name>
# 5. Re-run the eval harness — the corpus count goes up
```

---

## How to add a new metric

```bash
# 1. Implement the metric function
# src/eval/metrics/<metric_name>.py:
#    def compute(predictions: list[RetrievalResult], labels: list[Label]) -> float

# 2. Register it in the runner
# src/eval/runner.py:
#    METRICS = {
#        "mrr": compute_mrr,
#        "nDCG@10": compute_ndcg_at_10,
#        ...
#        "new_metric": compute_new_metric,
#    }

# 3. Add a column to LeaderboardRow
# src/eval/schemas.py:
#    class LeaderboardRow(BaseModel):
#        ...
#        new_metric: float

# 4. Run eval — the new column appears
```

---

## Production migration path

**Note:** This is documented but not built. The project is a demo. If you ever wanted to deploy this for real, here's the path.

### Temporal: from dev server to a real cluster

**Note:** This path is documented, not built. The repo currently runs `temporal server start-dev` for local development; the section below is "what I would do when I'm ready to ship this for real," not a working deploy.

#### 1. Provision a Postgres 16 instance (separate from the priorart pgvector DB)

The Temporal server needs its own Postgres — do not co-locate it on the `priorart-pgvector` database. pgvector's HNSW index and Temporal's visibility/workflow tables have very different write patterns.

```bash
# Provision Postgres 16 (RDS, Cloud SQL, Neon, Supabase, or a self-hosted instance).
# Then create the two databases Temporal needs:
psql -h <postgres-host> -U postgres -c "CREATE DATABASE temporal;"
psql -h <postgres-host> -U postgres -c "CREATE DATABASE temporal_visibility;"
```

#### 2. Stand up Temporal via the upstream Helm chart

Use the [upstream Temporal Helm repo](https://github.com/temporalio/helm-charts):

```bash
helm repo add temporal https://temporal.github.io/helm-charts
helm repo update
```

For a production-grade install, follow the [Temporal Helm production deployment guide](https://docs.temporal.io/self-hosted-guide/kubernetes) — the snippet below is the minimum-viable shape, not a hardened configuration (TLS, mTLS, Elasticsearch visibility, Prometheus metrics, etc. all live outside this snippet). Upstream chart source: <https://github.com/temporalio/helm-charts>.

`values-prod.yaml`:

```yaml
# Postgres-backed persistence + visibility, Cassandra explicitly disabled.
# The two SQL stores point at SEPARATE databases on the same Postgres 16
# instance — Temporal uses one for workflows/history and the other for
# the visibility store. Co-locating with pgvector is operationally
# painful (different write patterns, different backup cadence).
server:
  config:
    persistence:
      driver: "sql"
      sql:
        driver: "postgres16"
        host: "temporal-postgres.internal"
        database: "temporal"
        user: "temporal_user"
    visibility:
      driver: "sql"
      sql:
        driver: "postgres16"
        host: "temporal-postgres.internal"
        database: "temporal_visibility"
        user: "temporal_user"
    cassandra:
      clusterSize: 0    # explicitly disable Cassandra
```

```bash
helm install temporal temporal/temporal --values values-prod.yaml
```

#### 3. Create the namespace

The convention is `<project>-<env>` so multiple environments can share one cluster without colliding:

```bash
temporal operator namespace create priorart-prod \
  --address temporal.<host>:7233 \
  --description "PriorArt idea-dedup workflow (production)"

# Register the search attributes this workflow reads
temporal operator search-attribute create \
  --name CustomKeywordField \
  --type Keyword
# Repeat for every CustomKeywordField / CustomIntField the workflow indexes on.
```

#### 4. Wire the worker to the new cluster

The worker reads its connection from three env vars (which `src/config.py` already honors); flip them in the deployment manifest, no code change needed:

```bash
# In the worker's deployment / .env:
TEMPORAL_ADDRESS=temporal.<host>:7233       # was: 127.0.0.1:7233 (dev server)
TEMPORAL_NAMESPACE=priorart-prod            # was: default
TEMPORAL_TASK_QUEUE=priorart-idea-analysis  # same value as the dev-server default; explicit in prod
# Plus the mTLS certs if the cluster is TLS-terminated:
TEMPORAL_TLS_CERT=/etc/temporal/ca.pem
TEMPORAL_TLS_KEY=/etc/temporal/ca.key
```

```python
# src/config.py — already honors these three as env-overridable defaults
TEMPORAL_ADDRESS      = os.getenv("TEMPORAL_ADDRESS",      "127.0.0.1:7233")
TEMPORAL_NAMESPACE    = os.getenv("TEMPORAL_NAMESPACE",    "default")
TEMPORAL_TASK_QUEUE   = os.getenv("TEMPORAL_TASK_QUEUE",   "priorart-idea-analysis")
```

The `src/workflow/worker.py` long-lived process picks these up at start; restart the worker after flipping the env vars. Same shape applies if you want to run it explicitly: `python -m src.workflow.worker --address temporal.<host>:7233 --namespace priorart-prod --task-queue priorart-idea-analysis`.

In the Helm release, the worker runs as a long-lived Deployment (one replica is enough at demo scale; scale `replicas: N` for prod throughput). All worker fleet + UI tctl commands target `priorart-prod` by default.

#### 5. Verify

```bash
temporal operator namespace describe priorart-prod --address temporal.<host>:7233
# Expect: Name: priorart-prod, State: Registered, Description: ...

temporal workflow list --query 'WorkflowType="IdeaAnalysisWorkflow"' --limit 5
# Expect: a healthy history of completed workflows
```

### Postgres: from Docker to a managed instance

```bash
# 1. Provision a managed Postgres (RDS, Cloud SQL, Supabase, Neon, etc.)
# 2. Enable the pgvector extension
# 3. Restore from the latest committed snapshot
# 4. Update the connection string in src/data/db.py
# 5. Re-run the eval harness — the numbers should match
```

### Frontend: from `pnpm dev` to a real deployment

```bash
# 1. Build
cd src/frontend && pnpm build
# Output: dist/

# 2. Serve
# Option A: Static host (Vercel, Netlify, Cloudflare Pages)
# Option B: Nginx in front of the FastAPI server
# Option C: Behind the same FastAPI as a mounted static directory
```

### Observability: from local to managed

```bash
# Langfuse: managed cloud (langfuse.com) or self-host on a real cluster
# MLflow: managed (Databricks) or self-host on a real cluster
# Dagster: Dagster Cloud (managed) or self-host via the official
#   dagster-k8s / dagster-celery-k8s images (docker.io/dagster/*)
# The decision is operational, not technical — pick based on the operator's preferences
```

---

## Backup and recovery

The eval set, the corpus snapshots, the configs, the prompt templates — all committed to git. **The system is reproducible from git alone.** The only stateful thing is the Postgres database, and that's a derived artifact (re-ingestable from the snapshots).

Backup strategy:
- **Git is the source of truth** for code, configs, prompts, eval set, corpus snapshots.
- **Postgres is a cache.** Backing it up is nice-to-have, not critical. Loss-of-DB means re-running `make ingest` from the committed snapshots.
- **No production data of value.** This is a portfolio project on public data. Backups are about reproducibility, not about recovery from disaster.

---

## What we don't do

- **No Kubernetes.** Single host, Docker Compose. The user already operates Honcho, Langfuse, Firecrawl this way. Don't introduce a new operational model for a portfolio project.
- **No multi-region.** Single region, single host.
- **No CDN.** Static assets served from the same host as the API.
- **No secrets manager.** `.env` files for the dev environment. Production would need a real solution (Vault, AWS Secrets Manager, etc.) but that's not in scope.
- **No CI/CD deploy pipeline.** The user handles deploy separately. The eval-regression GitHub Action is the only Action.
