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
| Dagster assets not materializing | Sensor hasn't fired | Check the sensor's cursor in the Dagster UI; manually trigger via `dagster asset materialize --asset <name>` |
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

```bash
# 1. Stand up a real Temporal cluster (Helm)
helm repo add temporal https://temporal.github.io/temporal-helm
helm install temporal temporal/temporal \
  --set server.config.persistence.driver=sql \
  --set server.config.persistence.sql.driver=postgres \
  --set server.config.persistence.sql.host=<postgres-host> \
  --set cassandra.config.clusterSize=0  # not using Cassandra

# 2. Create the namespaces
temporal operator namespace create priorart
temporal operator namespace create priorart-visibility

# 3. Update the worker config
# src/workflow/config.py:
#    TEMPORAL_ADDRESS = "priorart.<host>.com:7233"
#    TEMPORAL_NAMESPACE = "priorart"

# 4. Register the worker
# src/workflow/worker.py runs as a long-lived service
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
# Dagster: Dagster Cloud (managed) or self-host
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
