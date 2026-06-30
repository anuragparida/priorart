# PriorArt — Makefile
#
# This is the operator's one-stop shop for Phase 1 + 2.1. The intent
# is that a reader can ``make eval`` after a fresh clone and see the
# same numbers in ``results/leaderboard.csv`` that the project
# README claims, and ``make smoke`` after ``make up && make
# temporal-up && make dev`` to see the full Temporal-backed stack
# run end-to-end.
#
# Every target uses the project's ``uv``-managed Python — no
# system Python, no system pip. The same Python version (3.12)
# is pinned in ``.python-version`` and ``pyproject.toml``.
#
# Phase 1 targets:
#   - up:     docker compose up -d (postgres + pgvector)
#   - down:   docker compose down
#   - health: verify the API on 18001 is up
#   - scrape: refresh the YC snapshot (Phase 1.2)
#   - ingest: load the snapshot into Postgres (Phase 1.3)
#   - eval:   run the eval harness against the live API (Phase 1.6)
#   - smoke:  end-to-end smoke test against /healthz + /search +
#             /ideas/analyze + /workflows/{id} + /workflows/{id}/result
#             (Phase 2.1)
#   - dev:    run uvicorn (API on 18001) + temporal-worker +
#             pnpm dev (frontend on 15174) in parallel (Phase 2.1)
#   - test:   run the pytest suite
#   - clean:  remove build / cache artifacts
#
# Phase 2.1 targets:
#   - temporal-up:    start the Temporal dev server on 7233 (gRPC)
#                     + 8233 (Web UI) — in-memory store, ephemeral
#   - temporal-down:  stop the Temporal dev server
#   - temporal-ui:    print the Temporal UI URL
#   - worker:         run the Temporal worker in the foreground
#
# Phase 2.3 targets:
#   - langfuse-up:    start the self-hosted Langfuse container (Phase 2.3)
#   - langfuse-down:  stop the Langfuse container
#   - langfuse-logs:  tail the Langfuse container logs
#   - langfuse-health:check whether Langfuse is reachable on localhost:13000
#
# Notes on port numbers:
# - 18000 is squatted by a local clausecraft stack on this host.
#   The priorart API runs on 18001 (Phase 1.4 ships on 18001).
# - 8233 is the Temporal Web UI; 7233 is the Temporal gRPC endpoint.
#   Both are the Temporal CLI defaults — change TEMPORAL_ADDRESS in
#   src/config.py if you need a non-default port.
# - Postgres on 15433 (Phase 1.1 ships on 15433).
# - See docker-compose.yml for the actual port mappings.

PY ?= uv run python
PYTHON ?= python3
PORT ?= 18001

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help. Default target.
	@awk 'BEGIN {FS = ":.*##"; printf "PriorArt Phase 1 + Phase 2 Makefile\n\nUsage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: up
up: ## docker compose up -d — start postgres + pgvector.
	docker compose up -d

.PHONY: down
down: ## docker compose down — stop the stack.
	docker compose down

.PHONY: health
health: ## curl the API on $(PORT)/healthz. Exits 0 on HTTP 200.
	@curl -fsS http://localhost:$(PORT)/healthz && echo

.PHONY: scrape
scrape: ## Refresh the YC snapshot at data/snapshots/yc_<date>.jsonl.
	$(PY) -m src.data.scrape_yc

.PHONY: scrape-hn
scrape-hn: ## Refresh the HN "Show HN" snapshot at data/snapshots/hn_show_<date>.jsonl (3-year lookback, Firecrawl-scrape external URLs).
	$(PY) -m src.data.scrape_hn

.PHONY: hn-scrape
hn-scrape: scrape-hn ## Alias for `scrape-hn` (card t_56b10368 acceptance).
	@true

.PHONY: ingest
ingest: ## Ingest the latest snapshot into Postgres + pgvector.
	$(PY) -m src.data.ingest --snapshot data/snapshots/yc_$$(ls -1 data/snapshots/ | grep '\.jsonl$$' | sort | tail -1)

.PHONY: eval
EXPERIMENT ?= phase-2-baseline
MLFLOW_TRACKING_URI ?= http://localhost:15000
# Run the eval harness with MLflow logging ON by default. Set
# ``make eval NO_MLFLOW=1`` to skip MLflow logging. Set
# ``make eval EXPERIMENT=bm25-baseline`` to land the run in a
# different MLflow experiment (handy for 2.9 BM25 + Hybrid RRF runs).
ifeq ($(NO_MLFLOW),1)
EVAL_MLFLOW_FLAGS := --no-mlflow
else
EVAL_MLFLOW_FLAGS := --experiment-name $(EXPERIMENT)
endif
eval: ## Run the eval harness against the live priorart API. Writes results/leaderboard.csv + results/eval.duckdb AND logs the run to MLflow (override with NO_MLFLOW=1).
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) $(PY) -m eval.run \
		--benchmark evals/labeled_v100.jsonl \
		--config configs/dense_bge_m3.yaml \
		--output results/leaderboard.csv \
		--db results/eval.duckdb \
		--markdown-out results/leaderboard.md \
		--mlflow-tracking-uri $(MLFLOW_TRACKING_URI) \
		$(EVAL_MLFLOW_FLAGS)

.PHONY: screenshot
screenshot: ## Re-render docs/assets/leaderboard-v1.png from results/leaderboard.csv (Phase 1 dense config).
	$(PY) scripts/render_leaderboard_screenshot.py

.PHONY: screenshot-v2
screenshot-v2: ## Re-render docs/assets/leaderboard-v2.png from results/leaderboard.csv (Phase 2.11, 3 configs on labeled_v300).
	$(PY) scripts/render_leaderboard_v2_screenshot.py

.PHONY: smoke
smoke: ## End-to-end smoke test: hits /healthz + /search + /ideas/analyze. Exits 0 on success.
	$(PY) scripts/smoke.py --api-url http://localhost:$(PORT)

.PHONY: dev
dev: ## Run uvicorn (API on $(PORT)) + temporal-worker + pnpm dev (frontend on 15174) in parallel.
	@echo "Starting uvicorn on :$(PORT), temporal-worker on :7233, and pnpm dev on :15174 in parallel."
	@echo "API logs        → /tmp/priorart-api.log"
	@echo "Worker logs     → /tmp/priorart-worker.log"
	@echo "Frontend logs   → /tmp/priorart-frontend.log"
	@echo "Stop with Ctrl-C. (All processes will receive the signal.)"
	@echo "Prereq: 'make temporal-up' must have been run once already."
	@trap 'kill 0' EXIT INT TERM; \
	  (cd $(CURDIR) && .venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port $(PORT) > /tmp/priorart-api.log 2>&1 &) ; \
	  (cd $(CURDIR) && .venv/bin/python -m src.workflow.worker > /tmp/priorart-worker.log 2>&1 &) ; \
	  (cd $(CURDIR)/src/frontend && pnpm dev > /tmp/priorart-frontend.log 2>&1 &) ; \
	  wait

.PHONY: test
test: ## Run the pytest suite.
	$(PY) -m pytest

.PHONY: lint
lint: ## Run ruff on src/ and tests/.
	$(PY) -m ruff check src/ tests/ scripts/

.PHONY: clean
clean: ## Remove __pycache__, .pytest_cache, and stale result files.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
	rm -f results/eval.duckdb results/leaderboard.csv

# ---------------------------------------------------------------------------
# langfuse (Phase 2.3)
# ---------------------------------------------------------------------------

.PHONY: langfuse-up
langfuse-up: ## Start the self-hosted Langfuse container on 13000/13001 (Phase 2.3). On this host the clausecraft stack already runs Langfuse on 13000/13001 — this command will fail with a port conflict in that case. The wrapper (src/observability/langfuse.py) is happy to point at the existing instance via LANGFUSE_HOST=http://localhost:13000.
	docker compose --profile langfuse up -d langfuse
	@echo "Langfuse UI:  http://localhost:13000"
	@echo "Langfuse API: http://localhost:13000/api/public/health"
	@echo ""
	@echo "Next: open the UI, create a project (e.g. 'priorart'), and copy the pk-lf-* / sk-lf-* keys into .env as LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY."

.PHONY: langfuse-down
langfuse-down: ## Stop the Langfuse container.
	docker compose --profile langfuse stop langfuse

.PHONY: langfuse-logs
langfuse-logs: ## Tail the Langfuse container logs.
	docker compose --profile langfuse logs -f langfuse

.PHONY: langfuse-health
langfuse-health: ## Check whether a self-hosted Langfuse is reachable at $$LANGFUSE_HOST (default http://localhost:13000).
	@echo "Probing $${LANGFUSE_HOST:-http://localhost:13000}/api/public/health ..."
	@curl -sS -m 5 $${LANGFUSE_HOST:-http://localhost:13000}/api/public/health || { echo "Langfuse NOT reachable"; exit 1; }
	@echo ""

# ---------------------------------------------------------------------------
# Temporal dev server (Phase 2.1)
# ---------------------------------------------------------------------------
#
# Temporal's "dev server" is an in-memory + SQLite-backed single-node
# cluster intended for local development. It is *not* a production
# deployable. The CLI (`temporal server start-dev`) downloads a small
# set of artifacts on first run (~200 MB into ~/.temporalio/); after
# that the server starts in < 2 s.
#
# Why we run it as a background process (and not a docker-compose service)
# ---------------------------------------------------------------------------
# The Temporal team strongly recommends `temporal server start-dev` over
# the docker image for development. Reasons:
# - The CLI version is *always* in sync with the temporalio SDK version
#   we pin in pyproject.toml. The docker image lags by 2-4 weeks per
#   release.
# - The CLI starts in 2 s; the docker container takes 10-15 s on the
#   first boot (and ~5 s after that).
# - No port mapping conflict to manage.
#
# Ports
# -----
# - 7233 — gRPC API the temporalio SDK + worker talk to.
# - 8233 — Web UI (http://localhost:8233).
#
# The defaults match src/config.py's TEMPORAL_ADDRESS / TEMPORAL_NAMESPACE.
# If you change them, change src/config.py too (or set TEMPORAL_ADDRESS /
# TEMPORAL_NAMESPACE in the environment).

TEMPORAL_PIDFILE ?= /tmp/priorart-temporal.pid
TEMPORAL_LOGFILE ?= /tmp/priorart-temporal.log

.PHONY: temporal-up
temporal-up: ## Start the Temporal dev server in the background (gRPC 7233 + Web UI 8233).
	@if [ -f $(TEMPORAL_PIDFILE) ] && kill -0 $$(cat $(TEMPORAL_PIDFILE)) 2>/dev/null; then \
	  echo "Temporal is already running (pid $$(cat $(TEMPORAL_PIDFILE))). Use 'make temporal-down' first."; \
	  exit 1; \
	fi
	@command -v temporal >/dev/null 2>&1 || { echo "temporal CLI not found. Install: https://temporal.download/cli/archive/latest?platform=linux&arch=amd64"; exit 1; }
	@echo "Starting Temporal dev server (logs → $(TEMPORAL_LOGFILE))..."
	@nohup temporal server start-dev --db-filename /tmp/priorart-temporal.db > $(TEMPORAL_LOGFILE) 2>&1 &
	@echo $$! > $(TEMPORAL_PIDFILE)
	@echo "Temporal pid: $$(cat $(TEMPORAL_PIDFILE))"
	@echo "Waiting for Temporal to become reachable on 127.0.0.1:7233..."
	@for i in $$(seq 1 30); do \
	  if temporal operator namespace list >/dev/null 2>&1; then \
	    echo "Temporal is up (took ~$${i}s)."; \
	    $(MAKE) --no-print-directory temporal-ui; \
	    exit 0; \
	  fi; \
	  sleep 1; \
	done; \
	echo "Temporal did not come up within 30s. Tail $(TEMPORAL_LOGFILE) for details."; \
	exit 1

.PHONY: temporal-down
temporal-down: ## Stop the Temporal dev server started by 'make temporal-up'.
	@if [ ! -f $(TEMPORAL_PIDFILE) ]; then \
	  echo "No Temporal pidfile at $(TEMPORAL_PIDFILE). Nothing to stop."; \
	  exit 0; \
	fi
	@PID=$$(cat $(TEMPORAL_PIDFILE)); \
	if kill -0 $$PID 2>/dev/null; then \
	  echo "Stopping Temporal (pid $$PID)..."; \
	  kill $$PID; \
	  for i in $$(seq 1 10); do \
	    if ! kill -0 $$PID 2>/dev/null; then \
	      rm -f $(TEMPORAL_PIDFILE); \
	      echo "Temporal stopped."; \
	      exit 0; \
	    fi; \
	    sleep 1; \
	  done; \
	  echo "PID $$PID did not exit cleanly; force-killing."; \
	  kill -9 $$PID; \
	  rm -f $(TEMPORAL_PIDFILE); \
	else \
	  echo "Stale pidfile (pid $$PID not running); removing."; \
	  rm -f $(TEMPORAL_PIDFILE); \
	fi

.PHONY: temporal-ui
temporal-ui: ## Print the Temporal Web UI URL.
	@echo "Temporal Web UI: http://localhost:8233"
	@echo "Temporal gRPC:    127.0.0.1:7233 (matches src/config.py TEMPORAL_ADDRESS)"

.PHONY: temporal-status
temporal-status: ## Show whether the Temporal dev server is up.
	@if [ -f $(TEMPORAL_PIDFILE) ] && kill -0 $$(cat $(TEMPORAL_PIDFILE)) 2>/dev/null; then \
	  echo "Temporal is RUNNING (pid $$(cat $(TEMPORAL_PIDFILE)))"; \
	  temporal operator namespace list 2>&1 | head -20; \
	else \
	  echo "Temporal is NOT running. Use 'make temporal-up' to start it."; \
	  exit 1; \
	fi

.PHONY: worker
worker: ## Run the Temporal worker in the foreground (used by 'make dev' but exposed for debugging).
	$(PY) -m src.workflow.worker

# ---------------------------------------------------------------------------
# Product Hunt archive scraper (Phase 2.5)
# ---------------------------------------------------------------------------

PH_DATE ?= $(shell date -u +%Y-%m-%d)
PH_YC_SNAPSHOT ?= data/snapshots/yc_2026-06-08.jsonl

.PHONY: ph-scrape
ph-scrape: ## Scrape the PH archive (top 5K upvoted, last 3 years, dedup vs YC). Writes data/snapshots/producthunt_<date>.{jsonl,manifest.json,borderline.jsonl}.
	$(PY) -m src.data.scrape_ph \
		--date $(PH_DATE) \
		--yc-snapshot $(PH_YC_SNAPSHOT) \
		--max-records 5000 \
		--out data/snapshots

.PHONY: ph-scrape-fast
ph-scrape-fast: ## PH scrape with --skip-dedup (no bge-m3 cosine step; faster for smoke testing).
	$(PY) -m src.data.scrape_ph \
		--date $(PH_DATE) \
		--skip-dedup \
		--max-records 5000 \
		--out data/snapshots

.PHONY: ph-scrape-test
ph-scrape-test: ## PH scrape with --skip-dedup --max-records 50; for the test suite.
	$(PY) -m src.data.scrape_ph \
		--date $(PH_DATE) \
		--skip-dedup \
		--max-records 50 \
		--out /tmp/ph-scrape-test

.PHONY: ph-scrape-clean
ph-scrape-clean: ## Remove PH snapshot artifacts + the YC name embeddings cache.
	rm -f data/snapshots/producthunt_*.jsonl
	rm -f data/snapshots/producthunt_*.manifest.json
	rm -f data/snapshots/producthunt_*.borderline.jsonl
	rm -rf data/cache

# ---------------------------------------------------------------------------
# MLflow tracking server (Phase 2.4)
# ---------------------------------------------------------------------------
#
# PHASE-2.md §Pitfall rule: "Do not put MLflow in the same container as
# the API." The eval harness reaches for the tracking server on every
# `make eval`; the service is therefore always-on (no profile gate).
#
# Port choice: 15000, never 5000. Honcho is squatting 5000 on this host
# (AGENTS.md memory note); using 5000 here would make `make mlflow-up`
# fail with a port-bind error.
#
# The eval harness wrapper (`src.eval.mlflow_logger`) falls back to a
# per-process tmp-dir file-store when the server is unreachable, so the
# eval doesn't crash on a downed tracker — the leaderboard CSV / DuckDB
# still get written.

MLFLOW_PORT ?= 15000

.PHONY: mlflow-up
mlflow-up: ## Start the self-hosted MLflow container (Phase 2.4) on port 15000. SQLite backend in the named volume `mlflow-data`.
	docker compose up -d mlflow
	@echo "MLflow UI:    http://localhost:$(MLFLOW_PORT)"
	@echo "MLflow API:   http://localhost:$(MLFLOW_PORT)/api/2.0/mlflow/experiments/list"
	@echo "SQLite DB:    <named volume mlflow-data>/mlflow-data/mlflow.db"
	@echo "Artifact root:<named volume mlflow-data>/mlflow-data/artifacts"

.PHONY: mlflow-down
mlflow-down: ## Stop the MLflow container (preserves the named volume; SQLite DB + artifacts survive).
	docker compose stop mlflow

.PHONY: mlflow-logs
mlflow-logs: ## Tail the MLflow container logs.
	docker compose logs -f mlflow

.PHONY: mlflow-health
mlflow-health: ## Probe MLflow on http://localhost:$(MLFLOW_PORT)/health. Exits 0 if 200.
	@curl -fsS http://localhost:$(MLFLOW_PORT)/health && echo

.PHONY: mlflow-ls
mlflow-ls: ## List experiments in the MLflow tracking server.
	@curl -sS -X POST http://localhost:$(MLFLOW_PORT)/api/2.0/mlflow/experiments/search \
	  -H "Content-Type: application/json" -d '{"max_results":100}' \
	  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(e['experiment_id'],'|',e['name']) for e in d.get('experiments',[])]"

.PHONY: mlflow-runs
mlflow-runs: ## List runs in the phase-2-baseline experiment (override with EXPERIMENT=name).
	@EXP="$${EXPERIMENT:-phase-2-baseline}"; \
	echo "Listing runs in experiment '$$EXP'..."; \
	curl -sS -X POST http://localhost:$(MLFLOW_PORT)/api/2.0/mlflow/runs/search \
	  -H "Content-Type: application/json" \
	  -d "{\"experiment_names\":[\"$$EXP\"],\"max_results\":50}" \
	  | python3 -c "import json,sys; d=json.load(sys.stdin); runs=d.get('runs',[]); \
	    [print(r['info']['run_id'][:12], '|', r['info']['run_name'], '|', r['info']['status'], '|', \
	      [(m['key'], m['value']) for m in r['data']['metrics']]) for r in runs]"

.PHONY: mlflow-reset
mlflow-reset: ## DESTRUCTIVE — drop the MLflow SQLite DB + artifact store. Requires explicit confirmation.
	@echo "This will DELETE the named volume 'mlflow-data' on the docker host."
	@echo "All MLflow experiments / runs / artifacts will be lost."
	@read -p "Continue? (type 'yes' to proceed) " answer && [ "$$answer" = "yes" ] || { echo "aborted"; exit 1; }
	docker compose down -v mlflow
# ---------------------------------------------------------------------------
# Phase 2.7 — corpus build (merge + dedup + ingest + HNSW rebuild)
# ---------------------------------------------------------------------------
#
# Reads the latest snapshot per source under data/snapshots/:
#   - yc_<date>.jsonl
#   - producthunt_<date>.jsonl
#   - hn_show_<date>.jsonl
#
# Then: runs the schema migration, loads + normalises each source,
# cross-source dedups by name cosine ≥ 0.85 (bge-m3), upserts the
# merged companies, embeds + upserts the bge-m3 vectors, and writes
# a manifest at data/snapshots/corpus_<date>.manifest.json.
#
# Idempotent — re-running with the same inputs is a no-op on both
# the companies and the embeddings tables.

CORPUS_DATE ?= $(shell date -u +%Y-%m-%d)
CORPUS_THRESHOLD ?= 0.85

.PHONY: corpus-build
corpus-build: ## Phase 2.7 — merge YC + PH + HN, dedup by name cosine ≥ 0.85, rebuild HNSW. Idempotent.
	$(PY) -m src.data.corpus_build \
		--snapshots-dir data/snapshots \
		--out-manifest data/snapshots/corpus_$(CORPUS_DATE).manifest.json \
		--threshold $(CORPUS_THRESHOLD)

.PHONY: corpus-build-smoke
corpus-build-smoke: ## Phase 2.7 smoke — same as corpus-build but skips the bge-m3 embed (CI / quick check).
	$(PY) -m src.data.corpus_build \
		--snapshots-dir data/snapshots \
		--no-embed \
		--out-manifest data/snapshots/corpus_smoke.manifest.json \
		--threshold $(CORPUS_THRESHOLD)

.PHONY: migrate
migrate: ## Phase 2.7 — run the schema migrations only (idempotent).
	$(PY) -m src.data.migrate
