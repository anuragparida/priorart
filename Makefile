# PriorArt — Makefile
#
# This is the operator's one-stop shop for Phase 1. The intent is
# that a reader can `make eval` after a fresh clone and see the
# same numbers in `results/leaderboard.csv` that the project
# README claims.
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
#   - test:   run the pytest suite
#   - clean:  remove build / cache artifacts
#
# Notes on port numbers:
# - 18000 is squatted by a local clausecraft stack on this host.
#   The priorart API runs on 18001 (Phase 1.4 ships on 18001).
# - Postgres on 15433 (Phase 1.1 ships on 15433).
# - See docker-compose.yml for the actual port mappings.

PY ?= uv run python
PYTHON ?= python3
PORT ?= 18001

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help. Default target.
	@awk 'BEGIN {FS = ":.*##"; printf "PriorArt Phase 1 Makefile\n\nUsage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

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

.PHONY: ingest
ingest: ## Ingest the latest snapshot into Postgres + pgvector.
	$(PY) -m src.data.ingest --snapshot data/snapshots/yc_$$(ls -1 data/snapshots/ | grep '\.jsonl$$' | sort | tail -1)

.PHONY: eval
eval: ## Run the eval harness against the live priorart API. Writes results/leaderboard.csv + results/eval.duckdb.
	$(PY) -m eval.run \
		--benchmark evals/labeled_v100.jsonl \
		--config configs/dense_bge_m3.yaml \
		--output results/leaderboard.csv \
		--db results/eval.duckdb \
		--markdown-out results/leaderboard.md

.PHONY: screenshot
screenshot: ## Re-render docs/assets/leaderboard-v1.png from results/leaderboard.csv.
	$(PY) scripts/render_leaderboard_screenshot.py

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