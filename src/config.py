"""Centralized config: env vars, paths, model versions.

Single source of truth so the API, CLI scripts, and eval harness all
agree on which Postgres to talk to and which embedding model to load.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: src/../  =  <repo>
# ``src/config.py`` -> ``src/`` -> ``priorart/`` (repo root) -> ``workspace/`` (parent).
# So ``parents[1]`` is the repo root, not ``parents[2]``.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = REPO_ROOT / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
EVAL_DATA_DIR = DATA_DIR / "eval"
EVALS_DIR = REPO_ROOT / "evals"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"

# Database
# Default port 15433 to avoid clashing with clausecraft-postgres on 15432.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://priorart:priorart@localhost:15433/priorart",
)

# Embedding model — bge-m3 (1024-dim, multilingual). The CV line is
# "pgvector + bge-m3 retrieval", so we pin this version explicitly.
EMBEDDING_MODEL = os.getenv("PRIORART_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

# LLM — Anthropic Claude Sonnet 4.5 for the structured-comparison call.
ANTHROPIC_MODEL = os.getenv("PRIORART_ANTHROPIC_MODEL", "claude-sonnet-4-5")
