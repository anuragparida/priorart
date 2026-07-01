"""Dagster assets for the PriorArt corpus ingestion pipeline.

Phase 3.1 (card t_7928b3e2). The five assets model the batch data
platform boundary described in docs/ARCHITECTURE.md §Dagster:

    yc_directory       (asset, scrape YC public directory)
    product_hunt_archive (asset, scrape PH top launches)
    hn_show_posts      (asset, scrape HN "Show HN" stream)
    company_embeddings (asset, merge + dedup + embed, writes HNSW)
    eval_benchmark     (asset, track eval-set version + staleness)

Lineage
-------
The PHASE-3.md §3.1 lineage graph (also documented in
docs/ARCHITECTURE.md) is:

    yc_directory ─┐
    product_hunt_archive ─┼─→ company_embeddings
    hn_show_posts ─┘
                       (eval_benchmark is independent — it's a
                       staleness signal on the eval set, not part
                       of the corpus DAG.)

Asset materialization strategy
------------------------------
Each asset runs the existing Phase 1.2 / 2.5 / 2.6 / 2.7 scrapers
and ingester as a subprocess via ``python -m src.data.<module>``.
We do NOT reimplement the scrapers — that would fork the
``make ph-scrape`` CLI. Subprocess keeps one source of truth.

Asset return values are :class:`MaterializeResult` with a
``metadata`` payload — Dagster surfaces the JSONL row count and
the snapshot path in the asset catalog so operators can see what
changed without leaving the UI.

Idempotency
-----------
All five assets are idempotent. Re-materialization with no
upstream changes is a no-op at the database level (the
``(source, external_id)`` unique constraint on ``companies`` and
the ``(company_id, model_version, chunk_index)`` unique constraint
on ``company_embeddings`` cover it). The scrapers write
deterministic JSONL, so re-runs produce byte-identical files.

Skip-on-no-change
-----------------
``company_embeddings`` short-circuits when no snapshot has changed
since the last materialization. The freshness check compares the
mtime of the input JSONLs against the mtime of the manifest
written by the last successful build.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dagster import (
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    get_dagster_logger,
)

from src.config import REPO_ROOT

logger = get_dagster_logger(__name__)


# ---------------------------------------------------------------------------
# Constants — pulled from models.yaml + the data layout.
# ---------------------------------------------------------------------------

# SNAPSHOTS_DIR / EVALS_DIR / MODELS_YAML_PATH are derived from
# src.config.REPO_ROOT so they stay in sync with the rest of the
# project (the API, the eval runner, the workflow worker).
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
EVALS_DIR = REPO_ROOT / "evals"
MODELS_YAML_PATH = REPO_ROOT / "models.yaml"

# A scrape that returns no records is treated as a hard failure.
# The card body says "no upstream — Dagster is independent", but
# the contract is: an empty corpus build is wrong.
MIN_RECORDS_THRESHOLD = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_executable() -> str:
    """Return the python interpreter to invoke for subprocess assets.

    Dagster-materialized subprocesses run inside the same venv as the
    Dagster code server. ``sys.executable`` is the right answer in
    both ``dagster dev`` (uvicorn subprocess) and the dockerized
    ``dagster dev`` (entrypoint sets ``sys.executable``).
    """
    return sys.executable


def _read_jsonl_count(path: Path) -> int:
    """Count non-empty lines in a JSONL file. Returns 0 if missing."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _read_manifest_count(path: Path) -> int:
    """Pull the ``count`` field out of a Phase 1.2/2.5/2.6 manifest."""
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("count", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _read_corpus_manifest(path: Path) -> dict[str, Any]:
    """Read a Phase 2.7 ``corpus_<date>.manifest.json`` (best-effort)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _latest_snapshot(source_prefix: str, snapshots_dir: Path | None = None) -> Path | None:
    """Return the most recent ``<prefix>_<YYYY-MM-DD>.jsonl`` in ``snapshots_dir``.

    Defaults to the module-level ``SNAPSHOTS_DIR`` *at call time* (not at
    function-definition time) so tests can monkey-patch the constant and
    the change will be observed. Mirrors the discovery logic in
    ``src.data.corpus_build.discover_snapshots`` but is single-source-prefix
    and tolerant of a missing source (returns None instead of raising) —
    an asset should fail loudly at materialization, not at module import.
    """
    if snapshots_dir is None:
        snapshots_dir = SNAPSHOTS_DIR
    candidates: list[tuple[date, Path]] = []
    for entry in snapshots_dir.iterdir() if snapshots_dir.exists() else []:
        if not entry.is_file():
            continue
        name = entry.name
        if not name.startswith(f"{source_prefix}_") or not name.endswith(".jsonl"):
            continue
        # Extract YYYY-MM-DD between the prefix and the extension.
        date_part = name[len(source_prefix) + 1 : -len(".jsonl")]
        try:
            d = date.fromisoformat(date_part)
        except ValueError:
            continue
        candidates.append((d, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def _run_subprocess(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, surfacing stderr on non-zero exit.

    Used by the asset materializers. We capture stdout + stderr and
    raise ``RuntimeError`` (which Dagster surfaces as a failed asset
    materialization) so the UI gets a readable error message.
    """
    logger.info("running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if result.returncode != 0:
        logger.error("subprocess failed (rc=%d): %s", result.returncode, result.stderr)
        raise RuntimeError(
            f"command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- stdout ---\n{result.stdout}"
        )
    return result


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


@asset(
    group_name="corpus",
    description=(
        "Scrape the YC public directory via the Algolia-backed search "
        "key. Writes ``data/snapshots/yc_<date>.jsonl`` + manifest. "
        "Phase 1.2 scraper, wrapped."
    ),
    metadata={"source": "yc", "phase": "1.2"},
)
def yc_directory(context) -> MaterializeResult:
    """Wrap ``src.data.scrape_yc`` as a Dagster asset.

    The scraper is invoked via ``python -m src.data.scrape_yc``.
    Idempotent — Algolia pagination is deterministic, the JSONL is
    byte-stable across re-runs.
    """
    py = _python_executable()
    cmd = [py, "-m", "src.data.scrape_yc", "--out", str(SNAPSHOTS_DIR)]
    _run_subprocess(cmd)

    latest = _latest_snapshot("yc", SNAPSHOTS_DIR)
    if latest is None:
        raise RuntimeError("scrape_yc returned no JSONL snapshot in data/snapshots/")

    record_count = _read_jsonl_count(latest)
    if record_count < MIN_RECORDS_THRESHOLD:
        raise RuntimeError(
            f"yc_directory produced {record_count} records — "
            f"expected at least {MIN_RECORDS_THRESHOLD} (snapshot: {latest.name})"
        )

    manifest_path = latest.with_suffix("").with_name(latest.stem + ".manifest.json")
    manifest_count = _read_manifest_count(manifest_path)

    context.add_output_metadata(
        {
            "snapshot_path": MetadataValue.path(str(latest)),
            "manifest_path": MetadataValue.path(str(manifest_path)),
            "record_count": record_count,
            "manifest_count": manifest_count,
            "snapshot_date": MetadataValue.text(latest.stem.split("_", 1)[1]),
        }
    )
    return MaterializeResult(
        metadata={
            "snapshot_path": MetadataValue.path(str(latest)),
            "record_count": record_count,
            "manifest_count": manifest_count,
        }
    )


@asset(
    group_name="corpus",
    description=(
        "Scrape the Product Hunt archive via the Wayback Machine. "
        "Wraps ``src.data.scrape_ph``. Phase 2.5."
    ),
    metadata={"source": "producthunt", "phase": "2.5"},
)
def product_hunt_archive(context) -> MaterializeResult:
    """Wrap ``src.data.scrape_ph`` as a Dagster asset.

    The scraper accepts ``--skip-dedup`` for fast smoke runs; the
    Dagster asset runs the full pipeline (with the bge-m3 cosine
    dedup against YC) because that's the production contract.
    """
    py = _python_executable()
    yc_snapshot = _latest_snapshot("yc", SNAPSHOTS_DIR)
    yc_arg = ["--yc-snapshot", str(yc_snapshot)] if yc_snapshot else []

    cmd = [
        py,
        "-m",
        "src.data.scrape_ph",
        "--out",
        str(SNAPSHOTS_DIR),
        "--max-records",
        "5000",
        *yc_arg,
    ]
    _run_subprocess(cmd)

    latest = _latest_snapshot("producthunt", SNAPSHOTS_DIR)
    if latest is None:
        raise RuntimeError("scrape_ph returned no JSONL snapshot in data/snapshots/")

    record_count = _read_jsonl_count(latest)
    if record_count < MIN_RECORDS_THRESHOLD:
        raise RuntimeError(
            f"product_hunt_archive produced {record_count} records — "
            f"expected at least {MIN_RECORDS_THRESHOLD} (snapshot: {latest.name})"
        )

    manifest_path = SNAPSHOTS_DIR / (latest.stem + ".manifest.json")
    manifest_count = _read_manifest_count(manifest_path)

    context.add_output_metadata(
        {
            "snapshot_path": MetadataValue.path(str(latest)),
            "manifest_path": MetadataValue.path(str(manifest_path)),
            "record_count": record_count,
            "manifest_count": manifest_count,
        }
    )
    return MaterializeResult(
        metadata={
            "snapshot_path": MetadataValue.path(str(latest)),
            "record_count": record_count,
            "manifest_count": manifest_count,
        }
    )


@asset(
    group_name="corpus",
    description=(
        "Scrape the HN \"Show HN\" stream via Algolia + Firecrawl. "
        "Wraps ``src.data.scrape_hn``. Phase 2.6."
    ),
    metadata={"source": "hn", "phase": "2.6"},
)
def hn_show_posts(context) -> MaterializeResult:
    """Wrap ``src.data.scrape_hn`` as a Dagster asset.

    The scraper paginates ``hn.algolia.com`` for ``tags=show_hn`` +
    ``points>=50`` and Firecrawl-scrapes the external URLs for one
    paragraph of description.
    """
    py = _python_executable()
    cmd = [py, "-m", "src.data.scrape_hn", "--out", str(SNAPSHOTS_DIR)]
    _run_subprocess(cmd)

    latest = _latest_snapshot("hn_show", SNAPSHOTS_DIR)
    if latest is None:
        raise RuntimeError("scrape_hn returned no JSONL snapshot in data/snapshots/")

    record_count = _read_jsonl_count(latest)
    if record_count < MIN_RECORDS_THRESHOLD:
        raise RuntimeError(
            f"hn_show_posts produced {record_count} records — "
            f"expected at least {MIN_RECORDS_THRESHOLD} (snapshot: {latest.name})"
        )

    manifest_path = SNAPSHOTS_DIR / (latest.stem + ".manifest.json")
    manifest_count = _read_manifest_count(manifest_path)

    context.add_output_metadata(
        {
            "snapshot_path": MetadataValue.path(str(latest)),
            "manifest_path": MetadataValue.path(str(manifest_path)),
            "record_count": record_count,
            "manifest_count": manifest_count,
        }
    )
    return MaterializeResult(
        metadata={
            "snapshot_path": MetadataValue.path(str(latest)),
            "record_count": record_count,
            "manifest_count": manifest_count,
        }
    )


@asset(
    group_name="corpus",
    description=(
        "Merge the three source snapshots, dedup by name cosine ≥ "
        "0.85, embed with bge-m3, write the HNSW index, and emit a "
        "manifest. Skips the bge-m3 embed when no input has changed "
        "since the last successful build. Wraps "
        "``src.data.corpus_build``. Phase 2.7."
    ),
    metadata={"phase": "2.7"},
)
def company_embeddings(
    context,
    yc_directory,  # noqa: ARG001 — used via Dagster lineage
    product_hunt_archive,  # noqa: ARG001
    hn_show_posts,  # noqa: ARG001
) -> MaterializeResult:
    """Merge + dedup + embed. Writes the HNSW-ready ``company_embeddings`` table.

    Skip-on-no-change: if the latest corpus manifest is newer than
    every input snapshot, the asset short-circuits and returns the
    cached metadata. This keeps the daily schedule a no-op on days
    when no source has refreshed.
    """
    # Discover the three input snapshots — the line above ensures
    # Dagster has materialized them before this asset runs.
    snapshots = {
        source: _latest_snapshot(prefix, SNAPSHOTS_DIR)
        for source, prefix in [
            ("yc", "yc"),
            ("producthunt", "producthunt"),
            ("hn", "hn_show"),
        ]
    }
    missing = [s for s, p in snapshots.items() if p is None]
    if missing:
        raise RuntimeError(f"company_embeddings: missing snapshots for sources: {missing}")

    # Skip-on-no-change check.
    latest_corpus_manifest = (
        max(SNAPSHOTS_DIR.glob("corpus_*.manifest.json"), default=None, key=lambda p: p.stat().st_mtime)
    )
    if latest_corpus_manifest is not None:
        manifest_mtime = latest_corpus_manifest.stat().st_mtime
        newest_input_mtime = max(p.stat().st_mtime for p in snapshots.values())  # type: ignore[union-attr]
        if newest_input_mtime <= manifest_mtime:
            cached = _read_corpus_manifest(latest_corpus_manifest)
            totals = cached.get("totals", {})
            context.add_output_metadata(
                {
                    "status": MetadataValue.text("skipped — no input change"),
                    "cached_manifest": MetadataValue.path(str(latest_corpus_manifest)),
                    "embedded_total": totals.get("embedded", 0),
                    "kept_total": totals.get("kept", 0),
                }
            )
            logger.info(
                "company_embeddings: skipped — newest input mtime %.0f <= manifest mtime %.0f",
                newest_input_mtime,
                manifest_mtime,
            )
            return MaterializeResult(
                metadata={
                    "status": "skipped — no input change",
                    "embedded_total": totals.get("embedded", 0),
                    "kept_total": totals.get("kept", 0),
                    "manifest_path": str(latest_corpus_manifest),
                }
            )

    # Run corpus_build via subprocess.
    out_manifest = SNAPSHOTS_DIR / f"corpus_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.manifest.json"
    cmd = [
        _python_executable(),
        "-m",
        "src.data.corpus_build",
        "--snapshots-dir",
        str(SNAPSHOTS_DIR),
        "--out-manifest",
        str(out_manifest),
        "--threshold",
        "0.85",
    ]
    _run_subprocess(cmd)

    manifest = _read_corpus_manifest(out_manifest)
    totals = manifest.get("totals", {})

    context.add_output_metadata(
        {
            "manifest_path": MetadataValue.path(str(out_manifest)),
            "embedded_total": totals.get("embedded", 0),
            "kept_total": totals.get("kept", 0),
            "records_in": totals.get("records_in", 0),
            "dropped_dedup": totals.get("dropped_dedup", 0),
            "embedding_model": MetadataValue.text(manifest.get("embedding_model", "BAAI/bge-m3")),
        }
    )
    return MaterializeResult(
        metadata={
            "manifest_path": str(out_manifest),
            "embedded_total": totals.get("embedded", 0),
            "kept_total": totals.get("kept", 0),
            "records_in": totals.get("records_in", 0),
            "dropped_dedup": totals.get("dropped_dedup", 0),
        }
    )


@asset(
    group_name="eval",
    description=(
        "Track the current eval-set version + surface staleness via "
        "``last_evaluated_at``. The card body explicitly defines "
        "this asset as the 'freshness check' called out in the "
        "parent — no separate asset for that."
    ),
    metadata={"phase": "1.5a / 2.8"},
)
def eval_benchmark(context) -> MaterializeResult:
    """Surface the current eval-set version + last-evaluated timestamp.

    No remote call. We read ``models.yaml`` for the current version
    label and stat the eval JSONL for mtime. Staleness is reported
    via ``MetadataValue.int`` (days since the JSONL was last
    modified) so Dagster can chart it on the asset detail page.
    """
    # 1. Version label from models.yaml (eval_benchmark section).
    version_label = "unknown"
    if MODELS_YAML_PATH.exists():
        # Lazy import — avoid a hard yaml dep on module import time.
        import yaml  # noqa: PLC0415 — local import keeps the dagster
        # asset module importable even if pyyaml is missing in a slim env.

        try:
            cfg = yaml.safe_load(MODELS_YAML_PATH.read_text()) or {}
        except yaml.YAMLError:
            cfg = {}
        version_label = (
            cfg.get("dagster", {}).get("eval_benchmark", {}).get("version", "unknown")
        )

    # 2. Eval-set file.
    candidates = sorted(EVALS_DIR.glob("labeled_v*.jsonl")) if EVALS_DIR.exists() else []
    if not candidates:
        # Stale-evals is a real failure mode. Surface it loudly.
        context.add_output_metadata(
            {
                "status": MetadataValue.text("no eval set found"),
                "version_label": MetadataValue.text(version_label),
            }
        )
        raise RuntimeError(f"eval_benchmark: no labeled_v*.jsonl in {EVALS_DIR}/")

    # Latest eval-set wins (matches the spec — labeled_v100 < labeled_v300).
    latest_eval = candidates[-1]
    eval_mtime = latest_eval.stat().st_mtime
    age_days = int((datetime.now(timezone.utc).timestamp() - eval_mtime) / 86400)

    context.add_output_metadata(
        {
            "version_label": MetadataValue.text(version_label),
            "eval_path": MetadataValue.path(str(latest_eval)),
            "record_count": _read_jsonl_count(latest_eval),
            "age_days": age_days,
            "last_modified_utc": MetadataValue.text(
                datetime.fromtimestamp(eval_mtime, tz=timezone.utc).isoformat()
            ),
        }
    )
    return MaterializeResult(
        metadata={
            "version_label": version_label,
            "eval_path": str(latest_eval),
            "record_count": _read_jsonl_count(latest_eval),
            "age_days": age_days,
        }
    )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

# Daily schedule fires the re-embedding at 02:30 UTC — well after
# the upstream scrapers (which are typically run in the evening)
# and well before the operator's morning so the leaderboard is
# fresh when they sit down. The card body says "@daily"; the
# 02:30 UTC pin keeps it deterministic and documentable.
#
# Dagster 1.13.x removed ``DailySchedule`` — use ``ScheduleDefinition``
# with a cron expression and the job reference instead.
nightly_re_embedding_schedule = ScheduleDefinition(
    name="nightly_re_embedding",
    job_name="nightly_re_embedding_job",
    execution_timezone="UTC",
    cron_schedule="30 2 * * *",
    description=(
        "Phase 3.1 nightly re-embedding. Triggers at 02:30 UTC "
        "every day; materializes the four corpus assets. The "
        "company_embeddings asset short-circuits when no input "
        "snapshot has changed since the last successful build."
    ),
)


__all__ = [
    "yc_directory",
    "product_hunt_archive",
    "hn_show_posts",
    "company_embeddings",
    "eval_benchmark",
    "nightly_re_embedding_schedule",
]