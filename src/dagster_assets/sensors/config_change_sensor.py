"""Config-change sensor + eval-regression job — Phase 3.2 (card t_877e48cd).

Lives under ``src/dagster_assets/sensors/`` (not ``src/dagster/sensors/``)
for the same reason the parent package avoids the ``src/dagster/`` name:
a project-local package named ``dagster`` shadows the real ``dagster``
package on ``sys.path`` because the project entry point is closer than
the venv's site-packages. See ``src/dagster_assets/__init__.py`` for the
long-form rationale. The card body's ``src/dagster/sensors/...`` path
was written before 3.1's package-name decision landed; this module
follows 3.1.

What this module ships:

* ``build_config_change_sensor(...)`` — a factory that returns a
  Dagster ``SensorDefinition`` bound to a specific repo root. Tests
  call it with a ``tmp_path``; production calls it once with the
  live ``$REPO_ROOT`` to produce the module-level
  ``config_change_sensor``.

* ``config_change_sensor`` — the default-decorated sensor that the
  ``Definitions`` object picks up.

* ``run_eval_for_config`` — an :func:`@op <dagster.op>` that runs
  ``python -m eval.run`` for one retrieval config and (if the
  workspace is clean) commits the regenerated leaderboard files
  to ``main``.

* ``eval_regression_job`` — an :func:`@job <dagster.job>` that
  the sensor fires. Currently a single-op job (one config per
  run). The shape is forward-compatible with adding
  ECE re-computation or per-config re-runs as separate ops in
  Phase 3.3+.

Hard rules (card body + PHASE-3.md §3.2):
* **Self-contained regression.** The op runs the same offline
  eval that ``make eval`` runs today (per Phase 2.9 the three
  leaderboard configs are all local-only: bge-m3 + rank_bm25).
  No Brave Search, no Cohere rerank, no Anthropic API.
* **Debounce.** Multiple file changes in a single tick collapse
  to one eval run per *affected config*, not N. The card's
  30-second debounce window is implemented as
  ``minimum_interval_seconds=30`` on the sensor (Dagster
  enforces a minimum tick rate) plus a cursor that records the
  ``mtime_ns`` of every watched file (so a file that hasn't
  moved is not re-fired even if the tick rate allows it).
* **Branch.** ``main``. No feature branches.
* **Commit.** If the workspace is clean (no uncommitted
  operator-local edits, on ``main``), commit the regenerated
  leaderboard files. If dirty, leave them on disk for the
  operator to stage + commit manually.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dagster import (
    OpExecutionContext,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    job,
    op,
)

from src.config import REPO_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Name of the job this sensor fires. Matches
#: ``models.yaml:dagster.config_change_sensor.target_job`` (Apollo
#: already pinned this name in 3.1). The 3.6 GitHub Actions workflow
#: will reuse the same job name — keep them in sync.
EVAL_REGRESSION_JOB_NAME = "eval_regression_job"

#: Default benchmark the regression runs against. Phase 2.8 brought
#: the v2 set (300 records, LLM-generated v2, hand-review pending).
DEFAULT_BENCHMARK = "evals/labeled_v300.jsonl"

#: How often Dagster will tick this sensor at most. The card body
#: says "30-second debounce window"; Dagster's
#: ``minimum_interval_seconds`` is the cleanest enforcement.
DEFAULT_DEBOUNCE_SECONDS = 30


# ---------------------------------------------------------------------------
# Watch surface (mirrors models.yaml + 3.6 GitHub Actions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchSpec:
    """One glob that the sensor scans.

    ``kind`` tells ``_affected_configs`` how to fan out when the
    glob matches new files:

    * ``"config"`` — a per-config file in ``configs/``; the path's
      stem (sans ``.yaml``) is the config name. Only that config
      is affected.
    * ``"global"`` — a path that can affect any / all configs
      (``models.yaml``, ``evals/labeled_v*.jsonl``,
      ``src/embedding/**``, ``src/llm/**``). All three configs
      re-run.
    """

    kind: str
    glob: str


def _default_watch_paths() -> list[WatchSpec]:
    """Return the sensor's default watch surface.

    Mirrors the broader path list under
    ``models.yaml:dagster.eval_regression.trigger_paths`` (which
    the 3.6 GitHub Actions workflow consumes) plus
    ``dagster.config_change_sensor.watch_paths`` in the same
    file. The card body explicitly says "matches the GitHub
    Actions path from 3.6 — keep them in sync".
    """
    return [
        WatchSpec(kind="config", glob="configs/*.yaml"),
        WatchSpec(kind="global", glob="models.yaml"),
        WatchSpec(kind="global", glob="evals/labeled_v*.jsonl"),
        WatchSpec(kind="global", glob="src/embedding/**/*.py"),
        WatchSpec(kind="global", glob="src/llm/**/*.py"),
    ]


def _all_retrieval_configs(repo_root: Path) -> list[str]:
    """Return the sorted list of retrieval-config names in ``configs/``.

    Tests override the repo root; production reads from
    ``$REPO_ROOT/configs/*.yaml`` (the same surface the eval
    harness uses). Hidden files (``.*.yaml``) and non-yaml files
    are skipped.
    """
    configs_dir = repo_root / "configs"
    if not configs_dir.exists():
        return []
    out: list[str] = []
    for path in sorted(configs_dir.glob("*.yaml")):
        if path.name.startswith("."):
            continue
        out.append(path.stem)
    return out


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


def _scan_mtimes(
    repo_root: Path,
    watch_specs: list[WatchSpec],
) -> dict[str, int]:
    """Return ``{path_str: mtime_ns}`` for every file matched by the specs.

    Paths are stored relative to ``repo_root`` so the cursor
    survives ``$REPO_ROOT`` renames (we don't have any today,
    but it's the right default).
    """
    snapshot: dict[str, int] = {}
    for spec in watch_specs:
        for path in sorted(repo_root.glob(spec.glob)):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                # Path wasn't under repo_root — shouldn't happen,
                # but don't crash the sensor over it.
                continue
            snapshot[rel] = path.stat().st_mtime_ns
    return snapshot


def _affected_configs(changed_paths: list[str], repo_root: Path) -> list[str]:
    """Reduce a set of changed paths to the set of affected config names.

    * A change to ``configs/<X>.yaml`` affects only config ``X``.
    * A change to any "global" path (``models.yaml``,
      ``evals/labeled_v*.jsonl``, ``src/embedding/**``,
      ``src/llm/**``) affects every retrieval config.
    * If a path is outside the watch surface entirely (shouldn't
      happen — the scan is spec-driven) it's ignored.

    The result is sorted so the RunRequest stream is
    deterministic (matters for the test suite and for
    ``git diff`` stability).
    """
    affected: set[str] = set()
    has_global = False
    for rel in changed_paths:
        if rel.startswith("configs/") and rel.endswith(".yaml"):
            stem = rel[len("configs/") : -len(".yaml")]
            if stem:
                affected.add(stem)
        elif rel == "models.yaml":
            has_global = True
        elif rel.startswith("evals/labeled_v") and rel.endswith(".jsonl"):
            has_global = True
        elif rel.startswith("src/embedding/"):
            has_global = True
        elif rel.startswith("src/llm/"):
            has_global = True
        else:
            logger.warning(
                "config_change_sensor: changed path %r not in watch surface",
                rel,
            )

    if has_global:
        affected.update(_all_retrieval_configs(repo_root))

    return sorted(affected)


def _diff_mtimes(
    prev: dict[str, int],
    curr: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Return ``(changed_paths, removed_paths)`` between two mtime snapshots.

    A file is "changed" if its mtime_ns moved forward (or it
    appears for the first time). A file is "removed" if it
    disappeared between ticks. The sensor doesn't fire on pure
    removals (deleting ``configs/bm25.yaml`` is a no-op for
    the surviving configs), but the cursor still drops the path
    so the next tick doesn't think it's still there.
    """
    changed: list[str] = []
    for path, mtime in curr.items():
        if path not in prev or prev[path] < mtime:
            changed.append(path)
    removed = [p for p in prev if p not in curr]
    return changed, removed


# ---------------------------------------------------------------------------
# Sensor factory (test-friendly + production)
# ---------------------------------------------------------------------------


def build_config_change_sensor(
    *,
    repo_root: Path,
    watch_specs: list[WatchSpec] | None = None,
    target_job_name: str = EVAL_REGRESSION_JOB_NAME,
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS,
):
    """Build a ``@sensor``-decorated function bound to a specific repo root.

    Tests call this with a ``tmp_path`` repo root and an isolated
    watch surface; production calls it once with the live
    ``$REPO_ROOT`` and the default watch list.

    The returned object is a Dagster ``SensorDefinition``; Dagster
    treats it the same as a sensor declared with the
    ``@sensor`` decorator at module top level.
    """
    if watch_specs is None:
        watch_specs = _default_watch_paths()

    def _sensor_fn(
        context: SensorEvaluationContext,
    ) -> Iterator[RunRequest | SkipReason]:
        """Scan the watch surface; yield one ``RunRequest`` per affected config.

        Tick semantics:

        1. Read the previous cursor (a JSON ``{path: mtime_ns}``
           blob) from ``context.cursor`` (Dagster persists it
           across ticks). On the first tick,
           ``context.cursor`` is ``None`` — we initialize it
           with the current mtime snapshot and yield a
           ``SkipReason`` ("no baseline"). This means a fresh
           deploy doesn't immediately flood the queue with
           eval jobs; the operator can trigger a baseline
           manually with ``make eval`` if they want a populated
           leaderboard on day one.
        2. Compute the current mtime snapshot.
        3. Diff against the previous cursor. Changed / removed
           paths are tracked.
        4. If no changes, persist the (unchanged) snapshot and
           yield ``SkipReason``.
        5. If changes exist, reduce them to the affected
           *config* list (``_affected_configs``), yield one
           ``RunRequest`` per config with the changed paths
           in the run tags, then persist the new snapshot.
        """
        prev: dict[str, int] = {}
        if context.cursor:
            try:
                prev = json.loads(context.cursor)
            except (TypeError, ValueError):
                logger.warning(
                    "config_change_sensor: corrupt cursor, re-baselining"
                )
                prev = {}

        curr = _scan_mtimes(repo_root, watch_specs)

        # First tick: bootstrap the cursor and skip. A fresh
        # deploy shouldn't fire eval jobs on its very first
        # tick just because Dagster started up — the operator
        # can trigger a baseline manually with ``make eval`` if
        # they want a populated leaderboard on day one. After
        # this tick, ``prev`` is non-empty and subsequent ticks
        # diff against it normally.
        if not prev:
            context.update_cursor(json.dumps(curr, sort_keys=True))
            return SkipReason(
                f"first tick — cursor bootstrapped with "
                f"{len(curr)} file(s); next real change will fire"
            )

        changed, removed = _diff_mtimes(prev, curr)

        if not changed and not removed:
            # Nothing moved. Persist the (identical) snapshot
            # anyway so a future tick that *adds* a new file
            # (e.g. a new config is dropped into ``configs/``)
            # starts from the correct baseline.
            context.update_cursor(json.dumps(curr, sort_keys=True))
            return SkipReason(
                f"no config changes (watched {len(curr)} files, "
                f"debounce={debounce_seconds}s)"
            )

        if not changed and removed:
            # Files were deleted but nothing moved forward.
            # Persist the new snapshot (without the removed
            # paths) and skip — removals don't change
            # retrieval quality on the surviving configs.
            new_curr = {k: v for k, v in curr.items()}
            context.update_cursor(json.dumps(new_curr, sort_keys=True))
            return SkipReason(
                f"{len(removed)} watched file(s) removed; no rerun "
                f"(removed: {', '.join(removed[:5])}"
                f"{'…' if len(removed) > 5 else ''})"
            )

        affected = _affected_configs(changed, repo_root)
        if not affected:
            # Path-level changes exist but no retrieval config
            # is affected. Persist + skip.
            context.update_cursor(json.dumps(curr, sort_keys=True))
            return SkipReason(
                f"{len(changed)} path(s) changed but no retrieval config "
                f"is affected (changed: {', '.join(changed[:3])}"
                f"{'…' if len(changed) > 3 else ''})"
            )

        # Persist BEFORE yielding so a downstream crash doesn't
        # leave the cursor behind the actual file state.
        context.update_cursor(json.dumps(curr, sort_keys=True))

        logger.info(
            "config_change_sensor: %d changed path(s) → %d config(s): %s",
            len(changed),
            len(affected),
            ", ".join(affected),
        )
        # Yield one RunRequest per affected config so the runs
        # are independent in the Dagster UI (each has its own
        # logs, timing, and per-config leaderboard).
        for idx, config_name in enumerate(affected):
            run_key = f"{config_name}:{max(curr[p] for p in changed)}"
            yield RunRequest(
                run_key=run_key,
                job_name=target_job_name,
                tags={
                    "config": config_name,
                    "config_change": "true",
                    "triggered_by": "config_change_sensor",
                    "changed_paths": ",".join(changed[:10]),
                    "batch_index": str(idx),
                },
            )

    from dagster import sensor as _sensor_decorator

    return _sensor_decorator(
        job_name=target_job_name,
        minimum_interval_seconds=debounce_seconds,
        name="config_change_sensor",
        description=(
            "Phase 3.2 (card t_877e48cd). Watches configs/*.yaml + "
            "models.yaml + evals/labeled_v*.jsonl + src/embedding/** + "
            "src/llm/**. Fires eval_regression_job for the affected "
            "retrieval config(s). 30s debounce via Dagster's "
            "minimum_interval_seconds + an mtime cursor that filters "
            "out files that haven't moved since the last tick."
        ),
    )(_sensor_fn)


# Module-level sensor: the default-decorated sensor the Dagster
# ``Definitions`` object picks up. Tests should use
# ``build_config_change_sensor(...)`` to get a sensor bound to a
# throwaway repo root.
config_change_sensor = build_config_change_sensor(repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Job + op
# ---------------------------------------------------------------------------


def _resolve_config_from_context(context: OpExecutionContext) -> str:
    """Read the config name from the run tags (set by the sensor).

    The sensor attaches ``tags={"config": "dense_bge_m3"}`` to the
    ``RunRequest``; Dagster carries those tags through to the
    ``Run`` that the job materializes, and the op can read them
    via ``context.run.tags``. This avoids a separate
    graph-input + run_config plumbing for a one-arg op.

    Falls back to a hard error if the tag is missing — that
    means someone launched the job manually without setting
    the tag, which is a real bug.
    """
    if context.run is None or not context.run.tags.get("config"):
        raise RuntimeError(
            "run_eval_for_config: no 'config' run tag; the job was "
            "launched without a config-change sensor RunRequest. "
            "Launch via the sensor or set run tags={'config': '<name>'}."
        )
    return context.run.tags["config"]


@op(
    description=(
        "Run the offline eval-harness regression for the config in "
        "``context.run.tags['config']``. Shells out to "
        "``python -m eval.run`` for one retrieval config; commits "
        "the regenerated leaderboard files to main if the workspace "
        "is clean. Self-contained — no external APIs."
    ),
)
def run_eval_for_config(context) -> dict[str, Any]:
    """Op body: run ``python -m eval.run`` for the config in ``context.run.tags['config']``.

    The op shells out to ``python -m eval.run`` with the same
    flags ``make eval`` uses, but with ``--output`` /
    ``--markdown-out`` pointed at the per-config file shape that
    Phase 2.9 established
    (``results/leaderboard.<config>.<benchmark>.{csv,md}``). The
    bench path is hard-coded to ``evals/labeled_v300.jsonl`` —
    the v2 set Phase 2.8 shipped. When a new eval-set version
    is released, the bench path is bumped here.

    Hard rules (card body):
    * **Self-contained regression.** No Brave Search, no
      Cohere rerank, no Anthropic API. The eval runner hits
      ``POST /search`` against the live local API (same surface
      as ``make eval``) — which is offline from the operator's
      perspective (no external network) but does need the
      API on ``localhost:18001`` to be up. The Makefile's
      ``make up`` and ``make dagster-up`` targets are the
      operator's responsibility.
    * **Branch.** ``main``. No feature branches.
    * **Commit.** If the workspace is clean (and on ``main``),
      commit the regenerated leaderboard files. If dirty, the
      op still runs the eval but leaves the commit to the
      operator — same surface as ``make eval`` not
      auto-committing.

    Returns a dict of metadata the Dagster UI surfaces on the
    op panel (leaderboard paths, commit SHA, run row count).
    """
    config = _resolve_config_from_context(context)
    cfg_path = REPO_ROOT / "configs" / f"{config}.yaml"
    if not cfg_path.exists():
        raise RuntimeError(
            f"run_eval_for_config: no config at {cfg_path} "
            f"(job was launched with config={config!r})"
        )

    bench_path = REPO_ROOT / DEFAULT_BENCHMARK
    bench_stem = bench_path.stem  # e.g. "labeled_v300"
    output_csv = REPO_ROOT / "results" / f"leaderboard.{config}.{bench_stem}.csv"
    output_md = REPO_ROOT / "results" / f"leaderboard.{config}.{bench_stem}.md"

    # Ensure the results dir exists (eval.run creates it via
    # the append-mode write, but committing later needs it to
    # exist for the ``git add`` step).
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "eval.run",
        "--benchmark",
        str(bench_path.relative_to(REPO_ROOT)),
        "--config",
        str(cfg_path.relative_to(REPO_ROOT)),
        "--output",
        str(output_csv.relative_to(REPO_ROOT)),
        "--markdown-out",
        str(output_md.relative_to(REPO_ROOT)),
        "--db",
        str(REPO_ROOT / "results" / "eval.duckdb"),
        "--mlflow-tracking-uri",
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:15000"),
    ]
    if os.environ.get("NO_MLFLOW"):
        cmd.append("--no-mlflow")

    context.log.info("run_eval_for_config(%s): %s", config, " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if result.returncode != 0:
        # Surface the eval-runner stderr in the Dagster UI;
        # the operator shouldn't have to dig into the
        # container logs to figure out why the regression
        # failed.
        raise RuntimeError(
            f"eval.run failed (rc={result.returncode}, config={config}):\n"
            f"--- stderr ---\n{result.stderr[-4000:]}\n"
            f"--- stdout ---\n{result.stdout[-2000:]}"
        )

    commit_sha = _maybe_commit_results(
        config=config,
        output_csv=output_csv,
        output_md=output_md,
        log=context.log.info,
    )

    return {
        "config": config,
        "benchmark": DEFAULT_BENCHMARK,
        "leaderboard_csv": str(output_csv.relative_to(REPO_ROOT)),
        "leaderboard_md": str(output_md.relative_to(REPO_ROOT)),
        "commit_sha": commit_sha or "",
    }


@job(
    name=EVAL_REGRESSION_JOB_NAME,
    description=(
        "Phase 3.2 (card t_877e48cd). Runs the offline eval-harness "
        "regression for one retrieval config. Triggered by the "
        "config_change_sensor. Per-config file shape: "
        "results/leaderboard.<config>.<benchmark>.{csv,md}. "
        "Self-contained — no external network calls (the live "
        "POST /search target is the local API on :18001, same "
        "surface as ``make eval``)."
    ),
    tags={"phase": "3.2", "trigger": "config_change"},
)
def eval_regression_job() -> None:
    """The job graph: one op per run. The op body is the work.

    Dagster 1.13.x requires a body to define the op graph; a
    zero-arg call into ``run_eval_for_config.alias(...)`` is
    the documented shape for a single-op job.
    """
    run_eval_for_config()


# ---------------------------------------------------------------------------
# Commit helper
# ---------------------------------------------------------------------------


def _maybe_commit_results(
    *,
    config: str,
    output_csv: Path,
    output_md: Path,
    log,
) -> str | None:
    """Commit the regenerated leaderboard files if the workspace is clean.

    Card rule: "sensor runs in CI / local-dev — write to a
    working branch if dirty, or commit directly to ``main`` if
    you confirm the workspace is clean."

    We treat "clean" as: ``git status --porcelain`` is empty
    AND the current branch is ``main``. The op DOES NOT
    ``git stash`` or ``git reset`` — the operator's local
    edits win. If the workspace is dirty, we log a warning
    and return ``None`` (the eval result is still on disk
    for the operator to stage and commit manually).
    """
    try:
        branch_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log("commit skipped: git unavailable: %s", exc)
        return None
    branch = branch_proc.stdout.strip()

    if branch != "main":
        log(
            "commit skipped: on branch %r (sensor only commits to main); "
            "leaderboard regenerated and on disk for manual commit",
            branch,
        )
        return None

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        log(
            "commit skipped: workspace dirty (operator-local edits "
            "present); leaderboard regenerated and on disk for manual "
            "commit. Porcelain:\n%s",
            status.stdout[:2000],
        )
        return None

    rel_csv = str(output_csv.relative_to(REPO_ROOT))
    rel_md = str(output_md.relative_to(REPO_ROOT))
    add = subprocess.run(
        ["git", "add", rel_csv, rel_md],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if add.returncode != 0:
        log(
            "commit skipped: git add failed (rc=%d): %s",
            add.returncode,
            add.stderr,
        )
        return None

    msg = f"Phase 3.2 (card t_877e48cd): {config} leaderboard refresh"
    commit = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        log(
            "commit skipped: git commit failed (rc=%d): %s",
            commit.returncode,
            commit.stderr,
        )
        return None

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    log("committed %s @ %s", msg, sha)
    return sha


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_DEBOUNCE_SECONDS",
    "EVAL_REGRESSION_JOB_NAME",
    "WatchSpec",
    "build_config_change_sensor",
    "config_change_sensor",
    "eval_regression_job",
    "run_eval_for_config",
    "_affected_configs",
    "_diff_mtimes",
    "_scan_mtimes",
    "_maybe_commit_results",
]
