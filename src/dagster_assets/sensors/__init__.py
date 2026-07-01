"""Dagster sensors — Phase 3.2 (card t_877e48cd).

Lives under ``src/dagster_assets/sensors/`` (not
``src/dagster/sensors/``) for the same reason the parent package
avoids the ``src/dagster/`` name: a project-local package named
``dagster`` shadows the real ``dagster`` package on ``sys.path``
because the project entry point is closer than the venv's
site-packages. See ``src/dagster_assets/__init__.py`` for the
long-form rationale.

Currently houses a single sensor:

* ``config_change_sensor`` — watches ``configs/*.yaml``,
  ``models.yaml``, ``evals/labeled_v*.jsonl``, and
  ``src/embedding/**`` + ``src/llm/**`` (the same surface the
  Phase 3.6 GitHub Actions workflow watches, per the card's
  explicit "keep them in sync" rule). On a change, fires the
  ``eval_regression_job`` so the leaderboard stays in step
  with the config / model / eval-set that drives it.
"""

from __future__ import annotations

from src.dagster_assets.sensors.config_change_sensor import (
    DEFAULT_BENCHMARK,
    DEFAULT_DEBOUNCE_SECONDS,
    EVAL_REGRESSION_JOB_NAME,
    WatchSpec,
    build_config_change_sensor,
    config_change_sensor,
    eval_regression_job,
    run_eval_for_config,
)

__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_DEBOUNCE_SECONDS",
    "EVAL_REGRESSION_JOB_NAME",
    "WatchSpec",
    "build_config_change_sensor",
    "config_change_sensor",
    "eval_regression_job",
    "run_eval_for_config",
]
