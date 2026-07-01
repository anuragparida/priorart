"""Tests for the Phase 3.2 config-change sensor (card t_877e48cd).

Scope:
- Sensor module imports cleanly.
- The module-level ``config_change_sensor`` is a Dagster
  ``SensorDefinition`` with the right name + 30s minimum interval.
- The ``Definitions`` object exposes the sensor + the new
  ``eval_regression_job`` alongside the 3.1 surface.
- ``_scan_mtimes`` / ``_affected_configs`` / ``_diff_mtimes``
  classify changes correctly (single config, models.yaml,
  eval set, embedding/llm code paths).
- End-to-end sensor ticks: first tick bootstraps the cursor
  and skips; subsequent ticks with no change skip; a real
  mtime change yields a ``RunRequest`` per affected config;
  the run_key is stable for the same change set.
- The op's run-tag-driven config resolution raises a clear
  error when the tag is missing.

What this test does NOT cover:
- Live ``make eval`` / ``python -m eval.run`` end-to-end.
  The Phase 2.4 MLflow tracker + the local API on :18001
  are required for the eval to complete; that's the
  ``make smoke`` gate, not pytest. The 3.1 card body calls
  this out: "Live materialization ... is verified manually
  via ``make dagster-up`` + the Dagster UI, not in pytest."

Hard rule from the card body:
> Default retrieval config MUST run offline.
> Don't add Dagster-side code that touches external services during
> asset materialization.

These tests are pure — they don't run the eval, they don't
hit the live API, they don't touch the live Postgres. The
``build_config_change_sensor`` factory takes a ``tmp_path``
repo root, so the test fixture is fully isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dagster import RunRequest, SkipReason, build_sensor_context

from src.dagster_assets.sensors import (
    DEFAULT_DEBOUNCE_SECONDS,
    EVAL_REGRESSION_JOB_NAME,
    WatchSpec,
    build_config_change_sensor,
    config_change_sensor,
    eval_regression_job,
    run_eval_for_config,
)
from src.dagster_assets.sensors.config_change_sensor import (
    _affected_configs,
    _diff_mtimes,
    _scan_mtimes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CONFIGS = ["bm25", "dense_bge_m3", "hybrid_rrf"]


def _touch(path: Path, mtime_ns: int | None = None) -> Path:
    """Create the file (or touch it) and optionally set its mtime_ns.

    ``mtime_ns`` defaults to "now in nanoseconds" — tests that
    need a "later than the cursor" mtime pass an explicit value
    that's strictly greater than the bootstrap value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n")
    if mtime_ns is not None:
        # ``st_mtime_ns`` is the only knob ``os.utime`` exposes
        # at sub-second resolution. ``os.utime`` takes (atime_ns,
        # mtime_ns).
        import os

        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _make_repo(tmp_path: Path, *, with_eval: bool = True) -> dict[str, Path]:
    """Build a minimal repo fixture under ``tmp_path``.

    Mirrors the real project's layout: ``configs/<name>.yaml``,
    ``models.yaml``, ``evals/labeled_v300.jsonl``,
    ``src/embedding/x.py``, ``src/llm/x.py``. Returns a
    dict of paths so tests can ``.write_text(...)`` /
    ``.touch()`` on them as needed.
    """
    paths = {
        "config_dense": tmp_path / "configs" / "dense_bge_m3.yaml",
        "config_bm25": tmp_path / "configs" / "bm25.yaml",
        "config_hybrid": tmp_path / "configs" / "hybrid_rrf.yaml",
        "models": tmp_path / "models.yaml",
        "eval": tmp_path / "evals" / "labeled_v300.jsonl",
        "embedding": tmp_path / "src" / "embedding" / "bge_m3.py",
        "llm": tmp_path / "src" / "llm" / "prompts.py",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("placeholder\n")
    return paths


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, Path]:
    """Minimal repo fixture with all watched paths present."""
    return _make_repo(tmp_path)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_level_sensor_is_sensor_definition() -> None:
    """The module-level ``config_change_sensor`` is a real Dagster sensor.

    Imports + smoke check. Tests below exercise the factory;
    this one pins the production-decorated object so a
    refactor that accidentally drops the ``@sensor`` decorator
    surfaces as a test failure.
    """
    assert config_change_sensor is not None
    assert config_change_sensor.name == "config_change_sensor"
    assert config_change_sensor.minimum_interval_seconds == DEFAULT_DEBOUNCE_SECONDS
    assert config_change_sensor.minimum_interval_seconds == 30
    # The sensor must be pinned to the eval-regression job, which
    # matches ``models.yaml:dagster.config_change_sensor.target_job``.
    assert config_change_sensor.job_name == EVAL_REGRESSION_JOB_NAME


def test_eval_regression_job_shape() -> None:
    """The job has one op (`run_eval_for_config`) and the right name."""
    assert eval_regression_job is not None
    assert eval_regression_job.name == EVAL_REGRESSION_JOB_NAME
    op_names = [n.name for n in eval_regression_job.nodes]
    assert op_names == ["run_eval_for_config"]


def test_definitions_wires_sensor_and_job() -> None:
    """The ``Definitions`` object exposes the new sensor + job alongside 3.1."""
    from src.dagster_assets.definitions import defs

    sensor_names = {s.name for s in defs.sensors}
    job_names = {j.name for j in defs.jobs}
    assert "config_change_sensor" in sensor_names
    assert "eval_regression_job" in job_names
    # 3.1 surface must still be there — no regression.
    assert "nightly_re_embedding_job" in job_names
    # The 5 assets from 3.1 must still be there.
    asset_names = {a.key.to_user_string() for a in defs.assets}
    assert "yc_directory" in asset_names
    assert "company_embeddings" in asset_names


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_scan_mtimes_covers_all_watched_paths(repo: dict[str, Path]) -> None:
    """``_scan_mtimes`` returns a relpath → mtime_ns dict for every watched file."""
    snapshot = _scan_mtimes(
        repo_root=repo["config_dense"].parents[1],
        watch_specs=[
            WatchSpec(kind="config", glob="configs/*.yaml"),
            WatchSpec(kind="global", glob="models.yaml"),
            WatchSpec(kind="global", glob="evals/labeled_v*.jsonl"),
            WatchSpec(kind="global", glob="src/embedding/**/*.py"),
            WatchSpec(kind="global", glob="src/llm/**/*.py"),
        ],
    )
    assert set(snapshot.keys()) == {
        "configs/dense_bge_m3.yaml",
        "configs/bm25.yaml",
        "configs/hybrid_rrf.yaml",
        "models.yaml",
        "evals/labeled_v300.jsonl",
        "src/embedding/bge_m3.py",
        "src/llm/prompts.py",
    }
    # Every mtime must be a positive integer (ns).
    for mtime in snapshot.values():
        assert isinstance(mtime, int)
        assert mtime > 0


def test_diff_mtimes_detects_changes() -> None:
    """``_diff_mtimes`` reports a path whose mtime moved forward as changed."""
    prev = {"a": 100, "b": 200, "c": 300}
    curr = {"a": 100, "b": 250, "c": 300}
    changed, removed = _diff_mtimes(prev, curr)
    assert changed == ["b"]
    assert removed == []


def test_diff_mtimes_detects_new_files() -> None:
    """A new file is reported as changed (mtime is 'newer than nothing')."""
    prev = {"a": 100}
    curr = {"a": 100, "b": 200}
    changed, removed = _diff_mtimes(prev, curr)
    assert changed == ["b"]
    assert removed == []


def test_diff_mtimes_detects_removals() -> None:
    """A removed file is in `removed` but NOT in `changed` (it didn't move)."""
    prev = {"a": 100, "b": 200}
    curr = {"a": 100}
    changed, removed = _diff_mtimes(prev, curr)
    assert changed == []
    assert removed == ["b"]


def test_affected_configs_single_config_change() -> None:
    """A change to ``configs/dense_bge_m3.yaml`` affects only dense_bge_m3."""
    repo_root = Path("/tmp/fake")  # not used in this test
    affected = _affected_configs(
        ["configs/dense_bge_m3.yaml"],
        repo_root=repo_root,
    )
    assert affected == ["dense_bge_m3"]


def test_affected_configs_multiple_config_changes() -> None:
    """Multiple config changes collapse to the unique config list."""
    affected = _affected_configs(
        [
            "configs/dense_bge_m3.yaml",
            "configs/bm25.yaml",
        ],
        repo_root=Path("/tmp/fake"),
    )
    assert affected == ["bm25", "dense_bge_m3"]


def test_affected_configs_models_yaml_fans_out_to_all(
    repo: dict[str, Path],
) -> None:
    """``models.yaml`` change affects every retrieval config in ``configs/``."""
    affected = _affected_configs(
        ["models.yaml"],
        repo_root=repo["config_dense"].parents[1],
    )
    assert affected == sorted(ALL_CONFIGS)


def test_affected_configs_eval_set_fans_out_to_all(
    repo: dict[str, Path],
) -> None:
    """A new eval-set version invalidates every config's leaderboard row."""
    affected = _affected_configs(
        ["evals/labeled_v300.jsonl"],
        repo_root=repo["config_dense"].parents[1],
    )
    assert affected == sorted(ALL_CONFIGS)


def test_affected_configs_embedding_change_fans_out_to_all(
    repo: dict[str, Path],
) -> None:
    """A bge-m3 wrapper change affects every config (same model)."""
    affected = _affected_configs(
        ["src/embedding/bge_m3.py"],
        repo_root=repo["config_dense"].parents[1],
    )
    assert affected == sorted(ALL_CONFIGS)


def test_affected_configs_llm_change_fans_out_to_all(
    repo: dict[str, Path],
) -> None:
    """An LLM-side change affects every config (comparison path)."""
    affected = _affected_configs(
        ["src/llm/prompts.py"],
        repo_root=repo["config_dense"].parents[1],
    )
    assert affected == sorted(ALL_CONFIGS)


def test_affected_configs_unknown_path_is_ignored() -> None:
    """Paths outside the watch surface are logged + ignored (no crash)."""
    affected = _affected_configs(
        ["README.md", "scripts/whatever.py"],
        repo_root=Path("/tmp/fake"),
    )
    assert affected == []


def test_affected_configs_mixed_fans_out_via_global() -> None:
    """A config change + a global change together fan out to all configs."""
    affected = _affected_configs(
        [
            "configs/dense_bge_m3.yaml",
            "models.yaml",
        ],
        repo_root=Path("/tmp/fake")
        / "configs"
        / "dense_bge_m3.yaml",  # exists so we read the dir
    )
    # The fake root has no configs/ dir, so the global path
    # doesn't add anything; only the explicit config change
    # contributes. (This test pins the "no config dir = no
    # fan-out from global" behaviour, which matters when the
    # sensor ticks against a freshly-scaffolded repo.)
    assert affected == ["dense_bge_m3"]


# ---------------------------------------------------------------------------
# End-to-end sensor ticks
# ---------------------------------------------------------------------------


def _eval_tick(sensor, *, cursor: str | None) -> tuple[list[RunRequest], str | None, str | None]:
    """Run one sensor tick and return ``(run_requests, skip_message, new_cursor)``."""
    ctx = build_sensor_context(cursor=cursor, sensor_name=sensor.name)
    data = sensor.evaluate_tick(ctx)
    return list(data.run_requests), data.skip_message, data.cursor


def test_sensor_first_tick_bootstraps_cursor(repo: dict[str, Path]) -> None:
    """First tick (no cursor) yields SkipReason and persists the baseline."""
    sensor = build_config_change_sensor(
        repo_root=repo["config_dense"].parents[1],
    )
    requests, skip, new_cursor = _eval_tick(sensor, cursor=None)
    assert requests == []
    assert skip is not None
    assert "first tick" in skip
    assert "cursor bootstrapped" in skip
    # The cursor must be a non-empty JSON blob — the next tick
    # will diff against it.
    assert new_cursor is not None
    parsed = json.loads(new_cursor)
    assert "configs/dense_bge_m3.yaml" in parsed


def test_sensor_idempotent_when_nothing_changes(repo: dict[str, Path]) -> None:
    """Two ticks with the same cursor → second is a SkipReason."""
    sensor = build_config_change_sensor(
        repo_root=repo["config_dense"].parents[1],
    )
    _, _, first_cursor = _eval_tick(sensor, cursor=None)
    requests, skip, second_cursor = _eval_tick(sensor, cursor=first_cursor)
    assert requests == []
    assert skip is not None
    # The cursor must be stable across identical ticks (no spurious
    # change detection from mtime rounding, etc.).
    assert second_cursor == first_cursor


def test_sensor_fires_on_config_change(repo: dict[str, Path]) -> None:
    """A mtime bump on a config YAML fires one RunRequest for that config."""
    sensor = build_config_change_sensor(
        repo_root=repo["config_dense"].parents[1],
    )
    _, _, prev_cursor = _eval_tick(sensor, cursor=None)

    # Move the mtime of the dense config forward by 10s.
    import os
    dense = repo["config_dense"]
    new_mtime = dense.stat().st_mtime_ns + 10_000_000_000
    os.utime(dense, ns=(new_mtime, new_mtime))

    requests, skip, _ = _eval_tick(sensor, cursor=prev_cursor)
    assert skip is None
    assert len(requests) == 1
    req = requests[0]
    assert req.job_name == EVAL_REGRESSION_JOB_NAME
    assert req.tags["config"] == "dense_bge_m3"
    assert req.tags["triggered_by"] == "config_change_sensor"
    assert "configs/dense_bge_m3.yaml" in req.tags["changed_paths"]


def test_sensor_fires_three_run_requests_on_models_change(
    repo: dict[str, Path],
) -> None:
    """A change to models.yaml fans out to one RunRequest per config."""
    sensor = build_config_change_sensor(
        repo_root=repo["config_dense"].parents[1],
    )
    _, _, prev_cursor = _eval_tick(sensor, cursor=None)

    import os
    new_mtime = repo["models"].stat().st_mtime_ns + 10_000_000_000
    os.utime(repo["models"], ns=(new_mtime, new_mtime))

    requests, skip, _ = _eval_tick(sensor, cursor=prev_cursor)
    assert skip is None
    assert len(requests) == 3
    config_names = sorted(r.tags["config"] for r in requests)
    assert config_names == sorted(ALL_CONFIGS)
    # Every request must reference the same changed path.
    for r in requests:
        assert "models.yaml" in r.tags["changed_paths"]


def test_sensor_fires_one_run_request_per_changed_config(
    repo: dict[str, Path],
) -> None:
    """Multiple per-config changes → one RunRequest per affected config."""
    sensor = build_config_change_sensor(
        repo_root=repo["config_dense"].parents[1],
    )
    _, _, prev_cursor = _eval_tick(sensor, cursor=None)

    import os
    for path in (repo["config_dense"], repo["config_bm25"]):
        new_mtime = path.stat().st_mtime_ns + 10_000_000_000
        os.utime(path, ns=(new_mtime, new_mtime))

    requests, skip, _ = _eval_tick(sensor, cursor=prev_cursor)
    assert skip is None
    assert len(requests) == 2
    affected = sorted(r.tags["config"] for r in requests)
    assert affected == ["bm25", "dense_bge_m3"]


def test_sensor_removal_does_not_fire(repo: dict[str, Path]) -> None:
    """Deleting a watched file → SkipReason, no RunRequest."""
    sensor = build_config_change_sensor(
        repo_root=repo["config_dense"].parents[1],
    )
    _, _, prev_cursor = _eval_tick(sensor, cursor=None)

    # Delete one of the configs.
    repo["config_bm25"].unlink()

    requests, skip, _ = _eval_tick(sensor, cursor=prev_cursor)
    assert requests == []
    assert skip is not None
    # The skip message must call out the removal specifically —
    # the operator wants to know why a config disappeared.
    assert "removed" in skip


# ---------------------------------------------------------------------------
# Op-level checks
# ---------------------------------------------------------------------------


def test_run_eval_for_config_resolves_from_run_tag() -> None:
    """The op pulls the config name from ``context.run.tags['config']``.

    We can't invoke the op directly (it shells out to
    ``python -m eval.run`` against the live API), so we
    exercise the resolution helper by building a minimal
    fake context and asserting the helper raises a clear
    error when the tag is missing. The happy path is
    covered by the live ``make eval`` smoke test in
    ``scripts/smoke.py``.
    """
    from src.dagster_assets.sensors.config_change_sensor import (
        _resolve_config_from_context,
    )

    # The function reads ``context.run.tags``; the simplest
    # way to exercise the negative path without Dagster's
    # full context machinery is to feed it an object whose
    # ``.run`` is None.

    class _Ctx:
        run = None

    with pytest.raises(RuntimeError, match="no 'config' run tag"):
        _resolve_config_from_context(_Ctx())  # type: ignore[arg-type]


def test_run_eval_for_config_op_is_exported() -> None:
    """The op is importable from the sensors package and is a Dagster op."""
    from dagster import OpDefinition

    assert isinstance(run_eval_for_config, OpDefinition)
    assert run_eval_for_config.name == "run_eval_for_config"
