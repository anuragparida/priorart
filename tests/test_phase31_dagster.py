"""Tests for the Phase 3.1 Dagster integration (card t_7928b3e2).

Scope:
- The ``Definitions`` object loads cleanly (no import errors).
- The asset graph has the 5 named assets + the right lineage.
- The nightly schedule + job wiring match ``models.yaml``.
- The asset helpers (snapshot discovery, JSONL count) work against
  a small temp-data fixture.

What this test does NOT cover:
- Live materialization. The scraper subprocesses hit Algolia /
  Wayback / HN Algolia + Firecrawl and bge-m3 takes ~20-30 min
  for 10K rows. End-to-end materialization is verified manually
  via ``make dagster-up`` + the Dagster UI, not in pytest.

Hard rule from the card body:
> Default retrieval config MUST run offline.
> Don't add Dagster-side code that touches external services during
> asset materialization.

These tests are pure — they don't scrape, they don't embed, they
don't touch the live Postgres. The pg_engine fixture (see conftest)
is unused here; that's deliberate.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.dagster_assets import assets as dagster_assets
from src.dagster_assets.definitions import defs


# ---------------------------------------------------------------------------
# Definitions object — smoke + structural shape
# ---------------------------------------------------------------------------


EXPECTED_ASSET_NAMES = {
    "yc_directory",
    "product_hunt_archive",
    "hn_show_posts",
    "company_embeddings",
    "eval_benchmark",
}


def test_definitions_load() -> None:
    """``defs`` is a Dagster ``Definitions`` instance — sanity check."""
    assert defs is not None
    assert hasattr(defs, "assets")
    assert hasattr(defs, "jobs")
    assert hasattr(defs, "schedules")


def test_asset_count_is_five() -> None:
    """Card body: 5 assets. Not 4, not 6."""
    # In Dagster 1.13.x, ``defs.assets`` is a list of ``AssetsDefinition``
    # objects (one per ``@asset`` decorated function).
    assert len(defs.assets) == 5, (
        f"expected 5 assets, got {len(defs.assets)}: "
        f"{[a.key.to_user_string() for a in defs.assets]}"
    )


def test_asset_names_match_card_body() -> None:
    """The 5 names match the card body verbatim."""
    asset_keys = {a.key.to_user_string() for a in defs.assets}
    assert asset_keys == EXPECTED_ASSET_NAMES, (
        f"asset name mismatch: got {sorted(asset_keys)}, "
        f"expected {sorted(EXPECTED_ASSET_NAMES)}"
    )


def test_lineage_matches_card_body() -> None:
    """Card body: YC -> PH -> HN -> companies -> embeddings.

    The exact DAG the architecture doc calls out:

        yc_directory ─┐
        product_hunt_archive ─┼─→ company_embeddings
        hn_show_posts ─┘

        eval_benchmark is independent (the freshness check — no
        corpus edges).
    """
    deps_by_asset: dict[str, set[str]] = {}
    for a in defs.assets:
        deps_by_asset[a.key.to_user_string()] = {
            d.to_user_string() for d in a.dependency_keys
        }

    assert deps_by_asset["yc_directory"] == set(), "yc_directory must be a leaf"
    assert deps_by_asset["product_hunt_archive"] == set(), "product_hunt_archive must be a leaf"
    assert deps_by_asset["hn_show_posts"] == set(), "hn_show_posts must be a leaf"
    assert deps_by_asset["eval_benchmark"] == set(), (
        "eval_benchmark must NOT depend on corpus assets — it's the "
        "freshness check on the eval set, not a corpus node."
    )

    assert deps_by_asset["company_embeddings"] == {
        "yc_directory",
        "product_hunt_archive",
        "hn_show_posts",
    }, (
        "company_embeddings must depend on all three source assets "
        "(this is the lineage DAG documented in ARCHITECTURE.md)."
    )


def test_nightly_job_targets_four_corpus_assets() -> None:
    """The nightly job materializes the 4 corpus assets; eval_benchmark is excluded."""
    jobs = {j.name: j for j in defs.jobs}
    assert "nightly_re_embedding_job" in jobs
    job = jobs["nightly_re_embedding_job"]

    # Dagster 1.13.x exposes ``job.selection`` as a ``KeysAssetSelection``
    # (an unresolved expression — only resolves once the asset graph is
    # built). We pull the keys off the selection's internal list — the
    # strings show up directly in the ``__repr__`` output above.
    # The stable, public way to read the selection is via ``str(selection)``
    # which Dagster formats as ``key:"yc_directory" or key:"product_hunt_archive" or ...``.
    sel_str = str(job.selection)
    expected_keys = {"yc_directory", "product_hunt_archive", "hn_show_posts", "company_embeddings"}
    selected = {
        token.split('"')[1] for token in sel_str.split() if token.startswith("key:")
    }
    assert selected == expected_keys, (
        f"nightly_re_embedding_job must select exactly the 4 corpus "
        f"assets (eval_benchmark excluded). got {selected}"
    )
    # Belt-and-suspenders: confirm eval_benchmark is NOT in the selection.
    assert "eval_benchmark" not in sel_str


def test_nightly_schedule_wiring() -> None:
    """The @daily schedule fires the nightly job at 02:30 UTC."""
    schedules = {s.name: s for s in defs.schedules}
    assert "nightly_re_embedding" in schedules
    sched = schedules["nightly_re_embedding"]

    assert sched.cron_schedule == "30 2 * * *", (
        f"schedule cron must be '30 2 * * *' per models.yaml; got {sched.cron_schedule!r}"
    )
    assert str(sched.execution_timezone) == "UTC", (
        f"schedule timezone must be UTC; got {sched.execution_timezone!r}"
    )
    assert sched.job_name == "nightly_re_embedding_job"


def test_models_yaml_dagster_section_in_sync() -> None:
    """The asset list in models.yaml matches the Definitions object.

    This is the contract: ``models.yaml: dagster.assets`` is the
    operator-facing record; ``src.dagster_assets.definitions`` is
    the Dagster-facing record. They must agree — otherwise the
    audit trail is wrong.
    """
    import yaml  # noqa: PLC0415 — keep the import local so a slim
    # env without pyyaml still loads the rest of the module.

    cfg = yaml.safe_load(Path("models.yaml").read_text())
    yaml_assets = sorted(cfg["dagster"]["assets"])
    defs_assets = sorted(a.key.to_user_string() for a in defs.assets)
    assert yaml_assets == defs_assets, (
        f"models.yaml dagster.assets ({yaml_assets}) doesn't match "
        f"Definitions ({defs_assets}). Update one or the other."
    )

    # Cron + timezone match too.
    assert cfg["dagster"]["nightly_re_embedding"]["cron"] == "30 2 * * *"
    assert cfg["dagster"]["nightly_re_embedding"]["execution_timezone"] == "UTC"
    assert cfg["dagster"]["webserver_port"] == 13002


# ---------------------------------------------------------------------------
# Helpers — _latest_snapshot, _read_jsonl_count
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_latest_snapshot_picks_most_recent_date(tmp_path: Path) -> None:
    """_latest_snapshot returns the file with the latest YYYY-MM-DD in its name."""
    _write_jsonl(tmp_path / "yc_2026-06-01.jsonl", [{"name": "old"}])
    _write_jsonl(tmp_path / "yc_2026-06-29.jsonl", [{"name": "new"}])
    _write_jsonl(tmp_path / "yc_2026-06-15.jsonl", [{"name": "mid"}])

    # Patch the module-level SNAPSHOTS_DIR to use tmp_path.
    original = dagster_assets.SNAPSHOTS_DIR
    dagster_assets.SNAPSHOTS_DIR = tmp_path
    try:
        latest = dagster_assets._latest_snapshot("yc")
    finally:
        dagster_assets.SNAPSHOTS_DIR = original

    assert latest is not None
    assert latest.name == "yc_2026-06-29.jsonl"


def test_latest_snapshot_returns_none_when_missing(tmp_path: Path) -> None:
    """_latest_snapshot returns None instead of raising for a missing prefix."""
    original = dagster_assets.SNAPSHOTS_DIR
    dagster_assets.SNAPSHOTS_DIR = tmp_path
    try:
        result = dagster_assets._latest_snapshot("does_not_exist")
    finally:
        dagster_assets.SNAPSHOTS_DIR = original
    assert result is None


def test_latest_snapshot_handles_hn_show_prefix(tmp_path: Path) -> None:
    """``hn_show_<date>.jsonl`` is the on-disk convention (matches the scraper)."""
    _write_jsonl(tmp_path / "hn_show_2026-06-29.jsonl", [{"object_id": "x"}])
    original = dagster_assets.SNAPSHOTS_DIR
    dagster_assets.SNAPSHOTS_DIR = tmp_path
    try:
        latest = dagster_assets._latest_snapshot("hn_show")
    finally:
        dagster_assets.SNAPSHOTS_DIR = original
    assert latest is not None
    assert latest.name == "hn_show_2026-06-29.jsonl"


def test_read_jsonl_count(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, [{"a": 1}, {"a": 2}, {"a": 3}])
    assert dagster_assets._read_jsonl_count(p) == 3

    # Empty lines should not be counted.
    p.write_text('{"a":1}\n\n{"a":2}\n\n\n')
    assert dagster_assets._read_jsonl_count(p) == 2

    # Missing file = 0.
    assert dagster_assets._read_jsonl_count(tmp_path / "missing.jsonl") == 0


def test_read_manifest_count(tmp_path: Path) -> None:
    """_read_manifest_count pulls the ``count`` field out of a Phase 1.2/2.5/2.6 manifest."""
    p = tmp_path / "yc_2026-06-29.manifest.json"
    p.write_text(json.dumps({"count": 5949, "schema_version": "1.0.0"}))
    assert dagster_assets._read_manifest_count(p) == 5949

    # Missing file = 0.
    assert dagster_assets._read_manifest_count(tmp_path / "missing.json") == 0

    # Malformed JSON = 0 (best-effort).
    (tmp_path / "bad.json").write_text("not json")
    assert dagster_assets._read_manifest_count(tmp_path / "bad.json") == 0


def test_read_corpus_manifest(tmp_path: Path) -> None:
    """_read_corpus_manifest returns the full Phase 2.7 manifest dict."""
    p = tmp_path / "corpus_2026-06-29.manifest.json"
    payload = {"totals": {"embedded": 10983, "kept": 10942}, "schema_version": "1.0.0"}
    p.write_text(json.dumps(payload))
    assert dagster_assets._read_corpus_manifest(p) == payload
    assert dagster_assets._read_corpus_manifest(tmp_path / "missing.json") == {}


# ---------------------------------------------------------------------------
# eval_benchmark — surface the version + freshness
# ---------------------------------------------------------------------------


def test_eval_benchmark_surfaces_version_label(tmp_path: Path) -> None:
    """The asset reads ``models.yaml: dagster.eval_benchmark.version``."""
    # Patch both MODELS_YAML_PATH and EVALS_DIR for an isolated read.
    (tmp_path / "models.yaml").write_text(
        "dagster:\n  eval_benchmark:\n    version: v300-llm-v2\n"
    )
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "labeled_v300.jsonl").write_text(
        '{"id": "ev-001"}\n{"id": "ev-002"}\n{"id": "ev-003"}\n'
    )

    original_models = dagster_assets.MODELS_YAML_PATH
    original_evals = dagster_assets.EVALS_DIR
    dagster_assets.MODELS_YAML_PATH = tmp_path / "models.yaml"
    dagster_assets.EVALS_DIR = tmp_path / "evals"
    try:
        from dagster import build_asset_context

        ctx = build_asset_context()
        result = dagster_assets.eval_benchmark(ctx)
    finally:
        dagster_assets.MODELS_YAML_PATH = original_models
        dagster_assets.EVALS_DIR = original_evals

    assert result.metadata["version_label"] == "v300-llm-v2"
    assert result.metadata["record_count"] == 3
    assert "age_days" in result.metadata
    assert result.metadata["age_days"] >= 0


def test_eval_benchmark_fails_when_no_eval_set(tmp_path: Path) -> None:
    """An empty evals/ dir is a hard failure — operator must fix it."""
    (tmp_path / "models.yaml").write_text("dagster:\n  eval_benchmark:\n    version: v300\n")
    (tmp_path / "evals").mkdir()  # exists but empty

    original_models = dagster_assets.MODELS_YAML_PATH
    original_evals = dagster_assets.EVALS_DIR
    dagster_assets.MODELS_YAML_PATH = tmp_path / "models.yaml"
    dagster_assets.EVALS_DIR = tmp_path / "evals"
    try:
        from dagster import build_asset_context

        ctx = build_asset_context()
        with pytest.raises(RuntimeError, match="no labeled_v\\*\\.jsonl"):
            dagster_assets.eval_benchmark(ctx)
    finally:
        dagster_assets.MODELS_YAML_PATH = original_models
        dagster_assets.EVALS_DIR = original_evals


# ---------------------------------------------------------------------------
# Asset subprocess failure path — make sure errors bubble up cleanly
# ---------------------------------------------------------------------------


def test_run_subprocess_raises_on_nonzero_exit(tmp_path: Path) -> None:
    """A failing subprocess must raise — Dagster surfaces that as a failed materialization."""
    with pytest.raises(RuntimeError, match="command failed"):
        dagster_assets._run_subprocess(["false"], cwd=tmp_path)


def test_run_subprocess_passes_on_zero_exit(tmp_path: Path) -> None:
    """A successful subprocess returns normally with stdout captured."""
    result = dagster_assets._run_subprocess(
        ["echo", "hello"], cwd=tmp_path
    )
    assert result.returncode == 0
    assert "hello" in result.stdout


# ---------------------------------------------------------------------------
# Date parsing — sanity check on the snapshot filename convention
# ---------------------------------------------------------------------------


def test_latest_snapshot_ignores_non_matching_files(tmp_path: Path) -> None:
    """Files that don't match the ``<prefix>_<date>.jsonl`` shape are ignored."""
    # A non-matching file in the same dir shouldn't trip _latest_snapshot.
    (tmp_path / "README.md").write_text("not a snapshot")
    (tmp_path / "yc_2026-06-29.jsonl").write_text('{"name": "Acme"}')

    original = dagster_assets.SNAPSHOTS_DIR
    dagster_assets.SNAPSHOTS_DIR = tmp_path
    try:
        latest = dagster_assets._latest_snapshot("yc")
    finally:
        dagster_assets.SNAPSHOTS_DIR = original

    assert latest is not None
    assert latest.name == "yc_2026-06-29.jsonl"
    # And the date parses back to the right day.
    d = date.fromisoformat(latest.stem.split("_", 1)[1])
    assert d == date(2026, 6, 29)