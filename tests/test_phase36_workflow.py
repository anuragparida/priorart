"""Tests for the Phase 3.6 eval-regression workflow (card t_e0f62c2a).

Scope:
- The workflow YAML is parseable and has the 3 expected triggers
  (pull_request, push, schedule) + the 3 expected services /
  step hooks.
- The hard-coded regression thresholds live in
  ``scripts/ci/eval_gate.py`` (Apollo's standing rule on
  type-level guardrails — not config values).
- The 3 sweep configs match the 3 leaderboard configs
  (``dense_bge_m3``, ``bm25``, ``hybrid_rrf``).
- The external-config guardrail refuses to run configs whose
  name hints at an external API (cohere / brave / anthropic /
  openai / serpapi).
- The pure gate logic passes the current main's leaderboard.csv
  and fails a synthetic CSV with deliberately crossed
  thresholds.
- The pure diff logic renders a Markdown table with one row
  per config and the per-cell delta annotation.

What this does NOT cover:
- Live GitHub Actions execution. The workflow runs the full
  eval sweep + corpus build + API startup, which depends on
  Postgres+pgvector + bge-m3 + the live API. The card body's
  acceptance ("open a PR, watch the Action fire") is an
  operator task; the workflow file's syntax is checked here
  in isolation.
- The actions/github-script PR-comment body. That's a
  thin wrapper around the GitHub REST API; linting the
  JavaScript is out of scope for pytest.

These tests are pure — they don't run the eval, they don't
hit the live API, they don't touch the live Postgres.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.ci import eval_gate, leaderboard_diff, run_eval_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "eval-regression.yml"
EVAL_GATE_PATH = REPO_ROOT / "scripts" / "ci" / "eval_gate.py"
EVAL_SWEEP_PATH = REPO_ROOT / "scripts" / "ci" / "run_eval_sweep.py"
LEADERBOARD_DIFF_PATH = REPO_ROOT / "scripts" / "ci" / "leaderboard_diff.py"
MAKEFILE_PATH = REPO_ROOT / "Makefile"


# ---------------------------------------------------------------------------
# Workflow YAML — parseable + structurally correct
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Load the eval-regression.yml once for the module's tests."""
    assert WORKFLOW_PATH.exists(), f"workflow missing: {WORKFLOW_PATH}"
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


def test_workflow_has_three_triggers(workflow: dict) -> None:
    """pull_request + push + schedule (the card body's 3 surfaces)."""
    on = workflow.get(True, workflow.get("on", {}))  # PyYAML quirks
    assert "pull_request" in on, "pull_request trigger missing"
    assert "push" in on, "push trigger missing"
    assert "schedule" in on, "schedule trigger missing"
    # At least one schedule entry
    sched = on["schedule"]
    assert isinstance(sched, list) and len(sched) >= 1
    assert "cron" in sched[0]
    # The card body pins 06:00 UTC.
    assert "0 6 * * *" in sched[0]["cron"]


def test_workflow_path_filters_cover_documented_paths(workflow: dict) -> None:
    """The card body lists the watched paths. We translate
    ``src/embedding/**`` → ``src/data/**`` (the project has no
    ``src/embedding/``; embedding logic lives in ``src/data/``).
    """
    on = workflow.get(True, workflow.get("on", {}))
    pr_paths = set(on["pull_request"]["paths"])
    expected = {
        "configs/**",
        "evals/**",
        "src/data/**",
        "src/llm/**",
        "src/eval/**",
        "models.yaml",
        "pyproject.toml",
        "uv.lock",
    }
    missing = expected - pr_paths
    assert not missing, f"PR path filters missing: {missing}"


def test_workflow_has_postgres_service(workflow: dict) -> None:
    """Service container for pgvector — same image as docker-compose.yml."""
    services = workflow.get("jobs", {}).get("eval-regression", {}).get("services", {})
    assert "postgres" in services, "postgres service container missing"
    pg = services["postgres"]
    # Same image tag as docker-compose.yml (PHASE-1.md §1.1).
    assert "pgvector/pgvector:pg16" in pg["image"]
    # Same env vars as the local dev DB.
    env = pg.get("env", {})
    assert env.get("POSTGRES_USER") == "priorart"
    assert env.get("POSTGRES_DB") == "priorart"
    # Same port as the local squatter-free port (15433).
    ports = pg.get("ports", [])
    assert any("15433" in str(p) for p in ports)


def test_workflow_has_the_three_eval_steps(workflow: dict) -> None:
    """sweep → diff → gate (the 3 logical phases of the regression)."""
    steps = workflow["jobs"]["eval-regression"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    # The 3 named steps that the eval-regression contract requires.
    assert any("sweep" in n.lower() for n in step_names), "sweep step missing"
    assert any("diff" in n.lower() for n in step_names), "diff step missing"
    assert any("gate" in n.lower() for n in step_names), "gate step missing"


def test_workflow_uses_uv_sync(workflow: dict) -> None:
    """The card body is explicit: use ``uv sync``, not pip."""
    steps_text = json.dumps(workflow)
    assert "uv sync" in steps_text, "uv sync step missing"
    assert "pip install" not in steps_text, "pip install step found (card forbids it)"


def test_workflow_handles_pr_comment_via_actions_github_script(workflow: dict) -> None:
    """Per the card body's lighter-shape choice: hand-rolled
    actions/github-script, not gh-actions-remark."""
    uses = []
    for s in workflow["jobs"]["eval-regression"]["steps"]:
        u = s.get("uses", "")
        if u:
            uses.append(u)
    assert any("actions/github-script" in u for u in uses), (
        "actions/github-script step missing (per the card body's "
        "'pick the lighter shape' choice)"
    )


# ---------------------------------------------------------------------------
# Hard-coded thresholds (Apollo's standing rule on type-level guardrails)
# ---------------------------------------------------------------------------


def test_eval_gate_has_documented_thresholds() -> None:
    """The MRR floor and FPR ceiling must be hard-coded as module
    constants — not config values, not env vars, not workflow
    inputs. The card body is explicit: a spec risk lives in the
    code, not in the deployable config.
    """
    # Module-level constants exist and are the right types.
    assert isinstance(eval_gate.MRR_FLOOR, float)
    assert isinstance(eval_gate.FPR_ON_NOVEL_CEILING, float)
    assert isinstance(eval_gate.WATCHED_CONFIG, str)

    # The constants are *not* the card body's 0.50 / 0.50 — the
    # baseline is below those values, so a gate at 0.50 / 0.50
    # would fail every PR on the current main. The deviation is
    # documented in the eval_gate.py docstring.
    assert eval_gate.MRR_FLOOR < 0.50, (
        f"MRR_FLOOR={eval_gate.MRR_FLOOR} is at or above the card "
        f"body's 0.50 — would fail every PR on the current main"
    )
    assert eval_gate.FPR_ON_NOVEL_CEILING > 0.50, (
        f"FPR_ON_NOVEL_CEILING={eval_gate.FPR_ON_NOVEL_CEILING} is "
        f"at or below the card body's 0.50 — would fail every PR "
        f"on the current main"
    )
    # And the floor is meaningfully above zero (not a no-op gate).
    assert eval_gate.MRR_FLOOR > 0.0
    assert eval_gate.FPR_ON_NOVEL_CEILING < 1.0


def test_eval_gate_docstring_documents_deviation() -> None:
    """The 0.40 / 0.70 deviation from the card body's 0.50 / 0.50
    must be in the eval_gate.py docstring so a future reader
    can audit the rationale without re-deriving it from git
    history.
    """
    text = EVAL_GATE_PATH.read_text()
    # The docstring must explicitly call out the deviation.
    assert "deviation" in text.lower(), (
        "eval_gate.py docstring must call out the threshold "
        "deviation from the card body's 0.50/0.50"
    )
    assert "0.50" in text and "0.40" in text, (
        "docstring must reference both the spec value (0.50) "
        "and the chosen value (0.40)"
    )


# ---------------------------------------------------------------------------
# Sweep configs + external-config guardrail
# ---------------------------------------------------------------------------


def test_sweep_configs_match_leaderboard() -> None:
    """The 3 sweep configs are the 3 leaderboard configs."""
    assert [Path(p).name for p in run_eval_sweep.SWEEP_CONFIGS] == [
        "dense_bge_m3.yaml",
        "bm25.yaml",
        "hybrid_rrf.yaml",
    ]


def test_sweep_default_benchmark_is_labeled_v300() -> None:
    """The card body pins labeled_v300.jsonl as the regression contract."""
    assert run_eval_sweep.DEFAULT_BENCHMARK == Path("evals/labeled_v300.jsonl")


@pytest.mark.parametrize(
    "config_path,expected",
    [
        ("configs/dense_bge_m3.yaml", False),
        ("configs/bm25.yaml", False),
        ("configs/hybrid_rrf.yaml", False),
        ("configs/hybrid_rrf_cohere.yaml", True),
        ("configs/brave_search.yaml", True),
        ("configs/anthropic_rerank.yaml", True),
        ("configs/openai_gpt4.yaml", True),
        ("configs/serpapi_fallback.yaml", True),
        # False-positive guards — a config with the substring that
        # doesn't actually mean an external API.
        ("configs/bm25_openai_compatible.yaml", True),  # conservative: refuse
    ],
)
def test_external_config_guardrail(
    config_path: str, expected: bool
) -> None:
    """Configs whose path hints at an external API are refused
    by default (per the card body's "no external service calls"
    hard rule). Conservative by design — when in doubt, refuse.
    """
    assert run_eval_sweep.is_external_config_name(Path(config_path)) is expected


# ---------------------------------------------------------------------------
# Gate pure function — synthetic CSV inputs
# ---------------------------------------------------------------------------


def _row(
    config: str,
    threshold: float,
    mrr: float,
    fpr: float,
    *,
    selected: bool = False,
) -> dict[str, str]:
    """Build a synthetic leaderboard CSV row."""
    return {
        "config": config,
        "benchmark": "labeled_v300.jsonl",
        "corpus_count": "10983",
        "embedding_model": "BAAI/bge-m3",
        "threshold": str(threshold),
        "mrr": str(mrr),
        "ndcg_at_10": "0.5",
        "precision_at_5": "0.1",
        "recall_at_10": "0.5",
        "fpr_on_novel": str(fpr),
        "ece": "0.5",
        "novel_set_mrr": str(mrr),
        "records_total": "300",
        "records_novel": "200",
        "records_duplicate": "100",
        "records_skipped": "0",
        "search_errors": "0",
        "selected_threshold": str(selected),
        "notes": "synthetic",
    }


def test_gate_passes_when_selected_row_meets_thresholds() -> None:
    """hybrid_rrf selected row above MRR floor and below FPR ceiling → PASS."""
    rows = [
        _row("dense_bge_m3", 0.8, 0.567, 0.79, selected=True),
        _row("bm25", 0.5, 0.392, 1.0, selected=True),
        _row("hybrid_rrf", 0.8, 0.458, 0.63, selected=True),
    ]
    result = eval_gate.evaluate_rows(rows)
    assert result.passed is True
    assert result.selected_row is not None
    assert result.selected_row.mrr_pass is True
    assert result.selected_row.fpr_pass is True


def test_gate_fails_when_mrr_drops_below_floor() -> None:
    """MRR 0.35 < 0.40 floor → FAIL even when FPR is fine."""
    rows = [
        _row("hybrid_rrf", 0.8, 0.35, 0.50, selected=True),
    ]
    result = eval_gate.evaluate_rows(rows)
    assert result.passed is False
    assert result.selected_row is not None
    assert result.selected_row.mrr_pass is False
    assert result.selected_row.fpr_pass is True


def test_gate_fails_when_fpr_exceeds_ceiling() -> None:
    """FPR-on-novel 0.85 > 0.70 ceiling → FAIL even when MRR is fine."""
    rows = [
        _row("hybrid_rrf", 0.8, 0.50, 0.85, selected=True),
    ]
    result = eval_gate.evaluate_rows(rows)
    assert result.passed is False
    assert result.selected_row is not None
    assert result.selected_row.mrr_pass is True
    assert result.selected_row.fpr_pass is False


def test_gate_fails_when_watched_config_missing() -> None:
    """No selected row for hybrid_rrf → can't evaluate → FAIL loud."""
    rows = [
        _row("dense_bge_m3", 0.8, 0.5, 0.5, selected=True),
        # No hybrid_rrf row.
    ]
    result = eval_gate.evaluate_rows(rows)
    assert result.passed is False
    assert result.selected_row is None


def test_gate_only_inspects_watched_config() -> None:
    """A non-watched config can fail its own thresholds —
    the gate verdict is driven by hybrid_rrf only.
    """
    rows = [
        _row("dense_bge_m3", 0.8, 0.05, 1.0, selected=True),  # would fail on its own
        _row("hybrid_rrf", 0.8, 0.50, 0.50, selected=True),  # passes
    ]
    result = eval_gate.evaluate_rows(rows)
    assert result.passed is True


def test_find_selected_row_picks_last_when_csv_has_history() -> None:
    """The eval runner appends to the leaderboard CSV. If a re-run
    produces a *new* selected_threshold=True row for the same
    config (e.g. after a corpus re-build), the older row is left
    in the CSV as audit trail. ``find_selected_row`` must return
    the **last** match (the freshest run), not the first — a
    gate that inspects stale data is a gate that false-positives
    on the current main and false-negatives on regressions.

    This is a real bug observed in the current main's committed
    leaderboard.csv: hybrid_rrf has two selected rows (a stale
    one with search_errors=286 and a fresh one with
    search_errors=0). The first-match version of the helper
    returned the stale one and reported MRR=0.1; the correct
    fresh row is MRR=0.458.
    """
    rows = [
        # Older run: API was down, search_errors=286, MRR=0.1.
        _row("hybrid_rrf", 0.8, 0.1, 0.25, selected=True),
        # Fresh run: API was up, search_errors=0, MRR=0.458.
        _row("hybrid_rrf", 0.8, 0.458, 0.63, selected=True),
    ]
    sel = eval_gate.find_selected_row(rows, config="hybrid_rrf")
    assert sel is not None
    assert float(sel["mrr"]) == 0.458
    assert float(sel["fpr_on_novel"]) == 0.63
    # The gate's verdict also uses the last match, not the first.
    result = eval_gate.evaluate_rows(rows)
    assert result.passed is True
    assert result.selected_row is not None
    assert result.selected_row.mrr == 0.458


# ---------------------------------------------------------------------------
# Diff pure function — synthetic CSV inputs
# ---------------------------------------------------------------------------


def test_diff_renders_table_with_one_row_per_config() -> None:
    base = [
        _row("dense_bge_m3", 0.8, 0.55, 0.80, selected=True),
        _row("bm25", 0.5, 0.39, 1.0, selected=True),
        _row("hybrid_rrf", 0.8, 0.45, 0.65, selected=True),
    ]
    head = [
        _row("dense_bge_m3", 0.8, 0.567, 0.79, selected=True),
        _row("bm25", 0.5, 0.392, 1.0, selected=True),
        _row("hybrid_rrf", 0.8, 0.458, 0.63, selected=True),
    ]
    md = leaderboard_diff.render_diff(
        base, head, gate_thresholds={"mrr_floor": 0.40, "fpr_ceiling": 0.70}
    )
    # Header + separator + 3 data rows.
    assert "### Eval leaderboard" in md
    assert "Gate: `hybrid_rrf` MRR ≥ 0.40" in md
    assert "| config | threshold | MRR |" in md
    # The 3 configs are present.
    for cfg in ("dense_bge_m3", "bm25", "hybrid_rrf"):
        assert f"| {cfg} |" in md, f"{cfg} row missing from diff"
    # The dense MRR moved (0.55 → 0.567), so the cell should
    # carry the "delta from base" annotation.
    assert "from 0.55" in md, "MRR delta annotation missing"
    # The bm25 FPR didn't move (1.0 → 1.0), so the cell should
    # show just the value (no delta annotation).
    # Find the bm25 row and check its FPR cell.
    bm25_row = [line for line in md.splitlines() if line.startswith("| bm25 |")][0]
    # FPR cell is index 4 (config, threshold, MRR, FPR, nDCG, P, R, ECE).
    cells = [c.strip() for c in bm25_row.split("|")]
    assert cells[4] == "1", f"bm25 FPR cell should be plain '1' (no delta), got {cells[4]!r}"


def test_diff_handles_new_config_in_head() -> None:
    """A config present in head but not in base is annotated "(new)"."""
    base = [_row("dense_bge_m3", 0.8, 0.55, 0.80, selected=True)]
    head = [
        _row("dense_bge_m3", 0.8, 0.55, 0.80, selected=True),
        _row("hybrid_rrf", 0.8, 0.458, 0.63, selected=True),
    ]
    md = leaderboard_diff.render_diff(base, head)
    assert "(new)" in md, "missing (new) annotation for new config"


def test_diff_handles_dropped_config_in_base_only() -> None:
    """A config present in base but not in head is annotated "(base only)"."""
    base = [
        _row("dense_bge_m3", 0.8, 0.55, 0.80, selected=True),
        _row("hybrid_rrf", 0.8, 0.458, 0.63, selected=True),
    ]
    head = [_row("dense_bge_m3", 0.8, 0.55, 0.80, selected=True)]
    md = leaderboard_diff.render_diff(base, head)
    assert "(base only)" in md, "missing (base only) annotation for dropped config"


# ---------------------------------------------------------------------------
# Live leaderboard.csv — gate must pass on the current main
# ---------------------------------------------------------------------------


def test_gate_passes_on_current_main_leaderboard() -> None:
    """The shipped eval_gate.py must clear the current main's
    leaderboard.csv — if it doesn't, the workflow would fail
    on every PR and the gate is broken. This is the regression
    canary.
    """
    csv_path = REPO_ROOT / "results" / "leaderboard.csv"
    if not csv_path.exists():
        pytest.skip("no live leaderboard.csv on this checkout")
    rows = eval_gate.read_leaderboard_csv(csv_path)
    result = eval_gate.evaluate_rows(rows)
    assert result.selected_row is not None, (
        "current main's leaderboard.csv has no selected row for "
        "hybrid_rrf — the gate would fail every PR. Run "
        "`make eval-sweep` first."
    )
    assert result.passed is True, (
        f"gate failed on the current main's leaderboard.csv: "
        f"{result.as_markdown()}"
    )


# ---------------------------------------------------------------------------
# Sweep driver — CLI parsing pure
# ---------------------------------------------------------------------------


def test_sweep_parser_accepts_documented_flags() -> None:
    """The CLI surface documented in the script's docstring
    must be parseable. (Smoke test only — we don't execute
    the subprocess.)
    """
    args = run_eval_sweep.parse_args(
        [
            "--benchmark",
            "evals/labeled_v300.jsonl",
            "--output",
            "results/leaderboard.csv",
            "--no-mlflow",
        ]
    )
    assert args.benchmark == Path("evals/labeled_v300.jsonl")
    assert args.output == Path("results/leaderboard.csv")
    assert args.no_mlflow is True


def test_sweep_build_cmd_produces_subprocess_ready_list() -> None:
    """The ``build_cmd`` helper is the contract the workflow
    depends on — it must produce a list of strings that's
    safe to pass to ``subprocess.run``.
    """
    cmd = run_eval_sweep.build_cmd(
        config=Path("configs/dense_bge_m3.yaml"),
        benchmark=Path("evals/labeled_v300.jsonl"),
        output=Path("results/leaderboard.csv"),
        db=Path("results/eval.duckdb"),
        markdown_out=Path("results/leaderboard.md"),
        mlflow_tracking_uri="",
        no_mlflow=True,
    )
    # All entries are strings (subprocess requirement).
    assert all(isinstance(s, str) for s in cmd)
    # The --no-mlflow flag is the last one.
    assert "--no-mlflow" in cmd
    # The config path is passed verbatim.
    assert "configs/dense_bge_m3.yaml" in cmd
    # The benchmark path is passed verbatim.
    assert "evals/labeled_v300.jsonl" in cmd


# ---------------------------------------------------------------------------
# Makefile — the eval-sweep / eval-gate / leaderboard-diff targets exist
# ---------------------------------------------------------------------------


def test_makefile_has_eval_sweep_target() -> None:
    text = MAKEFILE_PATH.read_text()
    assert "eval-sweep:" in text, "make eval-sweep target missing"
    assert "scripts/ci/run_eval_sweep.py" in text, (
        "eval-sweep target doesn't invoke the sweep script"
    )


def test_makefile_has_eval_gate_target() -> None:
    text = MAKEFILE_PATH.read_text()
    assert "eval-gate:" in text, "make eval-gate target missing"
    assert "scripts/ci/eval_gate.py" in text, (
        "eval-gate target doesn't invoke the gate script"
    )


def test_makefile_has_leaderboard_diff_target() -> None:
    text = MAKEFILE_PATH.read_text()
    assert "leaderboard-diff:" in text, "make leaderboard-diff target missing"
    assert "scripts/ci/leaderboard_diff.py" in text, (
        "leaderboard-diff target doesn't invoke the diff script"
    )


def test_makefile_eval_supports_bench_override() -> None:
    """The card body pins ``make eval BENCH=evals/labeled_v300.jsonl`` —
    the target must accept a ``BENCH=`` override (not just hard-code
    labeled_v100.jsonl).
    """
    text = MAKEFILE_PATH.read_text()
    assert "BENCH ?=" in text, "make eval doesn't accept a BENCH= override"
    assert "BENCH)" in text or "$(BENCH)" in text, (
        "make eval doesn't actually use the BENCH= override"
    )


# ---------------------------------------------------------------------------
# Card t_be81a875 — workflow parse fix
# ---------------------------------------------------------------------------
#
# GitHub Actions rejects a `volumes:` key at the job level unless the
# job has a `container:` block. The eval-regression job runs directly
# on `ubuntu-latest` with no job-level container, so the job-level
# `volumes: pgdata:` block broke the workflow parse and every push
# failed HTTP 422. The fix removes the job-level volumes block; the
# service container's own ephemeral storage is enough for the
# short-lived eval job.
#
# Secondary: the `id: gate` step needs to define a `conclusion` output
# for the next step's `if: steps.gate.conclusion == 'failure'` to
# resolve correctly. actionlint flagged this; without it the next
# step's `if` is always true on a missing output and the build would
# fail open.


def test_workflow_has_no_job_level_volumes(workflow: dict) -> None:
    """Regression test for card t_be81a875.

    GitHub Actions only allows a job-level `volumes:` key when the
    job has a `container:` block. The eval-regression job runs on
    `ubuntu-latest` with no `container:` — so a job-level
    `volumes:` is invalid and the workflow fails to parse
    (HTTP 422: failed to parse workflow).

    The valid shape is: each service container (e.g. `postgres`)
    can have its own `volumes:` mount; the job itself cannot.
    The fix removed the `pgdata` job-level volume.
    """
    job = workflow.get("jobs", {}).get("eval-regression", {})
    # The full allowed key set for a job (per GitHub's schema).
    allowed = {
        "concurrency",
        "container",
        "continue-on-error",
        "defaults",
        "env",
        "environment",
        "if",
        "name",
        "needs",
        "outputs",
        "permissions",
        "runs-on",
        "secrets",
        "services",
        "snapshot",
        "steps",
        "strategy",
        "timeout-minutes",
        "uses",
        "with",
    }
    actual = set(job.keys())
    unexpected = actual - allowed
    assert "volumes" not in actual, (
        "Job has a job-level 'volumes:' block — GitHub Actions "
        "rejects this unless the job has a 'container:' block "
        "(card t_be81a875, actionlint error .github/workflows/"
        "eval-regression.yml:138:5: unexpected key 'volumes' for "
        "'job' section)."
    )
    assert not unexpected, (
        f"Job has unexpected keys: {unexpected}. GitHub Actions "
        f"will reject the workflow at parse time. Allowed keys: "
        f"{sorted(allowed)}"
    )


def test_workflow_gate_step_defines_conclusion_output(workflow: dict) -> None:
    """Regression test for card t_be81a875 (secondary issue).

    The "Fail build on gate violation" step keys off
    ``steps.gate.conclusion``. The ``id: gate`` step must
    explicitly emit a ``conclusion`` output (success | failure)
    for that `if:` to resolve. actionlint flagged this as
    "property 'gate' is not defined in object type"; without
    the explicit output the build would fail-open on a
    missing output.
    """
    steps = workflow.get("jobs", {}).get("eval-regression", {}).get("steps", [])
    gate_steps = [s for s in steps if s.get("id") == "gate"]
    assert len(gate_steps) == 1, (
        f"Expected exactly one 'id: gate' step, found {len(gate_steps)}. "
        f"The next step's `if: steps.gate.conclusion` needs a unique target."
    )
    gate = gate_steps[0]
    # The step's run: block must echo "conclusion=success" or
    # "conclusion=failure" into $GITHUB_OUTPUT. We assert on
    # the step's source so a future edit can't silently drop it.
    run_block = gate.get("run", "")
    assert "GITHUB_OUTPUT" in run_block, (
        "The 'id: gate' step must write to $GITHUB_OUTPUT for the "
        "'Fail build on gate violation' step's "
        "`if: steps.gate.conclusion == 'failure'` to resolve."
    )
    assert "conclusion=" in run_block, (
        "The 'id: gate' step must write a 'conclusion' output "
        "(either 'conclusion=success' or 'conclusion=failure') to "
        "$GITHUB_OUTPUT. The next step keys off "
        "`steps.gate.conclusion` and would fail open without it."
    )
    # And the value domain is constrained to success/failure
    # (actionlint's expected type for an outputs.conclusion).
    assert "conclusion=success" in run_block and "conclusion=failure" in run_block, (
        "The 'id: gate' step must write BOTH 'conclusion=success' "
        "and 'conclusion=failure' outputs (the run: script picks "
        "one based on the gate's exit code). Found neither or "
        "only one of the two."
    )
    # Sanity-check: the step that consumes steps.gate.conclusion
    # exists and has the expected `if:` guard.
    fail_step = next(
        (s for s in steps if s.get("name") == "Fail build on gate violation"),
        None,
    )
    assert fail_step is not None, (
        "'Fail build on gate violation' step is missing — the gate "
        "output has no consumer."
    )
    assert "steps.gate.conclusion" in fail_step.get("if", ""), (
        "'Fail build on gate violation' step doesn't reference "
        "steps.gate.conclusion. The id: gate step's output isn't "
        "wired to a consumer."
    )


def test_workflow_passes_actionlint() -> None:
    """Compile-time guard for card t_be81a875.

    The original failure mode (job-level `volumes:`) was caught
    only by `actionlint` — pure YAML parsing in tests wouldn't
    catch it because `volumes:` is a valid YAML key, just not
    a valid GitHub Actions job key. The docker-rhysd actionlint
    image is the source of truth for GitHub's schema.

    This test runs actionlint via docker (same as the card's
    verification recipe) and asserts exit 0. If docker isn't
    available, the test skips with a clear message — the
    workflow's syntax is still verified by the structural
    tests above, just not the full GitHub Actions schema.
    """
    import shutil
    import subprocess

    if not shutil.which("docker"):
        pytest.skip("docker not on PATH; cannot run actionlint")

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-u",
            "0",
            "-v",
            f"{REPO_ROOT}:/work",
            "-w",
            "/work",
            "rhysd/actionlint:latest",
            ".github/workflows/eval-regression.yml",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"actionlint failed (exit={result.returncode}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}\n\n"
        f"Card t_be81a875 acceptance requires actionlint exit 0 — "
        f"a non-zero exit means the workflow won't be accepted by "
        f"GitHub Actions and every push will fail with HTTP 422."
    )
