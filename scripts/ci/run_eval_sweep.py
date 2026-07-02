"""3-config eval-harness sweep driver — Phase 3.6 (card t_e0f62c2a).

What this is
------------
A thin wrapper that runs ``python -m eval.run`` once per retrieval
config, in a fixed order, against a single labeled benchmark. Used
by the GitHub Actions regression workflow and (optionally) by a
``make eval-sweep`` target for local reproduction.

The 3 leaderboard configs (per ``PHASE-3.md`` §3.6) are::

    configs/dense_bge_m3.yaml
    configs/bm25.yaml
    configs/hybrid_rrf.yaml

Each ``eval.run`` invocation **appends** rows to the leaderboard
CSV (per ``src/eval/run.py``'s append-mode contract). The CSV is
the single source of truth that ``scripts/ci/eval_gate.py`` reads
and that ``scripts/ci/leaderboard_diff.py`` diffs against the
base branch's committed copy.

Hard rules respected
--------------------
* **No external service calls.** The 3 configs are all offline
  (bge-m3 + rank_bm25, both local). No Cohere rerank, no
  Anthropic, no Brave Search. The wrapper refuses to call any
  config that needs an API key.
* **Committed snapshots only** for reproducibility. The corpus
  is loaded into Postgres by the workflow *before* this script
  runs (via ``make corpus-build`` against the committed
  snapshots). We never scrape fresh data here.
* **Single benchmark path.** All 3 configs run against the same
  labeled benchmark — the workflow pins ``evals/labeled_v300.jsonl``
  as the regression contract.

CLI
---
::

    uv run python scripts/ci/run_eval_sweep.py \\
        --benchmark evals/labeled_v300.jsonl \\
        --output results/leaderboard.csv

Exit code is non-zero if any of the 3 ``eval.run`` invocations
fail — the workflow's gate step (``eval_gate.py``) is a separate
post-sweep check on the leaderboard CSV's *content*, not on the
sweep's exit code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per the card body: the 3 leaderboard configs are the regression
# contract. Order is dense → bm25 → hybrid so the leaderboard CSV
# reads in the same order the dashboard sorts by. Adding a 4th
# config here is a deliberate change to the regression surface
# and must update the gate thresholds and the workflow trigger
# list in tandem.
SWEEP_CONFIGS: Sequence[Path] = (
    Path("configs/dense_bge_m3.yaml"),
    Path("configs/bm25.yaml"),
    Path("configs/hybrid_rrf.yaml"),
)

# The default benchmark — labeled_v300.jsonl is the Phase 3 contract
# (300 hand-reviewed + LLM-augmented records, FPR-on-novel surfaced
# in 3.5, ECE in 3.3). The workflow pins this explicitly via the
# --benchmark flag so the path can't drift.
DEFAULT_BENCHMARK: Path = Path("evals/labeled_v300.jsonl")

# Substring gate: any config file whose path contains one of these
# substrings is refused unless explicitly opted in via
# ``--allow-external-config``. The 3 default configs are local-only;
# this is the safety net so a future PR that adds ``cohere`` or
# ``brave`` to the config name doesn't silently re-introduce
# external API calls.
_EXTERNAL_CONFIG_HINTS: Sequence[str] = (
    "cohere",
    "brave",
    "anthropic",
    "openai",
    "serpapi",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no I/O)
# ---------------------------------------------------------------------------


def is_external_config_name(config_path: Path) -> bool:
    """Return True if the config name hints at an external API call.

    Used as a guardrail so a future PR that drops a ``cohere_rerank``
    or ``brave_search`` config into the sweep list fails the CI
    step loudly rather than silently making the regression suite
    depend on a paid API.

    The match is a substring test on the path (case-insensitive) —
    conservative by design. ``dense_bge_m3`` / ``bm25`` / ``hybrid_rrf``
    all return False; ``hybrid_rrf_cohere`` returns True.
    """
    name = str(config_path).lower()
    return any(hint in name for hint in _EXTERNAL_CONFIG_HINTS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI args. Args are paths (relative to repo root)."""
    p = argparse.ArgumentParser(
        prog="run_eval_sweep",
        description=(
            "Phase 3.6 — run the eval harness against the 3 leaderboard "
            "configs (dense_bge_m3, bm25, hybrid_rrf) and append to a "
            "single leaderboard CSV."
        ),
    )
    p.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=(
            "Path to the labeled benchmark JSONL "
            "(default: evals/labeled_v300.jsonl)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/leaderboard.csv"),
        help="Path to the leaderboard CSV (append mode).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=Path("results/eval.duckdb"),
        help="Path to the DuckDB store (or 'none' to disable).",
    )
    p.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("results/leaderboard.md"),
        help="Path to the per-run Markdown summary.",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:15000"
        ),
        help=(
            "MLflow tracking URI (default: $MLFLOW_TRACKING_URI or "
            "http://localhost:15000). Pass an empty string to skip "
            "MLflow logging (sets --no-mlflow)."
        ),
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help=(
            "Skip MLflow logging. Default ON in CI: the regression "
            "suite is for *the eval itself*, not for the MLflow "
            "experiment tracker. CI has no MLflow server and the "
            "mlflow_logger's offline-file fallback would otherwise "
            "clutter the artifacts dir."
        ),
    )
    p.add_argument(
        "--allow-external-config",
        action="store_true",
        help=(
            "Opt out of the external-config-name guardrail. Only set "
            "if a 4th config has a name that *sounds* external but "
            "is actually local (e.g. a config that loads a local "
            "re-ranker model). The default refuses to run "
            "cohere/brave/anthropic/openai/serpapi configs because "
            "those would silently re-introduce API dependencies."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the commands that *would* run without actually "
            "executing them. Used by the workflow's `dry-run` step "
            "to surface the command list in the Actions log without "
            "paying the eval cost."
        ),
    )
    return p.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Subprocess orchestration
# ---------------------------------------------------------------------------


def build_cmd(
    *,
    config: Path,
    benchmark: Path,
    output: Path,
    db: Path | None,
    markdown_out: Path | None,
    mlflow_tracking_uri: str,
    no_mlflow: bool,
) -> list[str]:
    """Build the ``python -m eval.run`` command for one config.

    Pure (no I/O) — every argument is a function input. Unit tests
    assert on the exact command list.
    """
    cmd = [
        sys.executable,
        "-m",
        "eval.run",
        "--benchmark",
        str(benchmark),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if db is not None:
        cmd.extend(["--db", str(db)])
    if markdown_out is not None:
        cmd.extend(["--markdown-out", str(markdown_out)])
    if mlflow_tracking_uri:
        cmd.extend(
            ["--mlflow-tracking-uri", mlflow_tracking_uri]
        )
    if no_mlflow:
        cmd.append("--no-mlflow")
    return cmd


def run_one(
    cmd: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess:
    """Run one ``eval.run`` invocation. Returns the CompletedProcess.

    On non-zero exit the caller decides what to do — the sweep
    contract is that one failed config fails the sweep (we don't
    keep going; an MRR/FPR gate is meaningless if half the data
    is missing).
    """
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env={**os.environ, "PYTHONPATH": str(cwd)},
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    # Resolve + validate inputs (loud, not silent).
    benchmark = (repo_root / args.benchmark).resolve()
    output = (repo_root / args.output).resolve()
    db = (repo_root / args.db).resolve() if args.db else None
    output.parent.mkdir(parents=True, exist_ok=True)

    for path in (benchmark,):
        if not path.exists():
            print(
                f"[sweep] benchmark not found: {path}",
                file=sys.stderr,
            )
            return 2

    # External-config guardrail. Loud-fail so the workflow's
    # gate step doesn't have to chase a silent regression later.
    if not args.allow_external_config:
        for cfg in SWEEP_CONFIGS:
            if is_external_config_name(cfg):
                print(
                    f"[sweep] refusing to run {cfg} — the name "
                    f"hints at an external API. Pass "
                    f"--allow-external-config to override (and add "
                    f"the API key to the workflow's env block).",
                    file=sys.stderr,
                )
                return 3

    # Translate "mlflow-tracking-uri=''" into --no-mlflow for
    # parity with ``--no-mlflow`` (both produce the same outcome).
    no_mlflow = args.no_mlflow or not args.mlflow_tracking_uri
    mlflow_uri = (
        args.mlflow_tracking_uri if not no_mlflow else ""
    )

    print(
        f"[sweep] benchmark={benchmark.name} output={output.name} "
        f"db={'none' if db is None else db.name} "
        f"no_mlflow={no_mlflow} "
        f"configs={[str(c) for c in SWEEP_CONFIGS]}"
    )

    failed: list[Path] = []
    for cfg_rel in SWEEP_CONFIGS:
        cfg = (repo_root / cfg_rel).resolve()
        if not cfg.exists():
            print(
                f"[sweep] config not found: {cfg}",
                file=sys.stderr,
            )
            failed.append(cfg_rel)
            continue
        cmd = build_cmd(
            config=cfg_rel,
            benchmark=args.benchmark,
            output=args.output,
            db=args.db,
            markdown_out=args.markdown_out,
            mlflow_tracking_uri=mlflow_uri,
            no_mlflow=no_mlflow,
        )
        if args.dry_run:
            print(f"[sweep] DRY: would run: {' '.join(cmd)}")
            continue
        print(f"[sweep] running config={cfg_rel.name}")
        result = run_one(cmd, cwd=repo_root)
        # Echo the eval runner's stdout so the workflow log shows
        # the per-config summary; the runner already prints the
        # Markdown table to stdout. Stderr is forwarded verbatim.
        if result.stdout:
            sys.stdout.write(result.stdout)
            sys.stdout.flush()
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()
        if result.returncode != 0:
            print(
                f"[sweep] config={cfg_rel.name} FAILED "
                f"rc={result.returncode}",
                file=sys.stderr,
            )
            failed.append(cfg_rel)
            # Don't break — let the other configs run so the gate
            # step can still inspect partial data. The non-zero
            # exit at the end surfaces the failure.

    if args.dry_run:
        print("[sweep] dry-run complete; no commands executed.")
        return 0

    if failed:
        print(
            f"[sweep] FAILED — {len(failed)}/{len(SWEEP_CONFIGS)} "
            f"configs errored: {[str(c) for c in failed]}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[sweep] OK — {len(SWEEP_CONFIGS)}/{len(SWEEP_CONFIGS)} "
        f"configs wrote to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
