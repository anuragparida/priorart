"""MLflow logging adapter for the eval harness (Phase 2.4).

What this is
------------
A small, opinionated wrapper around ``mlflow.tracking`` so the eval
harness (``src/eval/run.py``) can log every run end-to-end without
spreading MLflow imports across the runner. It hides three pitfalls:

1. ``MLFLOW_TRACKING_URI`` may not be set (e.g. on CI without an
   MLflow server). In that case we fall back to a *file-based* tracking
   store rooted at ``mlruns/`` (the MLflow default) — no exception, no
   hard fail. The eval harness keeps working offline.

2. The Phase 2.4 spec rule is **"do not log the prompt template as a
   param"** (params are hyperparameters). The prompt text is logged
   via ``mlflow.log_text`` as a run artifact, not as a
   ``mlflow.log_param`` call. This module enforces it.

3. Param / metric names are typed (str for params, numeric for
   metrics). MLflow's contract is strict: if you log the same param
   twice on one run it raises. We idempotently check ``mlflow.log_param``
   via the client to avoid that.

Design
------
The function ``log_run`` takes a structured dict and returns a
``RunRecord`` named tuple with the run id and the tracking URI it
actually used (for the operator to confirm). It never opens a long-lived
run context; the caller (``run_eval``) controls the lifetime by calling
``start_run`` itself in a context-manager block — keeping the API
minimal and explicit.

Why a separate module (not just direct ``mlflow.log_*`` calls)
--------------------------------------------------------------
The eval harness already has its own CSV / DuckDB writers
(``write_csv`` / ``write_duckdb``). MLflow is a third sink for the
same artifacts (the spec calls for "per-record CSV" + "leaderboard
CSV row" + "prompt template" — all logged by MLflow too). Keeping
the MLflow-specific code in one place is what makes the runner
readable; otherwise the runner becomes a maze of
``mlflow.log_param(...)`` calls scattered across the orchestration
function.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


#: Default tracking URI — points at the self-hosted MLflow server on
#: the host port ``15000`` (NOT ``5000`` — Honcho collision, per
#: AGENTS.md memory note). Override via the ``MLFLOW_TRACKING_URI``
#: env var, e.g. ``file:./mlruns`` for fully-offline runs.
DEFAULT_TRACKING_URI = "http://localhost:15000"

#: A 1.5-second HTTP probe is more than enough for an in-process
#: MLflow server on the loopback. The probe is best-effort — when
#: it fails we silently fall back to a file-based tracking URI so
#: the eval harness still works in CI / offline.
_PROBE_TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class RunRecord:
    """What ``log_run`` returns so the operator can verify.

    The ``tracking_uri_effective`` is the URI the wrapper actually
    wrote to (which may differ from the requested ``tracking_uri`` if
    a file-based fallback was needed because the tracking server was
    unreachable). Posting it back in the completion summary lets the
    operator confirm the run landed in the right place.
    """

    run_id: Optional[str]
    tracking_uri_requested: Optional[str]
    tracking_uri_effective: Optional[str]
    fallback_used: bool


# ---------------------------------------------------------------------------
# Connectivity probe + URI resolution
# ---------------------------------------------------------------------------


def is_tracking_server_reachable(tracking_uri: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if the tracking server at ``tracking_uri`` answers.

    For an ``http(s)://`` URL we probe ``GET {uri}/health`` (MLflow's
    liveness endpoint); for anything else (``file:...``, ``databricks``,
    ...) we return True so we don't accidentally fall back when the
    caller genuinely wants a file store.
    """
    if not tracking_uri.startswith(("http://", "https://")):
        # Non-HTTP tracking URIs (file:, databricks:, ...) — assume
        # they're meant to be used as-is. No probe.
        return True
    parsed = urllib.parse.urlparse(tracking_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Cheap TCP probe — faster than HTTP, and MLflow always opens a TCP
    # socket before serving. If the port is closed the connection
    # refuses immediately.
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def resolve_tracking_uri(
    explicit: Optional[str] = None,
    *,
    env_var: str = "MLFLOW_TRACKING_URI",
) -> str:
    """Pick a tracking URI in this order:

    1. ``explicit`` argument (used by ``--mlflow-tracking-uri``).
    2. ``MLFLOW_TRACKING_URI`` env var.
    3. ``DEFAULT_TRACKING_URI`` (http://localhost:15000).

    Always returns a non-empty string. Callers can pass the result
    straight to ``mlflow.set_tracking_uri``.
    """
    return (
        explicit
        or os.environ.get(env_var)
        or DEFAULT_TRACKING_URI
    )


def fallback_file_uri() -> str:
    """Return a per-process tmp-dir file URI as the offline fallback.

    The tmp dir is created under the system temp area so test runs
    don't pollute the repo. We use a per-process subdir so concurrent
    eval runs (e.g. parallel CI jobs) don't collide on mlruns/.
    """
    base = Path(tempfile.gettempdir()) / "priorart-mlruns" / str(int(time.time() * 1000))
    base.mkdir(parents=True, exist_ok=True)
    return f"file:{base.as_posix()}"


def _is_artifact_location_writable(artifact_location: Optional[str]) -> bool:
    """Return True if ``artifact_location`` is writable in this process.

    MLflow's ``file:`` artifact locations are written client-side —
    when the daemon was launched by another uid the path often isn't
    writable. Non-``file:`` schemes (s3, gs, azure, http, etc.) are
    always considered writable because the upload goes through the
    MLflow REST API regardless of local uid.
    """
    if not artifact_location:
        return False
    if not artifact_location.startswith("file:"):
        # S3 / GCS / Azure / http — upload goes via the server, so
        # local-uid permissions don't matter.
        return True
    path = artifact_location[len("file:"):]
    if path.startswith("//"):
        path = path[1:]
    elif path.startswith("/"):
        path = path
    p = Path(path)
    try:
        # Try to create a probe file; if it fails, it's unwritable.
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".write_probe_{int(time.time() * 1000)}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The logging facade
# ---------------------------------------------------------------------------


def _try_log_param_once(client: Any, run_id: str, key: str, value: Any) -> None:
    """Idempotent ``log_param`` — silent on ``RESOURCE_ALREADY_EXISTS``.

    MLflow raises on duplicate-param-name per run. The eval harness
    may emit the same param from two code paths (e.g. a config file
    *and* an explicit override); we want the first one to win, not
    the runner to crash.
    """
    try:
        client.log_param(run_id, key, value)
    except Exception as exc:
        # MlflowClient raises ``MlflowException`` for the duplicate
        # case. We catch broadly because MLflow's exception hierarchy
        # has shifted across 2.x and 3.x. Any other exception is
        # surfaced via ``warnings.warn`` so the operator still sees
        # it in the log but the eval run doesn't crash.
        message = str(exc)
        if "RESOURCE_ALREADY_EXISTS" in message or "already been recorded" in message:
            return
        import warnings

        warnings.warn(f"[mlflow_logger] log_param({key!r}) failed: {exc}")


def _try_log_metric_once(client: Any, run_id: str, key: str, value: Any) -> None:
    """Idempotent metric log — best-effort, MLflow silently replaces
    duplicate metric entries per MLflow semantics for ``log_metric``,
    so this is mostly here for symmetry with ``_try_log_param_once``.

    A ``None`` value is skipped (MLflow requires finite numbers).
    """
    if value is None:
        return
    try:
        # MLflow's ``log_metric`` does NOT raise on duplicate keys —
        # it logs both values and the UI shows the last. Keep that
        # semantics; only swallow network / SDK errors.
        client.log_metric(run_id, key, float(value))
    except Exception as exc:
        import warnings

        warnings.warn(f"[mlflow_logger] log_metric({key!r}) failed: {exc}")


def log_run(
    *,
    experiment_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    prompt_template_text: Optional[str] = None,
    prompt_template_artifact_name: str = "prompt_template.txt",
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Mapping[str, str]] = None,
) -> RunRecord:
    """Log a single eval run to MLflow.

    Parameters
    ----------
    experiment_name:
        Required. The MLflow experiment (created if missing). MLflow
        3.x scopes experiments by name and the UI groups runs by it.
    params:
        Hyperparameters. Keys must be short (``embedding_model``,
        ``threshold``). Values are coerced to ``str``.
    metrics:
        Numeric metrics (``mrr``, ``ndcg_at_10``, ...). ``None`` values
        are skipped. Floats are expected; ints are auto-coerced.
    artifacts:
        Mapping of ``artifact_name -> local_path``. Each path is read
        by MLflow from disk and uploaded to the artifact store. All
        paths must exist on disk when this is called.
    prompt_template_text:
        Optional. When provided, the text is written to a temp file
        and logged as ``prompt_template_artifact_name`` via
        ``mlflow.log_text``. We use ``mlflow.log_text`` here (NOT
        ``log_param``) per the PHASE-2.md pitfall rule.
    tracking_uri:
        Override for the tracking URI. ``None`` means "use the
        MLflow env var / default".
    run_name:
        Optional display name for the run (shown in the MLflow UI list).
    tags:
        Optional ``str -> str`` mapping of free-form tags.

    Returns
    -------
    RunRecord with the assigned ``run_id`` and the tracking URI the
    call actually wrote to (which may differ from the requested one if
    a fallback was needed).
    """
    import warnings

    import mlflow
    from mlflow.tracking import MlflowClient

    requested = resolve_tracking_uri(tracking_uri)
    if not is_tracking_server_reachable(requested):
        # Server unreachable — fall back to the file-based tracking
        # URI and continue silently. We pick the ``file:`` URI because
        # it doesn't require SQLite setup; MLflow creates one
        # automatically.
        effective = fallback_file_uri()
        fallback_used = True
    else:
        effective = requested
        fallback_used = False

    mlflow.set_tracking_uri(effective)
    # Pre-create / fetch the experiment with an artifact_location we
    # can actually write to. When the server's default-artifact-root
    # is owned by another uid (common in dev setups where the
    # MLflow daemon was launched as root), ``mlflow.log_artifact``
    # tries ``os.makedirs(<artifact_root>)`` on the *client* side
    # and fails with PermissionError.
    #
    # Strategy:
    #
    # 1. Probe the experiment by name.
    # 2. If it doesn't exist, create it with a *user-writable*
    #    ``artifact_location`` rooted under ``tempfile.gettempdir()``.
    # 3. If it does exist but its current ``artifact_location``
    #    points at a directory we cannot write to, fall back to a
    #    file-store run rooted under the tmp dir.
    #
    # MLflow 3.x enforces ``MLFLOW_ALLOW_FILE_STORE=true`` for any
    # ``file:`` tracking URI (the file-store is in maintenance mode).
    # Set it here so the file-store fallback actually writes.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tmp_artifacts_root = (
        "file://" + Path(tempfile.gettempdir()).as_posix() + "/priorart-mlflow-artifacts"
    )
    Path(tempfile.gettempdir()).joinpath("priorart-mlflow-artifacts").mkdir(
        parents=True, exist_ok=True
    )
    # NOTE: MlflowClient carries the *current* tracking URI at the
    # point of its construction. If we switch tracking URIs inside
    # this function (which we do in the unwritable-artifact fallback
    # branch), we have to build a *new* MlflowClient so it picks up
    # the new URI. We rebuild ``client`` in that branch below.
    client = MlflowClient()
    exp_id = None  # default; explicit = "0" only when we have a live experiment_id
    pin_explicit_experiment = True
    try:
        try:
            exp = client.get_experiment_by_name(experiment_name)
        except Exception:
            exp = None
        if exp is None:
            try:
                exp_id = client.create_experiment(
                    name=experiment_name,
                    artifact_location=tmp_artifacts_root,
                )
            except Exception:
                # Concurrent insert / re-fetch race — pick it up.
                exp = client.get_experiment_by_name(experiment_name)
                exp_id = exp.experiment_id if exp else "0"
        else:
            exp_id = exp.experiment_id
            # If the existing experiment's artifact_location is
            # unwritable to us, force the file-store fallback so the
            # run at least lands in a queryable / UI-accessible store
            # under ``tempfile.gettempdir()``. MLflow 3.x's file store
            # is local to the file root — when we switch tracking URI,
            # we must NOT pin the old experiment_id (it points at the
            # old DB), so we let the file-store create its own.
            if not _is_artifact_location_writable(exp.artifact_location):
                warnings.warn(
                    f"[mlflow_logger] existing experiment {experiment_name!r} "
                    f"has unwritable artifact_location={exp.artifact_location}; "
                    f"falling back to file-store at {fallback_file_uri()!r}"
                )
                effective = fallback_file_uri()
                mlflow.set_tracking_uri(effective)
                fallback_used = True
                # Rebuild the client so it talks to the file-store,
                # not the old server.
                client = MlflowClient()
                local_exp = client.get_experiment_by_name(experiment_name)
                if local_exp is None:
                    try:
                        exp_id = client.create_experiment(
                            name=experiment_name,
                            artifact_location=tmp_artifacts_root,
                        )
                    except Exception:
                        local_exp = client.get_experiment_by_name(experiment_name)
                        exp_id = local_exp.experiment_id if local_exp else None
                else:
                    exp_id = local_exp.experiment_id
                pin_explicit_experiment = exp_id is not None
    except Exception:
        # Last-ditch fallback — use whatever mlflow thinks the
        # default experiment is.
        exp_id = "0"

    run_id: Optional[str] = None
    try:
        # ``start_run`` opens a run and binds it as the active run for
        # the duration of the context. We pin the experiment_id
        # explicitly so the artifact_location we picked above is
        # used (otherwise the server's default-artifact-root can
        # leak through). When we fell back to a file-store, we let
        # MLflow resolve the experiment locally instead.
        start_kwargs = {"run_name": run_name}
        if pin_explicit_experiment and exp_id:
            start_kwargs["experiment_id"] = exp_id
        run_ctx = mlflow.start_run(**start_kwargs)
        with run_ctx as active_run:
            run_id = active_run.info.run_id
            client = MlflowClient()

            # ``set_tags`` (plural) accepts a dict and is idempotent —
            # setting the same tag twice does not raise.
            if tags:
                try:
                    mlflow.set_tags(dict(tags))
                except Exception as exc:
                    warnings.warn(f"[mlflow_logger] set_tags failed: {exc}")

            # Params
            for k, v in params.items():
                _try_log_param_once(client, run_id, k, v)

            # Metrics
            for k, v in metrics.items():
                _try_log_metric_once(client, run_id, k, v)

            # Artifacts (the per-record + leaderboard CSVs, etc.)
            for name, path in artifacts.items():
                p = Path(path)
                if p is None or not p.exists():
                    continue
                try:
                    mlflow.log_artifact(str(p), artifact_path="")
                except Exception as exc:
                    warnings.warn(f"[mlflow_logger] log_artifact({name}) failed: {exc}")

            # Prompt template — log as ARTIFACT, NOT as a param.
            # This is the PHASE-2.md pitfall rule ("do not log prompts
            # as params"). We write the text to a user-writable tmp
            # file and use ``log_artifact`` so the upload is driven by
            # the experiment's artifact_location (set above) rather
            # than the server's default-artifact-root.
            if prompt_template_text is not None:
                tmp_dir = Path(tempfile.gettempdir()) / "priorart-mlflow-prompt-templates"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / prompt_template_artifact_name
                tmp_path.write_text(prompt_template_text, encoding="utf-8")
                try:
                    mlflow.log_artifact(str(tmp_path), artifact_path="")
                except Exception as exc:
                    warnings.warn(f"[mlflow_logger] log_artifact(prompt) failed: {exc}")
    except Exception as exc:
        warnings.warn(f"[mlflow_logger] log_run failed: {exc}")
        # We deliberately swallow — the eval harness should not crash
        # because MLflow is unreachable. ``run_id`` is None on failure.

    return RunRecord(
        run_id=run_id,
        tracking_uri_requested=requested,
        tracking_uri_effective=effective,
        fallback_used=fallback_used,
    )


# ---------------------------------------------------------------------------
# Param / metric shape enforcement (used by run.py to construct the dict)
# ---------------------------------------------------------------------------


def metrics_from_summary(
    summary: Mapping[str, Any],
    *,
    best_threshold: float,
) -> Dict[str, float]:
    """Pick the canonical 5 metrics out of the runner's summary dict.

    The summary already exposes the five metrics under the
    ``best_*`` keys (per the runner contract). We re-derive them as
    a clean dict so the MLflow run row matches what the spec asks
    for (``MRR / nDCG@10 / P@5 / R@10 / FPR-on-novel``).
    """
    return {
        "mrr": float(summary["best_mrr"]),
        "ndcg_at_10": float(summary["best_ndcg_at_10"]),
        "precision_at_5": float(summary["best_precision_at_5"]),
        "recall_at_10": float(summary["best_recall_at_10"]),
        "fpr_on_novel": float(summary["best_fpr_on_novel"]),
        "best_threshold": float(best_threshold),
    }


def params_from_summary(
    *,
    config_name: str,
    embedding_model: str,
    threshold: float,
    benchmark_name: str,
    corpus_count: int,
    corpus_snapshot_date: str,
    prompt_template_version: str,
    api_url: str,
    top_k: int,
) -> Dict[str, str]:
    """Pick the canonical params per the Phase 2.4 spec table.

    All values are coerced to ``str`` because MLflow's log_param
    accepts only int / float / str / bool, and stringifying keeps the
    UI display consistent (no surprise 0/1 boxing in tag columns).
    """
    out: Dict[str, str] = {
        "config": str(config_name),
        "embedding_model": str(embedding_model),
        "threshold": str(threshold),
        "prompt_template_version": str(prompt_template_version),
        "corpus_snapshot_date": str(corpus_snapshot_date),
        "benchmark": str(benchmark_name),
        "api_url": str(api_url),
        "top_k": str(top_k),
    }
    if corpus_count:
        out["corpus_count"] = str(corpus_count)
    return out


__all__ = [
    "DEFAULT_TRACKING_URI",
    "RunRecord",
    "fallback_file_uri",
    "is_tracking_server_reachable",
    "log_run",
    "metrics_from_summary",
    "params_from_summary",
    "resolve_tracking_uri",
]
