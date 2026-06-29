"""Temporal client routes — ``POST /ideas/analyze`` (start a workflow)
and ``GET /workflows/{id}`` (poll the workflow status).

Phase 2.1 (docs/PHASE-2.md §2.1) replaces the Phase 1.8 inline
pipeline with a Temporal client. The contract changes:

- ``POST /ideas/analyze`` now returns
  ``{"workflow_id": "...", "run_id": "...", "status": "running"}``
  synchronously (HTTP 200) — the workflow runs in the background.
- ``GET /workflows/{id}`` returns the workflow's current status
  (Temporal ``describe_workflow`` + an in-flight ``phase`` query).
- The verdict is now reachable via ``GET /workflows/{id}/result``
  (returns the final ``IdeaVerdict`` once the workflow completes).

This module exposes two route bodies (``analyze_start_endpoint``
and ``workflow_status_endpoint``) plus the ``WorkflowStatusResponse``
schema. ``app.py`` mounts the actual ``@app.post(...)`` /
``@app.get(...)`` decorators.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.config import TEMPORAL_TASK_QUEUE
from src.llm.schemas import IdeaVerdict
from src.workflow.client import get_temporal_client
from src.workflow.shared import IdeaAnalysisInput
from src.workflow.workflows import IdeaAnalysisWorkflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schemas (the new wire shapes from Phase 2.1)
# ---------------------------------------------------------------------------


class AnalyzeStartResponse(BaseModel):
    """The synchronous response from ``POST /ideas/analyze``.

    Returns the workflow id + run id + the immediate status. The
    client polls ``GET /workflows/{id}`` until ``status`` becomes
    ``"COMPLETED"`` (or a failure state), then reads ``result``.

    Same HTTP 200 contract as Phase 1.8's ``POST /ideas/analyze``
    — a successful start is a 200, not a 202, because the request
    was well-formed and the workflow was started.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(..., description="Temporal workflow id.")
    run_id: str = Field(..., description="Temporal run id (unique per execution).")
    status: str = Field(
        default="running",
        description="Immediate status; always ``running`` for a freshly-started workflow.",
    )
    task_queue: str = Field(
        default=TEMPORAL_TASK_QUEUE,
        description="The Temporal task queue the workflow was started on.",
    )


class WorkflowStatusResponse(BaseModel):
    """The wire shape of ``GET /workflows/{id}``.

    The base ``WorkflowStatus`` from ``src.workflow.shared`` is the
    canonical shape (matches the ``get_status`` query + the
    ``describe_workflow`` response). This class is just a re-export
    for FastAPI's response_model — keeping it as a separate class
    means we can extend the HTTP contract without touching the
    workflow's query handler.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    run_id: str
    status: str
    phase: str
    start_time: datetime
    close_time: datetime | None = None
    result: IdeaVerdict | None = None
    failure: dict[str, Any] | None = None
    task_queue: str = TEMPORAL_TASK_QUEUE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEMPORAL_STATUS_NAMES = {
    # Map Temporal's enum names onto the wire shape. The Temporal
    # Python SDK returns these via ``WorkflowExecutionStatus`` enum
    # (``OPEN`` / ``CLOSED``), but the underlying status string is
    # e.g. ``"Running"`` / ``"Completed"`` — we uppercase them for
    # the JSON response.
    "RUNNING": "RUNNING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "TIMED_OUT": "TIMED_OUT",
    "CANCELLED": "CANCELLED",
    "TERMINATED": "TERMINATED",
    "CONTINUED_AS_NEW": "CONTINUED_AS_NEW",
}


def _temporal_status_name(workflow_execution_status: Any) -> str:
    """Coerce a Temporal status enum / str into the canonical wire name."""
    name = getattr(workflow_execution_status, "name", None) or str(workflow_execution_status)
    return _TEMPORAL_STATUS_NAMES.get(name.upper(), name.upper())


# ---------------------------------------------------------------------------
# Route bodies
# ---------------------------------------------------------------------------


async def analyze_start_endpoint(
    request: IdeaAnalysisInput,
) -> AnalyzeStartResponse:
    """Start a fresh ``IdeaAnalysisWorkflow`` and return its handle.

    Returns
    -------
    AnalyzeStartResponse
        ``workflow_id`` + ``run_id`` + ``status="running"``.

    Raises
    ------
    HTTPException(503)
        If the Temporal client can't be reached. The Phase 1.8
        contract is "no 500s on a missing dependency, surface as a
        503 with a structured body".
    """
    client = await get_temporal_client()
    try:
        handle = await client.start_workflow(
            IdeaAnalysisWorkflow.run,
            request,
            id=request.request_id or f"idea-analysis-{datetime.now(UTC).isoformat()}",
            task_queue=TEMPORAL_TASK_QUEUE,
        )
    except Exception as exc:
        # Network blip, namespace not found, etc. — surface as 503
        # so the client knows to retry. We do *not* return a
        # structured AnalyzeError here because the user-facing
        # contract for this route is "started a workflow" or
        # "couldn't reach Temporal".
        logger.exception("analyze_start: failed to start workflow")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "temporal_unavailable",
                "details": {"message": str(exc), "type": type(exc).__name__},
            },
        ) from exc

    return AnalyzeStartResponse(
        workflow_id=handle.id,
        run_id=handle.result_run_id or "",
        status="running",
        task_queue=TEMPORAL_TASK_QUEUE,
    )


async def workflow_status_endpoint(
    workflow_id: str,
) -> WorkflowStatusResponse:
    """Describe a workflow and surface its current status.

    Two sources feed the response:

    1. ``client.get_workflow_handle(workflow_id).describe()`` —
       raw Temporal status (``RUNNING`` / ``COMPLETED`` / ``FAILED``,
       ``start_time``, ``close_time``, history length).

    2. ``handle.query(...)`` on the workflow's ``get_status`` query
       — the in-flight ``phase`` + ``task_queue`` (and any partial
       results the workflow wants to surface).

    For completed workflows, we also fetch ``handle.result()`` to
    populate ``result`` with the final ``IdeaVerdict``.

    Returns
    -------
    WorkflowStatusResponse
        Wire shape the client polls.

    Raises
    ------
    HTTPException(404)
        If the workflow id doesn't exist (Temporal returns a
        ``WorkflowExecutionNotFoundError``).
    HTTPException(503)
        If the Temporal client can't be reached.
    """
    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    try:
        description = await handle.describe()
    except Exception as exc:
        # Temporal raises ``WorkflowExecutionNotFoundError`` for
        # unknown ids; we map both that and a transport failure
        # to 503 (caller-side decision: retry). For "not found"
        # specifically the message will say so, but we don't
        # distinguish — the client's contract is "if you don't
        # recognise the id, retry until the Temporal UI tells
        # you it was archived".
        if "not found" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "workflow_not_found",
                    "details": {"workflow_id": workflow_id, "message": str(exc)},
                },
            ) from exc
        logger.exception("workflow_status: failed to describe workflow")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "temporal_unavailable",
                "details": {"message": str(exc), "type": type(exc).__name__},
            },
        ) from exc

    status_name = _temporal_status_name(description.status)
    start_time = description.start_time
    close_time = description.close_time

    # Pull the workflow's in-flight phase via the query handler.
    # Phase 2.1 doesn't fail the request if the query fails — we
    # default to ``STARTED`` and let the caller poll again.
    phase = "started"
    try:
        query_result = await handle.query("get_status")
        # ``query_result`` is the dict the workflow's get_status
        # returned — it has a ``phase`` field.
        if isinstance(query_result, dict) and "phase" in query_result:
            phase = query_result["phase"]
    except Exception:
        logger.debug(
            "workflow_status: get_status query failed (workflow may have completed)",
            exc_info=True,
        )

    # Pull the result for completed workflows.
    result: IdeaVerdict | None = None
    failure: dict[str, Any] | None = None
    if status_name == "COMPLETED":
        try:
            verdict_dict = await handle.result()
            if isinstance(verdict_dict, dict):
                result = IdeaVerdict.model_validate(verdict_dict)
        except Exception as exc:
            # If result() raises, the workflow is in an odd state —
            # the Temporal status said COMPLETED but the result
            # call failed. Surface as a structured failure so the
            # caller can investigate.
            failure = {"type": type(exc).__name__, "message": str(exc)}
    elif status_name == "FAILED":
        # ``describe().failure`` carries the Temporal error info
        # for failed workflows.
        desc_failure = getattr(description, "failure", None)
        if desc_failure is not None:
            failure = {
                "type": getattr(desc_failure, "type", "") or "",
                "message": getattr(desc_failure, "message", "") or "",
            }

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        run_id=description.run_id,
        status=status_name,
        phase=phase,
        start_time=start_time,
        close_time=close_time,
        result=result,
        failure=failure,
        task_queue=TEMPORAL_TASK_QUEUE,
    )


# ---------------------------------------------------------------------------
# Block-poll endpoint — the convenience route for /workflows/{id}/result
# ---------------------------------------------------------------------------


#: Per-request budget for the convenience result route. Each /result
#: call polls the workflow status in a loop with a small sleep; we cap
#: the budget so a hung workflow doesn't wedge the FastAPI worker.
#: 30 s is generous — the typical workflow completes in 5–10 s.
RESULT_POLL_TIMEOUT_SECONDS = 30.0
RESULT_POLL_INTERVAL_SECONDS = 0.5


async def workflow_result_endpoint(
    workflow_id: str,
) -> dict[str, Any]:
    """Block-poll the workflow until it reaches a terminal state.

    Convenience wrapper around ``workflow_status_endpoint``: callers
    that want a single URL that returns the final IdeaVerdict (e.g.
    ``make smoke``) hit this endpoint instead of polling
    ``/workflows/{id}`` themselves.

    The polling loop caps at ``RESULT_POLL_TIMEOUT_SECONDS`` so a
    stuck workflow turns into a 409 ("not done yet, give up and
    retry the polling route") rather than a hung request.

    Returns
    -------
    dict
        - ``{"status": "COMPLETED", "result": <IdeaVerdict>}`` on success
        - ``{"status": "FAILED", "failure": {...}}`` on workflow failure
        - ``{"status": "<running_phase>"}`` on timeout — the route
          raises 409 in that case

    Raises
    ------
    HTTPException(404)
        If the workflow id doesn't exist.
    HTTPException(409)
        If the workflow hasn't reached a terminal state within the
        per-request budget.
    HTTPException(503)
        If the Temporal client can't be reached.
    """
    import asyncio

    elapsed = 0.0
    while elapsed <= RESULT_POLL_TIMEOUT_SECONDS:
        # We re-use the status endpoint so all 404/503 handling is
        # identical to the polling route.
        status_response = await workflow_status_endpoint(workflow_id)

        if status_response.status == "COMPLETED":
            return {
                "status": "COMPLETED",
                "workflow_id": workflow_id,
                "result": status_response.result.model_dump(mode="json")
                if status_response.result is not None
                else None,
                "phase": status_response.phase,
                "close_time": status_response.close_time.isoformat()
                if status_response.close_time
                else None,
            }
        if status_response.status == "FAILED":
            return {
                "status": "FAILED",
                "workflow_id": workflow_id,
                "phase": status_response.phase,
                "failure": status_response.failure,
            }
        if status_response.status in ("TIMED_OUT", "CANCELLED", "TERMINATED"):
            # Terminal but non-success — surface as 409 so the
            # client knows it didn't get a verdict.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "workflow_terminal_no_result",
                    "details": {
                        "workflow_id": workflow_id,
                        "status": status_response.status,
                        "phase": status_response.phase,
                    },
                },
            )

        await asyncio.sleep(RESULT_POLL_INTERVAL_SECONDS)
        elapsed += RESULT_POLL_INTERVAL_SECONDS

    # Budget exhausted — surface as 409 with the last-known phase so
    # the client can resume polling on the dedicated status route.
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "error": "workflow_not_done",
            "details": {
                "workflow_id": workflow_id,
                "timeout_seconds": RESULT_POLL_TIMEOUT_SECONDS,
                "hint": "use GET /workflows/{id} to poll until completion",
            },
        },
    )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "AnalyzeStartResponse",
    "WorkflowStatusResponse",
    "analyze_start_endpoint",
    "workflow_result_endpoint",
    "workflow_status_endpoint",
]