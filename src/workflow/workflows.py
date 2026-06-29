"""Temporal workflows — the orchestration layer.

Phase 2.1 (docs/PHASE-2.md §2.1): wire the Phase 1.8 inline pipeline
into Temporal. The workflow is **pure orchestration**: each step is
an ``await workflow.execute_activity(...)`` call. There is no I/O
in the workflow body itself.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

from src.workflow.shared import (
    AnnSearchResult,
    IdeaAnalysisInput,
    WorkflowPhase,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default retry policy
# ---------------------------------------------------------------------------

_DEFAULT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------


@workflow.defn(name="IdeaAnalysisWorkflow")
class IdeaAnalysisWorkflow:
    """Per-idea analysis workflow — Phase 2.1."""

    def __init__(self) -> None:
        self._phase: WorkflowPhase = WorkflowPhase.STARTED
        self._embedding: list[float] | None = None
        self._ann_result: AnnSearchResult | None = None
        self._llm_verdict: dict[str, Any] | None = None
        self._final_verdict: dict[str, Any] | None = None
        self._task_queue: str = "priorart-idea-analysis"

    @workflow.run
    async def run(self, input: IdeaAnalysisInput) -> dict[str, Any]:
        """The workflow body — 5 sequential activity calls.

        Note on ``result_type`` — we pass the return-type explicitly
        to every ``workflow.execute_activity`` call. Without it, the
        workflow-sandbox proxy of the annotation can leak through the
        Pydantic data converter as a plain ``dict`` (TypeAdapter sees
        a ``_RestrictedProxy`` of the Pydantic class instead of the
        class itself and falls back to ``dict`` validation). Explicit
        ``result_type=AnnSearchResult`` / ``result_type=dict`` keeps
        the boundary deterministic across replay.
        """
        self._embedding = await workflow.execute_activity(
            "embed_idea",
            input.idea,
            # First call downloads bge-m3 (~2.3 GB) from HuggingFace
            # and loads it into memory; subsequent calls are <1s.
            # 60s is generous for the cold-load path; the warm path
            # has plenty of headroom.
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=_DEFAULT_RETRY,
            result_type=list[float],
        )
        self._phase = WorkflowPhase.EMBEDDED

        self._ann_result = await workflow.execute_activity(
            "ann_search",
            args=[input.idea, input.top_k],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_DEFAULT_RETRY,
            result_type=AnnSearchResult,
        )
        self._phase = WorkflowPhase.SEARCHED

        top_k_payload: list[dict[str, Any]] = [
            {
                "company_id": hit.company_id,
                "name": hit.name,
                "description": hit.description,
                "similarity": hit.similarity,
            }
            for hit in self._ann_result.hits[: input.top_k]
        ]

        self._llm_verdict = await workflow.execute_activity(
            "llm_compare_topk",
            args=[input.idea, top_k_payload, input.top_k],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=_DEFAULT_RETRY,
            result_type=dict,
        )
        self._phase = WorkflowPhase.LLM_COMPARED

        market_scope = await workflow.execute_activity(
            "market_scope_signal",
            self._llm_verdict,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=_DEFAULT_RETRY,
            result_type=dict,
        )
        self._phase = WorkflowPhase.MARKET_SCOPED

        self._final_verdict = await workflow.execute_activity(
            "assemble_verdict",
            args=[self._llm_verdict, market_scope, self._ann_result],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=_DEFAULT_RETRY,
            result_type=dict,
        )
        self._phase = WorkflowPhase.ASSEMBLED

        return self._final_verdict

    @workflow.query(name="get_status")
    def get_status(self) -> dict[str, Any]:
        """Return a WorkflowStatus-shaped dict for the HTTP route.

        The query handler runs inside the workflow's deterministic
        sandbox; it must not call out to non-deterministic code
        (network, file I/O). We return a plain ``dict`` rather
        than a Pydantic ``model_dump`` because Pydantic v2's
        core-schema machinery imports ``pydantic_core`` lazily,
        which trips the workflow-sandbox deadlock detector when
        the query fires for the first time. ``WorkflowStatus``
        construction + validation happens in
        ``workflow_status_endpoint`` (the HTTP route), not here.
        """
        info = workflow.info()
        return {
            "workflow_id": info.workflow_id,
            "run_id": info.run_id,
            "status": "RUNNING",
            "phase": self._phase.value,
            "start_time": info.start_time.isoformat()
            if info.start_time is not None
            else None,
            "close_time": None,
            "result": None,
            "failure": None,
            "task_queue": self._task_queue,
        }


__all__ = [
    "IdeaAnalysisWorkflow",
    "WorkflowPhase",
    "WorkflowStatus",
]