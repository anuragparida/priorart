"""Temporal workflows — the orchestration layer.

Phase 2.1 (docs/PHASE-2.md §2.1): wire the Phase 1.8 inline pipeline
into Temporal. The workflow is **pure orchestration**: each step is
an ``await workflow.execute_activity(...)`` call. There is no I/O
in the workflow body itself.

Phase 2.2 (docs/PHASE-2.md §2.2): adds

- **Activity-level retry policies**: exponential backoff, max 3
  attempts on transient LLM failures. ``SchemaViolationError``
  bypasses retries (fail-fast — a misbehaving model is not a
  network blip).
- **Web fallback activity**: SearXNG-backed re-ranking when the
  corpus returns nothing above the configured cosine threshold.
- **Signal channel for low-confidence verdicts**: cosine in
  0.55–0.70 OR LLM self-confidence < 0.7 parks the workflow on
  ``wait_condition`` until a human posts a review signal.

The workflow body is still pure orchestration — every external
call is an activity. The signal handler is a ``workflow.signal``
method that mutates ``self._review_signal``; the verdict assembly
checks the band and either returns the model's verdict or blocks
on ``wait_condition`` until ``self._review_signal`` is set.
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
    ReviewSignal,
    WorkflowPhase,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------
#
# PHASE-2.md §2.2: "Activity-level retry policies: exponential
# backoff, max 3 attempts on transient LLM failures (5xx, rate
# limit). No retry on schema-violation — fail fast, surface the
# error."
#
# We model this with TWO retry policies, selected per-activity:
#
# - ``_DEFAULT_RETRY``: the workhorse policy for embed, ANN search,
#   web fallback, market-scope, assemble. Backs off 1s → 2s → 4s.
#
# - ``_NO_RETRY_ON_SCHEMA``: identical except ``maximum_attempts=1``
#   so a schema-violating LLM call fails immediately instead of
#   burning the full retry budget. We use it on
#   ``llm_compare_topk`` because the LLM's first schema-violation
#   is *informative* (instructor's last error in the conversation
#   gives the model a self-correction signal — but instructor's
#   own retries are a separate mechanism we keep on inside the
#   activity). At the Temporal layer, retrying a schema-violation
#   is wasted compute.


_DEFAULT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


_NO_RETRY_ON_SCHEMA = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    # ``maximum_attempts=1`` means Temporal will not retry. The
    # activity's exception propagates straight to the workflow
    # failure handler. We rely on this for ``SchemaViolationError``
    # — see the docstring on ``llm_compare_topk`` for the
    # instructor-side retry mechanism that's still active inside
    # the activity.
    maximum_attempts=1,
)


# Low-confidence band — drives the signal channel decision.
# PHASE-2.md §2.2: cosine in 0.55–0.70 OR LLM self-confidence < 0.7.
_LOW_CONF_MIN_COSINE = 0.55
_LOW_CONF_MAX_COSINE = 0.70
_LOW_CONF_LLM_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------


@workflow.defn(name="IdeaAnalysisWorkflow")
class IdeaAnalysisWorkflow:
    """Per-idea analysis workflow — Phase 2.2.

    Phase 2.2 adds:
    - The ``review`` signal handler (``@workflow.signal``).
    - The ``enable_web_fallback`` + ``enable_low_confidence_review``
      branches from ``IdeaAnalysisInput``.
    - A ``web_fallback_if_empty`` activity call between ``ann_search``
      and ``llm_compare_topk``.
    - A low-confidence verdict check at the end that parks on the
      signal channel.
    """

    def __init__(self) -> None:
        self._phase: WorkflowPhase = WorkflowPhase.STARTED
        self._embedding: list[float] | None = None
        self._ann_result: AnnSearchResult | None = None
        self._llm_verdict: dict[str, Any] | None = None
        self._final_verdict: dict[str, Any] | None = None
        self._task_queue: str = "priorart-idea-analysis"
        # Phase 2.2 — observability fields surfaced via ``get_status``.
        self._web_fallback_fired: bool = False
        self._low_confidence: bool = False
        # Phase 2.2 — signal channel state. ``_review_signal`` is
        # populated by the ``review`` signal handler (called via
        # ``handle.signal("review", ReviewSignal(...))`` from the
        # HTTP route). The verdict-assembly step blocks on
        # ``wait_condition(lambda: self._review_signal is not None)``
        # until a human posts the signal.
        self._review_signal: ReviewSignal | None = None
        self._review_reason: str | None = None

    # ------------------------------------------------------------------
    # Signal channel
    # ------------------------------------------------------------------

    @workflow.signal(name="review")
    async def on_review_signal(self, signal: ReviewSignal) -> None:
        """Receive a human review signal and unblock the verdict.

        The HTTP route ``POST /workflows/{id}/signal/review`` calls
        ``handle.signal("review", ReviewSignal(decision=...))`` to
        resume a workflow that parked on a low-confidence verdict.
        The signal handler sets ``self._review_signal`` (and
        ``self._review_reason``); the workflow body's
        ``wait_condition`` then unblocks and the verdict is
        returned (or rejected) per the decision.
        """
        logger.info(
            "on_review_signal: decision=%s reason=%r",
            signal.decision,
            signal.reason,
        )
        self._review_signal = signal
        self._review_reason = signal.reason

    # ------------------------------------------------------------------
    # Workflow body
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, input: IdeaAnalysisInput) -> dict[str, Any]:
        """The workflow body — 5 sequential activities + the fallback + signal.

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

        # ------------------------------------------------------------------
        # Phase 2.2 — web fallback
        # ------------------------------------------------------------------
        #
        # Fires when ``enable_web_fallback`` is True AND the corpus
        # search returned nothing above ``web_fallback_threshold``.
        # The activity's threshold-check is the same as the
        # workflow's, so this is a no-op fast-path on the typical
        # duplicate-idea case (where the corpus has a strong match).
        if input.enable_web_fallback:
            fallback_result = await workflow.execute_activity(
                "web_fallback_if_empty",
                args=[
                    input.idea,
                    self._ann_result,
                ],
                # 90s covers the worst-case: 3 parallel scrapes
                # at 30s each + embed overhead. Sequential would
                # be 90s too; we keep the budget generous because
                # arxiv / sciencedirect are common scrape targets
                # that often hit the 30s limit.
                start_to_close_timeout=timedelta(seconds=90),
                retry_policy=_DEFAULT_RETRY,
                # ``result_type`` must be explicit: without it,
                # the workflow-sandbox _RestrictedProxy of the
                # ``AnnSearchResult`` annotation leaks through
                # the Pydantic data converter and the workflow
                # sees a plain dict (Phase 2.1 had the same
                # issue with ``ann_search``; we learnt the
                # lesson here).
                result_type=AnnSearchResult,
            )
            # Did the activity *change* the result? If so, the
            # fallback fired. We compare hit sets by id (scrape
            # hits have negative ids so the comparison is cheap).
            if fallback_result.hits and (
                len(fallback_result.hits) != len(self._ann_result.hits)
                or any(h.company_id < 0 for h in fallback_result.hits)
            ):
                self._web_fallback_fired = True
                self._phase = WorkflowPhase.WEB_FALLBACK_FETCHED
            else:
                # Activity returned the original result unchanged.
                # The activity's threshold check skipped the
                # fallback path; we leave ``_web_fallback_fired``
                # as False so the metric stays honest.
                pass
            self._ann_result = fallback_result

        # ------------------------------------------------------------------
        # Phase 1.8 path — LLM call + market scope + assemble
        # ------------------------------------------------------------------

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
            # Fail-fast on schema-violation: instructor's internal
            # retries already cover the model side; we don't burn
            # Temporal's retry budget on a deterministic Pydantic
            # failure.
            retry_policy=_NO_RETRY_ON_SCHEMA,
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

        # ------------------------------------------------------------------
        # Phase 2.2 — low-confidence signal channel
        # ------------------------------------------------------------------
        #
        # Check the verdict's confidence band. If the top-1 cosine is
        # in the 0.55–0.70 band OR the LLM self-reported confidence is
        # below 0.7, park on ``wait_condition`` until a human posts a
        # review signal. The signal handler ``on_review_signal`` sets
        # ``self._review_signal`` and the wait unblocks.
        if input.enable_low_confidence_review:
            top1_cosine = (
                max(h.similarity for h in self._ann_result.hits)
                if self._ann_result.hits
                else 0.0
            )
            # LLM self-reported confidence — best-effort extraction
            # from the top-competitor. Tolerates missing keys
            # gracefully (returns 0.0 → not low-confidence).
            llm_self_confidence = 0.0
            try:
                top_competitors = self._llm_verdict.get("top_competitors") or []
                if top_competitors and isinstance(top_competitors, list):
                    first = top_competitors[0]
                    if isinstance(first, dict):
                        llm_self_confidence = float(
                            first.get("confidence", 0.0) or 0.0
                        )
            except (AttributeError, ValueError, TypeError):
                llm_self_confidence = 0.0

            cosine_band = (
                top1_cosine >= _LOW_CONF_MIN_COSINE
                and top1_cosine <= _LOW_CONF_MAX_COSINE
            )
            llm_low = llm_self_confidence < _LOW_CONF_LLM_THRESHOLD

            if cosine_band or llm_low:
                self._low_confidence = True
                logger.info(
                    "low_confidence verdict: top1_cosine=%.3f llm_self_conf=%.3f; "
                    "parking on wait_condition for review signal",
                    top1_cosine,
                    llm_self_confidence,
                )
                self._phase = WorkflowPhase.WAITING_FOR_REVIEW
                # Park indefinitely. The signal handler resets
                # ``self._review_signal``; ``wait_condition`` polls
                # the predicate on the workflow's event-loop tick
                # (deterministic, replay-safe — no time-based
                # wakeups inside the condition).
                await workflow.wait_condition(
                    lambda: self._review_signal is not None
                )
                self._phase = WorkflowPhase.ASSEMBLED
                logger.info(
                    "low_confidence verdict: received review signal decision=%s",
                    self._review_signal.decision,
                )

                signal = self._review_signal
                if signal.decision == "confirm":
                    # Keep the model's verdict as-is. Fall through.
                    pass
                elif signal.decision == "override":
                    if not signal.corrected_verdict:
                        # The reviewer sent an override without a
                        # verdict. Fail loudly — silent fall-through
                        # would mask a UX bug.
                        raise ValueError(
                            "review decision='override' requires "
                            "'corrected_verdict' field"
                        )
                    self._final_verdict = dict(signal.corrected_verdict)
                elif signal.decision == "reject":
                    # Fail the workflow with a structured body. The
                    # HTTP status endpoint surfaces this via
                    # ``describe().failure``.
                    raise RuntimeError(
                        f"workflow rejected by human reviewer: {signal.reason or 'no reason'}"
                    )
                else:
                    raise ValueError(
                        f"unknown review decision: {signal.decision!r}; "
                        "expected 'confirm' / 'override' / 'reject'"
                    )

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
            "web_fallback_fired": self._web_fallback_fired,
            "low_confidence": self._low_confidence,
            "review_pending": (
                self._phase == WorkflowPhase.WAITING_FOR_REVIEW
            ),
        }


__all__ = [
    "IdeaAnalysisWorkflow",
    "ReviewSignal",
    "WorkflowPhase",
    "WorkflowStatus",
]