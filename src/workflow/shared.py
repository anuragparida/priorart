"""Shared dataclasses / Pydantic models for the Temporal workflow layer.

Why a separate ``shared`` module
--------------------------------
Workflows and activities are passed across the Temporal boundary
as plain-Python dataclasses / Pydantic models. Temporal serialises
them via its default JSON converter; both sides (workflow +
activity) need the same class so the wire shape matches. The
canonical place for those shapes is here, not in either the
workflow module or the activities module — that way ``activities``
doesn't import ``workflows`` (Temporal's determinism rules forbid
that anyway) and the test suite can import the shared shapes
without dragging in either side.

Phase 2.1 scope
---------------
This card is JUST the Temporal plumbing + port. The shapes here
mirror the Phase 1.8 wire shapes so the FastAPI route can return
the same JSON shape it did in Phase 1, only sourced via a
Temporal handle instead of an inline call.

Phase 2.2 (retry + fallback) and Phase 2.10 (web search) will
extend ``IdeaAnalysisInput`` and ``WorkflowStatus`` with the
``search_strategy`` / ``web_fallback_used`` / ``low_confidence``
fields they need. We deliberately *don't* add those now — the
spec is explicit: this step is "port verbatim, no behavior
changes".
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.llm.schemas import IdeaVerdict

# ---------------------------------------------------------------------------
# Wire shapes between the FastAPI layer and the workflow
# ---------------------------------------------------------------------------


class IdeaAnalysisInput(BaseModel):
    """Inputs for ``IdeaAnalysisWorkflow.run``.

    This is the shape the FastAPI route passes to
    ``client.start_workflow``. It's also what the activity code
    unpacks at the start of each step — every step takes only
    the fields it needs, but the workflow holds the full input
    so a retry from any activity doesn't lose context.

    ``request_id`` is the FastAPI-side correlation id (e.g. an
    ``X-Request-ID`` header). Phase 2.1 doesn't surface it yet —
    it's plumbed through so Phase 2.3 (Langfuse) can attach it as
    trace metadata without re-plumbing.

    Phase 2.2 adds:

    - ``enable_web_fallback`` (default ``True``): opt-in flag for the
      SearXNG-backed fallback when the corpus returns nothing above
      ``web_fallback_threshold``. AGENTS.md says "Web fallback is
      OPT-IN, not the default path" — but the *feature* defaults to
      on because the eval set requires it to fire on novel ideas
      (PHASE-2.md §2.2 acceptance criteria). Operators who want
      strict offline behaviour can pass ``enable_web_fallback=False``
      in the request body.

    - ``web_fallback_threshold`` (default 0.7 cosine): minimum top-1
      cosine similarity required from the corpus before the fallback
      is skipped. Below this, the workflow runs the web search path.

    - ``enable_low_confidence_review`` (default ``True``): opt-in
      signal channel. When ``True``, a low-confidence verdict
      (top-1 cosine in 0.55–0.70 OR LLM self-confidence < 0.7)
      parks the workflow on ``wait_condition`` for a human review
      signal.
    """

    idea: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Free-text startup idea (same as Phase 1.8).",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="How many competitors to include in the verdict.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional correlation id from the HTTP request.",
    )
    enable_web_fallback: bool = Field(
        default=True,
        description=(
            "Phase 2.2 — when True, the workflow runs a SearXNG-"
            "backed web search if the corpus returns nothing above "
            "``web_fallback_threshold``. Defaults to True so the "
            "Phase 2.2 acceptance criteria pass; operators who want "
            "strict offline behaviour can opt out per-request."
        ),
    )
    web_fallback_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Phase 2.2 — minimum top-1 cosine similarity required "
            "from the corpus before the web fallback is skipped. "
            "Below this, the workflow runs the SearXNG search path."
        ),
    )
    enable_low_confidence_review: bool = Field(
        default=True,
        description=(
            "Phase 2.2 — when True, a low-confidence verdict "
            "(top-1 cosine in 0.55–0.70 OR LLM self-confidence "
            "< 0.7) parks the workflow on a wait_condition until "
            "a human posts a review signal at "
            "POST /workflows/{id}/signal/review."
        ),
    )


class ReviewSignal(BaseModel):
    """The payload of the Phase 2.2 ``review`` signal.

    A human posts this to ``POST /workflows/{id}/signal/review`` to
    resume a workflow that parked on a low-confidence verdict. The
    shape is intentionally a tagged union:

    - ``decision="confirm"`` → keep the model's verdict as-is.
    - ``decision="override"`` → swap the verdict for the supplied
      ``corrected_verdict`` (an ``IdeaVerdict``-shaped dict).
    - ``decision="reject"`` → fail the workflow with a structured
      failure body (the human doesn't trust the model's read).

    The signal-handling logic in the workflow treats ``confirm`` and
    ``override`` as "workflow completes"; ``reject`` as "workflow
    fails with the supplied reason".
    """

    model_config = ConfigDict(extra="forbid")

    decision: str = Field(
        ...,
        description=(
            "``confirm`` (keep verdict), ``override`` (swap in "
            "``corrected_verdict``), or ``reject`` (fail the "
            "workflow with the supplied reason)."
        ),
    )
    corrected_verdict: dict | None = Field(
        default=None,
        description=(
            "Only required when ``decision=\"override\"``. An "
            "IdeaVerdict-shaped dict that replaces the model's "
            "verdict for this run."
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Optional free-text note from the reviewer. Echoed "
            "back in the Langfuse trace metadata (Phase 2.3) and "
            "in the workflow's final log line."
        ),
    )


class AnnSearchHit(BaseModel):
    """One ANN search hit, as passed across the Temporal boundary."""

    model_config = ConfigDict(extra="forbid")

    company_id: int
    name: str
    description: str
    similarity: float = Field(..., ge=-1.0, le=1.0)


class AnnSearchResult(BaseModel):
    """The output of the ``ann_search`` activity."""

    model_config = ConfigDict(extra="forbid")

    hits: list[AnnSearchHit]
    corpus_count: int


# ---------------------------------------------------------------------------
# Workflow status (returned by ``GET /workflows/{id}``)
# ---------------------------------------------------------------------------


class WorkflowPhase(StrEnum):
    """High-level phase the workflow is in.

    Phase 2.2 adds:

    - ``WEB_FALLBACK_FETCHED`` — the SearXNG-backed fallback ran and
      scraped/embedded the top-3 web results.
    - ``WAITING_FOR_REVIEW`` — the workflow parked on the
      low-confidence signal channel; ``GET /workflows/{id}`` will
      show ``status: "RUNNING"`` + ``phase: "waiting_for_review"``
      until a human posts a review signal.
    """

    STARTED = "started"
    EMBEDDED = "embedded"
    SEARCHED = "searched"
    LLM_COMPARED = "llm_compared"
    MARKET_SCOPED = "market_scoped"
    ASSEMBLED = "assembled"
    WEB_FALLBACK_FETCHED = "web_fallback_fetched"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStatus(BaseModel):
    """The wire shape returned by ``GET /workflows/{id}``."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(..., description="Temporal workflow id.")
    run_id: str = Field(..., description="Temporal run id (unique per execution).")
    status: str = Field(
        ...,
        description=(
            "Raw Temporal status: RUNNING / COMPLETED / FAILED / "
            "TIMED_OUT / CANCELLED / TERMINATED."
        ),
    )
    phase: WorkflowPhase = Field(
        default=WorkflowPhase.STARTED,
        description="The last workflow phase to complete.",
    )
    start_time: datetime = Field(..., description="When the workflow was started.")
    close_time: datetime | None = Field(
        default=None,
        description="When the workflow completed or failed. None while running.",
    )
    result: IdeaVerdict | None = Field(
        default=None,
        description="The final IdeaVerdict (only set when status == COMPLETED).",
    )
    failure: dict | None = Field(
        default=None,
        description=(
            "Structured failure info (only set when status == FAILED)."
        ),
    )
    task_queue: str = Field(
        default="priorart-idea-analysis",
        description="The task queue the workflow ran on.",
    )
    web_fallback_fired: bool = Field(
        default=False,
        description=(
            "Phase 2.2 — True when the SearXNG-backed "
            "``web_fallback_if_empty`` activity ran. The eval "
            "harness asserts this fires < 10% of the time on the "
            "labeled benchmark (PHASE-2.md pitfall)."
        ),
    )
    low_confidence: bool = Field(
        default=False,
        description=(
            "Phase 2.2 — True when the verdict hit the "
            "low-confidence band (top-1 cosine in 0.55–0.70 OR "
            "LLM self-confidence < 0.7)."
        ),
    )
    review_pending: bool = Field(
        default=False,
        description=(
            "Phase 2.2 — True while the workflow is parked on the "
            "low-confidence signal channel waiting for a human "
            "review signal. False otherwise."
        ),
    )


__all__ = [
    "AnnSearchHit",
    "AnnSearchResult",
    "IdeaAnalysisInput",
    "IdeaVerdict",
    "ReviewSignal",
    "WorkflowPhase",
    "WorkflowStatus",
]