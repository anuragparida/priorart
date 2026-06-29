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
from enum import Enum

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


class WorkflowPhase(str, Enum):
    """High-level phase the workflow is in."""

    STARTED = "started"
    EMBEDDED = "embedded"
    SEARCHED = "searched"
    LLM_COMPARED = "llm_compared"
    MARKET_SCOPED = "market_scoped"
    ASSEMBLED = "assembled"
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


__all__ = [
    "AnnSearchHit",
    "AnnSearchResult",
    "IdeaAnalysisInput",
    "IdeaVerdict",
    "WorkflowPhase",
    "WorkflowStatus",
]