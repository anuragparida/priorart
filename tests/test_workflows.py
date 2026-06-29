"""Tests for the Temporal workflow layer (Phase 2.1)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.workflows import (
    AnalyzeStartResponse,
    WorkflowStatusResponse,
    analyze_start_endpoint,
    workflow_status_endpoint,
)
from src.workflow.shared import (
    AnnSearchHit,
    AnnSearchResult,
    IdeaAnalysisInput,
    WorkflowPhase,
    WorkflowStatus,
)
from src.workflow.workflows import IdeaAnalysisWorkflow

# ---------------------------------------------------------------------------
# Wire-shape tests
# ---------------------------------------------------------------------------


class TestIdeaAnalysisInput:
    """``IdeaAnalysisInput`` is the workflow's input shape."""

    def test_minimum_input(self) -> None:
        """An idea string is required; top_k defaults to 3."""
        inp = IdeaAnalysisInput(idea="hello")
        assert inp.idea == "hello"
        assert inp.top_k == 3
        assert inp.request_id is None

    def test_top_k_bounds(self) -> None:
        """top_k must be in [1, 5]."""
        with pytest.raises(ValueError):
            IdeaAnalysisInput(idea="x", top_k=0)
        with pytest.raises(ValueError):
            IdeaAnalysisInput(idea="x", top_k=6)

    def test_request_id_propagates(self) -> None:
        inp = IdeaAnalysisInput(
            idea="x",
            request_id="req-abc-123",
        )
        assert inp.request_id == "req-abc-123"


class TestAnnSearchResult:
    """``AnnSearchResult`` is the wire shape from the ``ann_search`` activity."""

    def test_empty_hits(self) -> None:
        result = AnnSearchResult(hits=[], corpus_count=0)
        assert result.hits == []
        assert result.corpus_count == 0

    def test_with_hits(self) -> None:
        result = AnnSearchResult(
            hits=[
                AnnSearchHit(
                    company_id=42,
                    name="Acme",
                    description="acme description",
                    similarity=0.87,
                )
            ],
            corpus_count=5990,
        )
        assert result.hits[0].company_id == 42
        assert result.corpus_count == 5990

    def test_similarity_bounds(self) -> None:
        with pytest.raises(ValueError):
            AnnSearchHit(company_id=1, name="x", description="y", similarity=2.0)
        with pytest.raises(ValueError):
            AnnSearchHit(company_id=1, name="x", description="y", similarity=-2.0)


class TestWorkflowStatus:
    """``WorkflowStatus`` is the wire shape of ``GET /workflows/{id}``."""

    def test_running_workflow(self) -> None:
        ws = WorkflowStatus(
            workflow_id="wf-1",
            run_id="run-1",
            status="RUNNING",
            phase=WorkflowPhase.EMBEDDED,
            start_time=datetime(2026, 6, 29, 12, 0, 0),
            close_time=None,
        )
        assert ws.status == "RUNNING"
        assert ws.phase == WorkflowPhase.EMBEDDED
        assert ws.close_time is None
        assert ws.result is None
        assert ws.failure is None

    def test_completed_workflow(self) -> None:
        ws = WorkflowStatus(
            workflow_id="wf-2",
            run_id="run-2",
            status="COMPLETED",
            phase=WorkflowPhase.ASSEMBLED,
            start_time=datetime(2026, 6, 29, 12, 0, 0),
            close_time=datetime(2026, 6, 29, 12, 0, 30),
        )
        assert ws.status == "COMPLETED"
        assert ws.close_time is not None

    def test_failed_workflow(self) -> None:
        ws = WorkflowStatus(
            workflow_id="wf-3",
            run_id="run-3",
            status="FAILED",
            phase=WorkflowPhase.SEARCHED,
            start_time=datetime(2026, 6, 29, 12, 0, 0),
            close_time=datetime(2026, 6, 29, 12, 0, 30),
            failure={"type": "SchemaViolationError", "message": "bad"},
        )
        assert ws.status == "FAILED"
        assert ws.failure is not None
        assert ws.failure["type"] == "SchemaViolationError"


# ---------------------------------------------------------------------------
# Workflow orchestration order
# ---------------------------------------------------------------------------


class TestWorkflowOrchestration:
    """The workflow's activity call order must match the spec."""

    def test_workflow_class_is_registered(self) -> None:
        """The workflow class has the right ``@workflow.defn`` name."""
        for attr in vars(IdeaAnalysisWorkflow):
            if attr.endswith("temporal_workflow_definition"):
                defn = getattr(IdeaAnalysisWorkflow, attr)
                assert defn.name == "IdeaAnalysisWorkflow"
                return
        pytest.fail(
            "IdeaAnalysisWorkflow has no @workflow.defn registration"
        )

    def test_run_method_is_async(self) -> None:
        """``run`` is an async method (Temporal requirement)."""
        import inspect

        assert inspect.iscoroutinefunction(IdeaAnalysisWorkflow.run)

    def test_workflow_init_sets_initial_phase(self) -> None:
        """A freshly-constructed workflow starts in the STARTED phase."""
        wf = IdeaAnalysisWorkflow()
        assert wf._phase == WorkflowPhase.STARTED
        assert wf._embedding is None
        assert wf._ann_result is None
        assert wf._llm_verdict is None
        assert wf._final_verdict is None

    def test_get_status_query_is_registered(self) -> None:
        """``get_status`` is exposed as a Temporal query."""
        for attr in vars(IdeaAnalysisWorkflow):
            if attr.endswith("temporal_workflow_definition"):
                defn = getattr(IdeaAnalysisWorkflow, attr)
                names = {q.name for q in defn.queries.values()}
                assert "get_status" in names, (
                    f"expected 'get_status' query, got {names}"
                )
                return
        pytest.fail(
            "IdeaAnalysisWorkflow has no @workflow.defn registration"
        )


# ---------------------------------------------------------------------------
# HTTP route tests (Phase 2.1 contract)
# ---------------------------------------------------------------------------


@dataclass
class _FakeStartHandle:
    """Stand-in for ``WorkflowHandle``."""

    workflow_id: str
    run_id: str

    @property
    def id(self) -> str:
        return self.workflow_id

    @property
    def result_run_id(self) -> str:
        return self.run_id


@dataclass
class _FakeStartClient:
    """Minimal Temporal client stand-in for ``analyze_start_endpoint``."""

    start_workflow: AsyncMock = field(default_factory=AsyncMock)


class TestAnalyzeStartEndpoint:
    """``POST /ideas/analyze`` → ``{workflow_id, run_id, status, task_queue}``."""

    def test_returns_workflow_handle_on_success(self) -> None:
        async def _run() -> None:
            fake_handle = _FakeStartHandle(
                workflow_id="wf-abc",
                run_id="run-xyz",
            )
            fake_client = _FakeStartClient()
            fake_client.start_workflow.return_value = fake_handle
            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await analyze_start_endpoint(
                    IdeaAnalysisInput(idea="hello world"),
                )
            assert isinstance(resp, AnalyzeStartResponse)
            assert resp.workflow_id == "wf-abc"
            assert resp.run_id == "run-xyz"
            assert resp.status == "running"
            assert resp.task_queue == "priorart-idea-analysis"

            fake_client.start_workflow.assert_awaited_once()
            call_kwargs = fake_client.start_workflow.await_args.kwargs
            assert call_kwargs["id"].startswith("idea-analysis-")
            assert call_kwargs["task_queue"] == "priorart-idea-analysis"

        asyncio.run(_run())

    def test_temporal_unavailable_returns_503(self) -> None:
        from fastapi import HTTPException

        async def _run() -> None:
            fake_client = _FakeStartClient()
            fake_client.start_workflow.side_effect = ConnectionError(
                "Temporal is down"
            )
            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await analyze_start_endpoint(
                        IdeaAnalysisInput(idea="hello world"),
                    )
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "temporal_unavailable"

        asyncio.run(_run())


@dataclass
class _FakeDescribe:
    status: Any = None
    run_id: str = "run-xyz"
    start_time: datetime = field(
        default_factory=lambda: datetime(2026, 6, 29, 12, 0, 0)
    )
    close_time: datetime | None = None
    failure: Any = None


@dataclass
class _FakeStatusHandle:
    description: _FakeDescribe
    query_result: dict | None = None
    result_payload: Any = None
    query_side_effect: BaseException | None = None
    describe_side_effect: BaseException | None = None
    result_side_effect: BaseException | None = None

    async def describe(self) -> _FakeDescribe:
        if self.describe_side_effect is not None:
            raise self.describe_side_effect
        return self.description

    async def query(self, name: str) -> dict:
        if self.query_side_effect is not None:
            raise self.query_side_effect
        if self.query_result is None:
            return {}
        return self.query_result

    async def result(self) -> Any:
        if self.result_side_effect is not None:
            raise self.result_side_effect
        return self.result_payload


@dataclass
class _FakeStatusClient:
    handle: _FakeStatusHandle

    def get_workflow_handle(self, workflow_id: str) -> _FakeStatusHandle:
        return self.handle


class TestWorkflowStatusEndpoint:
    """``GET /workflows/{id}`` → ``WorkflowStatusResponse``."""

    def _make_client(
        self,
        *,
        status: Any,
        run_id: str = "run-xyz",
        start_time: datetime | None = None,
        close_time: datetime | None = None,
        query_result: dict | None = None,
        query_side_effect: BaseException | None = None,
        result_payload: Any = None,
        failure: Any = None,
        describe_side_effect: BaseException | None = None,
        result_side_effect: BaseException | None = None,
    ) -> _FakeStatusClient:
        if start_time is None:
            start_time = datetime(2026, 6, 29, 12, 0, 0)
        handle = _FakeStatusHandle(
            description=_FakeDescribe(
                status=status,
                run_id=run_id,
                start_time=start_time,
                close_time=close_time,
                failure=failure,
            ),
            query_result=query_result,
            query_side_effect=query_side_effect,
            describe_side_effect=describe_side_effect,
            result_payload=result_payload,
            result_side_effect=result_side_effect,
        )
        return _FakeStatusClient(handle=handle)

    def test_running_workflow(self) -> None:
        async def _run() -> None:
            from temporalio.client import WorkflowExecutionStatus

            fake_client = self._make_client(
                status=WorkflowExecutionStatus.RUNNING,
                query_result={"phase": "embedded"},
            )

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await workflow_status_endpoint("wf-1")
            assert isinstance(resp, WorkflowStatusResponse)
            assert resp.workflow_id == "wf-1"
            assert resp.status == "RUNNING"
            assert resp.phase == "embedded"
            assert resp.close_time is None
            assert resp.result is None

        asyncio.run(_run())

    def test_completed_workflow_returns_verdict(self) -> None:
        async def _run() -> None:
            from temporalio.client import WorkflowExecutionStatus

            verdict_dict = {
                "idea": "AI for SMB contract review",
                "top_competitors": [
                    {
                        "company_id": 1,
                        "name": "Alpha Co",
                        "similarity_axes": ["AI drafting"],
                        "key_differences": ["different focus"],
                        "likely_failure_modes": ["strong distribution"],
                        "evidence_links": [],
                        "confidence": 0.75,
                    }
                ],
                "market_scope": "crowded_but_growing",
                "market_scope_rationale": "3 similar YC launches",
                "supporting_evidence": [],
            }
            fake_client = self._make_client(
                status=WorkflowExecutionStatus.COMPLETED,
                close_time=datetime(2026, 6, 29, 12, 0, 30),
                query_result={"phase": "assembled"},
                result_payload=verdict_dict,
            )

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await workflow_status_endpoint("wf-1")
            assert resp.status == "COMPLETED"
            assert resp.phase == "assembled"
            assert resp.result is not None
            assert resp.result.idea == "AI for SMB contract review"
            assert resp.result.market_scope.value == "crowded_but_growing"
            assert len(resp.result.top_competitors) == 1

        asyncio.run(_run())

    def test_failed_workflow_returns_failure_info(self) -> None:
        async def _run() -> None:
            from temporalio.client import WorkflowExecutionStatus

            # Build a Temporal-style failure chain via the SDK's
            # ``ApplicationError`` (the leaf: a Pydantic / instructor
            # validation error) and a plain ``Exception`` for the
            # outer ActivityError wrapper. ``ActivityError`` is
            # awkward to construct in tests — it requires
            # ``scheduled_event_id`` + ``started_event_id`` from a
            # real event-history event — so we use a real
            # ``ApplicationError`` for the leaf (which is the
            # thing that surfaces in the message) and a plain
            # ``Exception`` for the outer.
            from temporalio.exceptions import ApplicationError

            app_err = ApplicationError(
                "MissingAPIKeyError: Anthropic API key not found. "
                "Set $ANTHROPIC_API_KEY or write the key to ~/.anthropic_key.",
                type="MissingAPIKeyError",
            )
            # The endpoint walks ``getattr(exc, 'cause', None)``
            # then ``__cause__``. We set ``.cause`` directly so the
            # walk goes one level deep into ``ApplicationError``.
            activity_err = Exception("Activity task failed")
            activity_err.cause = app_err  # type: ignore[attr-defined]

            fake_client = self._make_client(
                status=WorkflowExecutionStatus.FAILED,
                close_time=datetime(2026, 6, 29, 12, 0, 30),
                result_side_effect=activity_err,
            )

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await workflow_status_endpoint("wf-1")
            assert resp.status == "FAILED"
            assert resp.failure is not None
            # Top-level: the wrapper exception's class name + message.
            assert resp.failure["type"] == "Exception"
            assert "Activity" in resp.failure["message"]
            # Cause: ApplicationError with the LLM message.
            assert resp.failure["cause"] is not None
            assert resp.failure["cause"]["type"] == "ApplicationError"
            assert "MissingAPIKeyError" in resp.failure["cause"]["message"]

        asyncio.run(_run())

    def test_query_failure_does_not_fail_the_request(self) -> None:
        async def _run() -> None:
            from temporalio.client import WorkflowExecutionStatus

            fake_client = self._make_client(
                status=WorkflowExecutionStatus.RUNNING,
                query_side_effect=ConnectionError("query failed"),
            )

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await workflow_status_endpoint("wf-1")
            assert resp.status == "RUNNING"
            assert resp.phase == "started"

        asyncio.run(_run())

    def test_unknown_workflow_returns_404(self) -> None:
        from fastapi import HTTPException

        async def _run() -> None:
            fake_client = self._make_client(
                status=MagicMock(),
                describe_side_effect=RuntimeError("workflow execution not found"),
            )

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await workflow_status_endpoint("wf-unknown")
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"] == "workflow_not_found"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Failure-surface helper (Phase 2.2)
# ---------------------------------------------------------------------------


class TestFailureToDict:
    """``_failure_to_dict`` walks the Temporal FailureError chain."""

    def test_single_exception_no_cause(self) -> None:
        from src.api.workflows import _failure_to_dict

        out = _failure_to_dict(ValueError("boom"))
        assert out["type"] == "ValueError"
        assert out["message"] == "boom"
        assert out["cause"] is None

    def test_two_level_chain(self) -> None:
        from src.api.workflows import _failure_to_dict

        cause = RuntimeError("underlying")
        wrapper = Exception("wrapper")
        wrapper.cause = cause  # type: ignore[attr-defined]

        out = _failure_to_dict(wrapper)
        assert out["type"] == "Exception"
        assert out["message"] == "wrapper"
        assert out["cause"] is not None
        assert out["cause"]["type"] == "RuntimeError"
        assert out["cause"]["message"] == "underlying"
        assert out["cause"]["cause"] is None

    def test_temporal_application_error(self) -> None:
        """The real-world shape: ActivityError → ApplicationError."""
        from temporalio.exceptions import ApplicationError

        from src.api.workflows import _failure_to_dict

        app = ApplicationError(
            "MissingAPIKeyError: Anthropic API key not found.",
            type="MissingAPIKeyError",
        )
        wrapper = Exception("Activity task failed")
        wrapper.cause = app  # type: ignore[attr-defined]

        out = _failure_to_dict(wrapper)
        assert out["cause"]["type"] == "ApplicationError"
        assert "MissingAPIKeyError" in out["cause"]["message"]

    def test_none_input(self) -> None:
        from src.api.workflows import _failure_to_dict

        out = _failure_to_dict(None)
        assert out["type"] == "Unknown"
        assert out["message"] == ""
        assert out["cause"] is None

    def test_cycle_is_bounded(self) -> None:
        """A self-referencing exception chain must not loop forever."""
        from src.api.workflows import _failure_to_dict

        a = Exception("a")
        b = Exception("b")
        a.cause = b  # type: ignore[attr-defined]
        b.cause = a  # type: ignore[attr-defined]

        out = _failure_to_dict(a)
        # Walk should terminate. We don't assert exact shape
        # (the depth bound + cycle-detection might place the
        # CycleDetected marker at different levels); we just
        # assert it didn't hang and the structure is finite.
        depth = 0
        node: dict | None = out
        while node is not None and depth < 100:
            node = node.get("cause")
            depth += 1
        assert depth < 50, f"chain depth too deep ({depth}) — possible infinite walk"


# ---------------------------------------------------------------------------
# Activities module — module-level invariants
# ---------------------------------------------------------------------------


class TestActivitiesModule:
    """The activities module exposes the five core activities + the Phase 2.2 fallback."""

    def test_all_activities_exported(self) -> None:
        from src.workflow import activities

        # Phase 2.2 — added ``web_fallback_if_empty`` (SearXNG-backed
        # fallback when the corpus returns nothing above the threshold).
        for name in (
            "embed_idea",
            "ann_search",
            "llm_compare_topk",
            "market_scope_signal",
            "assemble_verdict",
            "web_fallback_if_empty",
        ):
            assert hasattr(activities, name), (
                f"src.workflow.activities.{name} is missing"
            )
            fn = getattr(activities, name)
            assert callable(fn)

    def test_activity_names_match_spec(self) -> None:
        """The Temporal activity names match the PHASE-2.md §2.1 + §2.2 spec."""
        from src.workflow import activities

        for attr_name in (
            "embed_idea",
            "ann_search",
            "llm_compare_topk",
            "market_scope_signal",
            "assemble_verdict",
            "web_fallback_if_empty",
        ):
            fn = getattr(activities, attr_name)
            def_attr = None
            for attr in vars(fn):
                if attr.endswith("temporal_activity_definition"):
                    def_attr = attr
                    break
            assert def_attr is not None, (
                f"{attr_name} is missing the @activity.defn registration"
            )
            defn = getattr(fn, def_attr)
            assert defn.name == attr_name


# ---------------------------------------------------------------------------
# Phase 1.8 backwards-compat
# ---------------------------------------------------------------------------


class TestPhase18BackwardsCompat:
    def test_analyze_endpoint_still_importable(self) -> None:
        from src.api.analyze import analyze_endpoint

        assert callable(analyze_endpoint)

    def test_analyze_request_still_importable(self) -> None:
        from src.api.analyze import AnalyzeRequest

        req = AnalyzeRequest(idea="hello")
        assert req.idea == "hello"
        assert req.top_k == 3

    def test_analyze_error_still_importable(self) -> None:
        from src.api.analyze import AnalyzeError

        e = AnalyzeError(error="no_competitors", details={"corpus_count": 0})
        assert e.error == "no_competitors"


# ---------------------------------------------------------------------------
# Worker module — basic argparse / registration invariants
# ---------------------------------------------------------------------------


class TestWorkerCLI:
    """The worker CLI builds the right defaults + has the right flags."""

    def test_arg_parser_defaults(self) -> None:
        from src.workflow.worker import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.address == "127.0.0.1:7233"
        assert args.namespace == "default"
        assert args.task_queue == "priorart-idea-analysis"
        assert args.log_level == "INFO"
        assert args.reset_engine is False

    def test_arg_parser_overrides(self) -> None:
        from src.workflow.worker import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--address",
                "127.0.0.1:9999",
                "--namespace",
                "staging",
                "--task-queue",
                "tq-x",
                "--log-level",
                "DEBUG",
            ]
        )
        assert args.address == "127.0.0.1:9999"
        assert args.namespace == "staging"
        assert args.task_queue == "tq-x"
        assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Phase 2.2 — retry policies
# ---------------------------------------------------------------------------


class TestRetryPolicies:
    """PHASE-2.md §2.2: activity-level retry policies, exponential
    backoff, max 3 attempts on transient failures, no retry on
    schema-violation."""

    def test_default_retry_has_three_attempts(self) -> None:
        from src.workflow.workflows import _DEFAULT_RETRY

        assert _DEFAULT_RETRY.maximum_attempts == 3, (
            "default retry must allow 3 attempts (PHASE-2.md §2.2)"
        )

    def test_default_retry_uses_exponential_backoff(self) -> None:
        from src.workflow.workflows import _DEFAULT_RETRY

        assert _DEFAULT_RETRY.backoff_coefficient == 2.0
        assert _DEFAULT_RETRY.initial_interval.total_seconds() == 1.0

    def test_no_retry_on_schema_has_one_attempt(self) -> None:
        """Schema violations fail fast — exactly 1 attempt."""
        from src.workflow.workflows import _NO_RETRY_ON_SCHEMA

        assert _NO_RETRY_ON_SCHEMA.maximum_attempts == 1, (
            "schema violations must NOT retry (PHASE-2.md §2.2)"
        )

    def test_workflow_routes_llm_activity_to_no_retry_policy(self) -> None:
        """The workflow must wire the LLM activity with the no-retry
        policy, while other activities use the default. We can't
        introspect Temporal's internal wiring directly, but we can
        confirm the two policy instances differ and exist as
        module-level constants on the workflows module — the
        workflow body uses both."""
        from src.workflow import workflows as w

        assert w._DEFAULT_RETRY is not w._NO_RETRY_ON_SCHEMA
        assert w._DEFAULT_RETRY.maximum_attempts != w._NO_RETRY_ON_SCHEMA.maximum_attempts


# ---------------------------------------------------------------------------
# Phase 2.2 — IdeaAnalysisInput extended fields
# ---------------------------------------------------------------------------


class TestIdeaAnalysisInputPhase22:
    """Phase 2.2 adds three fields to ``IdeaAnalysisInput`` for
    opt-in fallback + signal-channel behaviour."""

    def test_enable_web_fallback_default(self) -> None:
        from src.workflow.shared import IdeaAnalysisInput

        inp = IdeaAnalysisInput(idea="hello")
        assert inp.enable_web_fallback is True, (
            "fallback must default on (PHASE-2.md §2.2 acceptance)"
        )

    def test_web_fallback_threshold_default(self) -> None:
        from src.workflow.shared import IdeaAnalysisInput

        inp = IdeaAnalysisInput(idea="hello")
        assert inp.web_fallback_threshold == 0.7

    def test_enable_low_confidence_review_default(self) -> None:
        from src.workflow.shared import IdeaAnalysisInput

        inp = IdeaAnalysisInput(idea="hello")
        assert inp.enable_low_confidence_review is True

    def test_web_fallback_threshold_bounds(self) -> None:
        from src.workflow.shared import IdeaAnalysisInput

        with pytest.raises(ValueError):
            IdeaAnalysisInput(idea="x", web_fallback_threshold=-0.1)
        with pytest.raises(ValueError):
            IdeaAnalysisInput(idea="x", web_fallback_threshold=1.1)

    def test_opt_out_of_web_fallback(self) -> None:
        """Operators can pass ``enable_web_fallback=False`` to run
        strict-offline — useful for the eval set regression when
        you want to measure corpus-only behaviour."""
        from src.workflow.shared import IdeaAnalysisInput

        inp = IdeaAnalysisInput(idea="hello", enable_web_fallback=False)
        assert inp.enable_web_fallback is False


# ---------------------------------------------------------------------------
# Phase 2.2 — WorkflowPhase + WorkflowStatus + ReviewSignal
# ---------------------------------------------------------------------------


class TestWorkflowPhasePhase22:
    """Phase 2.2 adds two phase enum values."""

    def test_web_fallback_fetched_phase_exists(self) -> None:
        from src.workflow.shared import WorkflowPhase

        assert WorkflowPhase.WEB_FALLBACK_FETCHED.value == "web_fallback_fetched"

    def test_waiting_for_review_phase_exists(self) -> None:
        from src.workflow.shared import WorkflowPhase

        assert WorkflowPhase.WAITING_FOR_REVIEW.value == "waiting_for_review"


class TestWorkflowStatusPhase22:
    """Phase 2.2 adds three WorkflowStatus fields surfaced via
    ``get_status``."""

    def test_default_values(self) -> None:
        from src.workflow.shared import WorkflowStatus

        ws = WorkflowStatus(
            workflow_id="wf-1",
            run_id="run-1",
            status="RUNNING",
            start_time=datetime(2026, 6, 29, 12, 0, 0),
        )
        assert ws.web_fallback_fired is False
        assert ws.low_confidence is False
        assert ws.review_pending is False

    def test_fallback_fired_flag(self) -> None:
        from src.workflow.shared import WorkflowStatus

        ws = WorkflowStatus(
            workflow_id="wf-2",
            run_id="run-2",
            status="RUNNING",
            start_time=datetime(2026, 6, 29, 12, 0, 0),
            web_fallback_fired=True,
        )
        assert ws.web_fallback_fired is True


class TestReviewSignal:
    """The Phase 2.2 ``ReviewSignal`` Pydantic model — the body of
    ``POST /workflows/{id}/signal/review``."""

    def test_confirm_decision(self) -> None:
        from src.workflow.shared import ReviewSignal

        sig = ReviewSignal(decision="confirm")
        assert sig.decision == "confirm"
        assert sig.corrected_verdict is None
        assert sig.reason is None

    def test_override_decision_with_verdict(self) -> None:
        from src.workflow.shared import ReviewSignal

        sig = ReviewSignal(
            decision="override",
            corrected_verdict={
                "idea": "x",
                "top_competitors": [],
                "market_scope": "wide_open",
                "market_scope_rationale": "no real competitor",
                "supporting_evidence": [],
            },
            reason="model was wrong, here's the correct verdict",
        )
        assert sig.decision == "override"
        assert sig.corrected_verdict is not None
        assert sig.corrected_verdict["market_scope"] == "wide_open"

    def test_reject_decision(self) -> None:
        from src.workflow.shared import ReviewSignal

        sig = ReviewSignal(decision="reject", reason="garbage")
        assert sig.decision == "reject"
        assert sig.reason == "garbage"

    def test_extra_fields_rejected(self) -> None:
        from src.workflow.shared import ReviewSignal

        with pytest.raises(ValueError):
            ReviewSignal(decision="confirm", unknown_field="x")


# ---------------------------------------------------------------------------
# Phase 2.2 — workflow signal handler + status query fields
# ---------------------------------------------------------------------------


class TestWorkflowSignalHandler:
    """The workflow exposes a ``review`` signal handler that
    unblocks the low-confidence verdict assembly."""

    def test_on_review_signal_method_exists(self) -> None:
        from src.workflow.workflows import IdeaAnalysisWorkflow

        assert hasattr(IdeaAnalysisWorkflow, "on_review_signal")

    def test_on_review_signal_has_workflow_signal_decorator(self) -> None:
        from src.workflow.workflows import IdeaAnalysisWorkflow

        fn = IdeaAnalysisWorkflow.on_review_signal
        found = False
        for attr in vars(fn):
            if attr.endswith("temporal_signal_definition"):
                defn = getattr(fn, attr)
                assert defn.name == "review"
                found = True
                break
        assert found, "on_review_signal is missing the @workflow.signal decorator"

    def test_workflow_init_initializes_signal_state(self) -> None:
        """``_review_signal`` and ``_review_reason`` start as None
        so ``wait_condition`` parks until a signal arrives."""
        from src.workflow.workflows import IdeaAnalysisWorkflow

        wf = IdeaAnalysisWorkflow()
        assert wf._review_signal is None
        assert wf._review_reason is None
        assert wf._web_fallback_fired is False
        assert wf._low_confidence is False

    def test_get_status_includes_phase22_fields(self) -> None:
        """The ``get_status`` query handler must surface the three
        Phase 2.2 fields so the HTTP route can serialize them."""
        # We can't drive ``workflow.query`` outside of a real
        # Temporal environment, but we can introspect the
        # underlying ``get_status`` body via the function's
        # source — or just instantiate the workflow and check
        # the dict shape manually by calling ``get_status``
        # under a Temporal worker mock. Easier: just verify the
        # method exists and returns the expected keys when called
        # outside the workflow sandbox (the @workflow.query
        # decorator is a no-op outside the worker sandbox).
        # The query handler runs in the sandbox; calling it
        # directly outside raises WorkflowSandboxBlockedError
        # in production. We don't want to drag Temporal's
        # workflow sandbox setup into a unit test, so we just
        # assert the dict keys are *referenced* in the function
        # source — the bare minimum invariant the HTTP route
        # relies on.
        import inspect

        from src.workflow.workflows import IdeaAnalysisWorkflow

        source = inspect.getsource(IdeaAnalysisWorkflow.get_status)
        assert "web_fallback_fired" in source
        assert "low_confidence" in source
        assert "review_pending" in source


# ---------------------------------------------------------------------------
# Phase 2.2 — workflow body wires the fallback between ann_search and llm
# ---------------------------------------------------------------------------


class TestWorkflowFallbackWiring:
    """The workflow must call ``web_fallback_if_empty`` between
    ``ann_search`` and ``llm_compare_topk`` when
    ``enable_web_fallback`` is True."""

    def test_workflow_body_calls_web_fallback(self) -> None:
        import inspect

        from src.workflow.workflows import IdeaAnalysisWorkflow

        source = inspect.getsource(IdeaAnalysisWorkflow.run)
        # The activity call must appear in the workflow body.
        assert '"web_fallback_if_empty"' in source
        # And it must come AFTER ann_search but BEFORE
        # llm_compare_topk. Order matters for the data flow.
        ann_idx = source.find('"ann_search"')
        fallback_idx = source.find('"web_fallback_if_empty"')
        llm_idx = source.find('"llm_compare_topk"')
        assert ann_idx < fallback_idx < llm_idx, (
            "workflow must run ann_search → web_fallback_if_empty → "
            "llm_compare_topk in that order"
        )

    def test_workflow_body_uses_no_retry_on_schema_for_llm(self) -> None:
        """The LLM activity call must be wired with
        ``_NO_RETRY_ON_SCHEMA``, not ``_DEFAULT_RETRY``. We assert
        by source-grep — the retry_policy kwarg on the
        llm_compare_topk execute_activity call must reference the
        no-retry constant."""
        import inspect

        from src.workflow.workflows import IdeaAnalysisWorkflow

        source = inspect.getsource(IdeaAnalysisWorkflow.run)
        # Find the ``llm_compare_topk`` block + the next 30 lines
        # (the execute_activity call sits inside).
        start = source.find('"llm_compare_topk"')
        block = source[start : start + 800]
        assert "_NO_RETRY_ON_SCHEMA" in block, (
            "llm_compare_topk must use the no-retry policy"
        )

    def test_workflow_body_parks_on_low_confidence(self) -> None:
        """The workflow must call ``wait_condition`` after
        ``assemble_verdict`` to park on the signal channel."""
        import inspect

        from src.workflow.workflows import IdeaAnalysisWorkflow

        source = inspect.getsource(IdeaAnalysisWorkflow.run)
        assert "wait_condition" in source, (
            "workflow must park on wait_condition for low-confidence"
        )
        assert "WAITING_FOR_REVIEW" in source or "waiting_for_review" in source.lower()


# ---------------------------------------------------------------------------
# Phase 2.2 — low-confidence band constants
# ---------------------------------------------------------------------------


class TestLowConfidenceBand:
    """PHASE-2.md §2.2: cosine in 0.55–0.70 OR LLM self-confidence
    < 0.7. The workflow module exposes these as module-level
    constants so tests + dashboards can reference them."""

    def test_band_constants(self) -> None:
        from src.workflow.workflows import (
            _LOW_CONF_LLM_THRESHOLD,
            _LOW_CONF_MAX_COSINE,
            _LOW_CONF_MIN_COSINE,
        )

        assert _LOW_CONF_MIN_COSINE == 0.55
        assert _LOW_CONF_MAX_COSINE == 0.70
        assert _LOW_CONF_LLM_THRESHOLD == 0.7


# ---------------------------------------------------------------------------
# Phase 2.2 — HTTP route surface
# ---------------------------------------------------------------------------


class TestSignalReviewRoute:
    """``POST /workflows/{id}/signal/review`` is wired in
    ``src.api.app``."""

    def test_route_registered(self) -> None:
        from fastapi.routing import APIRoute

        from src.api.app import app

        matches = [
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path.endswith("/signal/review")
            and "POST" in route.methods
        ]
        assert matches, (
            "POST /workflows/{id}/signal/review route is missing"
        )
        # The handler must be wired to our endpoint body.
        assert (
            matches[0].endpoint.__name__
            == "workflows_signal_review"
        )

    def test_route_uses_review_signal_body(self) -> None:
        """The route must accept a ``ReviewSignal`` Pydantic body."""
        from fastapi.routing import APIRoute

        from src.api.app import app
        from src.workflow.shared import ReviewSignal

        matches = [
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path.endswith("/signal/review")
        ]
        assert matches
        route = matches[0]
        # FastAPI stores the body field type on
        # ``route.dependant.body_params`` as ``ModelField`` in
        # the version this project pins. We pull the type off
        # the field via the ``outer_type_`` attribute that
        # ``ModelField`` exposes for Pydantic models.
        assert route.dependant.body_params, (
            "POST /workflows/{id}/signal/review has no body params"
        )
        # The body field's annotation should resolve to ``ReviewSignal``.
        # We check via the ``field_info`` annotation, which FastAPI
        # sets from the function signature's parameter type.
        body_param = route.dependant.body_params[0]
        annotation = getattr(body_param, "annotation", None) or getattr(
            body_param.field_info, "annotation", None
        )
        assert annotation is ReviewSignal or (
            # ``ModelField`` in older FastAPI exposes ``type_`` via
            # ``outer_type_``; if neither matches, fall back to
            # stringifying for a sanity check.
            "ReviewSignal" in str(annotation)
        )

    def test_response_model_is_signal_review_response(self) -> None:
        from fastapi.routing import APIRoute

        from src.api.app import app
        from src.api.workflows import SignalReviewResponse

        matches = [
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path.endswith("/signal/review")
        ]
        assert matches
        assert matches[0].response_model is SignalReviewResponse


class TestStatusResponsePhase22:
    """``GET /workflows/{id}`` must include the Phase 2.2 fields."""

    def test_response_model_includes_phase22_fields(self) -> None:
        from src.api.workflows import WorkflowStatusResponse

        resp = WorkflowStatusResponse(
            workflow_id="wf-1",
            run_id="run-1",
            status="RUNNING",
            phase="waiting_for_review",
            start_time=datetime(2026, 6, 29, 12, 0, 0),
        )
        assert resp.web_fallback_fired is False
        assert resp.low_confidence is False
        assert resp.review_pending is False

        # Override one to True to verify the field actually plumbs.
        resp2 = WorkflowStatusResponse(
            workflow_id="wf-2",
            run_id="run-2",
            status="COMPLETED",
            phase="completed",
            start_time=datetime(2026, 6, 29, 12, 0, 0),
            web_fallback_fired=True,
            low_confidence=True,
            review_pending=False,
        )
        assert resp2.web_fallback_fired is True
        assert resp2.low_confidence is True


# ---------------------------------------------------------------------------
# Phase 2.2 — AnalyzeRequest → IdeaAnalysisInput conversion
# ---------------------------------------------------------------------------


class TestAnalyzeRequestPhase22:
    """The HTTP ``AnalyzeRequest`` body is converted to the
    Temporal ``IdeaAnalysisInput`` with the three new fields
    forwarded verbatim."""

    def test_default_values_match(self) -> None:
        from src.api.analyze import AnalyzeRequest
        from src.workflow.shared import IdeaAnalysisInput

        req = AnalyzeRequest(idea="hello")
        wf_in = IdeaAnalysisInput(
            idea=req.idea,
            top_k=req.top_k,
            request_id=None,
            enable_web_fallback=req.enable_web_fallback,
            web_fallback_threshold=req.web_fallback_threshold,
            enable_low_confidence_review=req.enable_low_confidence_review,
        )
        assert wf_in.enable_web_fallback == req.enable_web_fallback
        assert wf_in.web_fallback_threshold == req.web_fallback_threshold
        assert wf_in.enable_low_confidence_review == req.enable_low_confidence_review

    def test_opt_out_propagates(self) -> None:
        from src.api.analyze import AnalyzeRequest
        from src.workflow.shared import IdeaAnalysisInput

        req = AnalyzeRequest(
            idea="hello", enable_web_fallback=False, enable_low_confidence_review=False
        )
        wf_in = IdeaAnalysisInput(
            idea=req.idea,
            top_k=req.top_k,
            request_id=None,
            enable_web_fallback=req.enable_web_fallback,
            web_fallback_threshold=req.web_fallback_threshold,
            enable_low_confidence_review=req.enable_low_confidence_review,
        )
        assert wf_in.enable_web_fallback is False
        assert wf_in.enable_low_confidence_review is False


# ---------------------------------------------------------------------------
# Phase 2.2 — endpoint body: workflow_signal_review_endpoint
# ---------------------------------------------------------------------------


class TestWorkflowSignalReviewEndpoint:
    """Unit tests for ``workflow_signal_review_endpoint`` — the
    route-body function called by the FastAPI handler."""

    def test_invalid_decision_returns_422(self) -> None:
        """A ``decision`` value outside the enum must 422 the request
        before we even contact Temporal."""
        import asyncio

        from fastapi import HTTPException

        from src.api.workflows import workflow_signal_review_endpoint
        from src.workflow.shared import ReviewSignal

        async def _run() -> None:
            sig = ReviewSignal(decision="not-a-valid-value")
            with pytest.raises(HTTPException) as exc_info:
                await workflow_signal_review_endpoint("wf-1", sig)
            assert exc_info.value.status_code == 422
            assert exc_info.value.detail["error"] == "invalid_decision"

        asyncio.run(_run())

    def test_signal_delivered_calls_handle_signal(self) -> None:
        """Happy path: signal is delivered, response echoes the decision."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.workflows import workflow_signal_review_endpoint
        from src.workflow.shared import ReviewSignal

        async def _run() -> None:
            sig = ReviewSignal(decision="confirm")
            fake_handle = MagicMock()
            fake_handle.signal = AsyncMock()
            fake_handle.query = AsyncMock(
                return_value={"review_pending": True, "phase": "waiting_for_review"}
            )
            fake_client = MagicMock()
            fake_client.get_workflow_handle = MagicMock(return_value=fake_handle)

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await workflow_signal_review_endpoint("wf-1", sig)

            assert resp.workflow_id == "wf-1"
            assert resp.decision == "confirm"
            assert resp.delivered is True
            # The Temporal SDK was called with the right signal name + payload.
            fake_handle.signal.assert_awaited_once()
            args, _ = fake_handle.signal.call_args
            assert args[0] == "review"
            assert isinstance(args[1], ReviewSignal)
            assert args[1].decision == "confirm"

        asyncio.run(_run())

    def test_unknown_workflow_returns_404(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import HTTPException

        from src.api.workflows import workflow_signal_review_endpoint
        from src.workflow.shared import ReviewSignal

        async def _run() -> None:
            sig = ReviewSignal(decision="confirm")
            fake_handle = MagicMock()
            fake_handle.signal = AsyncMock(
                side_effect=RuntimeError("workflow execution not found")
            )
            fake_handle.query = AsyncMock(
                return_value={"review_pending": False, "phase": "started"}
            )
            fake_client = MagicMock()
            fake_client.get_workflow_handle = MagicMock(return_value=fake_handle)

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await workflow_signal_review_endpoint("wf-unknown", sig)
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"] == "workflow_not_found"

        asyncio.run(_run())

    def test_closed_workflow_returns_409(self) -> None:
        """A workflow that has already completed/failed/closed
        cannot accept signals. Surface as 409 Conflict (not 503),
        so callers know the workflow is terminal — they shouldn't
        retry.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import HTTPException

        from src.api.workflows import workflow_signal_review_endpoint
        from src.workflow.shared import ReviewSignal

        async def _run() -> None:
            sig = ReviewSignal(decision="confirm")
            fake_handle = MagicMock()
            fake_handle.signal = AsyncMock(
                side_effect=RuntimeError("workflow execution already completed")
            )
            fake_handle.query = AsyncMock(
                return_value={"review_pending": False, "phase": "completed"}
            )
            fake_client = MagicMock()
            fake_client.get_workflow_handle = MagicMock(return_value=fake_handle)

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await workflow_signal_review_endpoint("wf-closed", sig)
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail["error"] == "workflow_closed"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Phase 2.2 — cosine helper unit tests
# ---------------------------------------------------------------------------


class TestCosineHelper:
    """The ``_cosine`` pure-Python helper inside ``activities.py``
    must work for unit-norm vectors (bge-m3 output) without
    pulling in numpy."""

    def test_unit_norm_parallel_vectors(self) -> None:
        from src.workflow.activities import _cosine

        # Same vector — cosine == 1.0
        v = [0.6] * 10 + [0.8] * 0  # placeholder; replaced below
        # Construct a real unit-norm vector
        v = [1.0 / (10 ** 0.5)] * 10
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        from src.workflow.activities import _cosine

        # Unit-norm vectors along different axes
        v1 = [1.0] + [0.0] * 9
        v2 = [0.0] + [1.0] + [0.0] * 8
        assert abs(_cosine(v1, v2)) < 1e-6

    def test_opposite_vectors(self) -> None:
        from src.workflow.activities import _cosine

        v1 = [1.0] + [0.0] * 9
        v2 = [-1.0] + [0.0] * 9
        assert abs(_cosine(v1, v2) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self) -> None:
        from src.workflow.activities import _cosine

        v_zero = [0.0] * 10
        v_norm = [1.0 / (10 ** 0.5)] * 10
        assert _cosine(v_zero, v_norm) == 0.0

    def test_mismatched_lengths_truncate(self) -> None:
        """Defensive truncation: a length mismatch uses the shorter
        prefix. Identical-direction vectors of different lengths
        should be cos-1.0 (within the truncation tolerance)."""
        from src.workflow.activities import _cosine

        v1 = [1.0, 0.0, 0.0, 0.0, 0.0]  # length 5
        v2 = [1.0, 0.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0]  # length 8, ignored tail
        assert abs(_cosine(v1, v2) - 1.0) < 1e-6

    def test_empty_vectors_return_zero(self) -> None:
        from src.workflow.activities import _cosine

        assert _cosine([], []) == 0.0
        assert _cosine([], [1.0]) == 0.0
        assert _cosine([1.0], []) == 0.0


# ---------------------------------------------------------------------------
# Phase 2.2 — WebFallbackClient smoke test (no real Firecrawl)
# ---------------------------------------------------------------------------


class TestWebFallbackClientOffline:
    """Smoke tests for the Firecrawl client wrapper that don't
    touch the network — confirm shape, error classes, and config
    defaults."""

    def test_module_exports(self) -> None:
        from src.workflow import web_fallback

        for name in (
            "WebFallbackClient",
            "WebFallbackDoc",
            "WebFallbackError",
            "WebFallbackTransportError",
            "FIRECRAWL_URL",
        ):
            assert hasattr(web_fallback, name)

    def test_default_base_url(self) -> None:
        from src.workflow.web_fallback import FIRECRAWL_URL

        assert FIRECRAWL_URL.startswith("http")

    def test_default_timeout_is_30s(self) -> None:
        from src.workflow.web_fallback import WEB_FALLBACK_TIMEOUT_SECONDS

        assert WEB_FALLBACK_TIMEOUT_SECONDS == 30.0

    def test_default_top_n_is_three(self) -> None:
        """PHASE-2.md §2.2: 'scrape the top-3 results'."""
        from src.workflow.web_fallback import WEB_FALLBACK_TOP_N

        assert WEB_FALLBACK_TOP_N == 3

    def test_empty_query_rejected(self) -> None:
        import pytest

        from src.workflow.web_fallback import WebFallbackClient

        c = WebFallbackClient()
        with pytest.raises(ValueError):
            c.search("")
        c.close()

    def test_zero_limit_rejected(self) -> None:
        import pytest

        from src.workflow.web_fallback import WebFallbackClient

        c = WebFallbackClient()
        with pytest.raises(ValueError):
            c.search("x", limit=0)
        c.close()

    def test_search_transport_error_on_non_2xx(self) -> None:
        """A 5xx from Firecrawl surfaces as WebFallbackTransportError."""
        from unittest.mock import MagicMock, patch

        from src.workflow.web_fallback import (
            WebFallbackClient,
            WebFallbackTransportError,
        )

        c = WebFallbackClient()

        # Build a fake ``Response`` whose ``.status_code >= 400``
        fake_response = MagicMock()
        fake_response.status_code = 502
        fake_response.text = "Bad Gateway"

        # The httpx.Client.post returns the fake response.
        with patch.object(
            c._client, "post", return_value=fake_response
        ):
            with pytest.raises(WebFallbackTransportError) as exc_info:
                c.search("query", limit=1)
            assert exc_info.value.details["status_code"] == 502

        c.close()

    def test_scrape_transport_error_on_non_2xx(self) -> None:
        from unittest.mock import MagicMock, patch

        from src.workflow.web_fallback import (
            WebFallbackClient,
            WebFallbackTransportError,
        )

        c = WebFallbackClient()

        fake_response = MagicMock()
        fake_response.status_code = 500
        fake_response.text = "internal error"

        with patch.object(c._client, "post", return_value=fake_response):
            with pytest.raises(WebFallbackTransportError):
                c.scrape("https://example.com")

        c.close()


# ---------------------------------------------------------------------------
# Phase 2.2 — Worker registers the new activity
# ---------------------------------------------------------------------------


class TestWorkerActivityRegistration:
    """The Temporal worker must register ``web_fallback_if_empty``
    alongside the existing five activities."""

    def test_worker_imports_web_fallback_activity(self) -> None:
        # The import block must reference the new activity name.
        import inspect

        from src.workflow import worker

        source = inspect.getsource(worker)
        assert "web_fallback_if_empty" in source

    def test_worker_registers_six_activities(self) -> None:
        """The worker's ``activities=[...]`` list must include all six."""
        import inspect

        from src.workflow import worker

        source = inspect.getsource(worker._run)
        # The activities list is the one passed to Worker().
        # Just confirm each name appears in the source.
        for name in (
            "embed_idea",
            "ann_search",
            "llm_compare_topk",
            "market_scope_signal",
            "assemble_verdict",
            "web_fallback_if_empty",
        ):
            assert name in source, (
                f"worker._run is missing activity registration: {name}"
            )