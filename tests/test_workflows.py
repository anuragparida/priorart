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
            result_payload=result_payload,
            describe_side_effect=describe_side_effect,
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

            failure = MagicMock()
            failure.type = "SchemaViolationError"
            failure.message = "LLM response failed IdeaVerdict validation"
            fake_client = self._make_client(
                status=WorkflowExecutionStatus.FAILED,
                close_time=datetime(2026, 6, 29, 12, 0, 30),
                failure=failure,
            )

            with patch(
                "src.api.workflows.get_temporal_client",
                AsyncMock(return_value=fake_client),
            ):
                resp = await workflow_status_endpoint("wf-1")
            assert resp.status == "FAILED"
            assert resp.failure is not None
            assert resp.failure["type"] == "SchemaViolationError"
            assert "IdeaVerdict" in resp.failure["message"]

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
# Activities module — module-level invariants
# ---------------------------------------------------------------------------


class TestActivitiesModule:
    """The activities module exposes the five activities the spec asks for."""

    def test_all_five_activities_exported(self) -> None:
        from src.workflow import activities

        for name in (
            "embed_idea",
            "ann_search",
            "llm_compare_topk",
            "market_scope_signal",
            "assemble_verdict",
        ):
            assert hasattr(activities, name), (
                f"src.workflow.activities.{name} is missing"
            )
            fn = getattr(activities, name)
            assert callable(fn)

    def test_activity_names_match_spec(self) -> None:
        """The Temporal activity names match the PHASE-2.md §2.1 spec."""
        from src.workflow import activities

        for attr_name in (
            "embed_idea",
            "ann_search",
            "llm_compare_topk",
            "market_scope_signal",
            "assemble_verdict",
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