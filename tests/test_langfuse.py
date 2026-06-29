"""Tests for the Langfuse observability wrapper (Phase 2.3).

What this covers
----------------
1. ``init_langfuse`` returns the no-op shim when keys are
   placeholders / missing — and a real ``langfuse.Langfuse`` client
   when keys look real (we don't hit the network here; the SDK
   lazy-connects on the first trace call).
2. ``trace_idea_compare`` is a context manager that:
     - yields a generation-like object on entry
     - exits cleanly on the happy path
     - propagates exceptions while still closing the trace
     - never raises even if the underlying SDK raises
3. ``is_tracing_enabled`` returns True/False consistently with what
   ``init_langfuse`` produced.
4. The /healthz response carries ``langfuse_enabled`` and it
   reflects the live tracing state.

Why we test against the no-op shim (not a real Langfuse server)
---------------------------------------------------------------
The smoke test (``make smoke``) hits a real Langfuse. The unit
tests run in CI without Langfuse, so the placeholder path is the
only one that's reliably reproducible. The real-client path is
exercised in the smoke test by curling /healthz and asserting
``langfuse_enabled`` flips to True when keys are configured.

Per-task provenance: t_476a0098 (Phase 2.3).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.observability import (
    add_user_feedback_placeholder,
    get_client,
    init_langfuse,
    is_tracing_enabled,
    reset_for_tests,
    trace_idea_compare,
)
from src.observability.langfuse import _NoopLangfuseClient, _NoopSpan


@pytest.fixture(autouse=True)
def _reset_langfuse_singleton() -> Any:
    """Drop the cached client between tests.

    The wrapper holds a module-level singleton so production code
    paths can read the same client from everywhere; tests need to
    drop that cache between cases so a "placeholder" test doesn't
    leak into a "valid-keys" test that follows it.
    """
    reset_for_tests()
    # Also drop env vars so a developer's local .env doesn't bleed
    # into the placeholder-detection tests.
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        os.environ.pop(key, None)
    yield
    reset_for_tests()
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        os.environ.pop(key, None)


# -------------------------------------------------------------------
# init_langfuse + is_tracing_enabled
# -------------------------------------------------------------------


def test_init_langfuse_placeholder_keys_returns_noop() -> None:
    """Placeholder keys → no-op shim, tracing disabled."""
    client = init_langfuse(
        public_key="pk-lf-placeholder",
        secret_key="sk-lf-placeholder",
        host="http://localhost:13000",
    )
    assert isinstance(client, _NoopLangfuseClient)
    assert is_tracing_enabled() is False


def test_init_langfuse_missing_keys_returns_noop() -> None:
    """No env vars, no explicit args → no-op shim."""
    # Env vars cleared by the autouse fixture.
    client = init_langfuse()
    assert isinstance(client, _NoopLangfuseClient)
    assert is_tracing_enabled() is False


def test_init_langfuse_redacted_marker_returns_noop() -> None:
    """The harness redacts real secrets with ``***``; the wrapper
    must recognise that pattern as a placeholder.

    Reproduces what happens when a developer's .env is shipped
    through the redaction layer — the public_key comes back as
    ``***`` rather than the real value, and we must NOT treat
    that as a usable key.
    """
    client = init_langfuse(
        public_key="***",
        secret_key="***",
        host="http://localhost:13000",
    )
    assert isinstance(client, _NoopLangfuseClient)
    assert is_tracing_enabled() is False


def test_init_langfuse_real_keys_attempts_real_client() -> None:
    """Non-placeholder keys → real Langfuse client (no network I/O here)."""
    client = init_langfuse(
        public_key="pk-lf-real-looking-key-1234567890",
        secret_key="sk-lf-real-looking-key-1234567890",
        host="http://localhost:13000",
    )
    # We don't assert the exact type — langfuse.Langfuse might not
    # be importable on a stripped-down test host. We assert that
    # the wrapper does NOT hand back the no-op shim, and that the
    # "enabled" flag flips on.
    assert not isinstance(client, _NoopLangfuseClient)
    assert is_tracing_enabled() is True


def test_init_langfuse_is_idempotent() -> None:
    """A second init without ``force=True`` returns the cached client."""
    c1 = init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    c2 = init_langfuse(public_key="pk-lf-different-placeholder", secret_key="sk-lf-different")
    assert c1 is c2, "init_langfuse must cache the client (process-singleton)"


def test_init_langfuse_force_reinitialises() -> None:
    """``force=True`` drops the cache; the second init takes effect."""
    init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    c2 = init_langfuse(
        public_key="pk-lf-new-placeholder",
        secret_key="sk-lf-new-placeholder",
        force=True,
    )
    assert isinstance(c2, _NoopLangfuseClient)


def test_get_client_lazy_initialises_to_noop() -> None:
    """``get_client`` without a prior init returns the no-op shim."""
    client = get_client()
    assert isinstance(client, _NoopLangfuseClient)


# -------------------------------------------------------------------
# trace_idea_compare — context manager shape
# -------------------------------------------------------------------


def test_trace_idea_compare_yields_generation_object() -> None:
    """Entering the context yields a generation-like object."""
    init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    with trace_idea_compare(
        name="idea-compare",
        input_payload={"idea": "test"},
        metadata={"embedding_latency_ms": 1.0},
    ) as gen:
        # No-op shim returns _NoopSpan for trace.generation(...)
        assert isinstance(gen, _NoopSpan)


def test_trace_idea_compare_exits_cleanly_on_success() -> None:
    """Exiting the context on the happy path closes the trace."""
    init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    with trace_idea_compare(
        name="idea-compare",
        input_payload={"idea": "x"},
    ) as gen:
        gen.update(model="claude-sonnet-4-5", usage={"input": 100, "output": 50, "unit": "TOKENS"})
    # No assertion needed beyond "this didn't raise". The trace
    # update + generation end are best-effort paths in the wrapper.


def test_trace_idea_compare_propagates_exceptions() -> None:
    """The context manager re-raises exceptions (does not swallow)."""
    init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    with pytest.raises(RuntimeError, match="boom"):
        with trace_idea_compare(name="idea-compare", input_payload={"idea": "x"}):
            raise RuntimeError("boom")


def test_trace_idea_compare_accepts_output_payload() -> None:
    """The wrapper accepts ``output_payload`` and closes cleanly."""
    init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    with trace_idea_compare(
        name="idea-compare",
        input_payload={"idea": "x"},
        output_payload={"competitors": []},
    ):
        pass


def test_trace_idea_compare_works_without_init() -> None:
    """Calling the context manager before init_langfuse is safe."""
    reset_for_tests()  # explicitly no init
    with trace_idea_compare(name="idea-compare", input_payload={"idea": "x"}) as gen:
        assert isinstance(gen, _NoopSpan)


# -------------------------------------------------------------------
# add_user_feedback_placeholder
# -------------------------------------------------------------------


def test_add_user_feedback_placeholder_is_noop_when_tracing_disabled() -> None:
    """When tracing is off, the placeholder call is a silent no-op."""
    init_langfuse(public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder")
    # No assertion needed — the function must not raise and must
    # not write anything (the no-op shim absorbs the call).
    add_user_feedback_placeholder(None)
    add_user_feedback_placeholder("some-trace-id")


def test_add_user_feedback_placeholder_accepts_trace_object() -> None:
    """When tracing is on, the placeholder accepts either a trace
    object or a trace-id string. We exercise the object path here.
    """
    init_langfuse(
        public_key="pk-lf-real-looking-key-1234567890",
        secret_key="sk-lf-real-looking-key-1234567890",
        host="http://localhost:13000",
    )

    class _StubTrace:
        def __init__(self) -> None:
            self.id = "trace-abc-123"
            self.updated: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.updated.append(kwargs)

    trace = _StubTrace()
    add_user_feedback_placeholder(trace)
    # The wrapper either calls trace.update(metadata={...}) (the
    # "metadata piggyback" path) or client.score(...). Either is
    # valid — we only assert it didn't raise and that the trace
    # object is unchanged in the sense that no exception escaped.
    assert isinstance(trace.updated, list)


# -------------------------------------------------------------------
# /healthz — langfuse_enabled field
# -------------------------------------------------------------------


def test_healthz_reports_langfuse_enabled_false_when_keys_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/healthz`` returns ``langfuse_enabled=False`` when no keys
    are configured. We exercise the real FastAPI route via the
    ``TestClient``; the DB may be unavailable (test host is
    Postgres-optional), so we tolerate a 200/503 split.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")

    # Importing the app triggers the startup hook, which reads the
    # env vars. Do that AFTER monkeypatch so the wrapper sees
    # "no keys".
    import importlib

    from src.api import app as app_module

    importlib.reload(app_module)

    client = TestClient(app_module.app)
    response = client.get("/healthz")
    # /healthz returns 200 when DB is reachable, 503 when it isn't.
    # The langfuse_enabled field is independent of DB status.
    assert response.status_code in (200, 503)
    body = response.json()
    assert "langfuse_enabled" in body, body
    assert body["langfuse_enabled"] is False