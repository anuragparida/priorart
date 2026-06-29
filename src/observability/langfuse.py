"""Langfuse observability — wraps every LLM call with a trace (Phase 2.3).

Why this module
---------------
The task card (``t_476a0098``) requires every LLM call to land in
Langfuse with the metadata the eval harness / Phase 3 thumbs-up UI
will later read. The wrapper has two layers:

1. **Client init + key resolution.** :func:`init_langfuse` reads
   the Langfuse public/secret keys from the env (with placeholder
   detection), constructs a real ``Langfuse`` client, and stashes
   it as a module-level singleton. If the keys are placeholders
   (or missing), it falls back to a no-op shim — the rest of the
   code can import ``get_client`` / ``trace_idea_compare`` without
   caring whether the real service is up.

2. **Trace helper.** :func:`trace_idea_compare` is the
   ``langfuse_context``-shaped context manager the task spec
   asks for. The card text uses ``langfuse_context`` — that's the
   v3 SDK name. We're on the v2 SDK (Langfuse on this host is
   v2.95, no ClickHouse), where the equivalent is ``client.trace``
   + ``client.generation`` blocks. :func:`trace_idea_compare`
   gives the same shape: ``input=``, ``output=``, ``metadata=``,
   ``name=``, plus a child ``generation`` for the actual LLM
   call with model + token usage.

The card also asks for an empty ``user_feedback`` score field —
:func:`add_user_feedback_placeholder` adds it as a metadata field
on the trace. Phase 3's thumbs up/down UI wires a real value.

Why a no-op fallback (and not a hard error)
-------------------------------------------
The card says: "Wrap every LLM call in Langfuse tracing." But
the LLM call path is on the critical request path of
``/ideas/analyze`` and on every Temporal activity. If Langfuse
is down or the keys are missing, we MUST NOT break the API —
tracing is observability, not correctness. The no-op shim makes
"missing Langfuse" a configuration state, not a runtime state.

Why we don't call ``client.flush()`` per request
------------------------------------------------
The Langfuse SDK batches + sends on a background thread. Calling
``flush()`` per request would add ~50–200 ms to every LLM call
(via /ideas/analyze) for marginal benefit. The eval harness and
the smoke test do call ``flush()`` after each batch — those are
the only paths where flush latency matters.

Per-request failures inside the trace
-------------------------------------
If the Langfuse SDK raises while we're writing to a trace (e.g.
the network is down), we catch + log + continue. The LLM call
itself has already returned; surfacing the trace-write failure
as a 500 would be the wrong priority order.

Module location
---------------
This module lives under ``src/observability/`` (new for Phase 2.3)
to match the AGENTS.md / SPEC.md layout ("Langfuse traces LLM
calls"). The ``src.observability`` package is one module for now;
Phase 3 will add scoring helpers / UI-tie-in wiring here too.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Placeholder detection — matches the clausecraft convention
# -------------------------------------------------------------------


#: Strings that mark a Langfuse key as "definitely not a real key".
#: We check both lowercased substrings and a list of explicit sentinels.
_PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "placeholder",
    "your-key",
    "pk-lf-placeholder",
    "***",  # harness redaction marker
)


def _looks_like_placeholder(value: str) -> bool:
    """Return True if a Langfuse key string looks like a placeholder.

    Heuristic: any of the placeholder substrings appears (case-
    insensitive), OR the key contains a triple-asterisk marker
    (which is what the harness redaction layer inserts when it
    detects a real secret in a tool call). We don't have to be
    perfect — the goal is to avoid authenticating with a known-
    bad key.
    """
    if not value:
        return True
    lowered = value.lower()
    if "***" in value:
        return True
    return any(token in lowered for token in _PLACEHOLDER_TOKENS)


# -------------------------------------------------------------------
# Module-level singleton + no-op shim
# -------------------------------------------------------------------


class _NoopSpan:
    """Drop-in for a Langfuse ``generation`` / ``span`` when tracing is off.

    Accepts every method the real SDK exposes (``update``,
    ``end``, ``score``, ``generation``, ``__enter__``/``__exit__``,
    etc.) and returns ``None`` or another ``_NoopSpan``. The
    caller's code never branches on "did this get a real Langfuse
    client?" — every method the real SDK has, this shim has too.
    """

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, **_kwargs: Any) -> None:
        return None

    def end(self, **_kwargs: Any) -> None:
        return None

    def score(self, **_kwargs: Any) -> None:
        return None

    def generation(self, *_args: Any, **_kwargs: Any) -> _NoopSpan:
        # The real trace object exposes ``trace.generation(...)``
        # which returns a ``generation`` observation. The shim
        # returns another shim so chained ``.update(...)`` /
        # ``.end()`` calls don't raise.
        return _NoopSpan()

    def span(self, *_args: Any, **_kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def event(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @property
    def id(self) -> str:
        return ""


class _NoopLangfuseClient:
    """Drop-in for ``langfuse.Langfuse`` when tracing is off.

    Every method returns either ``None`` or a no-op span. The
    public surface area mirrors what ``compare_topk`` and the
    smoke test actually call, so the call sites stay one-line.
    """

    def trace(self, *_args: Any, **_kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def generation(self, *_args: Any, **_kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def span(self, *_args: Any, **_kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def score(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def auth_check(self) -> bool:
        # The shim "passes" auth_check — but we also explicitly
        # check ``is_tracing_enabled()`` so callers that want to
        # branch on this get a consistent answer.
        return True


#: Real client once :func:`init_langfuse` succeeds; ``_NoopLangfuseClient``
#: otherwise. Module-level so the wrapper is process-singleton.
_client: Any = None
_tracing_enabled: bool = False
#: Last successful init metadata. Useful for the /healthz endpoint
#: and the smoke test's "is Langfuse wired?" check.
_last_init: dict[str, Any] = {}


def is_tracing_enabled() -> bool:
    """Return True if real Langfuse tracing is wired up.

    Public so the /ideas/analyze route and the Temporal worker
    can log a single line at startup that says whether tracing
    is on, without re-implementing the placeholder heuristic.
    """
    return _tracing_enabled


def get_client() -> Any:
    """Return the module-level Langfuse client (real or no-op).

    Safe to call before :func:`init_langfuse` — returns the no-op
    shim in that case. The card's recommendation: "call
    ``init_langfuse`` at process startup, then read the same
    client from everywhere else via ``get_client()``."
    """
    global _client
    if _client is None:
        # Lazy default: return the no-op shim. The real client is
        # only created on a successful ``init_langfuse`` call.
        _client = _NoopLangfuseClient()
    return _client


def init_langfuse(
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    force: bool = False,
) -> Any:
    """Initialize the Langfuse client (real or no-op).

    Resolution order for each value:
      1. The explicit argument (if non-None).
      2. The environment variable (``LANGFUSE_PUBLIC_KEY`` /
         ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``).
      3. The reasonable default (``http://localhost:13000`` for
         ``LANGFUSE_HOST`` on this host — that's where the
         self-hosted Langfuse v2 container is already running per
         AGENTS.md).

    If either key looks like a placeholder, the function logs +
    installs the no-op shim. ``force=True`` re-initialises even
    if a previous attempt succeeded (useful for the test suite).

    Returns
    -------
    The active client (real ``langfuse.Langfuse`` or no-op shim).
    """
    global _client, _tracing_enabled, _last_init

    if _client is not None and not force:
        return _client

    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY") or ""
    host = host or os.environ.get("LANGFUSE_HOST") or "http://localhost:13000"

    if _looks_like_placeholder(public_key) or _looks_like_placeholder(secret_key):
        logger.info(
            "Langfuse keys look like placeholders; running in no-op tracing "
            "mode (host=%s, public_key_set=%s, secret_key_set=%s). "
            "Set real LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env "
            "to enable tracing.",
            host,
            bool(public_key),
            bool(secret_key),
        )
        _client = _NoopLangfuseClient()
        _tracing_enabled = False
        _last_init = {"mode": "noop", "host": host, "reason": "placeholder_keys"}
        return _client

    try:
        # Import inside the function so the dep is lazy: tests
        # that don't enable tracing never load the SDK.
        from langfuse import Langfuse  # type: ignore[import-not-found]

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        _client = client
        _tracing_enabled = True
        _last_init = {
            "mode": "real",
            "host": host,
            "public_key_prefix": public_key[:10] + "..." if public_key else "",
        }
        logger.info("Langfuse client initialized (host=%s)", host)
        return client
    except Exception as exc:  # noqa: BLE001 — see module docstring for why
        logger.exception(
            "Failed to initialize Langfuse; falling back to no-op. "
            "Exception: %s",
            exc,
        )
        _client = _NoopLangfuseClient()
        _tracing_enabled = False
        _last_init = {"mode": "noop", "host": host, "reason": "init_exception"}
        return _client


def reset_for_tests() -> None:
    """Test hook — drop the cached client so a new init takes effect.

    Production code paths never call this. The test suite uses it
    to simulate "Langfuse keys became valid mid-run" or "Langfuse
    is unreachable" without re-importing the module.
    """
    global _client, _tracing_enabled, _last_init
    _client = None
    _tracing_enabled = False
    _last_init = {}


# -------------------------------------------------------------------
# Trace context — the ``langfuse_context``-shaped wrapper
# -------------------------------------------------------------------


@contextmanager
def trace_idea_compare(
    *,
    name: str,
    input_payload: Any,
    output_payload: Any = None,
    metadata: dict[str, Any] | None = None,
    flush_on_exit: bool = False,
) -> Iterator[Any]:
    """Wrap the structured-comparison LLM call as a Langfuse trace.

    Mirrors the card's required shape:
      ``name="idea-compare"``, ``input=idea + top-3``,
      ``output=CompetitorVerdict list``, metadata fields per the
      card body. Yields the active ``generation`` (LLM
    observation) so the call site can attach model + token
    usage once the LLM call returns.

    Why a ``generation`` (not just a ``trace``) wrapper
    --------------------------------------------------
    A ``trace`` is the umbrella for the whole request; a
    ``generation`` is the LLM-specific child observation. The
    Langfuse UI shows model, token usage, latency per
    ``generation``. We wrap the actual instructor-wrapped
    Anthropic call in a ``generation`` and the surrounding
    request in a ``trace`` — that nesting is the v2 SDK's
    equivalent of the v3 SDK's ``langfuse_context`` block.

    Parameters
    ----------
    name:
        The trace name. The card says ``"idea-compare"``; we
        default to that but accept overrides for tests.
    input_payload:
        The wire input to the LLM call — typically the
        ``{idea: ..., top_k_companies: [...]}`` dict.
    output_payload:
        The wire output from the LLM call — typically the
        ``IdeaVerdict.model_dump()`` dict. Optional because
        the wrapper is entered BEFORE the LLM call returns;
        we update with output on exit.
    metadata:
        Dict of metadata fields to attach to the trace. The
        card lists seven keys; the wrapper accepts whatever
        the caller provides and forwards them.
    flush_on_exit:
        If True, call ``client.flush()`` after the trace is
        closed. Useful for tests and the smoke script; the
        /ideas/analyze request path leaves this False (the
        SDK batches and flushes on a worker thread, and
        forcing a flush per request adds 50–200 ms).

    Yields
    ------
    The active ``generation`` object. The call site uses this
    to attach ``model`` / ``usage`` / ``latency`` once the
    LLM call completes; if the call raises, the wrapper still
    closes the generation cleanly with ``end_time``.
    """
    client = get_client()
    safe_metadata = dict(metadata or {})

    trace = client.trace(name=name, input=input_payload, metadata=safe_metadata)

    try:
        # The generation is the LLM-specific child. We create it
        # up-front so the model name lands on it even if the
        # call raises — the Langfuse UI will still show the
        # model + the latency it took to fail.
        generation = trace.generation(name="llm-call")
        yield generation
    except Exception as exc:
        # Langfuse expects to know why a generation didn't
        # complete. We tag it with the error type + message and
        # close it. The trace update with the error lands below
        # in the ``finally`` block.
        try:
            trace.update(
                output={"error": type(exc).__name__, "message": str(exc)},
                metadata={**safe_metadata, "error_type": type(exc).__name__},
            )
        except Exception:  # noqa: BLE001
            logger.exception("langfuse: failed to update trace on error")
        raise
    finally:
        if output_payload is not None:
            try:
                trace.update(output=output_payload)
            except Exception:  # noqa: BLE001
                logger.exception("langfuse: failed to update trace output")
        try:
            generation.end()
        except Exception:  # noqa: BLE001
            logger.exception("langfuse: failed to end generation")

    if flush_on_exit:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            logger.exception("langfuse: flush on exit failed")


# -------------------------------------------------------------------
# User-feedback score placeholder
# -------------------------------------------------------------------


def add_user_feedback_placeholder(
    trace_or_id: Any,
    *,
    comment: str = "Phase 3 wires the UI thumbs up/down.",
) -> None:
    """Attach the empty ``user_feedback`` score field the card asks for.

    Langfuse v2 SDK requires a numeric or categorical value for
    scores — it rejects ``value=None``. We instead add the
    ``user_feedback`` placeholder as a metadata field on the
    trace so it shows up in the UI alongside the other metadata
    keys. Phase 3 replaces it with a real score when the UI
    sends the user's vote.

    The function is a no-op when tracing is disabled (the
    placeholder simply doesn't exist anywhere — that's fine,
    the no-op shim absorbs the call).

    Parameters
    ----------
    trace_or_id:
        Either a real Langfuse trace object (with ``.id`` and
        ``.update`` methods) or a trace-id string. The wrapper
        accepts both because some v2 SDK paths return the
        trace object and others just return the id.
    comment:
        The annotation string. Defaults to the Phase 3 hint so
        the comment is informative if a reader inspects the
        trace in the Langfuse UI.
    """
    client = get_client()
    if not _tracing_enabled:
        return

    # Resolve the trace id + a callable for ``update``.
    trace_id: str | None = None
    update_fn = None
    if hasattr(trace_or_id, "id") and hasattr(trace_or_id, "update"):
        trace_id = getattr(trace_or_id, "id", None)
        update_fn = trace_or_id.update
    elif isinstance(trace_or_id, str):
        trace_id = trace_or_id

    if update_fn is None or trace_id is None:
        # We don't have a live trace object; emit a stub score
        # via the client. v2 SDK requires a numeric value, so we
        # write 0.0 (neutral) and attach the comment so a reader
        # sees "this isn't a real vote".
        try:
            client.score(
                trace_id=trace_id or "unknown",
                name="user_feedback",
                value=0.0,
                comment=comment + " (value=0.0 is a placeholder, not a vote)",
            )
        except Exception:  # noqa: BLE001
            logger.exception("langfuse: failed to write user_feedback score stub")
        return

    # Preferred path: piggyback on the trace's metadata so the
    # field shows up in the UI alongside the other metadata
    # (Langfuse v2 has no separate "scoring schema" surfacing
    # for empty placeholders).
    try:
        update_fn(metadata={"user_feedback": {"value": None, "comment": comment}})
    except Exception:  # noqa: BLE001
        logger.exception("langfuse: failed to attach user_feedback metadata")


__all__ = [
    "add_user_feedback_placeholder",
    "get_client",
    "init_langfuse",
    "is_tracing_enabled",
    "reset_for_tests",
    "trace_idea_compare",
]