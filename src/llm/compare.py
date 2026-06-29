"""Structured-comparison LLM call (Phase 1.7).

This module wraps a single Claude Sonnet 4.5 call in an
``instructor``-validated Pydantic schema, returning an
``IdeaVerdict``. The contract is:

    compare_topk(idea, top_k) -> IdeaVerdict
    compare_topk(idea, top_k) -> {"error": "schema_violation", "details": ...}

The second shape is for the rare case where the LLM returns
something instructor cannot coerce into ``IdeaVerdict`` even after
``max_retries`` retries. The caller (Phase 1.8's ``/ideas/analyze``)
maps that to a 200 with a structured error, *not* a 500.

Why instructor (and not raw anthropic + manual JSON parsing)
------------------------------------------------------------
- The Pydantic model is the *single source of truth* for the
  contract. Instructor emits the JSON schema from the model, sends
  it to Claude as a tool input, and re-validates the response
  against the model. If validation fails, instructor retries
  (up to ``max_retries``) with the validation error in the
  conversation, so the model can self-correct.
- This is the exact pattern SPEC.md / AGENTS.md / PHASE-1.md
  recommend: "Pydantic-validated LLM structured output".

Why Claude Sonnet 4.5
---------------------
Per SPEC.md: "Anthropic Claude Sonnet 4.5 for the structured-
comparison call (good at long, nuanced comparisons)." We pin the
default via :data:`src.config.ANTHROPIC_MODEL`; the env var
``PRIORART_ANTHROPIC_MODEL`` overrides for tests.

Cost control: one call per request
----------------------------------
``compare_topk`` does exactly one LLM call. The top-K company
descriptions are all included in the prompt, and the response
shape (a list of ``CompetitorVerdict``) is generated in one shot.
This is the explicit requirement in PHASE-1.md §1.7: "ONE LLM call
per request, not per competitor." For K=3 we trade one larger call
for three smaller ones; in practice the larger call is both
faster (one round-trip) and produces more consistent verdicts
(the model sees all three competitors at once, so the
"differences" are calibrated against the same baseline).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import instructor
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError, AuthenticationError
from instructor.core import Instructor
from pydantic import ValidationError

from src.config import ANTHROPIC_MODEL
from src.llm.prompts.compare import PROMPT_TEMPLATE_VERSION, build_user_prompt
from src.observability.langfuse import (
    add_user_feedback_placeholder,
    trace_idea_compare,
)
from src.llm.schemas import (
    CompetitorVerdict,
    DEFAULT_TOP_K,
    IdeaVerdict,
    MarketScope,
    MAX_TOP_K,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: Hard cap on the number of competitors in a single LLM call. Even
#: if the caller passes a larger ``top_k``, we trim to this before
#: sending the prompt. The PHASE-1.md requirement is "top-3 default,
#: configurable"; we ship 3.
DEFAULT_MAX_COMPANIES = DEFAULT_TOP_K  # 3

#: How many times instructor should retry when the LLM returns
#: something that fails Pydantic validation. Three is a good
#: default — the first try is the long-shot, the retry usually
#: succeeds because the model has the validation error in the
#: conversation and self-corrects.
DEFAULT_MAX_RETRIES = 3

#: Anthropic client timeout, in seconds. The structured-comparison
#: call is one round-trip with a few-thousand-token prompt and a
#: ~500-token response; 60 s is generous. Bumping it past 120 s
#: almost always means the model is misconfigured, not slow.
DEFAULT_TIMEOUT_SECONDS = 60.0


#: Token-cost table — USD per million tokens. Source: Anthropic's
#: published pricing for Claude Sonnet 4.5 (Oct 2025 snapshot).
#: Multiplied by 1e-6 to convert to per-token cost. Used by the
#: Langfuse trace metadata to compute ``token_cost_usd``. Update
#: this when Anthropic revs their pricing.
#:
#: We use a single ``PRICE_PER_TOKEN_USD`` mapping rather than
#: per-model rates so the test suite can mock by patching the
#: module-level constant. If we ever add a 2nd model to the
#: structured-comparison path, this dict grows by one entry.
PRICE_PER_TOKEN_USD: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {
        "input": 3.0 / 1_000_000,   # $3 / MTok
        "output": 15.0 / 1_000_000,  # $15 / MTok
    },
}


def _compute_token_cost_usd(
    *,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """Compute USD cost from token usage. Returns None when no usage data.

    Some test mocks don't include usage. The trace metadata
    accepts a None — Langfuse just shows ``"token_cost_usd":
    null`` and an alert fires on the dashboard.
    """
    if input_tokens is None and output_tokens is None:
        return None
    rates = PRICE_PER_TOKEN_USD.get(model)
    if rates is None:
        return None
    cost = 0.0
    if input_tokens is not None:
        cost += input_tokens * rates["input"]
    if output_tokens is not None:
        cost += output_tokens * rates["output"]
    return round(cost, 6)


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull input/output token counts out of an instructor-wrapped response.

    Instructor (anthropic-tools mode) returns the validated
    Pydantic model directly, dropping the Anthropic Message on
    the floor. When called outside instructor (Phase 2.3 smoke
    tests sometimes do this), the response may BE a Message
    instead. We try both shapes; if we can't find a usage
    object we return an empty dict (Langfuse handles missing
    usage gracefully — token_cost_usd comes out None).
    """
    # Shape 1: response has ``_raw_response`` (instructor-attached).
    raw = getattr(response, "_raw_response", None)
    if raw is not None:
        usage = getattr(raw, "usage", None)
        if usage is not None:
            return {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
    # Shape 2: response is the Anthropic Message directly.
    usage = getattr(response, "usage", None)
    if usage is not None:
        return {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
    return {}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CompareTopKError(RuntimeError):
    """Base error for the compare_topk call. Subclasses cover specific
    failure modes so the /ideas/analyze route can return a useful
    200-with-structured-error rather than a 500.

    The class hierarchy mirrors the error shape we want the API to
    expose — see the docstring of :func:`compare_topk` for the
    JSON shape.
    """


class SchemaViolationError(CompareTopKError):
    """The LLM returned something instructor could not coerce into
    an ``IdeaVerdict`` even after ``max_retries`` retries.

    Carries the underlying ``pydantic.ValidationError`` (or
    instructor's wrapped equivalent) on ``self.details`` so the
    caller can surface it to the user.
    """

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


class MissingAPIKeyError(CompareTopKError):
    """``ANTHROPIC_API_KEY`` is not set (env or ``~/.anthropic_key``).

    We do not fall back to a mock or a "no LLM" stub — the
    /ideas/analyze endpoint is useless without a real LLM call,
    so the error is a hard 503 at the API layer and a hard
    exception at the library layer.
    """


class LLMTransportError(CompareTopKError):
    """The Anthropic SDK raised a non-validation error: timeout,
    auth failure, network, etc. Carries the original exception
    on ``self.details``.
    """


# ---------------------------------------------------------------------------
# API-key resolution
# ---------------------------------------------------------------------------


def _read_api_key() -> str:
    """Return the Anthropic API key, or raise ``MissingAPIKeyError``.

    Resolution order:

    1. ``$ANTHROPIC_API_KEY`` (env var, set by CI / runtime).
    2. ``~/.anthropic_key`` (file containing just the key, one
       line). This is the convention the AGENTS.md / task spec
       document for Anurag's setup.
    3. The anthropic SDK's own discovery — but we don't rely on
       that, because the SDK's error message is opaque. We
       surface a clear "set the env var or drop the file" instead.
    """
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key.strip()

    path = Path.home() / ".anthropic_key"
    if path.is_file():
        # The file convention is one-line, no trailing newline.
        # Strip defensively — a stray \n would break the auth header.
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text

    raise MissingAPIKeyError(
        "Anthropic API key not found. Set $ANTHROPIC_API_KEY or write the "
        "key to ~/.anthropic_key (one line, no newline)."
    )


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareClient:
    """A pre-configured instructor client + model name.

    Holding the client on a small dataclass (instead of as module
    globals) makes the dependency explicit and test-friendly:
    tests can pass a ``CompareClient(instructor_client=mock, ...)``
    directly to :func:`compare_topk` without monkey-patching
    module state.

    Use :func:`build_default_client` to construct the production
    one (env / file API key, default model, default timeout).
    """

    instructor_client: Instructor
    model: str
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def build_default_client() -> CompareClient:
    """Construct the production :class:`CompareClient`.

    Reads the API key from env / ~/.anthropic_key, builds an
    instructor-wrapped Anthropic client in
    ``Mode.ANTHROPIC_TOOLS`` (the cleanest mode for structured
    outputs — Claude's tool-use channel is the most reliable
    way to get deterministic JSON back).

    Raises
    ------
    MissingAPIKeyError
        If no key is configured. The caller should let this
        propagate (it's already a clear message).
    """
    api_key = _read_api_key()
    # The provider string tells instructor to use the Anthropic
    # SDK under the hood. ``Mode.ANTHROPIC_TOOLS`` is the
    # deterministic-JSON mode — Claude emits the response as a
    # tool call whose input matches our Pydantic schema.
    client = instructor.from_provider(
        "anthropic/" + ANTHROPIC_MODEL,
        api_key=api_key,
        mode=instructor.Mode.ANTHROPIC_TOOLS,
    )
    return CompareClient(
        instructor_client=client,
        model=ANTHROPIC_MODEL,
    )


# ---------------------------------------------------------------------------
# The public function
# ---------------------------------------------------------------------------


def _coerce_top_k(
    top_k: Sequence[dict],
    *,
    max_companies: int,
) -> list[dict]:
    """Defensive normalization of the top-K list.

    The caller (``/ideas/analyze``) is expected to pass dicts with
    at least ``company_id``, ``name``, ``description``, and
    ``similarity``. We trim, drop entries with a missing
    ``company_id``, and trust the rest of the shape — the prompt
    builder (``build_user_prompt``) does final string coercion.
    """
    if max_companies <= 0:
        raise ValueError(f"max_companies must be > 0, got {max_companies}")
    if max_companies > MAX_TOP_K:
        # The prompt template only knows how to render up to
        # MAX_PROMPT_COMPANIES companies (defined in
        # src/llm/prompts/compare.py). The default and the
        # caller's cap might disagree; the more restrictive wins.
        max_companies = MAX_PROMPT_COMPANIES

    cleaned: list[dict] = []
    for entry in top_k:
        if "company_id" not in entry:
            # Drop the row, log, and keep going. /search
            # guarantees the field is present, but a hand-rolled
            # test fixture might not.
            logger.warning(
                "compare_topk: dropping top-K entry without company_id: %r",
                entry,
            )
            continue
        cleaned.append(entry)
        if len(cleaned) >= max_companies:
            break
    return cleaned


def compare_topk(
    idea: str,
    top_k: Sequence[dict],
    *,
    client: Optional[CompareClient] = None,
    max_companies: int = DEFAULT_MAX_COMPANIES,
    embedding_latency_ms: Optional[float] = None,
    ann_search_latency_ms: Optional[float] = None,
    top_k_ids: Optional[Sequence[int]] = None,
) -> IdeaVerdict:
    """Run one structured-comparison LLM call.

    Parameters
    ----------
    idea:
        The user's free-text idea. Echoed back into the
        ``IdeaVerdict.idea`` field.
    top_k:
        A sequence of dicts (one per top company), each with at
        least ``company_id``, ``name``, ``description``,
        ``similarity`` (float in [-1, 1]). Order is preserved;
        the first entry is the nearest match. The function
        trims to ``max_companies`` entries before sending.
    client:
        Optional pre-built :class:`CompareClient`. If ``None``,
        a default client is built from env / file API key. Tests
        pass a mock client here to avoid network I/O.
    max_companies:
        Hard cap on the number of competitors in the response.
        Default :data:`DEFAULT_MAX_COMPANIES` (3).
    embedding_latency_ms:
        Wall-clock time the embedding call took before the
        LLM call. Forwarded into the Langfuse trace metadata
        so the eval harness / dashboards can see how much of
        the user-visible latency is pre-LLM overhead. Optional —
        callers that don't time it (the eval harness, the
        Temporal worker) pass None.
    ann_search_latency_ms:
        Wall-clock time the pgvector ANN query took. Same as
        ``embedding_latency_ms`` — optional, traced as
        metadata.
    top_k_ids:
        Ordered list of ``company_id``s that fed the prompt.
        Captured into the trace metadata so the Langfuse UI
        can link a trace back to the corpus rows it surfaced.
        Defaults to the ids inferred from ``top_k`` if not
        supplied.

    Returns
    -------
    IdeaVerdict
        A Pydantic-validated structured comparison.

    Raises
    ------
    MissingAPIKeyError
        If no API key is configured.
    SchemaViolationError
        If instructor cannot coerce the LLM response into
        ``IdeaVerdict`` after ``max_retries`` retries. Carries
        the underlying validation error in ``exc.details``.
    LLMTransportError
        If the Anthropic SDK raises a non-validation error
        (timeout, auth, network).

    Notes
    -----
    The function is *synchronous*. Async support is not in
    Phase 1 scope — the /ideas/analyze endpoint is a 1-RPS
    user-facing route, and adding async adds three failure
    modes (event loop, gather, cancellation) for no throughput
    win.

    Phase 2.3: every call is wrapped in a Langfuse trace named
    ``"idea-compare"`` with metadata fields per the task card.
    The wrapper is no-op when Langfuse isn't configured.
    """
    if not idea or not idea.strip():
        raise ValueError("idea must be a non-empty string")
    if not top_k:
        raise ValueError("top_k must be a non-empty sequence")

    cleaned = _coerce_top_k(top_k, max_companies=max_companies)
    if not cleaned:
        # Defensive — _coerce_topk drops entries without
        # company_id. If the caller passed a list of empty
        # dicts, we'd hit this. Surface as a schema violation
        # because the verdict would be empty either way.
        raise ValueError(
            "top_k has no entries with a 'company_id' field; "
            "cannot produce a verdict"
        )

    if client is None:
        client = build_default_client()

    user_prompt = build_user_prompt(idea, cleaned)
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    # Phase 2.3: build the Langfuse trace metadata BEFORE we
    # enter the trace context so the wrapper can attach it
    # cleanly. Required fields per the card are the seven
    # below; ``token_cost_usd`` is filled in once the LLM call
    # returns with usage data.
    resolved_top_k_ids: list[int] = list(top_k_ids) if top_k_ids else [
        int(entry["company_id"]) for entry in cleaned
    ]
    trace_metadata: dict[str, Any] = {
        "embedding_latency_ms": embedding_latency_ms,
        "ann_search_latency_ms": ann_search_latency_ms,
        "top_k_ids": resolved_top_k_ids,
        "model_version": client.model,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        # token_cost_usd is updated post-call below; default
        # to None so the schema is stable.
        "token_cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": None,
    }
    trace_input: dict[str, Any] = {
        "idea": idea,
        "top_k_companies": [
            {
                "company_id": int(entry["company_id"]),
                "name": str(entry.get("name", "")),
                "similarity": float(entry.get("similarity", 0.0)),
            }
            for entry in cleaned
        ],
    }

    started_at = time.perf_counter()
    try:
        with trace_idea_compare(
            name="idea-compare",
            input_payload=trace_input,
            metadata=trace_metadata,
        ) as generation:
            try:
                verdict: IdeaVerdict = client.instructor_client.create(
                    response_model=IdeaVerdict,
                    messages=messages,
                    max_retries=client.max_retries,
                    # Pass the model explicitly — instructor's
                    # ``from_provider`` default may not include the
                    # right model name for the underlying Anthropic
                    # call, and the SDK requires a model argument.
                    model=client.model,
                    # Bounded timeout — we don't want a hung LLM call
                    # to wedge the FastAPI worker.
                    timeout=client.timeout_seconds,
                )
            except ValidationError as exc:
                # Instructor raised because even after retries the
                # response didn't match the schema. Surface the details
                # so the API layer can return a useful error.
                raise SchemaViolationError(
                    "LLM response failed IdeaVerdict validation after "
                    f"{client.max_retries} retries",
                    details=_format_validation_error(exc),
                ) from exc
            except (AuthenticationError, APITimeoutError, AnthropicAPIError) as exc:
                # Transport-level failures. We intentionally do not
                # catch the broader ``Exception`` — a bug in our own
                # code (e.g. a bad prompt template) should propagate.
                raise LLMTransportError(
                    f"Anthropic call failed: {type(exc).__name__}: {exc}"
                ) from exc

            # Pull usage + compute cost for the Langfuse trace
            # metadata. We attach these to the generation (the
            # child observation) so the Langfuse UI shows model +
            # tokens + cost per call.
            usage = _extract_usage(verdict)
            token_cost = _compute_token_cost_usd(
                model=client.model,
                input_tokens=usage.get("input_tokens") or None,
                output_tokens=usage.get("output_tokens") or None,
            )
            latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

            try:
                generation.update(
                    model=client.model,
                    usage={
                        "input": usage.get("input_tokens"),
                        "output": usage.get("output_tokens"),
                        "unit": "TOKENS",
                    },
                    metadata={
                        "token_cost_usd": token_cost,
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "latency_ms": latency_ms,
                    },
                )
            except Exception:  # noqa: BLE001 — never fail the call on trace writes
                logger.exception(
                    "compare_topk: failed to update Langfuse generation with usage"
                )

            # Add the empty user_feedback score field. Phase 3
            # replaces this with the UI's thumbs-up/down value.
            add_user_feedback_placeholder(generation)
    except Exception:
        # The Langfuse trace context catches + tags errors
        # itself; we just re-raise so the route layer gets the
        # typed exception (SchemaViolationError / LLMTransportError /
        # MissingAPIKeyError) and maps it to the right response.
        raise

    return verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_validation_error(exc: ValidationError) -> Any:
    """Render a ``pydantic.ValidationError`` as a JSON-safe dict.

    The default ``exc.errors()`` returns a list of dicts whose
    values can include non-JSON-serialisable types (e.g. ``bytes``,
    custom exception instances). For the API boundary we want
    plain primitives only.
    """
    try:
        # ``json.loads(json.dumps(...))`` is the cheapest way
        # to coerce the structure to JSON-safe primitives.
        # We don't care about the perf here — the call is
        # rare (only on schema violation, not the happy path).
        return json.loads(json.dumps(exc.errors(), default=str))
    except (TypeError, ValueError):
        # Last-resort fallback — return the string repr so the
        # user sees *something* rather than a 500.
        return {"repr": repr(exc)}


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "CompetitorVerdict",
    "CompareClient",
    "CompareTopKError",
    "DEFAULT_MAX_COMPANIES",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "IdeaVerdict",
    "LLMTransportError",
    "MarketScope",
    "MAX_TOP_K",
    "MissingAPIKeyError",
    "SchemaViolationError",
    "build_default_client",
    "compare_topk",
]
