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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import instructor
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError, AuthenticationError
from instructor.core import Instructor
from pydantic import ValidationError

from src.config import ANTHROPIC_MODEL
from src.llm.prompts.compare import build_user_prompt
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
