"""Unit tests for src/llm/compare (Phase 1.7).

What this covers
----------------
- The import contract: ``compare_topk`` is importable from
  ``src.llm.compare``.
- The schema roundtrip: a hand-built ``IdeaVerdict`` parses and
  re-serialises through the Pydantic model.
- The prompt builder: ``build_user_prompt`` renders the top-K
  list as expected, and refuses empty input.
- The mocked LLM call: ``compare_topk`` with a mocked
  ``CompareClient`` returns a valid ``IdeaVerdict`` for a
  realistic payload.
- Cost-control: ``compare_topk`` makes *exactly one* LLM call,
  regardless of ``top_k`` length (verifies the "one call per
  request, not per competitor" requirement).
- The "no API key" path: a missing key surfaces as
  ``MissingAPIKeyError``, not a generic ``Exception``.
- Defensive normalisation: top-K entries missing
  ``company_id`` are dropped before the prompt is built (so a
  hand-rolled fixture doesn't 500 the call).

What this does NOT cover
------------------------
- A live LLM call. That's the smoke fixture in
  ``tests/fixtures/compare_smoke.json`` and is gated on a
  real ``ANTHROPIC_API_KEY`` (see ``tests/test_compare_smoke.py``).
- The HTTP layer. ``/ideas/analyze`` is Phase 1.8; this test
  module is the library layer only.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.llm.compare import (
    CompareClient,
    MissingAPIKeyError,
    SchemaViolationError,
    _coerce_top_k,
    compare_topk,
)
from src.llm.prompts.compare import MAX_PROMPT_COMPANIES, build_user_prompt
from src.llm.schemas import (
    CompetitorVerdict,
    DEFAULT_TOP_K,
    IdeaVerdict,
    MarketScope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_top_k(n: int = 3) -> list[dict]:
    """Return a list of n fake top-K company dicts.

    The numbers and names are not real YC companies — they're
    obviously synthetic so a reader can tell at a glance that
    this is test data. The name template is parameterised so the
    helper works for n > 6 (some tests use n=10 to exercise the
    cap).
    """
    return [
        {
            "company_id": 1000 + i,
            "name": f"FakeCo_{i:03d}",
            "description": f"Synthetic test company #{i}. Does something AI-related.",
            "similarity": 0.9 - 0.01 * i,
        }
        for i in range(n)
    ]


def _make_idea_verdict(top_k: list[dict]) -> IdeaVerdict:
    """Return a hand-built ``IdeaVerdict`` that mirrors ``top_k``."""
    return IdeaVerdict(
        idea="AI contract review for SMB law firms",
        top_competitors=[
            CompetitorVerdict(
                company_id=c["company_id"],
                name=c["name"],
                similarity_axes=[f"AI feature #{i}", f"market: SMB"],
                key_differences=[f"diff #{i}"],
                likely_failure_modes=[f"failure mode #{i}"],
                evidence_links=[],
                confidence=0.5 + 0.1 * i,
            )
            for i, c in enumerate(top_k)
        ],
        market_scope=MarketScope.CROWDED_BUT_GROWING,
        market_scope_rationale=(
            "5+ similar YC launches in the last 3 years, none dominant"
        ),
        supporting_evidence=[],
    )


def _make_mock_client(verdict: IdeaVerdict | None = None) -> CompareClient:
    """Return a ``CompareClient`` whose instructor client is a
    ``MagicMock`` that returns ``verdict`` from ``create``.
    """
    mock = MagicMock()
    if verdict is not None:
        mock.create.return_value = verdict
    return CompareClient(
        instructor_client=mock,
        model="claude-sonnet-4-5-test",
        max_retries=3,
        timeout_seconds=10.0,
    )


# ---------------------------------------------------------------------------
# Import / surface contract
# ---------------------------------------------------------------------------


def test_compare_topk_is_importable():
    """Acceptance criterion 1: ``from src.llm.compare import compare_topk`` works."""
    # If this test file imports at all, the import contract holds.
    # The explicit re-import here is a regression guard against
    # someone renaming the function without updating callers.
    from src.llm.compare import compare_topk as _compare  # noqa: F401
    assert callable(compare_topk)


def test_schemas_roundtrip():
    """A hand-built ``IdeaVerdict`` parses and re-serialises."""
    v = _make_idea_verdict(_make_top_k(3))
    dumped = v.model_dump()
    rebuilt = IdeaVerdict.model_validate(dumped)
    assert rebuilt == v
    assert rebuilt.market_scope == MarketScope.CROWDED_BUT_GROWING
    assert len(rebuilt.top_competitors) == 3


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_build_user_prompt_renders_top_k():
    """The prompt contains the idea text and each top-K name + id."""
    top_k = _make_top_k(3)
    prompt = build_user_prompt("AI contract review", top_k)
    assert "AI contract review" in prompt
    for c in top_k:
        assert c["name"] in prompt
        assert f"id={c['company_id']}" in prompt


def test_build_user_prompt_refuses_empty_idea():
    with pytest.raises(ValueError):
        build_user_prompt("", _make_top_k(1))
    with pytest.raises(ValueError):
        build_user_prompt("   ", _make_top_k(1))


def test_build_user_prompt_refuses_empty_top_k():
    with pytest.raises(ValueError):
        build_user_prompt("an idea", [])


def test_build_user_prompt_caps_top_k():
    """The prompt builder refuses to render more than MAX_PROMPT_COMPANIES."""
    over_cap = _make_top_k(MAX_PROMPT_COMPANIES + 5)
    # The function trims silently — that's the documented behaviour.
    # We verify the trim by checking the prompt doesn't contain
    # entries past the cap.
    prompt = build_user_prompt("an idea", over_cap)
    # The cap is MAX_PROMPT_COMPANIES; verify the trimmed entries
    # are absent by looking for a name we know is past the cap.
    last_dropped_name = over_cap[MAX_PROMPT_COMPANIES]["name"]
    assert last_dropped_name not in prompt


# ---------------------------------------------------------------------------
# Mocked LLM call — the acceptance criterion
# ---------------------------------------------------------------------------


def test_compare_topk_returns_idea_verdict_with_mock():
    """Acceptance criterion 2: a unit test with a mocked instructor
    client asserts the JSON parses and matches the schema.

    This is the *only* path that runs without a real API key.
    """
    top_k = _make_top_k(3)
    expected = _make_idea_verdict(top_k)
    client = _make_mock_client(expected)

    verdict = compare_topk(
        "AI contract review for SMB law firms",
        top_k,
        client=client,
    )

    # The verdict is the exact object the mock returned (no
    # transformation) — we want the wire shape to be the
    # Pydantic model directly, not a re-parse.
    assert verdict is expected
    assert isinstance(verdict, IdeaVerdict)
    assert verdict.idea == "AI contract review for SMB law firms"
    assert verdict.market_scope == MarketScope.CROWDED_BUT_GROWING
    assert len(verdict.top_competitors) == 3
    for cv in verdict.top_competitors:
        assert 0.0 <= cv.confidence <= 1.0


def test_compare_topk_makes_exactly_one_llm_call():
    """Cost control: ONE LLM call per request, regardless of top-K.

    Spec: ``top_competitors`` for K=3 must come from a single
    Anthropic call. The mock's call_count is the proof.
    """
    top_k = _make_top_k(3)
    expected = _make_idea_verdict(top_k)
    client = _make_mock_client(expected)

    compare_topk("idea", top_k, client=client)

    assert client.instructor_client.create.call_count == 1, (
        "compare_topk must make exactly one LLM call per request; "
        "got {n}".format(n=client.instructor_client.create.call_count)
    )


def test_compare_topk_trims_top_k_to_max_companies():
    """The default cap is 3; passing 5 yields a prompt that
    includes only 3 entries (or fewer if the prompt builder
    caps differently).
    """
    top_k = _make_top_k(5)
    expected = _make_idea_verdict(_make_top_k(3))  # mock returns 3
    client = _make_mock_client(expected)

    verdict = compare_topk("idea", top_k, client=client)
    assert len(verdict.top_competitors) == 3

    # Verify the call passed the trimmed top-K. We grab the
    # prompt message from the mock's call args and check the
    # names of entries past the cap are NOT in the prompt.
    call_args = client.instructor_client.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[1]
    user_message = messages[0]["content"]
    # First 3 (kept) — present.
    assert "FakeCo_000" in user_message
    assert "FakeCo_001" in user_message
    assert "FakeCo_002" in user_message
    # 4th and 5th (trimmed) — absent.
    assert "FakeCo_003" not in user_message
    assert "FakeCo_004" not in user_message


def test_compare_topk_uses_configured_model():
    """The model name from the client is forwarded to the
    instructor call. Pinning this is important so a model swap
    in the env var doesn't accidentally use the wrong one.
    """
    top_k = _make_top_k(1)
    expected = _make_idea_verdict(top_k)
    client = _make_mock_client(expected)

    compare_topk("idea", top_k, client=client)

    call_kwargs = client.instructor_client.create.call_args.kwargs
    assert call_kwargs.get("model") == "claude-sonnet-4-5-test"
    assert call_kwargs.get("response_model") is IdeaVerdict
    assert call_kwargs.get("max_retries") == 3


def test_compare_topk_response_model_idea_verdict():
    """The instructor call is told to coerce to IdeaVerdict.

    This is the structural contract — without it, instructor
    wouldn't validate the response against the Pydantic schema,
    and the whole point of Phase 1.7 (deterministic JSON
    output) would be lost.
    """
    top_k = _make_top_k(1)
    client = _make_mock_client(_make_idea_verdict(top_k))
    compare_topk("idea", top_k, client=client)

    call_kwargs = client.instructor_client.create.call_args.kwargs
    assert call_kwargs.get("response_model") is IdeaVerdict


# ---------------------------------------------------------------------------
# Defensive normalisation
# ---------------------------------------------------------------------------


def test_coerce_top_k_drops_entries_without_company_id():
    """A top-K entry missing company_id is dropped (not crashed on)."""
    dirty = _make_top_k(3) + [{"name": "ghost", "description": "no id"}]
    cleaned = _coerce_top_k(dirty, max_companies=5)
    assert len(cleaned) == 3
    assert all("company_id" in c for c in cleaned)


def test_coerce_top_k_respects_max_companies():
    """The cap is honoured."""
    cleaned = _coerce_top_k(_make_top_k(10), max_companies=2)
    assert len(cleaned) == 2


def test_coerce_top_k_refuses_zero_cap():
    with pytest.raises(ValueError):
        _coerce_top_k(_make_top_k(1), max_companies=0)


def test_compare_topk_drops_garbage_and_still_works():
    """End-to-end: a top-K list with one bad entry should still
    return a verdict (the bad one gets dropped before the LLM
    call). This is the "no 500s on schema-violation" guarantee
    generalised: the prompt builder is the first line of
    defence, and a stray missing field is not a 500.
    """
    dirty = _make_top_k(3) + [{"name": "ghost", "description": "no id"}]
    # The first 3 are real; verdict should still be valid.
    expected = _make_idea_verdict(_make_top_k(3))
    client = _make_mock_client(expected)
    verdict = compare_topk("idea", dirty, client=client)
    assert isinstance(verdict, IdeaVerdict)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_compare_topk_refuses_empty_idea():
    with pytest.raises(ValueError):
        compare_topk("", _make_top_k(1), client=_make_mock_client())


def test_compare_topk_refuses_empty_top_k():
    with pytest.raises(ValueError):
        compare_topk("idea", [], client=_make_mock_client())


def test_compare_topk_surfaces_schema_violation_as_typed_error():
    """When instructor cannot coerce the response, we raise
    ``SchemaViolationError`` (not a generic ``Exception``) so
    the API layer can map it to a 200 with
    ``{"error": "schema_violation", "details": ...}``.

    We simulate the failure by giving the mock a side effect
    that raises ``ValidationError``.
    """
    from pydantic import ValidationError

    client = _make_mock_client()
    client.instructor_client.create.side_effect = ValidationError.from_exception_data(
        "IdeaVerdict",
        [{"type": "missing", "loc": ("idea",), "input": {}, "msg": "Field required"}],
    )

    with pytest.raises(SchemaViolationError) as excinfo:
        compare_topk("idea", _make_top_k(1), client=client)

    # The error carries details the API layer can surface to
    # the user. We don't pin the exact shape (instructor can
    # wrap it), but it must be JSON-serialisable.
    assert excinfo.value.details is not None


def test_compare_topk_surfaces_anthropic_error_as_typed_error():
    """Anthropic SDK errors become ``LLMTransportError``, not bare
    ``Exception`` propagation.
    """
    from anthropic import APIConnectionError

    client = _make_mock_client()
    client.instructor_client.create.side_effect = APIConnectionError(request=MagicMock())

    # We import the LLMTransportError only for the isinstance
    # assertion to avoid a hard dependency on the symbol in
    # the test's top-level imports.
    from src.llm.compare import LLMTransportError
    with pytest.raises(LLMTransportError):
        compare_topk("idea", _make_top_k(1), client=client)


def test_missing_api_key_surfaces_typed_error(monkeypatch):
    """If no API key is configured, ``build_default_client``
    raises ``MissingAPIKeyError`` with a clear message — not
    a generic ``Exception`` that would land as a 500.
    """
    from src.llm.compare import build_default_client

    # Wipe both the env var and the file path resolution. The
    # function reads the file from $HOME; we monkey-patch
    # Path.home to a temp dir so ~/.anthropic_key can't be
    # found even if it exists on the test machine.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import src.llm.compare as compare_mod
    monkeypatch.setattr(compare_mod, "Path", _FakeHomePath)

    with pytest.raises(MissingAPIKeyError) as excinfo:
        build_default_client()
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


class _FakeHomePath:
    """``Path.home()`` returns a path that has no ``.anthropic_key``.

    Used by :func:`test_missing_api_key_surfaces_typed_error` to
    ensure the test is hermetic — the user's real
    ``~/.anthropic_key`` (if any) does not affect the result.
    """

    _Path = __import__("pathlib").Path

    @classmethod
    def home(cls):
        # /tmp is writable and guaranteed to have no
        # ``.anthropic_key`` on any sane test machine.
        return cls._Path("/tmp")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_default_top_k_is_three():
    """PHASE-1.md §1.7: 'hard cap at top-3 (configurable)'."""
    assert DEFAULT_TOP_K == 3
