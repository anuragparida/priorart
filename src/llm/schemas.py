"""Pydantic v2 schemas for the structured-comparison call (Phase 1.7 + Phase 4.1).

These are the public, Pydantic-validated wire shapes for the LLM call
in :mod:`src.llm.compare`. The fields are exactly what
``docs/PHASE-1.md`` §1.7 specifies — ``CompetitorVerdict``,
``MarketScope``, and ``IdeaVerdict`` — and the JSON Schema instructor
generates from them is the source of truth for the structured output
contract with Claude.

Phase 4.1 — replaces the Phase 1.7 stub; see PHASE-4.md for the rationale.
The Phase 1.7 envelope (``market_scope: MarketScope`` enum +
``market_scope_rationale: str``) is preserved verbatim for backward
compatibility with the existing frontend (``src/frontend/src/lib/marketScope.ts``
keys off the four enum strings). A new *additive* field
``market_scope_signal: Optional[MarketScopeSignal] = None`` carries
the corpus-grounded quantitative layer that Phase 4.2 will populate
when the deterministic rules in 4.2 fire. The frontend stays on
``verdict.market_scope`` (the enum); the new confidence badge in 4.6
will read ``verdict.market_scope_signal.confidence`` when present and
fall back to "directional" when null.

Why Pydantic v2 (and not dataclasses / TypedDict)
-------------------------------------------------
- ``instructor`` wraps Pydantic models natively — it inspects the
  model's JSON schema and feeds it to the LLM as a tool input
  contract, then validates the response against the model. Dataclasses
  and TypedDicts don't get the same first-class treatment.
- Pydantic v2 emits clean JSON Schema (no ``title``/``default``
  noise), so the LLM sees a compact, unambiguous contract.
- The schemas are reused on the API boundary (1.8) — FastAPI uses
  the same models for request/response validation, so a single
  class is the source of truth for "what does an IdeaVerdict look
  like" in both the LLM call and the HTTP layer.

Stability
---------
The field names and types here are part of the public contract.
Renaming a field is a breaking change to the LLM call, the API
response, and the test fixture ``tests/fixtures/compare_smoke.json``.
If you need to deprecate a field, add a new one and keep the old
one populated for one release.

Phase 4.1 backwards-compat decision (option b)
----------------------------------------------
The Phase 4 spec at PHASE-4.md §4.1 hard-rules "no breaking changes
to the ``IdeaVerdict`` schema" — preserve ``market_scope: MarketScope``
as the string accessor the frontend reads. Two options were
considered:

- **(a) Replace ``market_scope: MarketScope`` with
  ``market_scope: MarketScopeSignal``.**
  Cleaner long-term (one field, no duplication) but breaks the
  Phase 1.7/1.8/1.11/2.1/2.2/2.3/2.8/3.1/3.3 fixtures that read
  ``verdict.market_scope.value`` and the frontend that reads
  ``verdict.market_scope`` directly. The Phase 1.7 LLM prompt emits
  ``market_scope: <enum string>`` — switching to (a) without also
  re-prompting the LLM means instructor would fail to validate, so
  (a) implicitly requires LLM-prompt changes that PHASE-4.md §4.1
  scopes out ("no LLM code in this card").
- **(b) Add ``market_scope_signal: Optional[MarketScopeSignal] = None``;
  keep ``market_scope: MarketScope`` unchanged.** The new envelope
  is purely additive — populated by the 4.2/4.3/4.4 pipeline when
  it runs, null otherwise. Frontend ignores it for now; Phase 4.6
  wires up the confidence badge.

**Chosen: (b).** The Phase 4 spec's hard rule + the "no LLM code in
this card" rule together force (b). The duplication of the direction
field (``market_scope`` as the existing string and
``market_scope_signal.direction`` as the new one) is bounded — both
fields point at the same ``MarketScope`` enum, so a future card can
deprecate ``market_scope`` in a follow-up release by syncing the
4.4 LLM synthesis to populate both.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Market-scope enum
# ---------------------------------------------------------------------------


class MarketScope(str, Enum):
    """Four-state directional signal for the competitive landscape.

    Phase 1.7 — this was a *stub* populated by the LLM from the
    top-K density and the structured comparison, **not** by a real
    SEMrush/Ahrefs pipeline. The README labelled it as a directional
    signal; "future work: integrate SEMrush / SimilarWeb when budget
    allows."

    Phase 4.1 — the string values are unchanged. The new
    ``MarketScopeSignal`` envelope in Phase 4.2/4.3/4.4 carries the
    corpus-grounded quantitative layer; ``MarketScope`` stays as the
    four-value direction the existing frontend renders.

    The string values are part of the public API contract — the
    frontend in Phase 1.9 / Phase 4.6 keys its colour coding off them.
    """

    WIDE_OPEN = "wide_open"
    CROWDED_BUT_GROWING = "crowded_but_growing"
    SATURATED = "saturated"
    NICHE_BUT_REAL = "niche_but_real"


# ---------------------------------------------------------------------------
# Phase 4.1 — MarketScopeQuant + MarketScopeEvidence + MarketScopeSignal
# ---------------------------------------------------------------------------


class MarketScopeQuant(BaseModel):
    """Corpus-derived quantitative layer for the market-scope signal.

    Phase 4.1 (PHASE-4.md §4.1) — the new envelope that 4.2 populates
    when the deterministic direction rules fire. All fields are
    derived from the local 11K-company corpus (``companies`` +
    ``company_embeddings``); ``search_volume_proxy`` is the only
    optional field that depends on a SearXNG-augmented layer (4.3).

    Field semantics
    ---------------
    - ``competitor_count``: the size of the top-200 neighborhood
      used to compute the signal. Capped at 200 by the activity.
    - ``recent_3y_count``: how many of those competitors launched in
      the last 3 years. Drives ``growth_rate`` + the "still growing?"
      leg of the deterministic direction rules.
    - ``category_distribution``: histogram of ``business_category``
      over the neighborhood, top-8 categories + an "other" bucket
      so the dict stays bounded.
    - ``search_volume_proxy``: SearXNG-derived domain-count from
      4.3's web augmentation. Null when the corpus is dense enough
      that 4.3 doesn't fire (the deterministic rules in 4.2
      already settled the direction).
    - ``saturation_index``: ``competitor_count / 200`` clamped to
      [0, 1]. 1.0 means "we hit the 200-cap" — the neighborhood is
      so dense we're not even trying to count beyond it.
    - ``growth_rate``: ``recent_3y_count / max(competitor_count, 1)``
      — the fraction of launches in the last 3 years. Drives the
      "still growing?" check; null when ``competitor_count == 0``
      (degenerate empty-corpus case).
    """

    model_config = ConfigDict(extra="forbid")

    competitor_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of competitors in the top-200 neighborhood used to "
            "compute the signal. Capped at 200 by the 4.2 activity."
        ),
    )
    recent_3y_count: int = Field(
        ...,
        ge=0,
        description=(
            "How many of those competitors launched in the last 3 years. "
            "Drives the growth_rate + the 'still growing?' leg of the "
            "deterministic direction rules."
        ),
    )
    category_distribution: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Histogram of business_category over the neighborhood. "
            "Top-8 categories + an 'other' bucket so the dict stays bounded."
        ),
    )
    search_volume_proxy: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "SearXNG-derived domain count from 4.3's web augmentation. "
            "Null when the corpus is dense enough that 4.3 doesn't fire."
        ),
    )
    saturation_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "competitor_count / 200 clamped to [0, 1]. 1.0 = 'we hit "
            "the 200-cap, the neighborhood is so dense we're not "
            "even trying to count beyond it'."
        ),
    )
    growth_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "recent_3y_count / max(competitor_count, 1) — the fraction "
            "of launches in the last 3 years. Null when competitor_count "
            "is zero (degenerate empty-corpus case)."
        ),
    )


class MarketScopeEvidence(BaseModel):
    """One evidence entry supporting a market-scope claim.

    Phase 4.1 (PHASE-4.md §4.1, hard rule on §Pitfalls) — the LLM
    synthesis step in 4.4 is allowed to *cite* an evidence URL but
    never to *invent* one. Every entry with ``source="web"`` MUST
    come from a real SearXNG + Firecrawl scrape (4.3); every entry
    with ``source="corpus"`` carries a real ``company_id`` from
    the local ``companies`` table.

    The ``as_of`` timestamp is the wall-clock at which the evidence
    was captured, so a reviewer can tell how fresh the underlying
    signal is. Pydantic v2 serialises ``datetime`` to ISO-8601
    strings on ``model_dump(mode="json")``.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["corpus", "web"] = Field(
        ...,
        description=(
            "Where the evidence came from. 'corpus' = a row in the local "
            "companies table (carries company_id). 'web' = a SearXNG + "
            "Firecrawl scrape (carries url + snippet)."
        ),
    )
    url: Optional[str] = Field(
        default=None,
        description=(
            "URL of the scraped page. Set when source='web'. Null when "
            "source='corpus'."
        ),
    )
    company_id: Optional[int] = Field(
        default=None,
        description=(
            "Numeric id of the company in the local companies table. Set "
            "when source='corpus'. Null when source='web'."
        ),
    )
    snippet: Optional[str] = Field(
        default=None,
        description=(
            "Short quote from the source supporting the direction claim. "
            "Optional — a corpus-source may omit it if the company name "
            "is enough; a web-source SHOULD include the snippet that "
            "justifies why the result was kept."
        ),
    )
    as_of: datetime = Field(
        ...,
        description=(
            "Wall-clock timestamp at which the evidence was captured. "
            "ISO-8601 serialised by Pydantic v2 on model_dump(mode='json')."
        ),
    )


#: Confidence tiers for the ``MarketScopeSignal`` envelope. Phase 4.1
#: (PHASE-4.md §4.1) — three levels that match the source strength:
#:
#: - ``"directional"``: the LLM synthesis fallback (4.4) when corpus
#:   + web aren't enough. Equivalent to the Phase 1.7 stub.
#: - ``"evidence_backed"``: ≥1 corpus source AND ≥1 web source
#:   (SearXNG-augmented, 4.3). The 4.4 path populates this tier.
#: - ``"quantitative"``: the deterministic direction rules in 4.2
#:   fired and ``MarketScopeQuant`` is fully populated.
MarketScopeConfidence = Literal["directional", "evidence_backed", "quantitative"]


class MarketScopeSignal(BaseModel):
    """The Phase 4 market-scope envelope.

    Phase 4.1 (PHASE-4.md §4.1) — wraps the existing
    ``MarketScope`` enum with a quantitative layer, a confidence
    tier, and an evidence trail. Populated by the 4.2/4.3/4.4
    pipeline; ``null`` on the ``IdeaVerdict.market_scope_signal``
    field when the 4.2 deterministic rules don't fire (the
    pipeline defaults back to the Phase 1.7 stub shape — the
    existing ``market_scope`` enum + ``market_scope_rationale``
    fields continue to carry the directional signal).

    Field semantics
    ---------------
    - ``direction``: the four-state directional signal. Same enum as
      the Phase 1.7 ``market_scope`` field; we duplicate the value
      here so the envelope is self-contained for downstream readers
      that read only ``market_scope_signal.direction``.
    - ``rationale``: the existing rationale field, renamed for
      clarity. Carries the 1–2 sentence explanation of why the
      direction was picked; the 4.4 LLM synthesis contract requires
      it to cite at least one of ``competitor_count``,
      ``recent_3y_count``, or a web-snippet URL.
    - ``quantitative``: the ``MarketScopeQuant`` layer. Null when
      confidence is ``"directional"``; populated when
      confidence is ``"quantitative"`` (deterministic rules
      fired); partially populated (``search_volume_proxy`` set,
      the rest null) when confidence is ``"evidence_backed"``.
    - ``confidence``: one of ``"directional"`` /
      ``"evidence_backed"`` / ``"quantitative"``.
    - ``evidence``: ordered list of supporting evidence. Up to 5
      ``{source: "corpus", company_id}`` entries for the
      ``"quantitative"`` tier; up to 3 ``{source: "web", url,
      snippet}`` entries appended for the ``"evidence_backed"``
      tier.

    Why a separate envelope (rather than collapsing onto the enum)
    -------------------------------------------------------------
    Keeps the Phase 1.7 LLM contract stable — the structured-
    comparison call still emits ``market_scope`` as a plain enum
    string and ``market_scope_rationale`` as a plain string. The
    envelope is populated by a *separate* activity (the
    ``market_scope_signal`` activity in 4.2/4.3/4.4) and stored
    as an additive ``IdeaVerdict.market_scope_signal`` field. The
    4.4 LLM synthesis step writes both ``market_scope`` /
    ``market_scope_rationale`` AND ``market_scope_signal`` so the
    legacy readers and the new envelope stay in sync.
    """

    model_config = ConfigDict(extra="forbid")

    direction: MarketScope = Field(
        ...,
        description=(
            "Four-state directional signal. Same enum as IdeaVerdict.market_scope "
            "— duplicated here so the envelope is self-contained for downstream "
            "readers that only see market_scope_signal."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "1-2 sentence explanation of why the direction was picked. The "
            "4.4 LLM synthesis contract requires it to cite at least one of "
            "competitor_count, recent_3y_count, or a web-snippet URL."
        ),
    )
    quantitative: Optional[MarketScopeQuant] = Field(
        default=None,
        description=(
            "Corpus-derived quantitative layer. Null when confidence is "
            "'directional'; populated when 'quantitative' (deterministic "
            "rules fired); partially populated (search_volume_proxy set, "
            "the rest null) when 'evidence_backed'."
        ),
    )
    confidence: MarketScopeConfidence = Field(
        default="directional",
        description=(
            "Confidence tier. 'directional' = LLM synthesis fallback "
            "(Phase 1.7 stub shape). 'evidence_backed' = >=1 corpus + "
            ">=1 web source. 'quantitative' = 4.2 deterministic rules "
            "fired and MarketScopeQuant is fully populated."
        ),
    )
    evidence: list[MarketScopeEvidence] = Field(
        default_factory=list,
        description=(
            "Ordered list of supporting evidence. Up to 5 corpus entries "
            "for 'quantitative'; up to 3 web entries appended for "
            "'evidence_backed'."
        ),
    )


# ---------------------------------------------------------------------------
# Per-competitor structured verdict
# ---------------------------------------------------------------------------


class CompetitorVerdict(BaseModel):
    """Structured comparison of the idea vs. one similar company.

    All five list fields are *strings*, not nested objects. The LLM
    is asked to write 2–4 short, declarative phrases per field — the
    frontend renders them as chips, bullets, and tags. The contract
    is "short, evidence-anchored, no marketing fluff", which is
    easier to enforce in the system prompt than to validate
    structurally.

    ``confidence`` is the model's self-reported 0–1 confidence. It
    is *not* the cosine similarity from /search — the LLM sees both
    the cosine scores and the company descriptions, and returns a
    holistic "how much of a real competitor is this" signal. The
    eval harness in Phase 1.6 will measure how well it calibrates.
    """

    # ``extra='forbid'`` means a misbehaving LLM that emits an
    # unexpected field raises a validation error instead of being
    # silently coerced. We want loud failures, not quiet drift.
    model_config = ConfigDict(extra="forbid")

    company_id: int = Field(
        ...,
        description="Numeric id of the company in the local `companies` table.",
    )
    name: str = Field(..., description="Human-readable company name.")
    similarity_axes: list[str] = Field(
        default_factory=list,
        description=(
            "Short phrases naming the dimensions on which this company "
            "is similar to the idea (e.g. 'AI-assisted drafting', 'SMB "
            "market', 'subscription pricing'). 2–4 phrases."
        ),
    )
    key_differences: list[str] = Field(
        default_factory=list,
        description=(
            "Short phrases naming the dimensions on which this company "
            "differs from the idea (e.g. 'enterprise-only', 'no AI "
            "drafting', 'larger entity-extraction model'). 1–3 phrases."
        ),
    )
    likely_failure_modes: list[str] = Field(
        default_factory=list,
        description=(
            "Why this competitor might lose to the user's idea, or "
            "might be a hard incumbent. 1–3 phrases. Honest — 'they "
            "have strong distribution' is a valid answer."
        ),
    )
    evidence_links: list[str] = Field(
        default_factory=list,
        description=(
            "URLs the model used as evidence (company url, blog "
            "post, founder interview, etc.). Empty list if the model "
            "relied on the supplied description alone."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model self-reported 0–1 confidence that this company is a "
            "real competitor to the idea. Not the cosine similarity — "
            "the LLM's holistic judgment after reading both."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level IdeaVerdict
# ---------------------------------------------------------------------------


#: Hard cap on the number of competitors in a verdict. PHASE-1.md §1.7
#: says "hard cap at top-3 (configurable)"; this constant is the
#: default. The /ideas/analyze endpoint (1.8) can override it via
#: env or a query param.
DEFAULT_TOP_K = 3
MAX_TOP_K = 5  # the prompt template enforces up to 5; we still cap to 3 by default


class IdeaVerdict(BaseModel):
    """The full structured response to ``POST /ideas/analyze``.

    This is the wire shape returned to the frontend and the unit of
    work for the eval harness in Phase 1.6.

    ``top_competitors`` is the *structured* result. The cosine
    similarities from /search are not preserved here — Phase 1.6's
    eval harness can re-derive them from the IDs if needed.

    ``supporting_evidence`` is the model's *general* evidence (blog
    posts, market reports) for the market-scope claim, distinct
    from the per-competitor ``evidence_links`` lists.

    Phase 4.1 — ``market_scope_signal`` is the new optional envelope.
    See the module docstring for the option-(b) backwards-compat
    rationale. The legacy ``market_scope`` (MarketScope enum) +
    ``market_scope_rationale`` (str) fields are unchanged so the
    Phase 1.7 / 1.8 / 1.11 / 2.1 / 2.2 / 2.3 / 2.8 / 3.1 / 3.3
    fixtures that read ``verdict.market_scope`` continue to work.
    When the 4.2 / 4.3 / 4.4 pipeline populates
    ``market_scope_signal``, the ``direction`` field inside mirrors
    ``market_scope`` — the 4.4 LLM synthesis writes both to keep
    the legacy + new readers in sync.
    """

    model_config = ConfigDict(extra="forbid")

    idea: str = Field(
        ...,
        description="The original user-supplied idea text, echoed back.",
    )
    top_competitors: list[CompetitorVerdict] = Field(
        default_factory=list,
        description=(
            "Ranked list of similar past launches. Length is the "
            "configured top-K, default 3. Same order as /search "
            "returned."
        ),
    )
    market_scope: MarketScope = Field(
        ...,
        description=(
            "Four-state directional signal: wide_open, "
            "crowded_but_growing, saturated, niche_but_real. Phase 4.1 "
            "adds the optional `market_scope_signal` envelope for the "
            "corpus-grounded quantitative layer; this enum field is "
            "unchanged for backward compat with the Phase 1.7 / 1.8 "
            "frontend and the existing test fixtures."
        ),
    )
    market_scope_rationale: str = Field(
        ...,
        description=(
            "1–2 sentence explanation of why the model picked the "
            "market_scope value. Anchors the stub in evidence so "
            "users can tell when the model is hand-waving. Phase 4.1 "
            "still reads this directly for the 'directional' confidence "
            "tier; the 4.4 LLM synthesis writes both this field and "
            "`market_scope_signal.rationale`."
        ),
    )
    market_scope_signal: Optional[MarketScopeSignal] = Field(
        default=None,
        description=(
            "Phase 4.1 — optional corpus-grounded envelope. Null when "
            "the 4.2 deterministic rules don't fire (the verdict "
            "relies on the Phase 1.7 stub shape); populated by the "
            "4.2/4.3/4.4 pipeline when the corpus + web signal is "
            "strong enough. When populated, `direction` mirrors "
            "`market_scope` and `confidence` is one of 'directional' / "
            "'evidence_backed' / 'quantitative'."
        ),
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description=(
            "URLs that support the market_scope verdict as a whole "
            "(market reports, HN threads, blog posts). Distinct "
            "from per-competitor `evidence_links`."
        ),
    )
