"""Prompt template for the structured-comparison LLM call (Phase 1.7).

This module is a *deliberate* sibling of ``constants.ts``-style prompt
config that AGENTS.md warns against: the prompt lives in Python
source, not in a JSON/YAML/TS file. The reasons:

- The prompt is *code* — it gets versioned, code-reviewed, and
  unit-tested like the rest of the module. A config file would
  hide it from ``git blame`` and from the eval-harness regression
  suite (Phase 1.6 / 3.2).
- Sub-string interpolation of the top-K company descriptions needs
  Python f-string semantics, not a templating language. (Templating
  languages work too, but then the prompt is two languages away
  from the data it consumes, which makes diffs noisy.)
- A/B-testing the prompt template is a real Phase 2/3 concern
  (MLflow tracks ``prompt_template_version`` as a run parameter).
  A module constant in Python is easier to version than a string
  in a side file.

Why two top-level constants (SYSTEM_PROMPT, build_user_prompt)
--------------------------------------------------------------
SYSTEM_PROMPT is the model's standing instructions — written once,
stable across requests, big (a few hundred tokens). It's the part
the model is most likely to over-fit to, so isolating it makes A/B
testing it cheap.

``build_user_prompt(...)`` is a *function*, not a constant,
because it's per-request: the idea text and the top-K company
descriptions vary. It returns the rendered user message.

Together, these two are the only prompt surface — keep them in
this file.
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Keep the system prompt under ~600 tokens. The longer it is, the
# more tokens per request we pay for *every* call, and the more
# the model has to attend to. Claude Sonnet 4.5 is good at long
# prompts but the signal-to-noise ratio drops fast past ~800 tokens.
SYSTEM_PROMPT: str = """\
You are a startup-idea research analyst. A user has given you a new \
startup idea and a ranked list of similar past launches from a public \
corpus (the Y Combinator directory). Your job is to produce a \
structured comparison and a market-scope signal.

# Output contract

You will be given a JSON schema and you must produce JSON that \
matches it exactly. Do not include any prose, no markdown fences, \
no commentary. Return only the JSON object.

# Fields per competitor

- `similarity_axes` (2–4 short phrases): the dimensions on which \
  this company is similar to the user's idea. Each phrase should be \
  evidence-anchored — based on the company's description, not a \
  guess. Good: "AI-assisted contract drafting". Bad: "AI company".
- `key_differences` (1–3 short phrases): the dimensions on which \
  this company *differs* from the user's idea. Same standard.
- `likely_failure_modes` (1–3 short phrases): why this competitor \
  might lose to the user's idea, or why they are hard to displace. \
  Be honest — "they have a 5-year head start and strong distribution" \
  is a valid answer.
- `evidence_links`: URLs the model actually used. If you relied on \
  the supplied description alone, return an empty list. Do not \
  fabricate URLs.
- `confidence` (0–1): your holistic judgment that this company is a \
  real, actionable competitor. Cosine similarity is a *signal*, not \
  the answer — a high-similarity result that is in a different market \
  segment is not actually a competitor. Use the full 0–1 range \
  liberally; 0.95 means "if you build this idea, you will lose to \
  them", 0.3 means "tangentially related at best".

# Market-scope signal

Pick exactly one of: `wide_open`, `crowded_but_growing`, `saturated`, \
`niche_but_real`. This is a directional signal based on the density \
and recency of similar launches in the corpus, **not** a SEMrush \
estimate. In `market_scope_rationale`, name the 1–2 most-decisive \
data points you used.

# Style

Short, declarative phrases. No marketing language. No "revolutionary" \
or "cutting-edge". No hedging ("could potentially", "may possibly"). \
If you don't know, write "unclear" or omit the phrase.

# Honesty

If the top-K is empty, return top_competitors as an empty list and \
pick the market_scope that best matches the absence of evidence \
(usually `wide_open` or `niche_but_real`). If two companies in the \
top-K do the same thing, the verdict for the lower-ranked one should \
note that in `key_differences`.
"""


# ---------------------------------------------------------------------------
# User-prompt builder
# ---------------------------------------------------------------------------

#: Cap on the number of top-K companies the prompt includes. The
#: function refuses to render prompts that exceed this, even if
#: the caller asks for more — protects against accidental cost
#: blowups if a /ideas/analyze caller passes top_k=50.
MAX_PROMPT_COMPANIES = 5


def build_user_prompt(idea: str, top_k: Sequence[dict]) -> str:
    """Render the user-message string for the structured comparison.

    Parameters
    ----------
    idea:
        The user's free-text idea.
    top_k:
        A sequence of dicts, one per top company, each with at
        least these keys: ``company_id`` (int), ``name`` (str),
        ``description`` (str), and ``similarity`` (float in
        [-1, 1]). Order matters — the first entry is the
        nearest match.

    Returns
    -------
    str
        The rendered user message. The first occurrence of the
        substring ``"{top_k_companies}"`` in the prompt template
        is replaced with the formatted company list.

    Notes
    -----
    The function is pure (no I/O). It does not call the LLM. The
    only validation it does is the length cap — callers that want
    richer validation (e.g. checking ``similarity`` is in [-1, 1])
    should do it before calling this.
    """
    if not idea or not idea.strip():
        raise ValueError("idea must be a non-empty string")
    if not top_k:
        raise ValueError("top_k must be a non-empty sequence")

    # Cap defensively. compare_topk() in src/llm/compare.py does
    # the same cap, but doing it here too means a unit test of
    # build_user_prompt alone can't accidentally render a 50-company
    # prompt and pay for it.
    items = list(top_k)[:MAX_PROMPT_COMPANIES]
    rendered = _format_company_list(items)
    return _USER_PROMPT_TEMPLATE.format(idea=idea.strip(), top_k_companies=rendered)


def _format_company_list(items: Sequence[dict]) -> str:
    """Render the top-K list as a numbered block.

    The format is intentionally structured (id, name, similarity,
    description on separate lines) so the LLM has a stable
    contract to parse — and so a regression in the format is
    easy to spot in the test fixtures.
    """
    lines: list[str] = []
    for i, company in enumerate(items, start=1):
        # Defensive type coercion — we trust the caller (compare_topk
        # in src/llm/compare.py) but a stray test fixture can pass
        # the wrong type. Better to render something than to 500.
        cid = int(company.get("company_id", 0))
        name = str(company.get("name", ""))
        sim = company.get("similarity", 0.0)
        try:
            sim_f = float(sim)
            sim_str = f"{sim_f:.4f}"
        except (TypeError, ValueError):
            sim_str = "n/a"
        description = str(company.get("description", ""))
        # Truncate long descriptions to keep the prompt size bounded.
        # 800 chars is roughly 200 tokens — well within the model's
        # comfort zone for K=3, and even K=5 lands around 1000 tokens
        # total for the user message.
        if len(description) > 800:
            description = description[:797] + "..."
        lines.append(
            f"[{i}] id={cid}\n"
            f"    name: {name}\n"
            f"    cosine_similarity: {sim_str}\n"
            f"    description: {description}"
        )
    return "\n\n".join(lines)


_USER_PROMPT_TEMPLATE: str = """\
# The user's idea

{idea}

# Top-K similar past launches from the YC public directory

These are the most similar companies to the idea, ranked by cosine \
similarity of their description embedding. Cosine is a *signal*, not \
the answer — use the descriptions to judge the actual overlap.

{top_k_companies}

# What to return

A single JSON object matching the schema provided. No prose, no \
markdown, no commentary. JSON only.
"""
