"""LLM layer — Claude wrappers, Pydantic schemas, prompt templates.

Phase 1.7 (this file): the structured-comparison call
(:mod:`src.llm.compare`) ships the Pydantic schemas, the prompt
template, and the ``compare_topk`` function that wraps Claude
Sonnet 4.5 with instructor.

Phase 1.8 will add the ``/ideas/analyze`` endpoint, which uses
:func:`src.llm.compare.compare_topk` together with the /search
results.

Public surface
--------------
- :class:`src.llm.schemas.IdeaVerdict` — the wire shape returned
  by the LLM call and the /ideas/analyze endpoint.
- :func:`src.llm.compare.compare_topk` — the one public function.
- :class:`src.llm.compare.CompareClient` — a small dataclass that
  bundles the instructor client + model name, for explicit
  dependency injection (tests pass a mock here).
"""
