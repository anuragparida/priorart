"""LLM prompt templates (Phase 1.7).

Each public-facing LLM call gets its own module here. Keep the
prompts as module-level constants (SYSTEM_PROMPT) and small builder
functions (build_user_prompt) — no JSON / YAML / TS side files.
See ``src/llm/prompts/compare.py`` for the rationale.
"""
