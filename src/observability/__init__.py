"""Observability package — Langfuse tracing for LLM calls (Phase 2.3).

Re-exports the public surface from :mod:`src.observability.langfuse`
so callers can ``from src.observability import trace_idea_compare``
without having to know the submodule name.
"""

from src.observability.langfuse import (
    add_user_feedback_placeholder,
    get_client,
    init_langfuse,
    is_tracing_enabled,
    reset_for_tests,
    trace_idea_compare,
)

__all__ = [
    "add_user_feedback_placeholder",
    "get_client",
    "init_langfuse",
    "is_tracing_enabled",
    "reset_for_tests",
    "trace_idea_compare",
]