"""Top-level ``eval`` package — a re-export of ``src.eval``.

Why this exists
---------------
The Phase 1.6 task spec (``docs/PHASE-1.md`` §1.6) pins the
runner as ``python -m eval.run``. The actual implementation lives
under ``src/eval/`` (the project's ``src/`` layout) — that's where
the other modules (``src.api``, ``src.data``, ``src.llm``) live
and where ``pyproject.toml``'s ``[tool.hatch.build.targets.wheel]``
packages point.

To honor the spec while keeping the ``src/`` layout, this
top-level package re-exports everything from ``src.eval``. The two
names point at the same code, so a reader looking at
``python -m eval.run`` sees what they expect, and the import
machinery is unambiguous.

We don't do any logic here — every name is sourced from
``src.eval``. If you add a module, add it here too.
"""

from __future__ import annotations

from src.eval import benchmark, config, metrics  # noqa: F401
from src.eval.run import app, main, run_eval  # noqa: F401