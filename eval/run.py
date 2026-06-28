"""``eval.run`` — re-export of ``src.eval.run``.

See ``eval/__init__.py`` for the why. This module exists so
``python -m eval.run`` works exactly as written in the Phase 1.6
spec, while the real implementation stays under ``src/eval/`` to
match the project's ``src/`` layout.
"""

from __future__ import annotations

from src.eval.run import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    app()