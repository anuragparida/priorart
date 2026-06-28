"""Shared FastAPI dependencies.

This module exists so ``/search`` and ``/ideas/analyze`` can both
depend on the same ``get_engine`` factory without one of them
having to import from ``app.py`` (which would create a circular
import — ``app.py`` imports the route modules, and the route
modules import the dependency factory).

The factory itself is the one ``app.py`` already used; it just
lives here now, with the route module importing it. ``app.py`` is
updated to re-export it so existing tests that monkeypatch
``app.get_engine`` still work.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL


def get_engine() -> Engine:
    """One engine per process. Cheap to create, fine for Phase 1.

    Phase 2 should swap this for a connection pool sized to the
    Temporal worker's expected concurrency.
    """
    return create_engine(DATABASE_URL, pool_pre_ping=True, future=True)