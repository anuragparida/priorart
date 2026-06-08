"""Database engine + session helpers.

One engine per process. The ``init_schema`` helper creates the
``pgvector`` extension (must be done outside a transaction) and runs
``Base.metadata.create_all`` + the HNSW index DDL.

Why a separate ``db.py`` from ``models.py``
-------------------------------------------
Keeping engine construction out of the model module means tests can
import the models without triggering engine creation (which requires
the DATABASE_URL env var to be set). The CLI and the API both call
:func:`get_engine` lazily — so importing the modules for inspection
or schema introspection is free.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL
from src.data.models import Base, HNSW_INDEX_SQL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def get_engine(url: str | None = None) -> Engine:
    """Return a SQLAlchemy engine.

    ``pool_pre_ping=True`` recycles dead connections — important when
    the API and ingest both run on the same Postgres container and
    that container restarts. ``future=True`` is the SQLAlchemy 2.x
    default and is set explicitly for clarity.
    """
    return create_engine(
        url or DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a sessionmaker bound to ``engine``."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Context manager that commits on success, rolls back on error.

    Use this from CLI scripts and short-lived workers. The API uses
    ``Depends(get_session)`` instead — see ``src/api/app.py``.
    """
    SessionLocal = session_factory(engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def ensure_pgvector_extension(engine: Engine) -> None:
    """Create the pgvector extension if missing.

    The CREATE EXTENSION statement can't run inside a transaction
    block, so we use ``AUTOCOMMIT`` for this single statement. The
    pgvector image (``pgvector/pgvector:pg16``) ships with the
    extension files installed, but the extension must still be
    enabled per-database.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except ProgrammingError as e:
            # The extension files aren't installed in this image. The
            # pgvector/pgvector:pg16 image should not hit this; bail
            # loud if it does so we don't silently lose vector
            # semantics.
            raise RuntimeError(
                "pgvector extension is not installed in this Postgres image. "
                "Use pgvector/pgvector:pg16 (see docker-compose.yml)."
            ) from e


def init_schema(engine: Engine) -> None:
    """Create the extension, tables, and the HNSW index.

    Idempotent. Safe to run on every ingest — a fresh database is
    brought to a queryable state in a single call.
    """
    ensure_pgvector_extension(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(HNSW_INDEX_SQL))
    logger.info("schema initialised: pgvector extension + companies + company_embeddings + hnsw")
