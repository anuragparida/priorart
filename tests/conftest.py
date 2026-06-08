"""Test fixtures for the priorart test suite.

The DB-backed tests need a real Postgres+pgvector instance (we use
the running docker-compose service). To keep the live corpus
untouched, every test gets a per-test schema: ``test_<name>_<rand>``.
The schema is dropped on teardown.

For unit tests that don't touch the DB (chunking, embedder-shape),
this file's fixtures are inert.
"""

from __future__ import annotations

import os
import random
import string
from typing import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.config import DATABASE_URL
from src.data.db import init_schema
from src.data.models import Base


def _random_schema() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"test_{suffix}"


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    """Yield a SQLAlchemy engine bound to a per-test Postgres schema.

    On enter: create schema, set search_path on every connection,
    create tables + HNSW index in the test schema. On exit: drop
    schema. Yields a plain Engine — the test is responsible for
    sessions, transactions, etc.

    Why we set search_path on the *connection* (via event listener)
    rather than in a transaction
    ---------------------------------------------------------------
    ``SET search_path`` inside ``engine.begin()`` is per-transaction
    in Postgres — it doesn't survive past COMMIT. We use a
    ``connect`` event so the search_path is set on every new
    connection the engine creates. We also keep ``public`` in the
    path so the ``vector`` type installed by the pgvector extension
    is still findable.

    Why we create tables inside the test schema with a connection
    that has the search_path set
    -----------------------------------------------------------
    ``Base.metadata.create_all(engine)`` does not honour the
    ``connect`` event for the connection it uses internally to
    emit ``CREATE TABLE`` — the tables end up in ``public`` and
    the test would see live-corpus rows. We work around this by
    opening an explicit connection with the search_path set, then
    running ``create_all`` against that connection.
    """
    schema = _random_schema()
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, connection_record):  # noqa: ANN001
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}", public')

    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        with engine.begin() as conn:
            # Extension install is idempotent and global.
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create tables inside the test schema. We use the engine
        # form (``create_all(engine)``) — passing a raw connection
        # here would emit DDL on whatever connection happens to be
        # in the pool, but the engine form honours the search_path
        # we set on every new connection via the connect event.
        # This was a real bug discovered the hard way: with
        # ``create_all(conn)`` the tables were created in ``public``
        # and tests saw live-corpus rows.
        Base.metadata.create_all(engine, checkfirst=False)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_company_embeddings_embedding_hnsw "
                    "ON company_embeddings "
                    "USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                )
            )

        yield engine
    finally:
        try:
            with engine.begin() as conn:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            engine.dispose()


@pytest.fixture
def pg_session(pg_engine: Engine) -> Iterator[Session]:
    """Yield a Session bound to the test schema. Closes on teardown."""
    with Session(bind=pg_engine) as session:
        yield session
