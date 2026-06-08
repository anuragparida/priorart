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

    On enter: create schema, set search_path, run init_schema. On
    exit: drop schema. Yields a plain Engine — the test is
    responsible for sessions, transactions, etc.
    """
    schema = _random_schema()
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            conn.execute(text(f'SET search_path TO "{schema}"'))

        # init_schema runs CREATE EXTENSION (which is global) +
        # create_all (which respects search_path). After the SET
        # above, create_all() creates tables in the test schema.
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            # Inline the HNSW DDL with the right search_path
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
        # Best-effort drop. The schema may already be gone if a test
        # truncated it explicitly.
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
