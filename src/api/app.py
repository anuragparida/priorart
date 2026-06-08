"""FastAPI app entrypoint.

Phase 1.3: ``/healthz`` returns the real ``corpus_count`` from
``company_embeddings`` once ingest has run. The schema is in place
(Phase 1.3) and the search / analyze routes land in Phase 1.4 / 1.8.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, status
from pydantic import BaseModel
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config import DATABASE_URL, EMBEDDING_MODEL
from src.data.models import CompanyEmbedding

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PriorArt",
    description=(
        "Startup-idea deduplication & competitor-research service. "
        "Takes a free-text idea, returns ranked similar YC launches + "
        "structured comparison + market-scope signal."
    ),
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_engine() -> Engine:
    """One engine per process. Cheap to create, fine for Phase 1.

    Phase 2 should swap this for a connection pool sized to the
    Temporal worker's expected concurrency.
    """
    return create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    status: str
    db: str
    model: str
    corpus_count: int | None  # None only when the table is missing or unreadable


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _count_corpus(engine: Engine) -> int | None:
    """Count rows in ``company_embeddings``.

    Returns None when the table doesn't exist yet (Phase 1.3
    pre-ingest) or when the DB is unreachable. The API treats
    None and 0 differently: 0 means "ingest has run on an empty
    snapshot", None means "ingest hasn't been wired up".
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(select(func.count()).select_from(CompanyEmbedding))
            return int(result.scalar_one())
    except SQLAlchemyError:
        return None


@app.get("/healthz", response_model=HealthStatus, status_code=status.HTTP_200_OK)
def healthz(engine: EngineDep) -> HealthStatus:
    """Liveness + dependency check.

    Returns 200 only when postgres is reachable. ``corpus_count`` is
    the row count of ``company_embeddings`` — the number of embedded
    chunks in the index. ``None`` means the table doesn't exist yet
    (run the ingest pipeline).
    """
    db_status = "ok"
    corpus_count: int | None = None
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        corpus_count = _count_corpus(engine)
    except SQLAlchemyError:
        db_status = "down"

    overall = "ok" if db_status == "ok" else "degraded"

    return HealthStatus(
        status=overall,
        db=db_status,
        model=EMBEDDING_MODEL,
        corpus_count=corpus_count,
    )
