"""FastAPI app entrypoint.

Phase 1.1: minimal app with /healthz. The real routes (POST /search,
POST /ideas/analyze) land in 1.4 and 1.8.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, status
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config import DATABASE_URL, EMBEDDING_MODEL

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
    version="0.1.0",
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
    corpus_count: int | None  # None until ingest (Phase 1.3) lands


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthStatus, status_code=status.HTTP_200_OK)
def healthz(engine: EngineDep) -> HealthStatus:
    """Liveness + dependency check.

    Returns 200 only when postgres is reachable. Corpus count is None
    until the YC scraper + ingest (1.2, 1.3) lands.
    """
    db_status = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        db_status = "down"

    overall = "ok" if db_status == "ok" else "degraded"

    return HealthStatus(
        status=overall,
        db=db_status,
        model=EMBEDDING_MODEL,
        corpus_count=None,
    )
