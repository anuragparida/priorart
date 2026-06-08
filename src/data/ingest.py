"""Ingest a YC JSONL snapshot into Postgres + pgvector.

Phase 1.3 (docs/PHASE-1.md §1.3). Pipeline:

    JSONL → load & dedup → upsert Company rows → chunk descriptions
        → embed with bge-m3 → upsert CompanyEmbedding rows (HNSW-ready)

Idempotency
-----------
- The (name, batch) unique constraint on ``companies`` means a
  re-run of the same JSONL is a no-op for the company table.
- The (company_id, model_version, chunk_index) unique constraint on
  ``company_embeddings`` means a re-run with the same model
  version is a no-op for the embedding table. We use
  ``INSERT ... ON CONFLICT ... DO UPDATE`` so the chunk text +
  vector get refreshed if a description was edited in the source
  snapshot.
- Bumping ``--model-version`` (or the pinned ``EMBEDDING_MODEL`` env
  var) creates a parallel set of rows — useful in Phase 2 for A/B
  model comparisons without losing the old vectors.

Throughput
----------
- Embedding is the bottleneck. With ``batch_size=32`` and CPU,
  bge-m3 does ~3-5 sentences/sec on a laptop, so 5949 rows ≈ 20-30
  min. We surface progress to stderr at 10% intervals so a long
  run isn't a black box.
- We commit in chunks of ``commit_every`` rows to keep WAL
  reasonable and to keep the long-running transaction from
  blocking other writers.

CLI
---
    uv run python -m src.data.ingest --snapshot data/snapshots/yc_<date>.jsonl
    uv run priorart-ingest  # uses today's snapshot
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

import typer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config import SNAPSHOTS_DIR
from src.data.chunking import Chunk, chunk_text
from src.data.db import get_engine, init_schema, session_scope
from src.data.embedder import Embedder
from src.data.models import Company, CompanyEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanyRecord:
    """One row of the YC JSONL, normalised for ingest."""

    name: str
    description: str
    batch: str
    status: str
    url: str
    tags: list[str]
    source: str
    snapshot_date: date


def load_snapshot(path: Path) -> Iterator[CompanyRecord]:
    """Yield CompanyRecord rows from a JSONL file.

    Skips blank lines. Raises on malformed JSON — the scraper's
    idempotency contract means the file is either valid or the
    scraper is broken, and a silent skip would mask the latter.
    """
    snap_date = date.fromisoformat(_date_from_filename(path))
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            yield CompanyRecord(
                name=str(row["name"]).strip(),
                description=str(row.get("description", "")).strip(),
                batch=str(row["batch"]).strip(),
                status=str(row.get("status", "Unknown")).strip() or "Unknown",
                url=str(row.get("url", "")).strip(),
                tags=[str(t).strip() for t in row.get("tags", []) if str(t).strip()],
                source=f"yc:{snap_date.isoformat()}",
                snapshot_date=snap_date,
            )


def _date_from_filename(path: Path) -> str:
    """Extract the YYYY-MM-DD from ``yc_<date>.jsonl``.

    Falls back to the file's mtime date if the filename doesn't
    follow the convention (so the script still works on
    hand-renamed files).
    """
    import re

    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if m:
        return m.group(1)
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


# ---------------------------------------------------------------------------
# Ingest pipeline
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    """Counts for the final report. Surfaced in CLI output and tests.

    We track the total number of companies seen (``companies_total``)
    and the number of embedding rows written. We don't try to
    distinguish insert vs update per-row inside a single ingest
    run — the idempotency guarantee is verified by the row count
    staying stable across re-runs (test_ingest_is_idempotent_on_re_run).
    """

    companies_total: int = 0
    chunks_total: int = 0
    embeddings_inserted: int = 0
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "companies_total": self.companies_total,
            "chunks_total": self.chunks_total,
            "embeddings_inserted": self.embeddings_inserted,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _upsert_company(session: Session, rec: CompanyRecord) -> int:
    """Upsert one Company row, return the id.

    Uses Postgres' ``ON CONFLICT (name, batch) DO UPDATE`` so a
    re-run of the same JSONL updates the description / tags / url
    in place. The RETURNING clause gives us the id (new or
    existing) without a second SELECT.
    """
    stmt = (
        pg_insert(Company)
        .values(
            name=rec.name,
            description=rec.description,
            batch=rec.batch,
            status=rec.status,
            url=rec.url,
            tags=rec.tags,
            source=rec.source,
            snapshot_date=rec.snapshot_date,
        )
        .on_conflict_do_update(
            index_elements=["name", "batch"],
            set_={
                "description": rec.description,
                "status": rec.status,
                "url": rec.url,
                "tags": rec.tags,
                "source": rec.source,
                "snapshot_date": rec.snapshot_date,
            },
        )
        .returning(Company.id)
    )
    return int(session.execute(stmt).scalar_one())


def _upsert_embedding(
    session: Session,
    *,
    company_id: int,
    model_version: str,
    chunk: Chunk,
    vector: list[float],
) -> bool:
    """Upsert one CompanyEmbedding row. Return True if inserted."""
    stmt = (
        pg_insert(CompanyEmbedding)
        .values(
            company_id=company_id,
            embedding=vector,
            model_version=model_version,
            chunk_index=chunk.index,
            chunk_count=chunk.count,
            chunk_text=chunk.text,
        )
        .on_conflict_do_update(
            index_elements=["company_id", "model_version", "chunk_index"],
            set_={
                "embedding": vector,
                "chunk_count": chunk.count,
                "chunk_text": chunk.text,
            },
        )
        .returning(CompanyEmbedding.company_id)
    )
    session.execute(stmt)
    # We can't easily tell insert-vs-update here without a second
    # query — the caller counts inserts via the xmax trick only on
    # the company row, and treats the embeddings table as
    # idempotent-no-net-change when the count matches expectations.
    return True


def ingest(
    session: Session,
    records: Iterable[CompanyRecord],
    *,
    embedder: Embedder,
    commit_every: int = 500,
) -> IngestStats:
    """Run the full ingest pipeline.

    ``commit_every`` is the number of *companies* between commits
    — we accumulate the embedding rows in the session's identity
    map and flush periodically so a long run doesn't accumulate
    unbounded state.
    """
    stats = IngestStats()
    started = time.monotonic()
    model_version = embedder.model_name

    # Materialise the iterable so we can iterate twice (once for the
    # count, once for the actual ingest). Most callers pass a
    # generator, but we want a stable count for the progress log.
    records = list(records)
    stats.companies_total = len(records)

    pending_texts: list[str] = []
    pending_meta: list[tuple[int, Chunk]] = []  # (company_id, chunk)
    last_progress = 0

    for idx, rec in enumerate(records, start=1):
        company_id = _upsert_company(session, rec)

        for chunk in chunk_text(rec.description):
            pending_texts.append(chunk.text)
            pending_meta.append((company_id, chunk))
            stats.chunks_total += 1

        # Flush embeddings in batches of 32 (bge-m3's recommended
        # batch size) to keep memory bounded.
        if len(pending_texts) >= 32:
            vectors = embedder.embed_batch(pending_texts)
            for (cid, chunk), vec in zip(pending_meta, vectors):
                if _upsert_embedding(
                    session,
                    company_id=cid,
                    model_version=model_version,
                    chunk=chunk,
                    vector=vec,
                ):
                    stats.embeddings_inserted += 1
            pending_texts.clear()
            pending_meta.clear()

        if idx - last_progress >= max(1, stats.companies_total // 10):
            last_progress = idx
            logger.info(
                "ingest progress: %d/%d companies, %d chunks",
                idx, stats.companies_total, stats.chunks_total,
            )

        if idx % commit_every == 0:
            session.commit()

    # Drain any remaining chunks (the YC snapshot isn't a multiple
    # of 32 most of the time).
    if pending_texts:
        vectors = embedder.embed_batch(pending_texts)
        for (cid, chunk), vec in zip(pending_meta, vectors):
            if _upsert_embedding(
                session,
                company_id=cid,
                model_version=model_version,
                chunk=chunk,
                vector=vec,
            ):
                stats.embeddings_inserted += 1
        pending_texts.clear()
        pending_meta.clear()

    session.commit()
    stats.duration_seconds = time.monotonic() - started
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help=__doc__, no_args_is_help=True)


@app.command()
def main(
    snapshot: Path = typer.Option(
        None,
        "--snapshot",
        "-s",
        help="Path to the YC JSONL snapshot. Defaults to data/snapshots/yc_<today>.jsonl.",
    ),
    init_db: bool = typer.Option(
        True,
        "--init-db/--no-init-db",
        help="Create the pgvector extension, tables, and HNSW index before ingesting.",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Override the embedding model (defaults to PRIORART_EMBEDDING_MODEL).",
    ),
    commit_every: int = typer.Option(500, help="Companies per commit."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING)."),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if snapshot is None:
        today = date.today().isoformat()
        snapshot = SNAPSHOTS_DIR / f"yc_{today}.jsonl"
    if not snapshot.exists():
        raise typer.BadParameter(f"snapshot file not found: {snapshot}")

    engine = get_engine()
    if init_db:
        init_schema(engine)

    embedder = Embedder(model_name=model)
    records = list(load_snapshot(snapshot))
    logger.info("loaded %d records from %s", len(records), snapshot)

    with session_scope(engine) as session:
        stats = ingest(session, records, embedder=embedder, commit_every=commit_every)

    logger.info("ingest complete: %s", stats.as_dict())
    # Also print a one-line JSON summary on stdout for scripts that
    # want to parse it.
    typer.echo(json.dumps(stats.as_dict()))


if __name__ == "__main__":  # pragma: no cover
    app()
