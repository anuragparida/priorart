"""Idempotent schema migrations for PriorArt.

Phase 2.7 (docs/PHASE-2.md §2.7) changed the ``companies`` table:

- Old dedup key: ``UNIQUE (name, batch)``.
- New dedup key: ``UNIQUE (source, external_id)``.
- ``source`` format changed from ``"yc:2026-06-08"`` to just
  ``"yc"`` (the date moved to ``snapshot_date``).
- New column ``external_id`` holds each source's natural primary key
  (YC url, PH id, HN object_id).

This module owns the SQL ALTERs. It is idempotent — running it twice
is a no-op. New phases should append their migration here, named
``migrate_phase_N_M``.

Why no Alembic
--------------
Alembic is in the lockfile (transitive via SQLAlchemy) but the repo
has no ``alembic.ini`` / ``alembic/`` directory yet — Phase 1
shipped without it. Adding the entire migration framework for a
single schema change is more risk than reward. The pattern below is
``SELECT ... FROM information_schema`` to check, ``ALTER TABLE IF
NOT EXISTS`` style with explicit ``IF NOT EXISTS`` where Postgres
supports it (and conditional branches where it does not). If/when
a second migration lands we can lift this into a proper Alembic
setup without rewriting the SQL itself.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Phase 2.7 — companies: source normalisation + external_id
# -----------------------------------------------------------------------


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    ).first()
    return row is not None


def _constraint_exists(conn, constraint: str) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM pg_constraint WHERE conname = :n"),
        {"n": constraint},
    ).first()
    return row is not None


def migrate_phase_2_7(engine: Engine) -> dict:
    """Phase 2.7 schema migration.

    Steps
    -----
    1. ``ALTER TABLE companies ADD COLUMN external_id VARCHAR(128)``
       (idempotent via ``IF NOT EXISTS`` — Postgres 9.6+).
    2. Backfill ``external_id`` for existing YC rows from ``url``
       (the canonical YC directory slug — last URL path segment, or
       the full URL if extraction fails). For YC legacy rows, ``url``
       is the stable YC id; for PH and HN rows that don't have
       ``external_id`` yet, we leave it NULL and let the corpus
       build pipeline write it.
    3. Backfill ``source`` from ``"yc:YYYY-MM-DD"`` to just
       ``"yc"``. YC rows with the old format get normalised.
    4. Drop the old unique constraint ``uq_companies_name_batch``.
    5. Add the new unique constraint
       ``uq_companies_source_external_id``.
    6. Add index ``ix_companies_source``.

    Returns a dict of counters for the CLI report.
    """
    stats = {
        "external_id_backfilled": 0,
        "source_normalised": 0,
        "old_constraint_dropped": False,
        "new_constraint_added": False,
        "index_added": False,
    }

    with engine.begin() as conn:
        # 1. Add external_id column (idempotent).
        if not _column_exists(conn, "companies", "external_id"):
            conn.execute(
                text("ALTER TABLE companies ADD COLUMN external_id VARCHAR(128)")
            )
            logger.info("migrate: added companies.external_id column")

        # 2. Backfill external_id for YC legacy rows that lack it.
        # Use url as the external_id (YC's slug lives at the end of
        # https://www.ycombinator.com/companies/<slug>). For rows
        # where url is empty, fall back to ``name``.
        result = conn.execute(
            text(
                """
                UPDATE companies
                SET external_id = COALESCE(
                    NULLIF(url, ''),
                    'name:' || name
                )
                WHERE external_id IS NULL OR external_id = ''
                """
            )
        )
        stats["external_id_backfilled"] = result.rowcount or 0

        # 3. Normalise source: strip the ":YYYY-MM-DD" suffix.
        result = conn.execute(
            text(
                """
                UPDATE companies
                SET source = split_part(source, ':', 1)
                WHERE source LIKE '%:%'
                """
            )
        )
        stats["source_normalised"] = result.rowcount or 0

        # 4. Drop old constraint.
        if _constraint_exists(conn, "uq_companies_name_batch"):
            conn.execute(
                text("ALTER TABLE companies DROP CONSTRAINT uq_companies_name_batch")
            )
            stats["old_constraint_dropped"] = True
            logger.info("migrate: dropped uq_companies_name_batch")

        # 5. Add new constraint (idempotent guard).
        if not _constraint_exists(conn, "uq_companies_source_external_id"):
            conn.execute(
                text(
                    "ALTER TABLE companies "
                    "ADD CONSTRAINT uq_companies_source_external_id "
                    "UNIQUE (source, external_id)"
                )
            )
            stats["new_constraint_added"] = True
            logger.info("migrate: added uq_companies_source_external_id")

        # 6. Add source index (IF NOT EXISTS makes this idempotent).
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_companies_source "
                "ON companies (source)"
            )
        )
        stats["index_added"] = True

    return stats


def run_all(engine: Engine) -> dict:
    """Run every migration in order. Idempotent across re-runs.

    New migrations should be appended here. Each function must be
    idempotent on its own.
    """
    combined: dict = {}
    combined["phase_2_7"] = migrate_phase_2_7(engine)
    return combined


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint — ``python -m src.data.migrate``."""
    import json
    import sys

    from src.data.db import get_engine

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    engine = get_engine()
    stats = run_all(engine)
    # Emit a one-line JSON summary for scripts that want to parse it.
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()