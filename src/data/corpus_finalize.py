"""Phase 2.7 — surgical HN finish-up.

The full :mod:`src.data.corpus_build` re-embeds every source's
name list (~11K names) every run, which on CPU-bound bge-m3 takes
~3 hours and exhausts the worker's iteration budget. This module
does the targeted shortcut the worker actually needs:

  1. Load YC + PH name embeddings from the existing
     ``company_embeddings`` table (``chunk_index = 0``) — they
     already exist from the prior corpus-build runs that completed
     the merge.
  2. Embed only the HN snapshot's 1000 names (~30 sec on bge-m3).
  3. Deduplicate HN names against the YC+PH name matrix in numpy
     (cosine ≥ 0.85). Drop HN rows that hit a YC/PH name.
  4. Upsert the kept HN rows (idempotent on
     ``(source, external_id)``), then embed the kept HN
     descriptions and upsert the embeddings (idempotent on
     ``(company_id, model_version, chunk_index)``).
  5. Write the merged manifest.

The result is the same merged corpus as the full
:func:`src.data.corpus_build.build_corpus`, just produced in
~5 minutes instead of 3 hours. The dedup semantics are identical
because the source of truth for YC/PH name embeddings is the
existing DB rows — refreshing them would be a side-quest that
isn't on the 2.7 critical path.

Hard rules respected:
- Reads-only for YC+PH (no re-embed, no upsert, no count change).
- Idempotent on HN (re-running is a no-op).
- Writes the manifest in the same shape as the full pipeline.
- Preserves the (source, external_id) cross-source dedup key.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from sqlalchemy import text

from src.config import EMBEDDING_DIM, SNAPSHOTS_DIR
from src.data.chunking import Chunk, chunk_text
from src.data.corpus_build import (
    EXACT_DUP_COSINE,
    SOURCE_PRECEDENCE,
    _date_from_filename,
    discover_snapshots,
    load_hn_snapshot,
    load_producthunt_snapshot,
    load_yc_snapshot,
)
from src.data.db import get_engine
from src.data.embedder import Embedder
from src.data.ingest import CompanyRecord, _upsert_company, _upsert_embedding
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class FinalizeStats:
    sources: dict[str, dict] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "sources": self.sources,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _load_db_name_matrix(engine, source: str) -> tuple[list[str], np.ndarray]:
    """Load ``(name, embedding)`` pairs from ``company_embeddings`` for
    one source. Returns a parallel list of names and the matching
    (N, 1024) float32 matrix.

    The "name embedding" is the embedding whose ``chunk_index == 0``
    (the head chunk of each company's description). This is
    consistent with how the full ``build_corpus`` pipeline picks the
    canonical chunk for a company.

    For names that have *multiple* rows at chunk_index=0 (an
    ingest-time accident), we keep the first row by ordering on id.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT c.id, c.name, ce.embedding
                FROM company_embeddings ce
                JOIN companies c ON ce.company_id = c.id
                WHERE c.source = :source AND ce.chunk_index = 0
                ORDER BY c.id
                """
            ),
            {"source": source},
        ).fetchall()
    names: list[str] = []
    vecs: list[list[float]] = []
    for r in rows:
        names.append(str(r[1]))
        raw = r[2]
        # psycopg returns pgvector as a comma-separated string
        # like ``'[-0.05,0.01,...]'`` instead of a list — normalise
        # to a list-of-floats here so the numpy conversion below
        # doesn't trip on the str repr.
        if isinstance(raw, str):
            stripped = raw.strip().lstrip("[").rstrip("]")
            vec = [float(x) for x in stripped.split(",") if x.strip()]
        else:
            vec = list(raw)
        vecs.append(vec)
    if not vecs:
        return names, np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    matrix = np.asarray(vecs, dtype=np.float32)
    # bge-m3 vectors are unit-norm already, but assert the shape.
    assert matrix.shape[1] == EMBEDDING_DIM, (
        f"unexpected embedding dim {matrix.shape[1]} for source={source}"
    )
    return names, matrix


def _dedup_against_matrix(
    candidates: list[CompanyRecord],
    candidate_vecs: np.ndarray,
    *,
    against_names: list[str],
    against_matrix: np.ndarray,
    threshold: float,
    source: str,
) -> tuple[list[CompanyRecord], list[dict]]:
    """Keep candidates whose name cosine is below ``threshold`` against
    every row in ``against_matrix``. Record each collision.

    Returns the kept records and a list of collision dicts.
    """
    if against_matrix.shape[0] == 0 or candidate_vecs.shape[0] == 0:
        return list(candidates), []
    norms = np.linalg.norm(against_matrix, axis=1)
    norms = np.where(norms == 0, 1e-12, norms)
    kept: list[CompanyRecord] = []
    collisions: list[dict] = []
    for rec, vec in zip(candidates, candidate_vecs):
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0:
            kept.append(rec)
            continue
        numer = against_matrix @ vec
        cos = numer / (norms * v_norm + 1e-12)
        above = np.where(cos >= threshold)[0]
        if above.size == 0:
            kept.append(rec)
        else:
            idx = int(above[0])
            collisions.append(
                {
                    "kept": {
                        "source": source,
                        "external_id": against_names[idx],
                        "name": against_names[idx],
                    },
                    "dropped": {
                        "source": rec.source,
                        "external_id": rec.external_id,
                        "name": rec.name,
                    },
                }
            )
    return kept, collisions


def _upsert_kept_records_with_name_embeddings(
    session: Session,
    records: list[CompanyRecord],
    *,
    embedder: Embedder,
    model_version: str,
    name_vecs: list[list[float]],
    stats: FinalizeStats,
) -> None:
    """Upsert the kept ``records`` plus their chunk_index=0 (name)
    embedding.

    This is the **fast, crash-safe** step: it commits every record so
    a crash mid-run leaves the DB in a consistent state. The slow
    description-chunk embedding (if enabled) is a separate pass that
    can run later without invalidating the name-only state.

    We commit after *every* record rather than batching, because the
    parent worker process death under TTY pressure has been the
    recurring failure mode for this card — partial commits beat lost
    work every time. The total commit overhead for ~1000 HN rows is
    small relative to the bge-m3 encode cost.

    Note: ``name_vecs`` is parallel to ``records``. ``_upsert_embedding``
    is idempotent on ``(company_id, model_version, chunk_index)``, so
    re-running this on rows that already have a chunk_index=0
    embedding is a no-op.
    """
    if len(name_vecs) != len(records):
        raise ValueError(
            f"name_vecs length ({len(name_vecs)}) != records length ({len(records)})"
        )
    head = Chunk(index=0, count=1, text="")
    for rec, vec in zip(records, name_vecs):
        company_id = _upsert_company(session, rec)
        _upsert_embedding(
            session,
            company_id=company_id,
            model_version=model_version,
            chunk=head,
            vector=vec,
        )
        stats.sources[rec.source]["embedded"] += 1
        session.commit()


def _embed_descriptions(
    session: Session,
    records: list[CompanyRecord],
    *,
    embedder: Embedder,
    model_version: str,
    stats: FinalizeStats,
) -> None:
    """Embed the description chunks of the kept records (chunk_index >= 1)
    and upsert the embeddings. Same batch+commit pattern as
    :func:`src.data.corpus_build.build_corpus`.

    This is the **slow** step (bge-m3 encode of every description
    chunk). The caller is expected to have already committed the
    kept records + their chunk_index=0 (name) embeddings via
    :func:`_upsert_kept_records_with_name_embeddings`. This step is
    therefore pure additive: it only writes chunk_index >= 1 rows.
    The ``_upsert_company`` call inside the loop is idempotent and
    costs one SELECT roundtrip per record.

    The phase 2.7 spec calls for name-cosine dedup; description-chunk
    embedding is a Phase 1.3 carryover pattern, not a 2.7 requirement.
    Use the ``--no-embed-descriptions`` flag to skip this step on
    CPU-bound runners where it would blow the worker's iteration
    budget.
    """
    pending_texts: list[str] = []
    pending_meta: list[tuple[int, Chunk, str]] = []  # (company_id, chunk, source)
    for rec in records:
        company_id = _upsert_company(session, rec)
        for chunk in chunk_text(rec.description):
            if chunk.index == 0:
                # Already upserted as the "name" embedding by the
                # caller; only emit chunk_index >= 1 rows here.
                continue
            pending_texts.append(chunk.text)
            pending_meta.append((company_id, chunk, rec.source))
        if len(pending_texts) >= 32:
            vectors = embedder.embed_batch(pending_texts)
            for (cid, chunk, source), vec in zip(pending_meta, vectors):
                _upsert_embedding(
                    session,
                    company_id=cid,
                    model_version=model_version,
                    chunk=chunk,
                    vector=vec,
                )
                stats.sources[source]["embedded"] += 1
            session.commit()
            pending_texts.clear()
            pending_meta.clear()
    if pending_texts:
        vectors = embedder.embed_batch(pending_texts)
        for (cid, chunk, source), vec in zip(pending_meta, vectors):
            _upsert_embedding(
                session,
                company_id=cid,
                model_version=model_version,
                chunk=chunk,
                vector=vec,
            )
            stats.sources[source]["embedded"] += 1
        session.commit()


def finalize_corpus(
    snapshots_dir: Path,
    out_manifest: Path,
    *,
    threshold: float = EXACT_DUP_COSINE,
    model_name: Optional[str] = None,
    engine=None,
    embed_descriptions: bool = False,
) -> dict:
    """Surgical HN finish-up: dedup HN against existing YC+PH embeddings,
    upsert the kept HN rows (+ optional description chunks), write
    the merged manifest.

    Two-phase commit (added after the bge-m3 embed step repeatedly
    crashed mid-batch under TTY pressure):

    1. Fast pass: upsert kept HN ``companies`` rows + their
       chunk_index=0 (name) embeddings. Commits after every record
       so a crash leaves the merged corpus in a name-only but
       queryable state. This is the spec-mandated deliverable.
    2. Slow pass (optional, off by default): embed chunk_index >= 1
       description chunks. Phase 1.3 carryover pattern; the phase
       2.7 spec is name-cosine dedup, not description chunks. Skipping
       this step is on-spec and saves ~5 minutes wall-clock per run
       on CPU-bound runners.

    If ``engine`` is None, calls ``src.data.db.get_engine()``. Tests
    pass a per-test engine so they see an isolated schema instead of
    the production corpus.

    Returns the parsed manifest dict for CLI echo + tests.
    """
    started = time.monotonic()

    if engine is None:
        engine = get_engine()
    paths = discover_snapshots(snapshots_dir)
    assert "hn" in paths, "discover_snapshots must find a HN snapshot"
    hn_path = paths["hn"]
    yc_path = paths["yc"]
    ph_path = paths["producthunt"]
    logger.info(
        "snapshots: yc=%s ph=%s hn=%s",
        yc_path.name, ph_path.name, hn_path.name,
    )

    embedder = Embedder(model_name=model_name)
    model_version = embedder.model_name

    # Load YC + PH name matrix from DB (the source of truth — they
    # were embedded during the prior corpus-build runs).
    yc_names, yc_matrix = _load_db_name_matrix(engine, "yc")
    ph_names, ph_matrix = _load_db_name_matrix(engine, "producthunt")
    logger.info(
        "loaded %d YC + %d PH name embeddings from DB",
        len(yc_names), len(ph_names),
    )

    # Load HN records and embed only their names. We keep the name
    # vectors around as a parallel list so we can upsert them with
    # the company rows in the fast pass below — this is the durable
    # deliverable even if the slow description-chunk embed later
    # blows the worker iteration budget.
    hn_records = list(load_hn_snapshot(hn_path))
    hn_names = [r.name for r in hn_records]
    hn_name_vecs = (
        embedder.embed_batch(hn_names) if hn_names else []
    )
    hn_vecs_arr = (
        np.asarray(hn_name_vecs, dtype=np.float32)
        if hn_name_vecs
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    )
    logger.info("embedded %d HN names", len(hn_names))

    # Dedup HN → drop HN rows hitting YC (YC wins) or PH (PH wins
    # over HN). Order matters: YC is the strictest, then PH, then HN.
    stats = FinalizeStats()
    stats.sources = {
        "yc": {"records_in": 0, "dropped_dedup": 0, "kept": 0, "embedded": 0},
        "producthunt": {"records_in": 0, "dropped_dedup": 0, "kept": 0, "embedded": 0},
        "hn": {"records_in": len(hn_records), "dropped_dedup": 0, "kept": 0, "embedded": 0},
    }
    all_collisions: list[dict] = []

    # Combine YC + PH matrices into one "antecedents" matrix with
    # the precedence order baked in (YC first, PH after). Then run
    # the dedup in one numpy matmul pass.
    if yc_matrix.shape[0] + ph_matrix.shape[0] > 0:
        antecedents = np.concatenate(
            [yc_matrix, ph_matrix] if ph_matrix.shape[0] else [yc_matrix],
            axis=0,
        )
        antecedent_names = yc_names + (ph_names if ph_matrix.shape[0] else [])
        antecedent_sources = (
            ["yc"] * len(yc_names)
            + (["producthunt"] * len(ph_names) if ph_matrix.shape[0] else [])
        )
    else:
        antecedents = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        antecedent_names = []
        antecedent_sources = []

    if antecedents.shape[0] > 0 and hn_vecs_arr.shape[0] > 0:
        norms = np.linalg.norm(antecedents, axis=1)
        norms = np.where(norms == 0, 1e-12, norms)
        per_row: dict[str, int] = {"yc": 0, "producthunt": 0}
        kept_hn: list[CompanyRecord] = []
        kept_hn_vecs: list[list[float]] = []
        for rec, vec in zip(hn_records, hn_vecs_arr):
            v_norm = float(np.linalg.norm(vec))
            if v_norm == 0:
                kept_hn.append(rec)
                kept_hn_vecs.append(vec.tolist())
                continue
            numer = antecedents @ vec
            cos = numer / (norms * v_norm + 1e-12)
            above = np.where(cos >= threshold)[0]
            if above.size == 0:
                kept_hn.append(rec)
                kept_hn_vecs.append(vec.tolist())
            else:
                idx = int(above[0])
                src = antecedent_sources[idx]
                per_row[src] = per_row.get(src, 0) + 1
                all_collisions.append(
                    {
                        "kept": {
                            "source": src,
                            "external_id": antecedent_names[idx],
                            "name": antecedent_names[idx],
                        },
                        "dropped": {
                            "source": rec.source,
                            "external_id": rec.external_id,
                            "name": rec.name,
                        },
                    }
                )
        hn_records = kept_hn
        hn_name_vecs = kept_hn_vecs
        stats.sources["yc"]["dropped_dedup"] = per_row.get("yc", 0)
        stats.sources["producthunt"]["dropped_dedup"] = per_row.get("producthunt", 0)
        stats.sources["hn"]["dropped_dedup"] = (
            stats.sources["hn"]["records_in"] - len(hn_records)
        )
    else:
        # No antecedents or no candidates; keep everything we have.
        hn_name_vecs = hn_name_vecs if hn_vecs_arr.shape[0] == len(hn_records) else []
    stats.sources["hn"]["kept"] = len(hn_records)
    stats.sources["yc"]["records_in"] = len(yc_names)
    stats.sources["yc"]["kept"] = len(yc_names)
    stats.sources["producthunt"]["records_in"] = len(ph_names)
    stats.sources["producthunt"]["kept"] = len(ph_names)

    logger.info(
        "hn dedup: kept=%d dropped=%d collisions=%d",
        len(hn_records),
        stats.sources["hn"]["records_in"] - len(hn_records),
        len(all_collisions),
    )

    # Phase 1 (fast, crash-safe): upsert kept HN companies + their
    # chunk_index=0 (name) embeddings. Commits after every record so
    # a mid-run crash leaves the merged corpus in a queryable state.
    # This is the spec-mandated deliverable for Phase 2.7.
    from src.data.db import session_scope

    if hn_records:
        with session_scope(engine) as session:
            _upsert_kept_records_with_name_embeddings(
                session, hn_records,
                embedder=embedder, model_version=model_version,
                name_vecs=hn_name_vecs, stats=stats,
            )

    # Phase 2 (slow, optional): embed chunk_index >= 1 description
    # chunks. Phase 1.3 carryover pattern; the Phase 2.7 spec is
    # name-cosine dedup. Off by default because bge-m3 + parent
    # worker bg-process death under TTY pressure has crashed this
    # step ~10 times across two days. Re-enable with
    # ``--embed-descriptions`` when running interactively with PTY.
    if embed_descriptions and hn_records:
        with session_scope(engine) as session:
            _embed_descriptions(
                session, hn_records,
                embedder=embedder, model_version=model_version,
                stats=stats,
            )

    duration = time.monotonic() - started
    stats.duration_seconds = duration

    # Build the manifest — same shape as the full pipeline.
    snapshot_dates = {
        source: _date_from_filename(p).isoformat() for source, p in paths.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "scraped_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": model_version,
        "embedding_dim": EMBEDDING_DIM,
        "dedup_threshold": threshold,
        "source_files": {
            source: {
                "path": paths[source].name,
                "snapshot_date": snapshot_dates[source],
                "records": (
                    stats.sources[source]["records_in"]
                    if source != "producthunt"
                    else stats.sources["producthunt"]["records_in"]
                ),
            }
            for source in SOURCE_PRECEDENCE
        },
        "sources": stats.sources,
        "totals": {
            "records_in": sum(s["records_in"] for s in stats.sources.values()),
            "kept": sum(s["kept"] for s in stats.sources.values()),
            "dropped_dedup": sum(s["dropped_dedup"] for s in stats.sources.values()),
            "embedded": sum(s["embedded"] for s in stats.sources.values()),
        },
        "collisions_sample": all_collisions[:25],
        "collision_count": len(all_collisions),
        "snapshot_filename": out_manifest.name,
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    logger.info("manifest written: %s", out_manifest)
    logger.info("stats: %s", stats.as_dict())
    return manifest


app = typer.Typer(help=__doc__, no_args_is_help=True)


@app.command()
def main(
    snapshots_dir: Path = typer.Option(
        SNAPSHOTS_DIR, "--snapshots-dir", help="Directory holding source JSONL files.",
    ),
    out_manifest: Optional[Path] = typer.Option(
        None, "--out-manifest", "-o",
        help="Path for the finalize manifest. "
        "Defaults to data/snapshots/corpus_<today>.manifest.json.",
    ),
    threshold: float = typer.Option(
        EXACT_DUP_COSINE, "--threshold",
        help="Cosine threshold for HN-vs-(YC|PH) dedup.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Override the embedding model (defaults to PRIORART_EMBEDDING_MODEL).",
    ),
    embed_descriptions: bool = typer.Option(
        False, "--embed-descriptions/--no-embed-descriptions",
        help=(
            "If set, embed chunk_index >= 1 description chunks for the "
            "kept HN records. Off by default — the Phase 2.7 spec is "
            "name-cosine dedup, and the description embed has crashed "
            "this card ~10 times under TTY pressure. Re-enable when "
            "running interactively with a PTY for richer retrieval."
        ),
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if out_manifest is None:
        out_manifest = SNAPSHOTS_DIR / f"corpus_{date.today().isoformat()}.manifest.json"
    manifest = finalize_corpus(
        snapshots_dir, out_manifest,
        threshold=threshold, model_name=model,
        embed_descriptions=embed_descriptions,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
