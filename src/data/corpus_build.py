"""Cross-source corpus build: merge YC + Product Hunt + HN, dedup, ingest.

Phase 2.7 (docs/PHASE-2.md §2.7). Pipeline:

    yc_*.jsonl            ┐
    producthunt_*.jsonl   ├─→ load & normalise ─→ dedup vs (name, cos≥0.85)
    hn_show_*.jsonl       ┘                          │
                                                   ▼
                          upsert Company (source, external_id)
                                                   │
                                                   ▼
                       chunk descriptions ─→ embed with bge-m3
                                                   │
                                                   ▼
                          upsert CompanyEmbedding (HNSW-ready)
                                                   │
                                                   ▼
                       write data/snapshots/corpus_<date>.manifest.json

Idempotency
-----------
- Each source's snapshot is independent; re-running with the same
  three files is a no-op on the companies and embeddings tables
  (the (source, external_id) and (company_id, model_version,
  chunk_index) unique constraints cover it).
- The dedup step is computed per run from the union of all three
  snapshot files — if a record is dropped from a snapshot, it will
  reappear on the next run. This is the spec's "ingest what's in
  the snapshots, dedup across them" contract.

Dedup precedence
----------------
When two records from different sources collide on name-cosine ≥
0.85, the precedence is:

    1. YC wins ties (most metadata — batch, tags, status, url).
    2. Product Hunt wins over HN (richer description / topics).
    3. HN is the default fallback.

The "winning" record keeps its (source, external_id) and is the
one we embed. The loser's record is **dropped** (not merged) — we
deliberately avoid materialising "combined" rows because the
description fields are schema-mismatched across sources.

CLI
---
    make corpus-build
    # or directly:
    python -m src.data.corpus_build
    python -m src.data.corpus_build --no-embed  # only merge + dedup
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

import typer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config import SNAPSHOTS_DIR
from src.data.chunking import Chunk, chunk_text
from src.data.db import get_engine, init_schema, session_scope
from src.data.embedder import Embedder
from src.data.ingest import CompanyRecord, _upsert_embedding
from src.data.migrate import run_all as run_migrations
from src.data.models import Company, CompanyEmbedding

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Dedup thresholds — pinned here so they're easy to find + tune.
# -----------------------------------------------------------------------

#: Cosine threshold on bge-m3(name) embeddings above which two
#: records from different sources are considered the same product.
#: The Phase 2.5 PH scraper uses the same threshold for its
#: YC-vs-PH dedup band (``EXACT_DUP_COSINE``); we reuse it verbatim
#: so the boundary is consistent across the whole pipeline.
EXACT_DUP_COSINE = 0.85

#: Source precedence when two records collide. Lower index = higher
#: precedence (wins the dedup). YC is the canonical public directory
#: and carries the most metadata, so it's first.
SOURCE_PRECEDENCE = ("yc", "producthunt", "hn")


# -----------------------------------------------------------------------
# Snapshot discovery
# -----------------------------------------------------------------------


_SNAPSHOT_GLOB = re.compile(
    r"^(?P<source>yc|producthunt|hn_show)_(?P<date>\d{4}-\d{2}-\d{2})\.jsonl$"
)


def discover_snapshots(snapshots_dir: Path) -> dict[str, Path]:
    """Return the latest snapshot per source, keyed by source prefix.

    Looks for files matching ``<source>_<date>.jsonl`` in
    ``snapshots_dir``. Picks the most recent date per source.
    Raises ``FileNotFoundError`` if no snapshots are found for any
    of the three sources — the corpus build can't proceed.
    """
    latest: dict[str, tuple[date, Path]] = {}
    for entry in snapshots_dir.iterdir():
        if not entry.is_file():
            continue
        m = _SNAPSHOT_GLOB.match(entry.name)
        if not m:
            continue
        source = m.group("source")
        # Normalise ``hn_show`` → ``hn`` so the manifest field is
        # the source-prefix the rest of the codebase uses.
        canonical = "hn" if source == "hn_show" else source
        d = date.fromisoformat(m.group("date"))
        prev = latest.get(canonical)
        if prev is None or d > prev[0]:
            latest[canonical] = (d, entry)

    if not latest:
        raise FileNotFoundError(
            f"no corpus snapshots found in {snapshots_dir}; "
            f"expected yc_*.jsonl, producthunt_*.jsonl, hn_show_*.jsonl"
        )

    # Require all three sources for a complete merge.
    missing = [s for s in SOURCE_PRECEDENCE if s not in latest]
    if missing:
        raise FileNotFoundError(
            f"missing snapshots for sources: {missing}; "
            f"found: {list(latest.keys())}"
        )

    return {source: path for source, (_, path) in latest.items()}


# -----------------------------------------------------------------------
# Per-source loaders — each yields CompanyRecord rows.
# -----------------------------------------------------------------------


def _date_from_filename(path: Path) -> date:
    """Extract the YYYY-MM-DD from ``<source>_<date>.jsonl``."""
    m = _SNAPSHOT_GLOB.match(path.name)
    if m:
        return date.fromisoformat(m.group("date"))
    # Fall back to mtime — defensive only; the snapshots ship with
    # the date in the filename.
    return date.fromtimestamp(path.stat().st_mtime)


def load_yc_snapshot(path: Path) -> Iterator[CompanyRecord]:
    """Load a YC snapshot — same shape as :mod:`src.data.ingest`.

    YC's natural id is the canonical directory slug in the url.
    """
    snap_date = _date_from_filename(path)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            url = str(row.get("url", "")).strip()
            name = str(row["name"]).strip()
            yield CompanyRecord(
                name=name,
                description=str(row.get("description", "")).strip(),
                batch=str(row.get("batch", "")).strip(),
                status=str(row.get("status", "Unknown")).strip() or "Unknown",
                url=url,
                tags=[str(t).strip() for t in row.get("tags", []) if str(t).strip()],
                source="yc",
                external_id=url or f"name:{name}",
                snapshot_date=snap_date,
            )


def load_producthunt_snapshot(path: Path) -> Iterator[CompanyRecord]:
    """Load a Product Hunt snapshot.

    PH's natural id is ``id`` (an integer-as-string). The snapshot
    row carries ``name``, ``tagline`` (one-liner), and optionally
    ``description`` (Firecrawl-scraped, may be empty for the bulk
    Algolia-fetched rows). We collapse tagline + description into
    a single ``description`` field for embedding, with the tagline
    first so the head of the chunk carries the most-signal text.

    PH has no YC-style batch; we synthesize one from the launch
    year so the column isn't empty.
    """
    snap_date = _date_from_filename(path)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            ph_id = str(row["id"]).strip()
            name = str(row["name"]).strip()
            tagline = str(row.get("tagline", "")).strip()
            description = str(row.get("description", "")).strip()
            # The PH scraper carries topics in addition to the
            # tagline; they make the embedding text richer.
            topics = [
                str(t).strip()
                for t in row.get("topics", [])
                if str(t).strip()
            ]
            topics_str = " · ".join(topics)

            # Compose the embed-text. The tagline is the most
            # reliable PH signal — Algolia always returns it.
            parts = [tagline] if tagline else []
            if topics_str:
                parts.append(topics_str)
            if description:
                parts.append(description)
            composed = " | ".join(parts).strip()

            # Synthesize a launch-year "batch" so the column isn't
            # NULL — the Phase 1.9 frontend reads batch.
            created_at = str(row.get("created_at", "")).strip()
            year_m = re.search(r"(\d{4})", created_at)
            launch_year = year_m.group(1) if year_m else str(snap_date.year)
            batch = f"PH {launch_year}"

            yield CompanyRecord(
                name=name,
                description=composed,
                batch=batch,
                status="Active",
                url=str(row.get("ph_url") or row.get("url") or "").strip(),
                tags=topics,
                source="producthunt",
                external_id=ph_id,
                snapshot_date=snap_date,
            )


def load_hn_snapshot(path: Path) -> Iterator[CompanyRecord]:
    """Load an HN "Show HN" snapshot.

    HN rows have a ``title`` (the post title — the product's name)
    and a ``description`` (the page's first-paragraph scrape, may
    be empty for the bulk Algolia-fetched rows). HN's natural id
    is ``object_id``.

    HN also has no YC-style batch; we synthesize ``"HN <year>"``.
    """
    snap_date = _date_from_filename(path)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            object_id = str(row["object_id"]).strip()
            title = str(row.get("title", "")).strip()
            # Strip the "Show HN: " / "Launch HN: " prefix — the
            # title prefix is meta-data, not part of the product
            # name, and would inflate the false-positive rate
            # against YC records (e.g. "Show HN: Nia (YC W36) – ..."
            # would cosine-match every YC company whose description
            # contains "Nia").
            cleaned_title = re.sub(
                r"^(Show|Launch|Ask)\s+HN:\s*", "", title, flags=re.IGNORECASE
            ).strip()
            # If the title embeds a YC batch marker like "(YC W36)"
            # or "(YC S22)", strip it too — same rationale.
            cleaned_title = re.sub(
                r"\s*\(YC\s+[A-Z]\d{2}\)\s*", " ", cleaned_title
            ).strip()
            cleaned_title = re.sub(r"\s+", " ", cleaned_title)

            description = str(row.get("description", "")).strip() or ""

            created_at = str(row.get("created_at", "")).strip()
            year_m = re.search(r"(\d{4})", created_at)
            launch_year = year_m.group(1) if year_m else str(snap_date.year)
            batch = f"HN {launch_year}"

            yield CompanyRecord(
                name=cleaned_title,
                description=description,
                batch=batch,
                status="Active",
                url=str(row.get("url") or row.get("hn_url") or "").strip(),
                tags=[],
                source="hn",
                external_id=object_id,
                snapshot_date=snap_date,
            )


_LOADERS = {
    "yc": load_yc_snapshot,
    "producthunt": load_producthunt_snapshot,
    "hn": load_hn_snapshot,
}


def load_all_snapshots(
    snapshots_dir: Path,
) -> tuple[dict[str, Path], list[CompanyRecord]]:
    """Discover and load all three snapshots.

    Returns the discovered paths (for the manifest) and the flat
    list of records (the dedup step operates on this list).
    """
    paths = discover_snapshots(snapshots_dir)
    records: list[CompanyRecord] = []
    for source, path in paths.items():
        loader = _LOADERS[source]
        loaded = list(loader(path))
        logger.info("loaded %d records from %s", len(loaded), path.name)
        records.extend(loaded)
    return paths, records


# -----------------------------------------------------------------------
# Cross-source dedup
# -----------------------------------------------------------------------


@dataclass
class DedupResult:
    """Per-source dedup accounting for the manifest."""

    kept: dict[str, int] = field(default_factory=dict)  # source → kept count
    dropped: dict[str, int] = field(default_factory=dict)  # source → dropped count
    collisions: list[dict] = field(default_factory=list)


def _name_for_dedup(rec: CompanyRecord) -> str:
    """The text we embed for the dedup step.

    For all sources, just the product name. We deliberately don't
    include the description: two records with the same name but
    different descriptions are still the same product (a YC name
    and a PH launch with the same product).
    """
    return rec.name.strip()


def cross_source_dedup(
    records: list[CompanyRecord],
    *,
    embedder: Embedder,
    threshold: float = EXACT_DUP_COSINE,
) -> tuple[list[CompanyRecord], DedupResult]:
    """Drop cross-source duplicate records on name cosine ≥ threshold.

    Within a single source, the snapshot's own dedup is the source
    of truth — we don't re-dedup inside a source. Across sources,
    we keep the record with the lowest SOURCE_PRECEDENCE index
    (YC wins ties).

    The algorithm:
    1. Embed all unique names with bge-m3 (cached at the string
       level so re-running with the same snapshot is free).
    2. Walk the records in SOURCE_PRECEDENCE order. For each
       record, compute the cosine similarity to every already-kept
       record's name embedding. If any cosine ≥ threshold, drop the
       current record.
    3. Record the collision for the manifest.

    Performance
    -----------
    With ~12K names and a threshold ≥ 0.85, the dedup walk is
    O(N * M) in kept records. We do it in numpy for O(M) per
    step (single matmul against the kept matrix). For our scale
    the whole walk is well under a minute on a laptop.
    """
    if not records:
        return [], DedupResult()

    # Group by source so we can walk in precedence order.
    by_source: dict[str, list[CompanyRecord]] = {s: [] for s in SOURCE_PRECEDENCE}
    for rec in records:
        by_source.setdefault(rec.source, []).append(rec)

    # Step 1: embed names per source (the embed call is naturally
    # batched; the bge-m3 cache means a re-run with identical
    # names is free).
    name_vectors: dict[tuple[str, str], list[float]] = {}
    for source in SOURCE_PRECEDENCE:
        names = [_name_for_dedup(r) for r in by_source.get(source, [])]
        if not names:
            continue
        vecs = embedder.embed_batch(names)
        for rec, vec in zip(by_source[source], vecs):
            name_vectors[(rec.source, rec.external_id)] = vec

    # Optional numpy acceleration — falls back to pure Python when
    # numpy is unavailable. The pure-Python path is still O(N * M)
    # in the number of records but each step is O(1024) — slow at
    # 12K rows but correct.
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return _cross_source_dedup_python(
            by_source, name_vectors, threshold=threshold,
        )

    # Step 2: walk precedence order, dedup against already-kept.
    # We maintain the kept matrix as a numpy array of shape (K, 1024)
    # and a parallel kept_norms vector of shape (K,). Both are
    # appended to in O(1) per keep (the alternative — np.vstack
    # on every step, or np.linalg.norm on the full slice every
    # step — is O(K) per step and dominates the runtime).
    kept: list[CompanyRecord] = []
    result = DedupResult()
    kept_capacity = sum(len(v) for v in by_source.values())
    kept_matrix = np.zeros((kept_capacity, 1024), dtype=np.float32)
    kept_norms = np.zeros(kept_capacity, dtype=np.float32)
    kept_count = 0
    for source in SOURCE_PRECEDENCE:
        kept_this_source = 0
        dropped_this_source = 0
        for rec in by_source.get(source, []):
            vec = name_vectors[(rec.source, rec.external_id)]
            vec_arr = np.asarray(vec, dtype=np.float32)
            v_norm = float(np.linalg.norm(vec_arr))
            if v_norm == 0:
                # Treat zero vectors as unique — they match nothing.
                kept.append(rec)
                kept_matrix[kept_count] = vec_arr
                kept_norms[kept_count] = 0.0
                kept_count += 1
                kept_this_source += 1
                continue
            if kept_count == 0:
                # First record — always keep.
                kept.append(rec)
                kept_matrix[kept_count] = vec_arr
                kept_norms[kept_count] = v_norm
                kept_count += 1
                kept_this_source += 1
                continue
            # Single matmul: shape (K, 1024) @ (1024,) → (K,).
            numer = kept_matrix[:kept_count] @ vec_arr
            cos = numer / (kept_norms[:kept_count] * v_norm + 1e-12)
            above = np.where(cos >= threshold)[0]
            if above.size == 0:
                kept.append(rec)
                kept_matrix[kept_count] = vec_arr
                kept_norms[kept_count] = v_norm
                kept_count += 1
                kept_this_source += 1
            else:
                dropped_this_source += 1
                winner = kept[int(above[0])]
                result.collisions.append(
                    {
                        "kept": {"source": winner.source, "external_id": winner.external_id, "name": winner.name},
                        "dropped": {"source": rec.source, "external_id": rec.external_id, "name": rec.name},
                    }
                )
        result.kept[source] = kept_this_source
        result.dropped[source] = dropped_this_source

    return kept, result


def _cross_source_dedup_python(
    by_source: dict[str, list[CompanyRecord]],
    name_vectors: dict[tuple[str, str], list[float]],
    *,
    threshold: float,
) -> tuple[list[CompanyRecord], DedupResult]:
    """Pure-Python fallback for :func:`cross_source_dedup`.

    Identical semantics to the numpy path but slower. Used when
    numpy isn't installed (CI image, dev laptop without
    ``uv sync --extra ml``).
    """
    kept: list[CompanyRecord] = []
    kept_vecs: list[list[float]] = []
    result = DedupResult()
    for source in SOURCE_PRECEDENCE:
        kept_this_source = 0
        dropped_this_source = 0
        for rec in by_source.get(source, []):
            vec = name_vectors[(rec.source, rec.external_id)]
            winner_idx = _nearest_above(vec, kept_vecs, threshold=threshold)
            if winner_idx is None:
                kept.append(rec)
                kept_vecs.append(vec)
                kept_this_source += 1
            else:
                dropped_this_source += 1
                winner = kept[winner_idx]
                result.collisions.append(
                    {
                        "kept": {"source": winner.source, "external_id": winner.external_id, "name": winner.name},
                        "dropped": {"source": rec.source, "external_id": rec.external_id, "name": rec.name},
                    }
                )
        result.kept[source] = kept_this_source
        result.dropped[source] = dropped_this_source

    return kept, result


def _nearest_above(
    query: list[float],
    candidates: list[list[float]],
    *,
    threshold: float,
) -> Optional[int]:
    """Index of the first candidate whose cosine to query ≥ threshold.

    Pure-Python linear scan with a vectorised cos() per step. The
    candidates list is iterated only once (so the per-call cost is
    O(M)), and the candidates list passed in is the live kept
    matrix — not a re-built numpy array on every call — so the
    dedup walk doesn't pay the per-step numpy-stack cost.

    The higher-level :func:`cross_source_dedup` rebuilds the
    numpy matrix once per source (not per record) so the overall
    cost is O(N * M / 1024) per source — well under a minute on
    12K rows.

    Returns None if no candidate passes the threshold.
    """
    if not candidates:
        return None
    q_norm = _norm(query)
    if q_norm == 0:
        return None
    for i, c in enumerate(candidates):
        c_norm = _norm(c)
        if c_norm == 0:
            continue
        cos = sum(a * b for a, b in zip(query, c)) / (q_norm * c_norm)
        if cos >= threshold:
            return i
    return None


def _norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


# -----------------------------------------------------------------------
# Pipeline: dedup → upsert companies → embed + upsert embeddings
# -----------------------------------------------------------------------


@dataclass
class BuildStats:
    """Per-source + aggregate counts for the final manifest."""

    sources: dict[str, dict] = field(default_factory=dict)
    # Each entry: {records_in, dropped, embedded, company_id}
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "sources": self.sources,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _upsert_company_source_ext(session: Session, rec: CompanyRecord) -> int:
    """Upsert one Company row by (source, external_id). Same as
    :func:`src.data.ingest._upsert_company` but kept here so the
    corpus-build path doesn't depend on the YC-only ingest module's
    private helper. Returns the company id.
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
            external_id=rec.external_id,
            snapshot_date=rec.snapshot_date,
        )
        .on_conflict_do_update(
            index_elements=["source", "external_id"],
            set_={
                "name": rec.name,
                "description": rec.description,
                "batch": rec.batch,
                "status": rec.status,
                "url": rec.url,
                "tags": rec.tags,
                "snapshot_date": rec.snapshot_date,
            },
        )
        .returning(Company.id)
    )
    return int(session.execute(stmt).scalar_one())


def build_corpus(
    session: Session,
    *,
    records: list[CompanyRecord],
    embedder: Embedder,
    commit_every: int = 500,
) -> BuildStats:
    """Upsert all kept records + their bge-m3 embeddings.

    Walks the deduped record list in deterministic order (source
    precedence, then name) so re-running with the same input gives
    the same id assignment order — keeps the migration
    deterministic and the manifest stable.

    Returns a :class:`BuildStats` with per-source counts.
    """
    started = time.monotonic()
    model_version = embedder.model_name

    stats = BuildStats()
    pending_texts: list[str] = []
    pending_meta: list[tuple[int, Chunk]] = []
    last_progress = 0

    # Walk in deterministic order: source precedence, then name.
    by_source: dict[str, list[CompanyRecord]] = {s: [] for s in SOURCE_PRECEDENCE}
    for rec in records:
        by_source.setdefault(rec.source, []).append(rec)
    for source in SOURCE_PRECEDENCE:
        by_source[source].sort(key=lambda r: r.name)

    # Per-source counters — initialise for all three sources so
    # the manifest has a stable shape even if a source is empty.
    for source in SOURCE_PRECEDENCE:
        stats.sources[source] = {"records_in": 0, "embedded": 0, "company_id": 0}

    for source in SOURCE_PRECEDENCE:
        for rec in by_source[source]:
            stats.sources[source]["records_in"] += 1
            company_id = _upsert_company_source_ext(session, rec)
            stats.sources[source]["company_id"] = company_id

            for chunk in chunk_text(rec.description):
                pending_texts.append(chunk.text)
                pending_meta.append((company_id, chunk))

            if len(pending_texts) >= 32:
                vectors = embedder.embed_batch(pending_texts)
                for (cid, chunk), vec in zip(pending_meta, vectors):
                    _upsert_embedding(
                        session,
                        company_id=cid,
                        model_version=model_version,
                        chunk=chunk,
                        vector=vec,
                    )
                    # Per-source attribution: increment whichever
                    # source the company_id belongs to. We walk
                    # source-by-source, so the last company_id in
                    # this batch belongs to ``source``; we count
                    # the whole batch under that source for
                    # simplicity (the alternative is a per-row
                    # source lookup, which is a wasted query).
                    stats.sources[source]["embedded"] += 1
                pending_texts.clear()
                pending_meta.clear()

            if (
                stats.sources[source]["records_in"] - last_progress
                >= max(1, sum(s["records_in"] for s in stats.sources.values()) // 10)
            ):
                last_progress = stats.sources[source]["records_in"]
                logger.info(
                    "corpus-build progress: %s -> %d records in",
                    source,
                    stats.sources[source]["records_in"],
                )

            if stats.sources[source]["records_in"] % commit_every == 0:
                session.commit()

        # Drain per-source leftover chunks (rare but cheap).
        if pending_texts:
            vectors = embedder.embed_batch(pending_texts)
            for (cid, chunk), vec in zip(pending_meta, vectors):
                _upsert_embedding(
                    session,
                    company_id=cid,
                    model_version=model_version,
                    chunk=chunk,
                    vector=vec,
                )
                stats.sources[source]["embedded"] += 1
            pending_texts.clear()
            pending_meta.clear()

    session.commit()

    stats.duration_seconds = time.monotonic() - started
    return stats


# -----------------------------------------------------------------------
# Manifest writer
# -----------------------------------------------------------------------


def write_manifest(
    out_path: Path,
    *,
    snapshots: dict[str, Path],
    load_counts: dict[str, int],
    dedup: DedupResult,
    build: BuildStats,
    model_version: str,
    embedding_dim: int,
    threshold: float,
) -> None:
    """Write the corpus-build manifest.

    Schema mirrors the Phase 2.5 / 2.6 manifests so the snapshot
    surface stays uniform: ``schema_version``, ``source_url`` (one
    per source), ``scrape_date``, ``count``, ``per_source``,
    ``dedup_stats``, ``embedding_model``, ``scraped_at_utc``,
    ``snapshot_filename``.
    """
    manifest = {
        "schema_version": "1.0.0",
        "scraped_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": model_version,
        "embedding_dim": embedding_dim,
        "dedup_threshold": threshold,
        "source_files": {
            source: {
                "path": snapshots[source].name,
                "records": load_counts[source],
                "snapshot_date": _date_from_filename(snapshots[source]).isoformat(),
            }
            for source in SOURCE_PRECEDENCE
        },
        "sources": {
            source: {
                "records_in": build.sources[source]["records_in"],
                "dropped_dedup": dedup.dropped.get(source, 0),
                "kept": dedup.kept.get(source, 0),
                "embedded": build.sources[source]["embedded"],
            }
            for source in SOURCE_PRECEDENCE
        },
        "totals": {
            "records_in": sum(load_counts.values()),
            "kept": sum(dedup.kept.values()),
            "dropped_dedup": sum(dedup.dropped.values()),
            "embedded": sum(s["embedded"] for s in build.sources.values()),
        },
        "collisions_sample": dedup.collisions[:25],  # cap so manifest stays small
        "collision_count": len(dedup.collisions),
        "snapshot_filename": out_path.name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


app = typer.Typer(help=__doc__, no_args_is_help=True)


@app.command()
def main(
    snapshots_dir: Path = typer.Option(
        SNAPSHOTS_DIR, "--snapshots-dir", help="Directory holding source JSONL files.",
    ),
    out_manifest: Optional[Path] = typer.Option(
        None,
        "--out-manifest",
        "-o",
        help="Path for the corpus-build manifest. "
        "Defaults to data/snapshots/corpus_<today>.manifest.json.",
    ),
    skip_embed: bool = typer.Option(
        False, "--no-embed", help="Merge + dedup only; skip the bge-m3 embed step."
    ),
    skip_dedup: bool = typer.Option(
        False, "--no-dedup", help="Skip the cross-source dedup step (NOT recommended — leaves duplicates)."
    ),
    threshold: float = typer.Option(
        EXACT_DUP_COSINE, "--threshold", help="Cosine threshold for the cross-source dedup step."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the embedding model (defaults to PRIORART_EMBEDDING_MODEL)."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if out_manifest is None:
        out_manifest = SNAPSHOTS_DIR / f"corpus_{date.today().isoformat()}.manifest.json"

    engine = get_engine()
    # Run any pending migrations first so the schema is up to date
    # for the (source, external_id) dedup key. ``run_all`` is
    # idempotent — safe to call on every corpus-build.
    migration_stats = run_migrations(engine)
    logger.info("migrations applied: %s", migration_stats)
    init_schema(engine)

    snapshots, records = load_all_snapshots(snapshots_dir)
    load_counts = {source: sum(1 for r in records if r.source == source)
                   for source in SOURCE_PRECEDENCE}
    logger.info(
        "loaded %d records across %d sources: %s",
        len(records), len(snapshots), load_counts,
    )

    if not skip_dedup:
        # Embedder is needed for the name-cosine dedup.
        embedder = Embedder(model_name=model)
        records, dedup_result = cross_source_dedup(records, embedder=embedder, threshold=threshold)
        logger.info(
            "dedup complete: kept=%s dropped=%s collisions=%d",
            dedup_result.kept, dedup_result.dropped, len(dedup_result.collisions),
        )
    else:
        dedup_result = DedupResult(kept=load_counts, dropped={s: 0 for s in SOURCE_PRECEDENCE})
        embedder = None  # type: ignore[assignment]

    if skip_embed:
        # For --no-embed we just write a manifest of what would be
        # ingested. Useful for CI / smoke tests that want to verify
        # the merge logic without paying for the embedding cost.
        build_stats = BuildStats(
            sources={
                s: {"records_in": dedup_result.kept[s], "embedded": 0, "company_id": 0}
                for s in SOURCE_PRECEDENCE
            }
        )
    else:
        assert embedder is not None
        with session_scope(engine) as session:
            build_stats = build_corpus(session, records=records, embedder=embedder)
        logger.info("build complete: %s", build_stats.as_dict())

    write_manifest(
        out_manifest,
        snapshots=snapshots,
        load_counts=load_counts,
        dedup=dedup_result,
        build=build_stats,
        model_version=(embedder.model_name if embedder else "skipped"),
        embedding_dim=1024,
        threshold=threshold,
    )

    # Also print a one-line JSON summary on stdout for scripts.
    print(json.dumps({
        "snapshots": {s: p.name for s, p in snapshots.items()},
        "load_counts": load_counts,
        "dedup": {
            "kept": dedup_result.kept,
            "dropped": dedup_result.dropped,
            "collisions": len(dedup_result.collisions),
        },
        "build": build_stats.as_dict(),
        "manifest": str(out_manifest),
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()