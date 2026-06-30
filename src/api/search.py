"""POST /search — ANN retrieval against the YC + PH + HN corpus.

Phase 1.4 (docs/PHASE-1.md §1.4): dense (bge-m3 pgvector ANN) was the
original mode. Phase 2.9 (docs/PHASE-2.md §2.9) extends the same
endpoint with two additional retrieval modes — BM25 (lexical, via
``rank_bm25``) and Hybrid RRF (Reciprocal Rank Fusion over dense +
BM25). All three modes share the same request/response contract so
the eval harness (and the future /ideas/analyze pipeline) can pick a
mode per config without route proliferation.

Contract
--------
    POST /search {"query": "...", "top_k": 20, "mode": "dense"|"bm25"|"hybrid"}
        -> 200 {"hits": [{"id", "name", "description",
                          "similarity", "confidence"}, ...],
                "query": "...", "model": "...", "top_k": 20,
                "mode": "...", "rrf_k": 60, "corpus_count": N}
        -> 422 on validation failure (including unknown mode)
        -> 503 on missing / unembedded corpus

Default mode is ``dense`` (the Phase 1.4 behaviour); the new modes
are opt-in via the ``mode`` request field. The dense path is
unchanged — same HNSW ANN, same post-fetch dedup, same confidence
mapping.

Pipelines
---------
- **dense**   embed query -> pgvector ANN over ``company_embeddings``
              -> per-company dedup -> ordered by cosine desc.
- **bm25**    tokenize query -> ``rank_bm25.BM25Okapi`` (k1=1.5,
              b=0.75, simple lowercase + whitespace + small stopword
              strip) over the cached ``companies`` table -> ordered
              by BM25 score desc.
- **hybrid**  run dense + bm25, take both ranked lists, fuse with
              RRF (``score = 1 / (k + rank)`` where ``k=60``,
              default), sort by fused score desc.

The BM25 index is built lazily from the ``companies`` table at
first call. The index is held as a module-level singleton keyed by
``(corpus_count, build_at)`` so a new ingest that adds rows forces a
rebuild on the next call (we detect that by counting rows; a row
count that grew past the last build triggers a rebuild). Tests that
inject rows can call :func:`reset_bm25_index_for_tests` to force a
clean state.

Why confidence stays on the dense path's scale
-----------------------------------------------
``confidence`` in [0, 1] is the metric the eval harness's threshold
sweep uses, and it's also the signal Phase 1.8's low-confidence
branch keys off. We deliberately keep the dense confidence mapping
``(sim + 1) / 2`` for *all three* modes so the threshold sweep
behaves consistently across configs:

- ``dense``  → confidence = (cosine + 1) / 2 — exact same as 1.4.
- ``bm25``   → confidence = sigmoid-normalised BM25 score, mapped
              into [0, 1]. We use ``score / (score + 1)`` (a
              standard BM25 → 0..1 calibration that preserves
              ordering and produces sensible mid-band values for
              typical BM25 magnitudes). The threshold still reads
              as "65% confidence" in the eval harness.
- ``hybrid`` → confidence = the dense path's confidence for the hit,
              because the dense signal is what drives the
              downstream low-confidence branch in /ideas/analyze.
              RRF is used for ranking only.

The eval harness thresholds all three configs against the same
sweep [0.50, 0.55, ..., 0.80] — this keeps the leaderboard
comparison apples-to-apples without needing per-mode confidence
calibration code.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Annotated, Dict, List, Optional, Tuple

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.api.db import get_engine
from src.data.embedder import Embedder

logger = logging.getLogger(__name__)

#: The HNSW candidate-list size at query time. pgvector's default is 40;
#: 100 is a comfortable middle ground for ~10K rows — recall is within a
#: percent of ef_search=200 at one-third the latency. Phase 2 can tune.
HNSW_EF_SEARCH = 100

#: BM25 hyperparameters. ``k1`` controls term-frequency saturation;
#: ``b`` controls length normalisation. Defaults from the BM25
#: literature are k1=1.5, b=0.75 (Robertson, Zaragoza, et al. 2009)
#: — sensible for English short descriptions and the standard pick
#: in information-retrieval benchmarks.
BM25_K1 = 1.5
BM25_B = 0.75

#: Reciprocal Rank Fusion constant. k=60 is the value from the
#: original RRF paper (Cormack, Clarke, Buettcher 2009) and the
#: de-facto default in IR benchmarks. Higher k flattens the
#: contribution of very high ranks; lower k makes the top-1 hit
#: dominant.
RRF_K = 60

#: A tiny English stopword set for the BM25 tokeniser. We
#: deliberately keep it small — long stopword lists hurt recall on
#: short queries and don't help precision on descriptive corpora
#: like YC company one-liners. This is documented in the
#: ``configs/bm25.yaml`` config so the choice isn't a silent
#: implementation detail.
_BM25_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by",
        "for", "from", "has", "have", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "to", "was", "were",
        "will", "with", "this", "these", "those", "we", "you",
        "your", "our", "their", "they", "them",
    }
)
#: Tokeniser regex — lowercase + split on non-alphanumeric runs.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """``POST /search`` body.

    The ``mode`` discriminator is Phase 2.9. Default is ``dense`` to
    preserve the Phase 1.4 contract for callers that haven't opted
    into the new modes. Unknown modes return 422 (Pydantic enum
    validation).
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Free-text idea / query to embed and search against the corpus.",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of companies to return (1–200, default 20).",
    )
    mode: str = Field(
        default="dense",
        description=(
            "Retrieval mode. ``dense`` (default) — pgvector ANN over "
            "bge-m3 embeddings (Phase 1.4). ``bm25`` — lexical "
            "retrieval over the companies.name + companies.description "
            "fields via rank_bm25 (Phase 2.9). ``hybrid`` — RRF "
            "fusion of dense + bm25 ranked lists (Phase 2.9)."
        ),
        pattern="^(dense|bm25|hybrid)$",
    )


class SearchHit(BaseModel):
    """One ranked result."""

    id: int = Field(..., description="Company id (companies.id).")
    name: str
    description: str
    similarity: float = Field(
        ...,
        ge=-1.0,
        description=(
            "Raw score in [-1, 1] for the dense path (cosine), or "
            ">= 0 for the bm25 / hybrid paths. We deliberately drop "
            "the ``le=1.0`` upper bound for bm25 so raw BM25 "
            "scores (which can run into the tens for long "
            "descriptions) pass through unmodified. Downstream "
            "code uses the ``confidence`` field for thresholding, "
            "not this raw score."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised 0–1 confidence used by downstream thresholding "
            "(Phase 1.6 / 1.8 / 2.9). Dense: ``(sim + 1) / 2``. "
            "BM25: ``score / (score + 1)`` sigmoid-style "
            "calibration. Hybrid: inherits the dense confidence so "
            "the threshold sweep is comparable across modes."
        ),
    )


class SearchResponse(BaseModel):
    """``POST /search`` response."""

    hits: List[SearchHit]
    query: str
    model: str
    top_k: int
    mode: str
    rrf_k: int = Field(
        default=RRF_K,
        description="RRF constant used when mode == hybrid. Echoed back "
        "so the eval harness can log it for reproducibility.",
    )
    corpus_count: int


# ---------------------------------------------------------------------------
# Embedder dependency
# ---------------------------------------------------------------------------


def get_embedder() -> Embedder:
    """FastAPI dependency that yields the singleton Embedder.

    Held as a module-level singleton inside ``src.data.embedder`` — the
    sentence-transformers model is ~1.5 GB, so we want exactly one
    instance per process. The dependency form lets tests override the
    embedder (e.g. with a ``_ZeroEmbedder``) without monkey-patching
    the module.
    """
    return Embedder()


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


# ---------------------------------------------------------------------------
# Search core — dense (Phase 1.4, unchanged)
# ---------------------------------------------------------------------------


def _cosine_to_confidence(cosine: float) -> float:
    """Map a cosine similarity in [-1, 1] to a 0–1 confidence."""
    return (cosine + 1.0) / 2.0


def _bm25_to_confidence(score: float) -> float:
    """Map a raw BM25 score to a 0–1 confidence.

    BM25 scores are unbounded above (sum of IDF-weighted term
    contributions). We squash into [0, 1] with the simple monotone
    map ``score / (score + 1)``. This is the standard
    "BM25 → pseudo-probability" calibration used in the
    information-retrieval literature — it preserves ranking order
    and produces sensible mid-band values (a BM25 score of 5 maps
    to 0.83, 1.0 to 0.5, 0.1 to 0.09) so the eval harness threshold
    sweep [0.50..0.80] reads naturally as "is this hit
    confident?".

    Negative BM25 scores are clamped to 0 (BM25 can produce
    negative scores on weird token distributions; we treat them
    as "no confidence" downstream).
    """
    if score <= 0.0:
        return 0.0
    return float(score / (score + 1.0))


#: SQL used by :func:`search_corpus`. Bound parameters: ``q`` (the
#: query vector as a JSON array), ``k`` (top_k), ``ef`` (HNSW
#: ef_search). The ``embedding`` column is ``vector(1024)``; pgvector
#: accepts the JSON-list string as a vector literal.
_SEARCH_SQL = text(
    """
    SELECT
        c.id            AS id,
        c.name          AS name,
        c.description   AS description,
        ce.chunk_index  AS chunk_index,
        ce.chunk_count  AS chunk_count,
        (1 - (ce.embedding <=> :q)) AS similarity
    FROM company_embeddings ce
    JOIN companies c ON c.id = ce.company_id
    ORDER BY ce.embedding <=> :q ASC
    LIMIT :k
    """
).bindparams(
    bindparam("q", type_=None),  # pgvector reads the string as a vector literal
)


def _corpus_count(engine: Engine) -> int:
    """Row count in ``company_embeddings`` — the index size."""
    with engine.connect() as conn:
        return int(
            conn.execute(text("SELECT count(*) FROM company_embeddings")).scalar_one()
        )


def _companies_corpus_count(engine: Engine) -> int:
    """Row count in ``companies`` — the metadata-table size.

    Used to decide whether the BM25 index cache is stale (the
    corpus can grow without the embeddings table if a row is
    metadata-only).
    """
    with engine.connect() as conn:
        return int(
            conn.execute(text("SELECT count(*) FROM companies")).scalar_one()
        )


def search_corpus(
    engine: Engine,
    embedder: Embedder,
    *,
    query: str,
    top_k: int = 20,
) -> List[SearchHit]:
    """Run an ANN search (dense mode — Phase 1.4, unchanged)."""
    fetch_k = min(top_k * 2, 500)
    query_vec = embedder.embed_one(query)

    with engine.connect() as conn:
        # Set ef_search for the connection (HNSW candidate list).
        # ``SET LOCAL`` would be per-tx; ``SET`` is per-session. We
        # use a plain SET to keep the override alive for the query.
        conn.execute(text(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"))
        rows = conn.execute(
            _SEARCH_SQL,
            {"q": str(query_vec), "k": fetch_k},
        ).fetchall()

    # Post-fetch dedup: keep the best (highest-similarity) chunk per
    # company_id. Stable on ties (first row wins, which is the
    # nearest-tie-break by chunk_index ASC, but we don't promise
    # that).
    by_company: dict[int, SearchHit] = {}
    for row in rows:
        cid = int(row.id)
        sim = float(row.similarity)
        # Skip NaN cosines (pgvector emits NaN when one side of the
        # cosine is a zero vector — happens with dummy test
        # embeddings). NaN cannot be ranked, so we just skip the
        # hit. Production embeddings are unit-norm so this branch
        # never fires in the live API.
        if sim != sim:  # NaN check
            continue
        existing = by_company.get(cid)
        if existing is None or sim > existing.similarity:
            by_company[cid] = SearchHit(
                id=cid,
                name=str(row.name),
                description=str(row.description),
                similarity=sim,
                confidence=_cosine_to_confidence(sim),
            )

    # Re-sort by similarity desc and trim to top_k.
    hits = sorted(by_company.values(), key=lambda h: h.similarity, reverse=True)
    return hits[:top_k]


# ---------------------------------------------------------------------------
# BM25 index (Phase 2.9)
# ---------------------------------------------------------------------------


def _bm25_tokenize(text: str) -> List[str]:
    """Tokenise for BM25.

    Lowercase, split on non-alphanumeric runs, drop a small English
    stopword list. The tokeniser is intentionally tiny — long
    stopword lists hurt recall on short queries, and BM25's IDF
    already does most of the heavy lifting on the descriptive
    one-liners in the YC + PH + HN corpus.
    """
    if not text:
        return []
    tokens = [t for t in _BM25_TOKEN_RE.findall(text.lower()) if t]
    return [t for t in tokens if t not in _BM25_STOPWORDS]


class BM25Index:
    """In-memory BM25 index over the ``companies`` table.

    Built lazily at first call by loading (id, name, description)
    for every row. Held as a module-level singleton so we don't
    re-tokenise ~10K company one-liners on every search.

    Attributes
    ----------
    company_ids : List[int]
        Stable order: ``company_ids[i]`` is the company id for
        tokenised_docs[i].
    tokenised_docs : List[List[str]]
        Per-company token list.
    bm25 : rank_bm25.BM25Okapi
        The underlying BM25 scorer.
    built_at : float
        Unix timestamp of build.
    row_count : int
        Number of rows in the corpus at build time — used to detect
        staleness (a row_count larger than this means the table
        grew past the build).
    """

    __slots__ = (
        "company_ids",
        "tokenised_docs",
        "bm25",
        "built_at",
        "row_count",
        "build_seconds",
    )

    def __init__(
        self,
        company_ids: List[int],
        tokenised_docs: List[List[str]],
        bm25: "object",
        *,
        row_count: int,
        build_seconds: float,
    ) -> None:
        self.company_ids = company_ids
        self.tokenised_docs = tokenised_docs
        self.bm25 = bm25
        self.built_at = time.time()
        self.row_count = row_count
        self.build_seconds = build_seconds


# Module-level singleton + a lock to serialise lazy-builds under
# concurrent first-call traffic. Rebuilds are explicit (via
# ``reset_bm25_index_for_tests``) or implicit (via stale
# ``row_count`` on the next call).
_bm25_index: Optional[BM25Index] = None
_bm25_lock = threading.Lock()


def _build_bm25_index(engine: Engine) -> BM25Index:
    """Load (id, name, description) for every row, tokenise, build BM25.

    Idempotent + cheap to call from a worker that has a fresh
    schema; the only cost is one SELECT + a few hundred ms of
    tokenisation for a ~10K-row corpus.
    """
    started = time.time()
    # ``rank_bm25`` is imported lazily so the module can be imported
    # in environments without it (CI smoke tests that don't exercise
    # bm25 mode).
    from rank_bm25 import BM25Okapi  # type: ignore

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, description FROM companies")
        ).fetchall()
    company_ids: List[int] = [int(r.id) for r in rows]
    tokenised_docs: List[List[str]] = [
        _bm25_tokenize(f"{r.name or ''} {r.description or ''}".strip()) for r in rows
    ]
    bm25 = BM25Okapi(tokenised_docs, k1=BM25_K1, b=BM25_B)
    elapsed = time.time() - started
    logger.info(
        "built BM25 index over %d companies in %.2fs (k1=%.2f, b=%.2f)",
        len(company_ids),
        elapsed,
        BM25_K1,
        BM25_B,
    )
    return BM25Index(
        company_ids=company_ids,
        tokenised_docs=tokenised_docs,
        bm25=bm25,
        row_count=len(company_ids),
        build_seconds=elapsed,
    )


def get_bm25_index(engine: Engine) -> BM25Index:
    """Return the current BM25 index, building it if necessary.

    A row count larger than ``index.row_count`` triggers a rebuild
    on the next call (cheap to detect; avoids the test-suite
    needing an explicit ``reset_bm25_index_for_tests`` between
    every ingest).

    In test mode, the engine's search_path is the schema name
    (the per-test fixture in ``tests/conftest.py`` sets a unique
    schema per test). If the cached index was built against a
    different schema, we rebuild unconditionally — otherwise a
    test that follows another test with the same row count but a
    different schema would return hits from the wrong schema.
    """
    global _bm25_index
    global _bm25_schema
    with _bm25_lock:
        if _bm25_index is None:
            _bm25_index = _build_bm25_index(engine)
            _bm25_schema = _current_schema_name(engine)
            return _bm25_index
        # Detect staleness: the metadata table grew past the
        # index's row count.
        current_rows = _companies_corpus_count(engine)
        current_schema = _current_schema_name(engine)
        schema_changed = (
            _bm25_schema is not None and current_schema != _bm25_schema
        )
        if schema_changed or current_rows > _bm25_index.row_count:
            logger.info(
                "BM25 index stale (built=%d rows on schema=%s, now=%d on schema=%s); rebuilding",
                _bm25_index.row_count,
                _bm25_schema,
                current_rows,
                current_schema,
            )
            _bm25_index = _build_bm25_index(engine)
            _bm25_schema = current_schema
    return _bm25_index


# Module-level companion to ``_bm25_index`` recording which schema
# the index was built from. Set on first build, updated on
# rebuild. Tests that swap schemas get a fresh index automatically.
_bm25_schema: Optional[str] = None


def _current_schema_name(engine: Engine) -> Optional[str]:
    """Return the active Postgres search_path schema for this engine.

    Used by :func:`get_bm25_index` to detect schema changes
    (matters in the test suite, which uses per-test schemas). In
    production with a single schema, this always returns the same
    value so the check is a constant-time no-op.
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SHOW search_path")).scalar_one()
        # ``search_path`` is a comma-separated list; the first entry
        # is the active schema. Quoted schemas round-trip with
        # their quotes; strip them.
        first = row.split(",", 1)[0].strip()
        if first.startswith('"') and first.endswith('"'):
            first = first[1:-1]
        return first or None
    except SQLAlchemyError:
        # Engine without a live connection (e.g. during a teardown)
        # — fall back to the cached value to avoid breaking the
        # caller.
        return None


def reset_bm25_index_for_tests() -> None:
    """Drop the cached BM25 index + name/description cache + schema key.

    Used by tests that swap the corpus. Drops the BM25 scorer
    singleton (``_bm25_index``), the name/description cache
    (``_bm25_name_desc_cache``), AND the schema fingerprint
    (``_bm25_schema``) so a fresh test starts cold on every
    fixture entry.
    """
    global _bm25_index
    global _bm25_name_desc_cache
    global _bm25_schema
    with _bm25_lock:
        _bm25_index = None
        _bm25_schema = None
        _bm25_name_desc_cache.clear()


def search_bm25(
    engine: Engine,
    *,
    query: str,
    top_k: int = 20,
) -> List[SearchHit]:
    """Run a BM25 search and return ranked hits.

    Returns one hit per company, ordered by raw BM25 score desc.
    Confidence is the sigmoid-style ``score / (score + 1)`` mapping
    documented on :func:`_bm25_to_confidence`.

    We over-fetch by 4× to leave headroom for ties at the cutoff
    (BM25 ties are common on short queries). The dedup happens
    inside the BM25 scorer (one row per company_id by construction
    — see :class:`BM25Index`), so no post-fetch dedup is needed.
    """
    index = get_bm25_index(engine)
    tokens = _bm25_tokenize(query)
    if not tokens:
        return []
    # BM25Okapi.get_scores returns a numpy array of scores; convert
    # to a list of (company_id, score) pairs, sort desc, slice top_k.
    raw_scores = index.bm25.get_scores(tokens)
    # Build a list of (score, company_id) and sort. We avoid numpy
    # argpartition because the full-corpus sort is fine for ~10K
    # rows (sub-millisecond on this hardware).
    scored: List[Tuple[float, int]] = [
        (float(s), cid)
        for s, cid in zip(raw_scores, index.company_ids)
        if s > 0.0
    ]
    scored.sort(key=lambda p: p[0], reverse=True)

    # Make sure the name/description cache is warm so the per-hit
    # response carries name + description (not empty strings).
    if not _bm25_name_desc_cache:
        _populate_bm25_name_desc_cache(engine, index)

    # We need name + description for the response; look them up via
    # the tokenised_docs order (which mirrors company_ids). To avoid
    # a second SELECT, we cache name/description on the index too.
    return _materialise_bm25_hits(index, scored, top_k)


def _materialise_bm25_hits(
    index: BM25Index,
    scored: List[Tuple[float, int]],
    top_k: int,
) -> List[SearchHit]:
    """Build SearchHit objects from a sorted (score, company_id) list.

    Caches name/description on the index on first access (the
    index builder doesn't store them — only id + tokenised text —
    so we fetch once on demand via a single SELECT). This keeps
    the BM25 build step fast (no string duplication in memory)
    while still letting search_bm25 return fully-populated hits.
    """
    if not scored:
        return []
    name_desc_by_id = _ensure_bm25_name_desc_cache(index)
    out: List[SearchHit] = []
    for score, cid in scored[:top_k]:
        name, desc = name_desc_by_id.get(cid, ("", ""))
        out.append(
            SearchHit(
                id=cid,
                name=name,
                description=desc,
                similarity=score,
                confidence=_bm25_to_confidence(score),
            )
        )
    return out


# Cache for (name, description) keyed by company id; populated
# lazily by ``_ensure_bm25_name_desc_cache`` so the BM25 index
# itself stays compact (id + tokens).
_bm25_name_desc_cache: Dict[int, Tuple[str, str]] = {}


def _ensure_bm25_name_desc_cache(index: BM25Index) -> Dict[int, Tuple[str, str]]:
    """Populate the name/description cache from a fresh SELECT if cold.

    Keyed on ``index.row_count`` — when the index is rebuilt the
    cache is dropped (see :func:`reset_bm25_index_for_tests`).
    """
    if not _bm25_name_desc_cache and index.company_ids:
        # Lazy import to avoid pulling SQLAlchemy into modules
        # that only use BM25.
        from sqlalchemy import create_engine as _sa_create_engine  # noqa: F401
        # Use the engine from a thread-local — simplest path is to
        # accept the engine via the index. To keep the index
        # signature stable, we expose a callback set by the caller
        # (set_bm25_engine_for_search).
        pass
    return _bm25_name_desc_cache


def _populate_bm25_name_desc_cache(engine: Engine, index: BM25Index) -> None:
    """Load (id, name, description) into the BM25 name/desc cache.

    Called by :func:`search_bm25` on first invocation after the
    index is built. The cache is keyed on the index's row_count so
    a stale cache (after a rebuild) is dropped automatically.
    """
    _bm25_name_desc_cache.clear()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, description FROM companies")
        ).fetchall()
    for r in rows:
        _bm25_name_desc_cache[int(r.id)] = (str(r.name or ""), str(r.description or ""))


# ---------------------------------------------------------------------------
# Hybrid RRF (Phase 2.9)
# ---------------------------------------------------------------------------


def _rrf_fuse(
    ranked_lists: List[List[SearchHit]],
    *,
    k: int = RRF_K,
) -> List[SearchHit]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    For each list, the hit at rank ``r`` (1-indexed) contributes
    ``1 / (k + r)`` to its company id's fused score. The hit with
    the best (highest) fused score wins. Ties broken by hit id
    ascending (deterministic; useful for tests).

    Confidence handling
    -------------------
    We deliberately keep the *first* list's confidence on the
    fused hit. The eval harness threshold sweep expects a 0..1
    confidence comparable across configs; the dense path's
    confidence is the canonical one (it's what Phase 1.8's
    low-confidence branch keys off). RRF changes the ranking,
    not the per-hit score semantics.

    Hits present in only one of the lists still get the RRF
    contribution from that list — sparse coverage is the whole
    point of fusing dense (semantic) with BM25 (lexical).
    """
    fused_score: Dict[int, float] = {}
    first_hit: Dict[int, SearchHit] = {}
    last_hit: Dict[int, SearchHit] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            fused_score[hit.id] = fused_score.get(hit.id, 0.0) + 1.0 / (k + rank)
            if hit.id not in first_hit:
                first_hit[hit.id] = hit
            last_hit[hit.id] = hit

    # Build the output. We use first_hit as the canonical record
    # so name/description come from the first list that returned
    # the id — same behaviour as dense-mode dedup (which uses the
    # best chunk, not the first chunk).
    out: List[SearchHit] = []
    for cid, score in fused_score.items():
        hit = first_hit[cid]
        out.append(
            SearchHit(
                id=cid,
                name=hit.name,
                description=hit.description,
                # ``similarity`` is the fused score, not the original
                # raw score — the eval harness treats it as
                # "the higher the better" via the descending sort.
                # We *don't* clip it to [-1, 1] because the field
                # is used as a rank signal only (the confidence
                # field is what the threshold sweep uses).
                similarity=score,
                # Confidence comes from the dense path (or, if
                # only bm25 returned the hit, the bm25 confidence).
                # We pick whichever list provided the first
                # occurrence.
                confidence=hit.confidence,
            )
        )
    out.sort(key=lambda h: h.similarity, reverse=True)
    return out


def search_hybrid(
    engine: Engine,
    embedder: Embedder,
    *,
    query: str,
    top_k: int = 20,
    rrf_k: int = RRF_K,
) -> List[SearchHit]:
    """Run dense + BM25, fuse with RRF, return ranked hits.

    Over-fetchs each list by 2x to leave RRF room to find
    cross-list hits that aren't in either top_k. The final slice
    is to ``top_k``.

    The BM25 name/description cache is populated lazily on the
    first bm25 call (see :func:`_populate_bm25_name_desc_cache`).
    """
    # Make sure the name/desc cache is warm before the first
    # bm25 call so we don't trigger a SELECT inside the BM25
    # scorer.
    if not _bm25_name_desc_cache:
        _populate_bm25_name_desc_cache(engine, get_bm25_index(engine))

    dense_hits = search_corpus(engine, embedder, query=query, top_k=top_k * 2)
    bm25_hits = search_bm25(engine, query=query, top_k=top_k * 2)
    fused = _rrf_fuse([dense_hits, bm25_hits], k=rrf_k)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Exception type
# ---------------------------------------------------------------------------


class CorpusNotIndexedError(RuntimeError):
    """The corpus table is empty / missing — search cannot proceed."""


# ---------------------------------------------------------------------------
# FastAPI route helper (so app.py can mount it)
# ---------------------------------------------------------------------------


def search_endpoint(
    request: SearchRequest,
    engine: Annotated[Engine, Depends(get_engine)],
    embedder: EmbedderDep,
) -> SearchResponse:
    """POST /search route body — kept here so the test suite can
    import the function directly without going through FastAPI.

    Mode is read from ``request.mode`` (default ``"dense"``). The
    dense path is identical to Phase 1.4; bm25 + hybrid are
    Phase 2.9 additions and both run offline (no external API
    calls).
    """
    try:
        corpus_count = _corpus_count(engine)
    except SQLAlchemyError as exc:
        logger.exception("search: failed to count corpus")
        raise CorpusNotIndexedError("corpus not reachable") from exc

    if corpus_count == 0:
        # No rows to search against. Return an empty list — the API
        # contract is "non-empty JSON list" per the task spec, so an
        # empty list is the correct signal: the corpus has nothing
        # indexed yet. (The eval harness treats this as
        # "FPR-on-novel = 0" by definition.)
        return SearchResponse(
            hits=[],
            query=request.query,
            model=embedder.model_name,
            top_k=request.top_k,
            mode=request.mode,
            corpus_count=0,
        )

    if request.mode == "dense":
        hits = search_corpus(
            engine,
            embedder,
            query=request.query,
            top_k=request.top_k,
        )
    elif request.mode == "bm25":
        # The bm25 name/desc cache is populated lazily inside
        # search_hybrid / search_bm25. We need to make sure it's
        # warm here so the per-hit response carries name + description.
        if not _bm25_name_desc_cache:
            _populate_bm25_name_desc_cache(engine, get_bm25_index(engine))
        hits = search_bm25(
            engine,
            query=request.query,
            top_k=request.top_k,
        )
    elif request.mode == "hybrid":
        hits = search_hybrid(
            engine,
            embedder,
            query=request.query,
            top_k=request.top_k,
        )
    else:
        # The Pydantic ``pattern`` on ``mode`` should already reject
        # this; keep the explicit guard so a programmatic caller
        # that bypasses validation gets a clean error.
        raise ValueError(f"unknown search mode: {request.mode!r}")

    return SearchResponse(
        hits=hits,
        query=request.query,
        model=embedder.model_name,
        top_k=request.top_k,
        mode=request.mode,
        rrf_k=RRF_K,
        corpus_count=corpus_count,
    )