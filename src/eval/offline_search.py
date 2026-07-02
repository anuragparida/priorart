"""In-process search backend that mirrors ``src.api.search`` for the
eval harness.

This is the Phase 3.6.2 (card t_68dd7a03) offline mode: instead of
calling the FastAPI ``/search`` endpoint over HTTP (which needs a live
API + bge-m3 for query embedding), the eval runner passes a
**precomputed query embedding** directly and the offline search backend
runs the same SQL the API runs, but in-process.

Why this exists
---------------
The eval-regression workflow failed on cold-cache runs because
bge-m3 couldn't be downloaded. The companion fixes ship precomputed
``corpus_embeddings.npz`` and ``eval_query_embeddings.npz`` so the
CI runner never touches HuggingFace. This module is the query-side
of that contract: the eval runner loads the precomputed query
embedding for the current record and hands it to ``offline_search``,
which executes the same SQL ANN / BM25 / hybrid paths the live API
runs.

Mirrors ``src.api.search`` semantics
------------------------------------
- **dense**:   SQL ANN over ``company_embeddings`` (1 - (emb <=> :q)),
               with the per-company dedup the API does post-fetch.
- **bm25**:    ``rank_bm25.BM25Okapi`` over the in-process tokenised
               ``companies`` table (k1=1.5, b=0.75, lowercase + the
               API's stopword list).
- **hybrid**:  RRF (k=60) over dense + bm25, two-pass over-fetch.

The offline path does NOT use the API's module-level BM25 singleton
(``get_bm25_index``); it builds a fresh index per call. This is
intentional — the offline path is short-lived, used only by the eval
runner, and a per-call build keeps the contract obvious from the
code (no hidden global state, no reset helpers needed in tests).

What this does NOT do
---------------------
- It does not call the live API, do network I/O, or load bge-m3.
- It does not return a ``SearchResponse`` Pydantic model — it returns
  a list of dicts the eval runner can iterate over (the contract is
  ``[{"id", "name", "description", "similarity", "confidence"}]``,
  same as the API's ``hits`` list).
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

# Repo path setup so this module is importable from /tmp/priorart-venv
# (the broken-venv workaround for the agent fleet).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.search import (  # noqa: E402
    RRF_K,
    HNSW_EF_SEARCH,
    _bm25_tokenize,
    _cosine_to_confidence,
    _rrf_fuse,
)
from src.data.db import get_engine  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL — kept identical to ``_SEARCH_SQL`` in src.api.search. The CI eval
# numbers must match the local API numbers, and the only way to guarantee
# that is to run the same SQL.
# ---------------------------------------------------------------------------

_DENSE_SQL = text(
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


# ---------------------------------------------------------------------------
# BM25 tokenisation — same regex + stopwords as the live API
# (``_BM25_TOKEN_RE`` / ``_BM25_STOPWORDS`` are module-level in
# ``src.api.search`` and we deliberately re-import them rather than
# copy-paste the regex).
# ---------------------------------------------------------------------------

from src.api.search import _BM25_STOPWORDS, _BM25_TOKEN_RE  # noqa: E402

_BM25_K1 = 1.5
_BM25_B = 0.75


# ---------------------------------------------------------------------------
# Public dataclass — the contract the eval runner consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineHit:
    """One ranked search hit, matching the API's ``SearchHit`` shape.

    Fields are JSON-serialisable so a caller can pickle / cache /
    log them without converting.
    """

    id: int
    name: str
    description: str
    similarity: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "similarity": self.similarity,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


def offline_dense(
    engine: Engine,
    query_vector: Sequence[float],
    *,
    top_k: int = 20,
    ef_search: Optional[int] = None,
) -> List[OfflineHit]:
    """Run the same SQL ANN search the live API runs, in-process.

    The query vector is the precomputed bge-m3 embedding of the
    query text — the caller is responsible for producing it (see
    ``scripts/build_eval_query_embeddings.py`` for the eval-set
    query-embedding cache).

    Mirrors ``search_corpus`` in ``src.api.search``:
    - HNSW ef_search is set per-session before the query.
    - fetch_k = min(top_k * 2, 500) (the API's over-fetch for ties).
    - Per-company dedup keeps the best (highest-similarity) chunk.
    """
    if query_vector is None or len(query_vector) == 0:
        return []
    if len(query_vector) != 1024:
        raise ValueError(
            f"query_vector must be 1024-dim (bge-m3); got {len(query_vector)}"
        )
    fetch_k = min(top_k * 2, 500)
    q_str = _vector_to_pgvector_literal(query_vector)
    with engine.connect() as conn:
        # Same SET-local HNSW override the API uses; fall back to the
        # module-level default if the caller didn't override.
        conn.execute(text(f"SET hnsw.ef_search = {ef_search or HNSW_EF_SEARCH}"))
        rows = conn.execute(
            _DENSE_SQL,
            {"q": q_str, "k": fetch_k},
        ).fetchall()

    by_company: Dict[int, OfflineHit] = {}
    for row in rows:
        sim = float(row.similarity)
        if sim != sim:  # NaN guard — the API does the same skip.
            continue
        cid = int(row.id)
        existing = by_company.get(cid)
        if existing is None or sim > existing.similarity:
            by_company[cid] = OfflineHit(
                id=cid,
                name=str(row.name or ""),
                description=str(row.description or ""),
                similarity=sim,
                confidence=_cosine_to_confidence(sim),
            )
    out = sorted(by_company.values(), key=lambda h: h.similarity, reverse=True)
    return out[:top_k]


def _load_companies_for_bm25(engine: Engine) -> Tuple[List[int], List[List[str]], Dict[int, Tuple[str, str]]]:
    """Read the (id, name, description) rows once for BM25 tokenisation.

    Returns:
        company_ids:  parallel list of int ids (in DB order)
        tokenised:    parallel list of token lists (one per company)
        name_desc:    dict id -> (name, description) for hit materialisation
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, description FROM companies")
        ).fetchall()
    company_ids: List[int] = []
    tokenised: List[List[str]] = []
    name_desc: Dict[int, Tuple[str, str]] = {}
    for r in rows:
        cid = int(r.id)
        name = str(r.name or "")
        desc = str(r.description or "")
        # BM25 tokenises ``name + " " + description`` so the index
        # can match query terms against either field with the same
        # IDF weight (the live API does the same in
        # ``get_bm25_index``).
        tokens = _bm25_tokenize(f"{name} {desc}")
        company_ids.append(cid)
        tokenised.append(tokens)
        name_desc[cid] = (name, desc)
    return company_ids, tokenised, name_desc


def _bm25_to_confidence_offline(score: float) -> float:
    """Mirror the API's ``_bm25_to_confidence`` mapping for the offline path."""
    if score <= 0.0:
        return 0.0
    return float(score / (score + 1.0))


def offline_bm25(
    engine: Engine,
    query_text: str,
    *,
    top_k: int = 20,
    _corpus_cache: Optional[Tuple[List[int], List[List[str]], Dict[int, Tuple[str, str]]]] = None,
) -> List[OfflineHit]:
    """Run BM25 in-process over the ``companies`` table.

    The query is plain text (not an embedding), so the eval runner
    doesn't need the precomputed query-embedding cache for this
    mode. The tokeniser is identical to the live API's
    (``_bm25_tokenize``).

    Builds a fresh ``rank_bm25.BM25Okapi`` index per call; ~10K rows
    is sub-second on a single core and avoids the module-singleton
    state the live API has to manage. Caches the tokenised corpus
    across calls within a single ``OfflineSearcher`` instance for
    amortised speed (the eval runner calls this 300 times per run).

    The ``_corpus_cache`` kwarg is a private hot-path optimisation
    for callers that already loaded the corpus (the eval runner
    uses it via ``OfflineSearcher._corpus_cache``). It's prefixed
    with an underscore to flag that it's not part of the public
    contract; module-level callers can ignore it and the function
    falls back to loading the corpus itself.
    """
    tokens = _bm25_tokenize(query_text)
    if not tokens:
        return []
    if _corpus_cache is not None:
        company_ids, tokenised, name_desc = _corpus_cache
    else:
        company_ids, tokenised, name_desc = _load_companies_for_bm25(engine)
    if not company_ids:
        return []
    # rank_bm25 is imported lazily so a unit test that only exercises
    # the dense path doesn't pay the import cost.
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenised, k1=_BM25_K1, b=_BM25_B)
    raw_scores = bm25.get_scores(tokens)
    scored: List[Tuple[float, int]] = [
        (float(s), cid)
        for s, cid in zip(raw_scores, company_ids)
        if s > 0.0
    ]
    scored.sort(key=lambda p: p[0], reverse=True)
    out: List[OfflineHit] = []
    for score, cid in scored[:top_k]:
        name, desc = name_desc.get(cid, ("", ""))
        out.append(
            OfflineHit(
                id=cid,
                name=name,
                description=desc,
                similarity=score,
                confidence=_bm25_to_confidence_offline(score),
            )
        )
    return out


def offline_hybrid(
    engine: Engine,
    query_vector: Sequence[float],
    query_text: str,
    *,
    top_k: int = 20,
    rrf_k: int = RRF_K,
    _corpus_cache: Optional[Tuple[List[int], List[List[str]], Dict[int, Tuple[str, str]]]] = None,
) -> List[OfflineHit]:
    """RRF over dense (precomputed vector) + BM25 (in-process).

    Mirrors ``search_hybrid`` in ``src.api.search``: over-fetches
    each list by 2× to leave RRF room, then fuses. Confidence is
    taken from whichever list returned the hit first (the dense
    list, by construction — same as the live API).

    The RRF step is shared with the live API (``_rrf_fuse``) so
    the fusion math is identical between offline and online paths.

    The ``_corpus_cache`` kwarg mirrors ``offline_bm25`` — when the
    caller has already tokenised the corpus (the ``OfflineSearcher``
    instance caches it for the eval runner), passing it in skips
    the per-record SQL round-trip + tokenisation.
    """
    dense_hits = offline_dense(engine, query_vector, top_k=top_k * 2)
    bm25_hits = offline_bm25(
        engine, query_text, top_k=top_k * 2, _corpus_cache=_corpus_cache
    )

    # ``_rrf_fuse`` is typed ``List[SearchHit]`` but is duck-typed on
    # the hit fields (id, name, description, confidence), so it
    # accepts ``OfflineHit`` instances and constructs ``SearchHit``
    # outputs. We then re-wrap the top-N in ``OfflineHit`` so the
    # offline contract is uniform across modes.
    fused: List[Any] = _rrf_fuse([dense_hits, bm25_hits], k=rrf_k)  # type: ignore[list-item]
    out: List[OfflineHit] = []
    for h in fused[:top_k]:
        out.append(
            OfflineHit(
                id=int(h.id),
                name=str(h.name or ""),
                description=str(h.description or ""),
                similarity=float(h.similarity),
                confidence=float(h.confidence),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vector_to_pgvector_literal(vec: Sequence[float]) -> str:
    """Format a Python list[float] as a pgvector literal.

    pgvector accepts the JSON-list string format ``'[1.0, 2.0, ...]'``
    and parses it as a ``vector(N)`` value when bound as a parameter.
    Building the string in Python is faster than going through the
    SQLAlchemy ``Vector`` type adapter for ~1.2K params per eval run.
    """
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


# ---------------------------------------------------------------------------
# Re-export the dataclass alias ``OfflineHit`` so a caller can `isinstance`
# against ``SearchHit`` if they want to — the fields match but the type
# is distinct (avoids a FastAPI / Pydantic v2 round-trip in the offline
# path).
# ---------------------------------------------------------------------------

__all__ = [
    "OfflineHit",
    "offline_dense",
    "offline_bm25",
    "offline_hybrid",
]
