"""Bulk-load a pre-baked corpus-embeddings .npz into Postgres.

This is the Phase 3.6.2 (card t_68dd7a03) CI-side companion to
``scripts/build_corpus_embeddings_npz.py``: the maintainer precomputes
the corpus embeddings on a machine with bge-m3 cached, commits the
``data/cache/corpus_embeddings.npz`` symlink to the repo, and the
eval-regression workflow bulk-loads them into a fresh CI database
without ever downloading bge-m3 from HuggingFace.

Why this exists
---------------
The eval-regression workflow needs the corpus embeddings in Postgres
to run ``/search`` against. The straightforward path
(``make corpus-build``) re-embeds the descriptions with bge-m3, which
downloads the 2.3 GB model on cold-cache runs and burns 30-60 min of
CI CPU. Both legs of that path are broken on the Actions runner
(``actions/cache`` returns HTTP 400 on the 2.3 GB cache key, and the
HF download trips ``OSError``). Bulk-loading from a committed ``.npz``
sidesteps both.

What this writes
----------------
The ``company_embeddings`` table for the (model_version) named in the
``.npz``. Companies must already exist (the .npz's ``company_id`` is
the FK target). The bulk-load is idempotent on
(company_id, model_version, chunk_index) — re-running on the same .npz
is a no-op.

CLI
---
::

    python -m src.data.load_corpus_embeddings \\
        --npz data/cache/corpus_embeddings.npz

The default ``--npz`` path is the ``data/cache/corpus_embeddings.npz``
symlink maintained by ``scripts/build_corpus_embeddings_npz.py``.

When to re-run
--------------
- After ``scripts/build_corpus_embeddings_npz.py`` is re-run (new
  snapshot hash → new .npz → bulk-load to refresh CI).
- The .npz's ``model_version`` array must match the
  ``EMBEDDING_MODEL`` the API is configured to use at query time, or
  pgvector will skip the rows on the (model_version) filter inside
  ``_SEARCH_SQL``. The script asserts this at load time.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

# Repo path setup so this module is importable from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EMBEDDING_MODEL  # noqa: E402
from src.data.db import get_engine, session_scope  # noqa: E402
from src.data.models import Company, CompanyEmbedding  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# .npz loading
# -----------------------------------------------------------------------


class CorpusEmbeddingsCache:
    """Read-only wrapper around a corpus_embeddings_*.npz file.

    The file is laid out as four numpy arrays:

    - ``company_id``  (int64)        — FK into ``companies.id``
    - ``chunk_index`` (int32)        — chunk index per (company, model)
    - ``model_version`` (object)     — the bge-m3 model id (all rows
                                       should be the same string)
    - ``embeddings``  (float32, N×D) — unit-norm 1024-dim vectors
    """

    def __init__(self, npz_path: Path) -> None:
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"corpus-embeddings .npz not found at {self.path}. "
                "Run `python scripts/build_corpus_embeddings_npz.py` first."
            )
        # ``allow_pickle=True`` is needed for the object arrays of
        # model_version strings that np.savez writes. The .npz is
        # produced by ``scripts/build_corpus_embeddings_npz.py`` (same
        # repo, same code path) — not by a third party — so this is
        # a project-local artifact, not an untrusted pickle. If you
        # want belt-and-braces, delete the .npz and re-run
        # ``scripts/build_corpus_embeddings_npz.py`` from scratch.
        data = np.load(self.path, allow_pickle=True)
        required = {"company_id", "chunk_index", "model_version", "embeddings"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"{self.path} is missing required arrays: {sorted(missing)}; "
                f"got: {list(data.keys())}"
            )
        self.company_id: np.ndarray = data["company_id"]
        self.chunk_index: np.ndarray = data["chunk_index"]
        self.model_version: np.ndarray = data["model_version"]
        self.embeddings: np.ndarray = data["embeddings"]
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 1024:
            raise ValueError(
                f"embeddings must be 2-D (N, 1024); got shape {self.embeddings.shape}"
            )
        unique_models = set(self.model_version.tolist())
        if len(unique_models) != 1:
            raise ValueError(
                f"corpus_embeddings.npz has mixed model_versions: {unique_models}; "
                "the .npz is supposed to be a single-model dump"
            )
        self.model_version_str: str = str(next(iter(unique_models)))

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])


# -----------------------------------------------------------------------
# Postgres bulk-load
# -----------------------------------------------------------------------


def _existing_company_ids(session: Session) -> set[int]:
    """Read the current ``companies.id`` set so the bulk-load can skip
    rows whose company doesn't exist (e.g. CI without the corpus
    build step).
    """
    rows = session.execute(select(Company.id)).fetchall()
    return {int(r[0]) for r in rows}


def bulk_load(
    npz_path: Path,
    *,
    batch_size: int = 500,
    expected_model_version: Optional[str] = None,
) -> int:
    """Upsert every row of the .npz into ``company_embeddings``.

    Returns the number of rows the bulk-load tried to insert (skipping
    rows whose company_id doesn't exist in the current database). The
    on-conflict clause makes the operation idempotent — re-running on
    the same .npz produces zero net rows.
    """
    cache = CorpusEmbeddingsCache(npz_path)
    if expected_model_version and cache.model_version_str != expected_model_version:
        raise ValueError(
            f"corpus_embeddings.npz model_version={cache.model_version_str!r} "
            f"does not match the API's expected model_version={expected_model_version!r}; "
            "the API will skip the rows. Re-run `scripts/build_corpus_embeddings_npz.py` "
            f"with --model-version {expected_model_version!r}."
        )
    engine = get_engine()
    t0 = time.time()
    with session_scope(engine) as session:
        existing_ids = _existing_company_ids(session)
        logger.info(
            "bulk_load: %d rows in .npz, %d companies in DB, model_version=%s",
            len(cache),
            len(existing_ids),
            cache.model_version_str,
        )
        # Filter to existing companies. The .npz was built from this
        # same DB so the sets should match exactly, but the CI DB
        # might be empty (no ``corpus_build`` step in the workflow)
        # if the operator forgot to populate the ``companies`` table —
        # we log a clear warning and skip the missing FKs.
        keep_mask = np.isin(cache.company_id, sorted(existing_ids))
        if not keep_mask.all():
            missing = int((~keep_mask).sum())
            logger.warning(
                "bulk_load: %d .npz rows have no matching company in the DB; "
                "those rows will be skipped (FK violation otherwise)",
                missing,
            )
        if not keep_mask.any():
            logger.warning("bulk_load: zero rows remain after FK filter; nothing to insert")
            return 0
        company_ids = cache.company_id[keep_mask]
        chunk_indices = cache.chunk_index[keep_mask]
        embeddings = cache.embeddings[keep_mask]

        # Upsert in batches — pgvector's `vector` type round-trips
        # through SQLAlchemy as a list, so 1024-dim vectors are
        # ~10 KB each. A 500-row batch is ~5 MB / insert, which is
        # well under Postgres's max_allowed_packet default and
        # keeps the round-trip count low.
        n_total = len(company_ids)
        n_inserted = 0
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_rows = []
            for i in range(start, end):
                batch_rows.append(
                    {
                        "company_id": int(company_ids[i]),
                        "embedding": embeddings[i].tolist(),
                        "model_version": cache.model_version_str,
                        "chunk_index": int(chunk_indices[i]),
                        # chunk_count and chunk_text fall through to
                        # the model defaults (1, ""). The pre-baked
                        # .npz doesn't carry chunk_text — chunk_text
                        # is only used for retrieval-debug display,
                        # not for the cosine path, so empty is fine.
                    }
                )
            stmt = pg_insert(CompanyEmbedding).values(batch_rows)
            # Idempotent on (company_id, model_version, chunk_index).
            # ``SET embedding = EXCLUDED.embedding`` so a re-run with
            # a different .npz (different snapshot) refreshes the
            # vector while keeping the row PK stable.
            stmt = stmt.on_conflict_do_update(
                index_elements=["company_id", "model_version", "chunk_index"],
                set_={"embedding": stmt.excluded.embedding},
            )
            session.execute(stmt)
            n_inserted += end - start
            logger.info("bulk_load: inserted %d / %d rows", n_inserted, n_total)
        session.commit()
    elapsed = time.time() - t0
    logger.info(
        "bulk_load: %d rows in %.2fs (%.0f rows/sec) — model_version=%s",
        n_inserted,
        elapsed,
        n_inserted / max(elapsed, 1e-6),
        cache.model_version_str,
    )
    return n_inserted


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--npz",
        default=str(REPO_ROOT / "data" / "cache" / "corpus_embeddings.npz"),
        help="Path to the corpus_embeddings .npz (default: %(default)s)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per INSERT batch (default: %(default)s)",
    )
    p.add_argument(
        "--expected-model-version",
        default=EMBEDDING_MODEL,
        help="Reject the .npz if its model_version doesn't match this "
        "(default: the configured EMBEDDING_MODEL = %(default)s)",
    )
    args = p.parse_args()
    n = bulk_load(
        Path(args.npz),
        batch_size=args.batch_size,
        expected_model_version=args.expected_model_version,
    )
    print(f"OK — bulk-loaded {n} corpus embeddings from {args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
