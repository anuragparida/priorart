"""Precompute the (company_id, model_version, chunk_index) -> embedding table to a committed .npz.

This is the Phase 3.6.2 (card t_68dd7a03) workaround for the eval-regression
workflow's bge-m3 download problem.

Why this exists
---------------
The eval-regression workflow needs the corpus embeddings in Postgres to run
``/search`` against. The straightforward path (``make corpus-build``)
re-embeds the descriptions with bge-m3, which:

1. Downloads the 2.3GB bge-m3 model from HuggingFace on the first cold
   cache run.
2. Burns ~30-60 min of CI CPU on the 11K rows × 1024-dim embedding pass.

The cold-cache run also fails: the ``actions/cache`` step returns HTTP 400
on the 2.3GB cache key, AND the HF download on the runner trips an
``OSError: We couldn't connect to 'https://huggingface.co'``. Both paths
are broken, so the workflow never reaches the "Post PR comment" step.

The fix
-------
Precompute the embeddings once on a maintainer's machine, save the table
to a committed ``.npz``, and bulk-load it into Postgres in CI. The CI
side never touches the model or HF. This is consistent with the existing
``data/cache/yc_name_embeddings.npz`` pattern (also committed, also
replaces a download).

What this writes
----------------
- ``data/cache/corpus_embeddings_<snapshot_hash>.npz`` (float32) with
  arrays ``company_id`` (int64), ``chunk_index`` (int32),
  ``model_version`` (object str), and ``embeddings`` (float32 (N, 1024)).
- A symlink ``data/cache/corpus_embeddings.npz`` pointing at the latest
  snapshot, so the workflow + scripts don't need to track the hash.

The bulk-load side lives in ``src.data.load_corpus_embeddings``.

When to re-run this
-------------------
Re-run when the underlying ``data/snapshots/*.jsonl`` changes (which
shifts the ``companies`` rows) OR when the bge-m3 model version is
bumped. The script's snapshot hash covers the former; the latter is the
``--model-version`` arg (default = current ``EMBEDDING_MODEL``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import select

# Repo path setup so we can import the priorart src tree from /tmp/priorart-venv.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"))

from src.config import EMBEDDING_MODEL  # noqa: E402
from src.data.corpus_build import discover_snapshots  # noqa: E402
from src.data.db import get_engine  # noqa: E402
from src.data.models import CompanyEmbedding  # noqa: E402


CACHE_DIR = REPO_ROOT / "data" / "cache"
LATEST_SYMLINK = CACHE_DIR / "corpus_embeddings.npz"


def _snapshot_hash() -> str:
    """Hash the contents of the three committed snapshot files.

    Stable across runs as long as the snapshots don't change. Used as
    the ``.npz`` filename suffix so a stale cache is obvious from the
    filename (and so the CI bulk-load refuses to load a .npz that
    doesn't match the current snapshots).
    """
    h = hashlib.sha256()
    snapshots = discover_snapshots(REPO_ROOT / "data" / "snapshots")
    for source in ("yc", "producthunt", "hn"):
        if source not in snapshots:
            raise FileNotFoundError(
                f"no committed snapshot for {source!r} under data/snapshots/"
            )
        h.update(snapshots[source].stem.encode("utf-8"))
    return h.hexdigest()[:12]


def _collect_embeddings(model_version: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read every (company_id, chunk_index) row for ``model_version`` from Postgres.

    Returns (company_ids, chunk_indices, embeddings) — all numpy arrays.
    Filters by model_version so a future second-model co-existing in the
    table doesn't pollute the cache.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                CompanyEmbedding.company_id,
                CompanyEmbedding.chunk_index,
                CompanyEmbedding.embedding,
            ).where(CompanyEmbedding.model_version == model_version)
        ).fetchall()
    if not rows:
        raise RuntimeError(
            f"no CompanyEmbedding rows for model_version={model_version!r} — "
            "run `make corpus-build` first to populate the table"
        )
    company_ids = np.array([r[0] for r in rows], dtype=np.int64)
    chunk_indices = np.array([r[1] for r in rows], dtype=np.int32)
    # pgvector returns a list per row; stack into (N, 1024).
    embeddings = np.stack([np.asarray(r[2], dtype=np.float32) for r in rows], axis=0)
    return company_ids, chunk_indices, embeddings


def build_npz(model_version: str = EMBEDDING_MODEL) -> Path:
    """Build the corpus-embeddings .npz and update the ``corpus_embeddings.npz`` symlink."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snap_hash = _snapshot_hash()
    out_path = CACHE_DIR / f"corpus_embeddings_{snap_hash}.npz"

    company_ids, chunk_indices, embeddings = _collect_embeddings(model_version)
    # Stable order: (model_version, company_id, chunk_index). The bulk
    # load is idempotent on (model_version, company_id, chunk_index)
    # already, but a stable order makes the .npz content byte-deterministic.
    order = np.lexsort((chunk_indices, company_ids))
    company_ids = company_ids[order]
    chunk_indices = chunk_indices[order]
    embeddings = embeddings[order]

    np.savez(
        out_path,
        company_id=company_ids,
        chunk_index=chunk_indices,
        model_version=np.array([model_version] * len(company_ids), dtype=object),
        embeddings=embeddings,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"wrote {out_path} — {len(company_ids)} rows, dim={embeddings.shape[1]}, "
        f"size={size_mb:.1f} MB, model_version={model_version}, snap_hash={snap_hash}"
    )

    # Refresh the ``corpus_embeddings.npz`` symlink to point at the
    # newest snapshot. If the symlink is broken / missing, create it.
    if LATEST_SYMLINK.is_symlink() or LATEST_SYMLINK.exists():
        LATEST_SYMLINK.unlink()
    LATEST_SYMLINK.symlink_to(out_path.name)
    print(f"symlinked {LATEST_SYMLINK} -> {out_path.name}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--model-version",
        default=EMBEDDING_MODEL,
        help="bge-m3 model identifier (default: %(default)s)",
    )
    args = p.parse_args()
    build_npz(model_version=args.model_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
