"""Precompute bge-m3 embeddings for every record in the labeled eval set.

This is the Phase 3.6.2 (card t_68dd7a03) companion to
``scripts/build_corpus_embeddings_npz.py``: it embeds the ``idea`` field
of every record in the eval set (currently 300 records) and writes the
resulting matrix to a tiny ``.npz``. The eval runner consumes the
precomputed query embeddings in ``--offline`` mode, so the workflow
never needs to load bge-m3 to evaluate the dense or hybrid configs.

Why this exists
---------------
The eval-regression workflow ran the eval against a live ``/search``
endpoint, which embeds the query with bge-m3 on every request. With
bge-m3 download broken on the CI runner (HTTP 400 on the actions/cache
restore + ``OSError`` on the HF download), the dense + hybrid configs
can't run. The fix is to precompute the query embeddings once on a
maintainer's machine and commit the result; the runner swaps the live
``embedder.embed_one(query)`` call for a dictionary lookup.

What this writes
----------------
- ``data/cache/eval_query_embeddings.npz`` (float32) with arrays
  ``record_id`` (object str, the eval record's ``id`` field) and
  ``embeddings`` (float32 (N, 1024)). The order matches
  ``evals/labeled_v300.jsonl`` line order; the runner uses ``record_id``
  for index alignment.

The eval-side consumer lives in ``src.eval.offline_search`` (see
``scripts/ci/run_eval_sweep.py`` --offline path).

When to re-run
--------------
Re-run when:
- ``evals/labeled_v300.jsonl`` (or whatever the pinned benchmark is)
  is updated.
- The bge-m3 model version is bumped (changes the embedding space, so
  corpus + query caches must be regenerated in lockstep).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"))

from src.config import EMBEDDING_MODEL  # noqa: E402
from src.data.embedder import Embedder  # noqa: E402


CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_BENCHMARK = REPO_ROOT / "evals" / "labeled_v300.jsonl"
OUTPUT_PATH = CACHE_DIR / "eval_query_embeddings.npz"


def _load_benchmark(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_npz(
    benchmark_path: Path = DEFAULT_BENCHMARK,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_benchmark(benchmark_path)
    if not records:
        raise RuntimeError(f"benchmark {benchmark_path} is empty")
    record_ids = [str(r["id"]) for r in records]
    ideas = [str(r.get("idea", "")).strip() for r in records]
    if not all(ideas):
        empty = [rid for rid, idea in zip(record_ids, ideas) if not idea]
        raise RuntimeError(
            f"benchmark {benchmark_path} has empty `idea` fields for: {empty[:5]}"
        )

    embedder = Embedder(model_name=EMBEDDING_MODEL)
    # batch_size 32 matches the corpus embedding pass; show_progress_bar
    # off because the script is short (~10s) and the bar adds noise.
    embeddings = embedder.embed_batch(ideas)

    np.savez(
        output_path,
        record_id=np.array(record_ids, dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_version=np.array([EMBEDDING_MODEL] * len(record_ids), dtype=object),
    )
    size_kb = output_path.stat().st_size / 1024
    print(
        f"wrote {output_path} — {len(record_ids)} records, dim={embeddings[0].__len__()}, "
        f"size={size_kb:.1f} KB, model_version={EMBEDDING_MODEL}, "
        f"benchmark={benchmark_path.name}"
    )
    return output_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK),
        help="Path to the labeled eval set (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to write the .npz (default: %(default)s)",
    )
    args = p.parse_args()
    build_npz(Path(args.benchmark), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
