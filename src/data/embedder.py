"""Embedding model wrapper.

Loads a sentence-transformers model once per process and exposes a
small surface: ``embed_one`` and ``embed_batch``. The default model
is ``BAAI/bge-m3`` (1024-dim, multilingual) — the model pinned in
``AGENTS.md`` and the phase plan.

Why a wrapper?
--------------
Sentence-transformers' ``SentenceTransformer.encode`` is the only
call we ever need. Wrapping it lets us:

- pin the model + version in one place (and call out the
  ``normalize_embeddings=True`` flag — bge-m3 expects unit-norm
  vectors for cosine similarity via pgvector);
- convert numpy arrays to plain Python lists (pgvector's SQLAlchemy
  type accepts lists, not ndarrays);
- guard against loading the heavy model at import time (lazy load on
  first ``embed_*`` call). This matters for tests that mock the
  embedder — they shouldn't pay the 1.5 GB model download cost.

The model version string used as the idempotency key in the
embeddings table is the HuggingFace model id (e.g. ``BAAI/bge-m3``).
Bumping to a different model = a different ``model_version`` = a
new row, so A/B comparisons are easy in Phase 2.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Sequence

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


# Module-level singleton — sentence-transformers' SentenceTransformer
# is thread-safe for ``encode`` (the underlying torch model is
# not, but ST serialises the forward pass). We hold a single lock
# around encode so concurrent ingest workers don't race the model's
# internal state.
_model = None
_model_lock = threading.Lock()


def _load_model():
    """Lazy-load the sentence-transformers model.

    Importing sentence_transformers is heavy (torch, transformers,
    tokenizers all come along), so we delay it until the first
    ``embed_*`` call. Tests that mock the embedder never trigger
    this.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("loading embedding model %s (one-time)", EMBEDDING_MODEL)
                _model = SentenceTransformer(EMBEDDING_MODEL)
                # bge-m3 emits 1024-dim vectors by default. We assert
                # rather than silently truncate — a dim mismatch is
                # a config bug, not a runtime condition.
                actual_dim = _model.get_sentence_embedding_dimension()
                if actual_dim != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"embedding model {EMBEDDING_MODEL!r} has dim={actual_dim}, "
                        f"expected {EMBEDDING_DIM} (set EMBEDDING_DIM in src/config.py "
                        f"and update the schema if you want to use this model)."
                    )
    return _model


def reset_model_for_tests() -> None:
    """Drop the cached model — used by tests that swap the embedder."""
    global _model
    with _model_lock:
        _model = None


class Embedder:
    """Thin facade over sentence-transformers.

    Holds a reference to the model after first use. Exposes
    ``embed_one`` (single text) and ``embed_batch`` (list of texts,
    encoded as a single batched call so we benefit from GPU/CPU
    batching on the 5949-row YC ingest).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or EMBEDDING_MODEL

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def embed_one(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = _load_model()
        # ``convert_to_numpy=True`` is the default, but we want the
        # list-of-lists shape for pgvector. We normalise because
        # pgvector's cosine operator expects unit vectors.
        vectors = model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
