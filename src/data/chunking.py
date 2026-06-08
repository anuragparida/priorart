"""Sentence-based chunker for company descriptions.

Why chunk at all?
-----------------
The YC public-directory descriptions are short — median 47 chars, p99
~315 chars. The vast majority fit in a single bge-m3 embedding window
(8192 tokens). But the long tail (a few thousand-char "long
description" entries) would either exceed the model's input window or
average too many semantic ideas into a single vector.

Phase 1 ingest is conservative: a description is a single chunk
unless it crosses a soft target length, in which case we split on
sentence boundaries. Each chunk is at most ``target_chars`` characters
(default 480 — well under the 8192-token bge-m3 window but well above
the median description length, so most descriptions stay as one
chunk). Empty descriptions get a single empty chunk (never zero
chunks) so the ingest pipeline never silently drops a row.

The chunker is *intentionally* simple — no sliding window, no
overlap, no tokeniser. Phase 2 can layer in smarter chunking once we
know the eval harness is happy with this one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# Matches sentence-ending punctuation followed by whitespace and a new
# capital letter / digit / opening quote / opening bracket. This is
# good-enough for English YC descriptions; multilingual is a Phase 2
# concern. We also split on ``\n`` because the YC page sometimes
# renders line-broken descriptions.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


@dataclass(frozen=True)
class Chunk:
    """One chunk of a description.

    ``index`` is 0-based; ``count`` is the total number of chunks for
    the description. Persisted into ``CompanyEmbedding.chunk_index``
    and ``chunk_count`` so the ANN search can later reconstruct the
    original description if needed (Phase 1.4 doesn't, but the schema
    is ready).
    """

    index: int
    count: int
    text: str


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter.

    Empty / whitespace-only input returns an empty list (caller decides
    whether to emit a single empty chunk or skip).
    """
    if not text or not text.strip():
        return []
    # Normalise whitespace so the regex behaves on multi-line input.
    normalised = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_SPLIT.split(normalised) if s.strip()]


def chunk_text(text: str, *, target_chars: int = 480) -> List[Chunk]:
    """Split a description into chunks of ~``target_chars`` characters.

    Algorithm
    ---------
    1. Split into sentences.
    2. Greedily concatenate sentences into chunks; emit a chunk when
       the next sentence would push the chunk past ``target_chars``.
    3. Always emit at least one chunk, even for empty text — the
       ingest pipeline expects a 1:1 mapping between (company,
       chunks).
    """
    sentences = split_sentences(text)
    if not sentences:
        # Single empty chunk so the row still gets an embedding row.
        # Use a placeholder text so the model has *something* to
        # embed — bge-m3 produces a well-defined vector for short
        # strings.
        return [Chunk(index=0, count=1, text="(no description)")]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        # If this single sentence is longer than the target, emit the
        # current chunk (if any) then the sentence as its own chunk.
        # We don't split inside a sentence — the only thing longer
        # than ``target_chars`` on YC is a URL, and splitting a URL is
        # semantically lossy.
        if sentence_len > target_chars and not current:
            chunks.append(sentence)
            current = []
            current_len = 0
            continue
        if current and current_len + 1 + sentence_len > target_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len if not current else 1 + sentence_len

    if current:
        chunks.append(" ".join(current))

    total = len(chunks)
    return [Chunk(index=i, count=total, text=c) for i, c in enumerate(chunks)]
