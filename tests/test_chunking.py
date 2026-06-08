"""Unit tests for the sentence-based chunker.

Pure-Python, no DB.
"""

from __future__ import annotations

from src.data.chunking import chunk_text, split_sentences


def test_empty_text_returns_one_placeholder_chunk() -> None:
    chunks = chunk_text("")
    assert len(chunks) == 1
    assert chunks[0].count == 1
    assert chunks[0].index == 0
    # The placeholder text is non-empty so the embedder always has
    # something to work with.
    assert chunks[0].text


def test_whitespace_only_returns_one_placeholder_chunk() -> None:
    chunks = chunk_text("   \n  \t ")
    assert len(chunks) == 1
    assert chunks[0].count == 1


def test_short_text_returns_one_chunk() -> None:
    chunks = chunk_text("Agent simulation and RL for researchers.")
    assert len(chunks) == 1
    assert chunks[0].count == 1
    assert chunks[0].text == "Agent simulation and RL for researchers."


def test_multi_sentence_stays_single_chunk_when_short() -> None:
    text = (
        "Bookkeeping, compliance and tax for founders. "
        "We file in 50 states. "
        "Our team includes ex-Stripe engineers."
    )
    chunks = chunk_text(text, target_chars=480)
    assert len(chunks) == 1
    # count metadata reflects chunk-count, not sentence-count
    assert chunks[0].count == 1


def test_long_text_splits_into_multiple_chunks() -> None:
    # Build 5 sentences, each ~200 chars, target 300 → should split.
    sentences = ["This is a long sentence about " + ("alpha " * 30) + "."] * 5
    text = " ".join(sentences)
    chunks = chunk_text(text, target_chars=300)
    assert len(chunks) >= 2
    assert chunks[0].index == 0
    assert chunks[-1].index == len(chunks) - 1
    assert chunks[-1].count == len(chunks)
    # Every chunk is non-empty
    for c in chunks:
        assert c.text.strip()


def test_single_giant_sentence_is_kept_intact() -> None:
    """A single sentence longer than target_chars is kept whole.

    We don't split inside a sentence because the only thing longer
    than the target on YC is a URL, and splitting a URL is
    semantically lossy.
    """
    text = "x" * 1500
    chunks = chunk_text(text, target_chars=300)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 1500


def test_split_sentences_normalises_whitespace() -> None:
    raw = "First sentence.   \n  Second sentence.\nThird sentence."
    parts = split_sentences(raw)
    assert len(parts) == 3
    # No leading/trailing whitespace in any part
    for p in parts:
        assert p == p.strip()


def test_split_sentences_handles_question_and_exclamation() -> None:
    parts = split_sentences("Why now? Because the market is ready. Let's go!")
    assert len(parts) == 3


def test_chunk_indices_are_dense_and_zero_indexed() -> None:
    text = "a. " * 200  # way more chunks than 1
    chunks = chunk_text(text, target_chars=50)
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)
