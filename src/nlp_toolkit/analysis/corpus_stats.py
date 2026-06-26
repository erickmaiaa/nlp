"""Corpus-level statistics (exercises 1 and 2).

The notebook computed these counts with three parallel copies of the same code
(one per Shakespeare play). Here the logic lives once in a typed dataclass and a
single function, eliminating duplication and the ``sc_``/``sh_``/``sm_`` global
sprawl.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..preprocessing.text import normalize_words


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Summary statistics for a single text."""

    total_words: int
    total_sentences: int
    unique_words: int
    repeated_words: int
    words_per_sentence: float


def compute_corpus_stats(
    words: Sequence[str],
    sentences: Sequence[object],
    *,
    already_normalized: bool = False,
) -> CorpusStats:
    """Compute word/sentence statistics for one text.

    Args:
        words: Raw or normalized tokens of the text.
        sentences: Sentence sequence (only its length is used).
        already_normalized: Set ``True`` to skip re-normalizing ``words``.

    Returns:
        A populated :class:`CorpusStats`.

    Notes:
        ``repeated_words`` follows the original definition: total words minus
        the number of distinct words (i.e. how many tokens are "duplicates").
    """
    norm = list(words) if already_normalized else normalize_words(words)
    total_words = len(norm)
    total_sentences = len(sentences)
    unique = len(set(norm))
    repeated = total_words - unique
    avg = total_words / total_sentences if total_sentences else 0.0
    return CorpusStats(
        total_words=total_words,
        total_sentences=total_sentences,
        unique_words=unique,
        repeated_words=repeated,
        words_per_sentence=avg,
    )


def gutenberg_word_counts() -> dict[str, int]:
    """Return normalized word counts for every Gutenberg document.

    Returns:
        Mapping ``file_id -> word_count`` preserving corpus order. Use ``max``/
        ``min`` on ``counts.items()`` keyed by value to find the largest and
        smallest documents.
    """
    from nltk.corpus import gutenberg

    counts: dict[str, int] = {}
    for file_id in gutenberg.fileids():
        counts[file_id] = len(normalize_words(gutenberg.words(file_id)))
    return counts


def word_set_difference(words_a: Sequence[str], words_b: Sequence[str]) -> set[str]:
    """Return words present in ``words_a`` but absent from ``words_b``."""
    return set(words_a) - set(words_b)
