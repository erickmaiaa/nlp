"""Corpus analysis: statistics, regex search and frequency distributions."""

from .corpus_stats import (
    CorpusStats,
    compute_corpus_stats,
    gutenberg_word_counts,
    word_set_difference,
)
from .frequency import most_common_words, word_frequencies
from .regex_search import (
    count_occurrences,
    count_words_matching,
    count_words_of_length,
)

__all__ = [
    "CorpusStats",
    "compute_corpus_stats",
    "gutenberg_word_counts",
    "word_set_difference",
    "most_common_words",
    "word_frequencies",
    "count_occurrences",
    "count_words_matching",
    "count_words_of_length",
]
