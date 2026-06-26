"""Tests for corpus statistics, regex search and frequency helpers."""

from __future__ import annotations

import re

from nlp_toolkit.analysis.corpus_stats import compute_corpus_stats, word_set_difference
from nlp_toolkit.analysis.frequency import most_common_words, word_frequencies
from nlp_toolkit.analysis.regex_search import (
    count_occurrences,
    count_words_ending_with,
    count_words_matching,
    count_words_of_length,
)


def test_compute_corpus_stats_basic():
    words = ["the", "cat", "sat", "the", "cat"]
    sentences = [["the", "cat", "sat"], ["the", "cat"]]
    stats = compute_corpus_stats(words, sentences, already_normalized=True)
    assert stats.total_words == 5
    assert stats.total_sentences == 2
    assert stats.unique_words == 3
    assert stats.repeated_words == 2
    assert stats.words_per_sentence == 2.5


def test_compute_corpus_stats_handles_empty_sentences():
    stats = compute_corpus_stats(["a", "b"], [], already_normalized=True)
    assert stats.words_per_sentence == 0.0


def test_word_set_difference():
    assert word_set_difference(["a", "b", "c"], ["b"]) == {"a", "c"}


def test_count_words_matching_pattern():
    words = ["river", "car", "sky", "star"]
    assert count_words_matching(words, re.compile(r"\w+r\b")) == 3


def test_count_words_ending_with():
    assert count_words_ending_with(["car", "bar", "sky"], "r") == 2


def test_count_words_of_length():
    assert count_words_of_length(["abcde", "abc", "hello"], 5) == 2


def test_count_occurrences_counts_every_match():
    # "err" appears once in "error" and twice in "errerr".
    assert count_occurrences(["error", "errerr", "ok"], "err") == 3


def test_frequency_helpers():
    words = ["a", "b", "a", "c", "a", "b"]
    assert word_frequencies(words)["a"] == 3
    assert most_common_words(words, 2) == [("a", 3), ("b", 2)]
