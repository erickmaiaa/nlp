"""Tests for the lexicon-based sentiment analyzer.

A trivial whitespace tokenizer is injected so these tests run without NLTK.
"""

from __future__ import annotations

import pytest

from nlp_toolkit.sentiment.lexicon import LexiconSentimentAnalyzer, SentimentLexicon

LEXICON = SentimentLexicon(
    positive=frozenset({"good", "great", "love"}),
    negative=frozenset({"bad", "terrible", "hate"}),
)


def _words(text: str) -> list[str]:
    return text.replace(".", " ").split()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in text.split(".") if s.strip()]


@pytest.fixture
def analyzer() -> LexiconSentimentAnalyzer:
    return LexiconSentimentAnalyzer(
        LEXICON, sentence_tokenizer=_sentences, word_tokenizer=_words
    )


def test_positive_only(analyzer):
    assert analyzer.score_sentence("this is good and great") == 1


def test_single_negative_flips(analyzer):
    assert analyzer.score_sentence("this is bad") == -1


def test_double_negative_cancels(analyzer):
    assert analyzer.score_sentence("bad and terrible") == 1


def test_neutral(analyzer):
    assert analyzer.score_sentence("this is a camera") == 0


def test_polarity_is_mean(analyzer):
    # sentences: good(+1), bad(-1) -> mean 0.0
    assert analyzer.polarity("good. bad") == 0.0


def test_corpus_polarity_sums_documents(analyzer):
    assert analyzer.corpus_polarity(["good", "good"]) == 2.0
