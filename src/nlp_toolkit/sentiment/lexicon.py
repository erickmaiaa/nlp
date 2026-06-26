"""Lexicon-based sentiment polarity (exercise 7).

Refactor highlights versus the notebook version:

- The positive/negative lexicons are stored as sets, turning the original
  O(words x lexicon) nested scan into O(words) membership tests.
- Tokenizer is injectable, so the core scoring logic is unit-testable without
  downloading NLTK data.
- The (intentionally preserved) parity-based scoring rule is documented so the
  business logic is explicit rather than buried in nested ``if`` branches.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True, slots=True)
class SentimentLexicon:
    """Positive and negative word sets used for scoring."""

    positive: frozenset[str]
    negative: frozenset[str]


def _default_sentence_tokenizer(text: str) -> list[str]:
    from nltk import sent_tokenize

    return sent_tokenize(text)


def _default_word_tokenizer(text: str) -> list[str]:
    from nltk import word_tokenize

    return word_tokenize(text)


class LexiconSentimentAnalyzer:
    """Score sentences and documents with a positive/negative word lexicon.

    The per-sentence rule is preserved from the original coursework solution:

    - positive words present and **no** negative words -> ``+1``;
    - an **odd** number of negative words -> ``-1`` (a single negation flips
      polarity);
    - an **even**, non-zero number of negative words -> ``+1`` (double negation
      cancels out);
    - otherwise (no sentiment words) -> ``0``.
    """

    def __init__(
        self,
        lexicon: SentimentLexicon,
        *,
        sentence_tokenizer: Tokenizer | None = None,
        word_tokenizer: Tokenizer | None = None,
    ) -> None:
        self._lexicon = lexicon
        self._split_sentences = sentence_tokenizer or _default_sentence_tokenizer
        self._split_words = word_tokenizer or _default_word_tokenizer

    def score_sentence(self, sentence: str) -> int:
        """Return ``-1``, ``0`` or ``+1`` for a single sentence."""
        positive_hits = 0
        negative_hits = 0
        for word in self._split_words(sentence):
            token = word.lower()
            if token in self._lexicon.positive:
                positive_hits += 1
            if token in self._lexicon.negative:
                negative_hits += 1

        if positive_hits > 0 and negative_hits == 0:
            return 1
        if negative_hits % 2 == 1:
            return -1
        if negative_hits > 0:  # even and non-zero
            return 1
        return 0

    def sentence_scores(self, text: str) -> list[int]:
        """Score every sentence in ``text``."""
        return [self.score_sentence(sentence) for sentence in self._split_sentences(text)]

    def polarity(self, text: str) -> float:
        """Return the mean sentence score for a document (0.0 if empty)."""
        scores = self.sentence_scores(text)
        return statistics.fmean(scores) if scores else 0.0

    def corpus_polarity(self, documents: Iterable[str]) -> float:
        """Return the summed document polarities across a corpus."""
        return sum(self.polarity(doc) for doc in documents)
