"""Word frequency distributions (exercise 2e/2f)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def word_frequencies(words: Iterable[str]) -> Counter[str]:
    """Return a :class:`collections.Counter` of word frequencies.

    Using the standard-library ``Counter`` avoids a hard dependency on
    ``nltk.FreqDist`` for what is plain counting, while remaining compatible
    (``FreqDist`` subclasses ``Counter``).
    """
    return Counter(words)


def most_common_words(words: Iterable[str], n: int = 5) -> list[tuple[str, int]]:
    """Return the ``n`` most frequent ``(word, count)`` pairs."""
    return word_frequencies(words).most_common(n)
