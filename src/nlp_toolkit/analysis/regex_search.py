"""Regular-expression based word searches (exercise 3).

The original notebook defined a one-off ``find_words_with_*`` helper per item,
each compiling a pattern inside a Python loop over every word. These functions
generalize the idea, compile patterns once, and expose clear semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def count_words_matching(words: Iterable[str], pattern: str | re.Pattern[str]) -> int:
    """Count words that contain a match for ``pattern``.

    Args:
        words: Tokens to test.
        pattern: A regex string or pre-compiled pattern.

    Returns:
        Number of words with at least one match.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    return sum(1 for word in words if compiled.search(word))


def count_words_ending_with(words: Iterable[str], suffix: str) -> int:
    """Count words ending with ``suffix`` (e.g. ``"r"``)."""
    return count_words_matching(words, re.compile(re.escape(suffix) + r"$"))


def count_words_of_length(words: Iterable[str], length: int) -> int:
    """Count words whose length is exactly ``length`` characters."""
    return sum(1 for word in words if len(word) == length)


def count_occurrences(words: Iterable[str], substring: str, *, ignore_case: bool = True) -> int:
    """Count total occurrences of ``substring`` across all words.

    Unlike :func:`count_words_matching` (which counts words), this counts every
    occurrence, so ``"err"`` appearing twice in one word adds two.
    """
    flags = re.IGNORECASE if ignore_case else 0
    compiled = re.compile(re.escape(substring), flags)
    return sum(len(compiled.findall(word)) for word in words)
