"""Reusable text-preprocessing primitives.

These functions replace several near-identical blocks that were copy-pasted
across the notebook (one per text, with the same ``[w.lower() for w in words
if w.isalpha()]`` comprehension). Each function does one thing, is typed, and is
independently testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def normalize_words(words: Iterable[str], *, keep_numbers: bool = False) -> list[str]:
    """Lower-case tokens and drop punctuation-only tokens.

    Args:
        words: Raw tokens (e.g. from ``gutenberg.words(...)``).
        keep_numbers: If ``True``, keep alphanumeric tokens (``isalnum``);
            otherwise keep only alphabetic tokens (``isalpha``), matching the
            original notebook behaviour.

    Returns:
        A list of normalized tokens.
    """
    predicate = str.isalnum if keep_numbers else str.isalpha
    return [word.lower() for word in words if predicate(word)]


@lru_cache(maxsize=8)
def _stopword_set(language: str) -> frozenset[str]:
    """Return (cached) the NLTK stopword set for a language."""
    from nltk.corpus import stopwords

    return frozenset(stopwords.words(language))


def remove_stopwords(words: Iterable[str], language: str | None = None) -> list[str]:
    """Remove stopwords from a token sequence.

    Unlike the original implementation, this operates on *tokens* and returns
    *tokens*, so the caller keeps a meaningful word count. The previous version
    joined tokens into a string and then measured ``len(string)`` (characters),
    which is why exercise 5 reported inflated numbers.

    Args:
        words: Input tokens (assumed already normalized/lower-cased upstream,
            but comparison is case-insensitive regardless).
        language: Stopword language; defaults to the configured language.

    Returns:
        Tokens with stopwords removed.
    """
    language = language or get_settings().language
    stop = _stopword_set(language)
    return [word for word in words if word.lower() not in stop]


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Apply the Porter stemmer to each token."""
    from nltk import PorterStemmer

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    """Lemmatize each token with the WordNet lemmatizer."""
    from nltk import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_for_tfidf(text: str, language: str | None = None) -> str:
    """Normalize, remove stopwords and lemmatize raw text for TF-IDF.

    Pipeline: tokenize -> keep alphabetic & lower-case -> drop stopwords ->
    lemmatize -> re-join. Returns a single space-joined string, which is what
    :class:`sklearn.feature_extraction.text.TfidfVectorizer` expects.

    Args:
        text: Raw document text.
        language: Stopword language; defaults to the configured language.
    """
    from nltk import word_tokenize

    language = language or get_settings().language
    tokens = normalize_words(word_tokenize(text))
    tokens = remove_stopwords(tokens, language=language)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)
