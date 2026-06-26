"""TF-IDF feature extraction and top-term ranking (exercise 8)."""

from __future__ import annotations

from collections.abc import Sequence

from ..logging_config import get_logger

logger = get_logger(__name__)


def top_tfidf_words(
    documents: Sequence[str],
    labels: Sequence[str] | None = None,
    top_n: int = 5,
) -> dict[str, list[str]]:
    """Rank the most relevant words per document via TF-IDF.

    Args:
        documents: Pre-processed documents (already normalized / stopword-free /
            lemmatized; see :func:`nlp_toolkit.preprocessing.preprocess_for_tfidf`).
        labels: Optional human-readable names for each document; defaults to
            ``"doc_0"``, ``"doc_1"``, ...
        top_n: Number of top words to return per document.

    Returns:
        Mapping ``label -> [top words, highest TF-IDF first]``.

    Raises:
        ValueError: If ``labels`` is given but its length differs from
            ``documents``.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    if labels is not None and len(labels) != len(documents):
        raise ValueError("labels and documents must have the same length")
    labels = list(labels) if labels is not None else [f"doc_{i}" for i in range(len(documents))]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    result: dict[str, list[str]] = {}
    for row_index, label in enumerate(labels):
        row = matrix[row_index]
        # Indices of the highest-scoring (largest TF-IDF) non-zero entries.
        order = np.argsort(row.data)[::-1][:top_n]
        top_indices = row.indices[order]
        result[label] = [feature_names[idx] for idx in top_indices]
    return result
