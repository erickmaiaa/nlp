"""Named-entity recognition helpers (exercise 6).

The notebook re-ran the spaCy pipeline implicitly and looped once per entity
label. Here the document is processed once and entity labels are tallied in a
single pass with :class:`collections.Counter`.
"""

from __future__ import annotations

from collections import Counter

from ..resources import load_spacy


def count_entities_by_label(text: str, model: str | None = None) -> Counter[str]:
    """Return a count of named entities grouped by label for a text.

    Args:
        text: Raw input text.
        model: Optional spaCy model name; defaults to the configured model.

    Returns:
        ``Counter`` mapping spaCy entity label (``"GPE"``, ``"LOC"``,
        ``"PERSON"`` ...) to its number of occurrences.
    """
    nlp = load_spacy(model)
    doc = nlp(text)
    return Counter(ent.label_ for ent in doc.ents)


def count_entities(text: str, label: str, model: str | None = None) -> int:
    """Count entities of a single ``label`` (e.g. ``"PERSON"``) in ``text``."""
    return count_entities_by_label(text, model=model)[label]
