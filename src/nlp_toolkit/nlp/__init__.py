"""spaCy-based natural-language processing (named-entity recognition)."""

from .ner import count_entities, count_entities_by_label

__all__ = ["count_entities", "count_entities_by_label"]
