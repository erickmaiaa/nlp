"""Data loading utilities (local files and NLTK corpora)."""

from .loaders import (
    load_lexicon,
    load_reviews,
    load_text_file,
    read_gutenberg_raw,
)

__all__ = [
    "load_lexicon",
    "load_reviews",
    "load_text_file",
    "read_gutenberg_raw",
]
