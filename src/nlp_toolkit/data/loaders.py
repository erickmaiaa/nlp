"""Loading of local data files and NLTK corpora.

Centralizing I/O keeps file paths, encodings and error handling in one place
instead of being repeated (and slightly inconsistent) across notebook cells.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def load_text_file(path: str | Path, encoding: str | None = None) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: Path to the file.
        encoding: Override the configured default encoding.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    encoding = encoding or get_settings().encoding
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    logger.debug("Reading text file %s", path)
    return path.read_text(encoding=encoding)


def load_lexicon(path: str | Path, encoding: str | None = None) -> frozenset[str]:
    """Load a one-word-per-row CSV lexicon into a fast-membership set.

    The original code stored the lexicon as a list of rows and scanned it with
    nested loops (O(n*m)). A :class:`frozenset` makes look-ups O(1) and signals
    immutability.

    Args:
        path: Path to the CSV file (first column holds the word).
        encoding: Override the configured default encoding.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    encoding = encoding or get_settings().encoding
    if not path.is_file():
        raise FileNotFoundError(f"Lexicon file not found: {path}")

    words: set[str] = set()
    with path.open("r", encoding=encoding, newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip():
                words.add(row[0].strip().lower())
    logger.debug("Loaded %d words from lexicon %s", len(words), path)
    return frozenset(words)


def load_reviews(path: str | Path, encoding: str | None = None) -> list[str]:
    """Load reviews (one per CSV row) as a list of strings.

    Args:
        path: Path to the reviews CSV.
        encoding: Override the configured default encoding.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    encoding = encoding or get_settings().encoding
    if not path.is_file():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    reviews: list[str] = []
    with path.open("r", encoding=encoding, newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip():
                reviews.append(row[0].strip())
    logger.info("Loaded %d reviews from %s", len(reviews), path)
    return reviews


def read_gutenberg_raw(file_id: str) -> str:
    """Return the raw text of an NLTK Gutenberg corpus document.

    Thin wrapper kept here so callers depend on the toolkit rather than on
    NLTK's import path directly.
    """
    from nltk.corpus import gutenberg

    return gutenberg.raw(file_id)
