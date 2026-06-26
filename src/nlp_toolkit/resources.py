"""One-time bootstrap for third-party NLP resources (NLTK data, spaCy model).

Downloading NLTK corpora and loading the spaCy pipeline are slow, side-effectful
operations. Here they are centralized, made idempotent and cached so the rest of
the toolkit can simply ask for ``load_spacy()`` without worrying about setup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .config import Settings, get_settings
from .logging_config import get_logger

if TYPE_CHECKING:  # avoid importing spacy at module import time
    from spacy.language import Language

logger = get_logger(__name__)


def ensure_nltk_data(settings: Settings | None = None) -> None:
    """Download the NLTK packages declared in settings if they are missing.

    Uses ``find`` to avoid re-downloading on every run, which keeps notebook
    startup fast and works offline once the data is present.
    """
    import nltk

    settings = settings or get_settings()
    # Map package name -> resource path used by nltk.data.find.
    lookup = {
        "gutenberg": "corpora/gutenberg",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for package in settings.nltk_packages:
        resource = lookup.get(package, f"corpora/{package}")
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info("Downloading NLTK package: %s", package)
            nltk.download(package, quiet=True)


@lru_cache(maxsize=2)
def load_spacy(model: str | None = None) -> Language:
    """Load and cache a spaCy pipeline.

    Args:
        model: Model name; defaults to ``Settings.spacy_model``.

    Raises:
        RuntimeError: If the model is not installed, with an actionable hint.
    """
    import spacy

    model = model or get_settings().spacy_model
    try:
        return spacy.load(model)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            f"spaCy model '{model}' is not installed. "
            f"Run: python -m spacy download {model}"
        ) from exc
