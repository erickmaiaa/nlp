"""Centralized, environment-aware configuration.

All paths, magic numbers and tunables that were previously scattered across
notebook cells live here. Values can be overridden via environment variables
(optionally loaded from a ``.env`` file) so the same code runs unchanged on a
laptop, in CI, or in a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional dependency; configuration still works without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when python-dotenv is absent
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


def _project_root() -> Path:
    """Return the repository root, regardless of the current working directory.

    Walks upwards from this file until a directory containing ``data`` (the
    bundled corpora) is found, falling back to three levels up
    (``src/nlp_toolkit/config.py`` -> repo root).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw.strip() else default


def _resolve_dir(name: str, default: Path, root: Path) -> Path:
    """Resolve a directory override.

    Relative values (e.g. ``NLP_DATA_DIR=data``) are interpreted relative to the
    repository root rather than the current working directory, so the same
    ``.env`` works from a notebook, a script, or CI regardless of where the
    process was launched.
    """
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings.

    Prefer :func:`get_settings` over instantiating this directly so the result
    is cached and ``.env`` is loaded exactly once.
    """

    project_root: Path
    data_dir: Path
    output_dir: Path

    #: Fixed seed for any stochastic step, guaranteeing reproducibility.
    random_seed: int = 42
    #: Default language used for stopwords and NLP models.
    language: str = "english"
    #: spaCy model used for tokenization and named-entity recognition.
    spacy_model: str = "en_core_web_sm"
    #: Default text encoding for reading/writing files.
    encoding: str = "utf-8"
    #: How many "top" items to report (frequencies, TF-IDF words, ...).
    top_n: int = 5
    #: NLTK packages required by the toolkit.
    nltk_packages: tuple[str, ...] = field(
        default_factory=lambda: (
            "gutenberg",
            "wordnet",
            "omw-1.4",
            "stopwords",
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger_eng",
        )
    )

    @property
    def positive_words_path(self) -> Path:
        return self.data_dir / "positive_words.csv"

    @property
    def negative_words_path(self) -> Path:
        return self.data_dir / "negative_words.csv"

    @property
    def reviews_path(self) -> Path:
        return self.data_dir / "reviews.csv"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (once) and return the cached :class:`Settings` instance."""
    root = _project_root()
    load_dotenv(root / ".env")

    data_dir = _resolve_dir("NLP_DATA_DIR", root / "notebooks" / "data", root)
    output_dir = _resolve_dir("NLP_OUTPUT_DIR", root / "outputs", root)
    output_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        random_seed=_get_int("NLP_RANDOM_SEED", 42),
        language=os.environ.get("NLP_LANGUAGE", "english"),
        spacy_model=os.environ.get("NLP_SPACY_MODEL", "en_core_web_sm"),
        encoding=os.environ.get("NLP_ENCODING", "utf-8"),
        top_n=_get_int("NLP_TOP_N", 5),
    )
