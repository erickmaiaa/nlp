"""nlp_toolkit: a small, production-grade toolkit for classic NLP coursework.

The package turns a previously notebook-only set of exercises into reusable,
typed and tested modules. The notebook in ``notebooks/`` is now a thin
orchestrator that imports from here.

Sub-packages
------------
- :mod:`nlp_toolkit.config`        Centralized, environment-aware settings.
- :mod:`nlp_toolkit.resources`     One-time NLTK/spaCy bootstrap helpers.
- :mod:`nlp_toolkit.data`          File and corpus loading.
- :mod:`nlp_toolkit.preprocessing` Text normalization, stopwords, lemmatization.
- :mod:`nlp_toolkit.analysis`      Corpus statistics, regex search, frequency.
- :mod:`nlp_toolkit.nlp`           spaCy-based named-entity recognition.
- :mod:`nlp_toolkit.features`      TF-IDF feature extraction.
- :mod:`nlp_toolkit.sentiment`     Lexicon-based sentiment polarity.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
