# Architecture & Design Decisions

This document explains *why* the refactored project is structured the way it is.

## Goals

1. **Preserve behaviour** of the original coursework notebook.
2. Move logic out of the notebook into reusable, typed, tested modules.
3. Centralize configuration; remove magic values and global variables.
4. Make the core logic testable without heavy downloads (NLTK data, spaCy).

## Layering

```
ingestion            data/loaders.py
   │
preprocessing        preprocessing/text.py
   │
analysis / features  analysis/*, features/tfidf.py, nlp/ner.py, sentiment/lexicon.py
   │
orchestration        notebooks/main.ipynb
```

Cross-cutting concerns — configuration (`config.py`), logging
(`logging_config.py`), resource bootstrap (`resources.py`) and reproducibility
(`utils/reproducibility.py`) — sit beside the layers and are used by all of them.

## Key decisions

- **Pure functions + dependency injection.** Statistics, regex and sentiment
  logic operate on plain token sequences and accept injectable tokenizers, so
  they are unit-tested without NLTK/spaCy. Heavy imports (`nltk`, `spacy`,
  `sklearn`, `numpy`) are deferred to call time, keeping `import nlp_toolkit`
  fast and side-effect-free.
- **Immutable, typed config.** A frozen `Settings` dataclass, cached via
  `functools.lru_cache`, replaces scattered literals. Overridable by env vars.
- **Sets over nested loops.** Lexicons and stopwords are `frozenset`s, turning
  the sentiment scan from O(words × lexicon) into O(words).
- **`collections.Counter`** for frequency/entity tallies instead of bespoke
  loops, computed in a single pass.
- **Idempotent resource bootstrap.** `ensure_nltk_data` checks before
  downloading; `load_spacy` is cached and raises an actionable error.

## Preserved-but-documented behaviour

- **Exercise 4** applies *stemming then lemmatization* (as in the original).
  This is unusual but kept to reproduce the original output; documented in the
  notebook.
- **Exercise 7** keeps the parity-based negation rule (odd negatives → −1, even
  non-zero → +1), now stated explicitly in the analyzer docstring.

## Fixed bug

- **Exercise 5** originally measured `len(joined_string)` (character count)
  instead of the number of remaining words. The refactor keeps tokens
  throughout and reports a correct word count.
