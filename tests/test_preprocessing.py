"""Tests for dependency-free preprocessing primitives."""

from __future__ import annotations

from nlp_toolkit.preprocessing.text import normalize_words, remove_stopwords


def test_normalize_words_lowercases_and_drops_punctuation():
    assert normalize_words(["Hello", "WORLD", ",", "123", "!"]) == ["hello", "world"]


def test_normalize_words_keep_numbers():
    assert normalize_words(["Cam3ra", "123", "!"], keep_numbers=True) == ["cam3ra", "123"]


def test_remove_stopwords_uses_injected_set(monkeypatch):
    # Patch the cached stopword set so the test needs no NLTK download.
    import nlp_toolkit.preprocessing.text as mod

    monkeypatch.setattr(mod, "_stopword_set", lambda lang: frozenset({"the", "a", "of"}))
    tokens = ["the", "quick", "brown", "fox", "of", "a", "den"]
    assert remove_stopwords(tokens, language="english") == ["quick", "brown", "fox", "den"]
