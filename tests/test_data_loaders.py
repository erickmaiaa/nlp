"""Tests for the data-loading utilities (using temporary files)."""

from __future__ import annotations

import pytest

from nlp_toolkit.data.loaders import load_lexicon, load_reviews, load_text_file


def test_load_text_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello world", encoding="utf-8")
    assert load_text_file(path) == "hello world"


def test_load_text_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_file(tmp_path / "nope.txt")


def test_load_lexicon_deduplicates_and_lowercases(tmp_path):
    path = tmp_path / "lex.csv"
    path.write_text("Good\ngreat\nGood\n\n", encoding="utf-8")
    assert load_lexicon(path) == frozenset({"good", "great"})


def test_load_reviews(tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text("first review\nsecond review\n", encoding="utf-8")
    assert load_reviews(path) == ["first review", "second review"]
