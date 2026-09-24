"""Loader tests that don't require network/API access."""

from __future__ import annotations

from pathlib import Path

from src.loaders.structured_loader import load_csv, load_json
from src.loaders.text_loader import load_text

DATA_DIR = Path(__file__).parent.parent / "data"


def test_load_text():
    docs = load_text(DATA_DIR / "text")
    assert len(docs) == 2
    assert all(doc.metadata["modality"] == "text" for doc in docs)


def test_load_json():
    docs = load_json(DATA_DIR / "json")
    assert len(docs) >= 1
    assert all(doc.metadata["modality"] == "json" for doc in docs)


def test_load_csv():
    docs = load_csv(DATA_DIR / "csv")
    assert len(docs) >= 1
    assert all(doc.metadata["modality"] == "csv" for doc in docs)


def test_load_text_empty_dir(tmp_path):
    assert load_text(tmp_path) == []
