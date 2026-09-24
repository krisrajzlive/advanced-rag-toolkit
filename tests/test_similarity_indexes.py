"""Smoke test for the three native similarity-metric retrievers.

Requires a configured provider (.env) since it makes real embedding calls;
skips instead of failing if none is available (e.g. in CI without secrets).
"""

from __future__ import annotations

import pytest

from src.config import configure_llamaindex_settings, get_llm_provider
from src.indexing.similarity_indexes import build_vector_index, compare_metrics
from llama_index.core.schema import Document


def _provider_available() -> bool:
    try:
        get_llm_provider()
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _provider_available(), reason="No LLM/embedding provider configured")
def test_compare_metrics_returns_all_three_modes():
    configure_llamaindex_settings()

    documents = [
        Document(text="Cosine similarity ignores vector magnitude."),
        Document(text="Dot product is sensitive to vector magnitude."),
        Document(text="Euclidean distance measures straight-line distance."),
    ]
    index = build_vector_index(documents)

    results = compare_metrics(index, "Which metric ignores magnitude?", similarity_top_k=2)

    assert set(results.keys()) == {"cosine", "dot_product", "euclidean"}
    for nodes in results.values():
        assert len(nodes) == 2
        assert all(isinstance(n.score, float) for n in nodes)
