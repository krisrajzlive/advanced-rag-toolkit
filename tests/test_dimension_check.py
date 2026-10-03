import pytest
from qdrant_client import QdrantClient

from src.indexing.qdrant_store import ensure_collection


def test_ensure_collection_rejects_different_embedding_dimension():
    client = QdrantClient(":memory:")
    ensure_collection(client, "docs", 1536)
    ensure_collection(client, "docs", 1536, recreate=False)  # same model: fine
    with pytest.raises(ValueError, match="1536.*384"):
        ensure_collection(client, "docs", 384, recreate=False)
    ensure_collection(client, "docs", 384, recreate=True)  # explicit rebuild is allowed
