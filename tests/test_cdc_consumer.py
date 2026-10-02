from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient

from src.cdc.consumer import apply_event, open_index


def _row(i, body, tenant="acme"):
    return {"id": i, "tenant_id": tenant, "title": f"doc {i}", "body": body}


def test_debezium_events_update_the_vector_index():
    Settings.embed_model = MockEmbedding(embed_dim=8)
    client = QdrantClient(":memory:")
    index = open_index("rag_cdc_test", client=client)
    count = lambda: client.count("rag_cdc_test", exact=True).count

    apply_event(index, {"op": "r", "before": None, "after": _row(1, "first")})   # snapshot
    apply_event(index, {"op": "c", "before": None, "after": _row(2, "second", "globex")})
    assert count() == 2

    apply_event(index, {"op": "u", "before": _row(1, "first"), "after": _row(1, "first, revised")})
    assert count() == 2  # replaced, not duplicated
    texts = [p.payload["_node_content"] for p in client.scroll("rag_cdc_test", limit=10)[0]]
    assert any("revised" in t for t in texts)

    apply_event(index, {"op": "d", "before": _row(2, "second", "globex"), "after": None})
    assert count() == 1

    # replaying the same delete (at-least-once delivery) is harmless
    apply_event(index, {"op": "d", "before": _row(2, "second", "globex"), "after": None})
    assert count() == 1
