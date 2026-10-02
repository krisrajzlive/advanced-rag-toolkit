from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient, models

from src.cdc.pipeline_sync import sync_directory


def _count(client, collection="rag_synced", tenant=None):
    flt = None
    if tenant:
        flt = models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant))])
    return client.count(collection, count_filter=flt, exact=True).count


def test_sync_detects_new_changed_unchanged_deleted(tmp_path):
    root, state = tmp_path / "docs", tmp_path / "state"
    for tenant in ("acme", "globex"):
        (root / tenant).mkdir(parents=True)
    (root / "acme" / "a.txt").write_text("alpha document about anvils")
    (root / "globex" / "g.txt").write_text("globex document about analytics")
    client, emb = QdrantClient(":memory:"), MockEmbedding(embed_dim=8)
    run = lambda: sync_directory(root, client=client, state_dir=state, embed_model=emb)

    r1 = run()
    assert r1["new"] == ["acme/a.txt", "globex/g.txt"] and r1["unchanged"] == 0
    assert _count(client) == 2

    r2 = run()  # nothing changed -> nothing re-embedded
    assert r2["new"] == r2["changed"] == r2["deleted"] == [] and r2["unchanged"] == 2

    (root / "acme" / "a.txt").write_text("alpha document, now revised")
    r3 = run()
    assert r3["changed"] == ["acme/a.txt"] and r3["new"] == []
    assert _count(client, tenant="acme") == 1  # old vectors replaced, not duplicated

    (root / "globex" / "g.txt").unlink()
    r4 = run()
    assert r4["deleted"] == ["globex/g.txt"]
    assert _count(client, tenant="globex") == 0
