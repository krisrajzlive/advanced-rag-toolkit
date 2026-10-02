"""Hash-based incremental sync (LlamaIndex `IngestionPipeline` + docstore).

The pipeline keeps a docstore of document hashes. On every `sync_directory`
call it compares the source files against that docstore and, using
`DocstoreStrategy.UPSERTS_AND_DELETE`:

- new document       -> chunk, embed, insert into Qdrant
- changed document   -> old vectors deleted, re-chunked, re-embedded, inserted
- unchanged document -> skipped (no embedding calls)
- removed document   -> its vectors are deleted from Qdrant

This is polling (run it when you want to refresh), not log-based CDC; see
`src/cdc/consumer.py` for the database-level, event-driven variant.
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.indexing.qdrant_store import TENANT_FIELD, ensure_collection, get_qdrant_client

COLLECTION = "rag_synced"
STATE_DIR = Path(os.environ.get("CDC_STATE_DIR", ".cdc_state"))


def load_tenant_files(root: Path):
    """Load `<root>/<tenant>/*` with stable ids (relative path) + tenant_id tags."""
    docs = []
    for tenant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(str(f) for f in tenant_dir.rglob("*") if f.is_file())
        if not files:  # tenant folder emptied -> its documents are "deleted"
            continue
        for doc in SimpleDirectoryReader(input_files=files, filename_as_id=True).load_data():
            doc.id_ = f"{tenant_dir.name}/{Path(doc.metadata['file_name']).name}"
            doc.metadata[TENANT_FIELD] = tenant_dir.name
            docs.append(doc)
    return docs


def sync_directory(
    root: Path,
    collection: str = COLLECTION,
    client: QdrantClient | None = None,
    state_dir: Path = STATE_DIR,
    embed_model=None,
) -> dict:
    """Sync `root` into Qdrant; returns counts of new/changed/unchanged/deleted docs."""
    client = client or get_qdrant_client()
    embed_model = embed_model or Settings.embed_model
    dim = len(embed_model.get_text_embedding("dimension probe"))
    ensure_collection(client, collection, dim, multitenant=True, normalized=True, recreate=False)

    state_dir.mkdir(parents=True, exist_ok=True)
    docstore_path = state_dir / f"{collection}.docstore.json"
    docstore = (
        SimpleDocumentStore.from_persist_path(str(docstore_path))
        if docstore_path.exists()
        else SimpleDocumentStore()
    )
    before = dict(docstore.get_all_document_hashes())  # {hash: doc_id}

    documents = load_tenant_files(root)
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50), embed_model],
        vector_store=QdrantVectorStore(client=client, collection_name=collection),
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    pipeline.run(documents=documents)
    docstore.persist(str(docstore_path))

    after = dict(docstore.get_all_document_hashes())
    ids_before, ids_after = set(before.values()), set(after.values())
    new_ids = ids_after - ids_before
    deleted_ids = ids_before - ids_after
    changed_ids = {after[h] for h in set(after) - set(before)} - new_ids
    return {
        "new": sorted(new_ids),
        "changed": sorted(changed_ids),
        "deleted": sorted(deleted_ids),
        "unchanged": len(ids_after) - len(new_ids) - len(changed_ids),
    }
