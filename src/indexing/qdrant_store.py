"""Qdrant-backed `VectorStoreIndex` with HNSW, quantization and multi-tenancy.

The collection is created here (not by LlamaIndex) so we control its HNSW
graph, quantization and tenant payload index; `QdrantVectorStore` then simply
attaches to it.

Multi-tenancy follows Qdrant's recommended "one collection, many tenants"
layout: every point carries a `tenant_id` payload, that field has a keyword
index flagged `is_tenant=True` (co-locates a tenant's vectors), and the global
HNSW graph is disabled (`m=0`) in favour of per-tenant graphs (`payload_m`).
"""

from __future__ import annotations

import os

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.indexing.hnsw import HnswParams, hnsw_config
from src.indexing.quantization import quantization_config, wait_until_indexed

TENANT_FIELD = "tenant_id"


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    *,
    quantization: str = "none",
    hnsw: HnswParams | None = None,
    multitenant: bool = False,
    normalized: bool = False,
    recreate: bool = True,
) -> None:
    """Create `name` with the requested HNSW / quantization / tenancy setup."""
    if client.collection_exists(name):
        if not recreate:
            return
        client.delete_collection(name)

    hnsw = hnsw or HnswParams()
    if multitenant:
        # No global graph; one HNSW graph per tenant_id value.
        hnsw = HnswParams(m=0, ef_construct=hnsw.ef_construct, payload_m=hnsw.m)

    client.create_collection(
        name,
        # Unit-length vectors make DOT == cosine and is the cheaper metric.
        vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.DOT if normalized else models.Distance.COSINE
        ),
        hnsw_config=hnsw_config(hnsw),
        quantization_config=quantization_config(quantization),
    )
    if multitenant:
        client.create_payload_index(
            name,
            field_name=TENANT_FIELD,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
        )


def build_qdrant_index(
    documents: list[Document],
    collection: str,
    *,
    quantization: str = "none",
    hnsw: HnswParams | None = None,
    multitenant: bool = False,
    normalized: bool = False,
    recreate: bool = True,
    client: QdrantClient | None = None,
    transformations: list | None = None,
) -> VectorStoreIndex:
    """Embed `documents` (with `Settings.embed_model`) into a Qdrant collection."""
    from llama_index.core import Settings

    client = client or get_qdrant_client()
    dim = len(Settings.embed_model.get_text_embedding("dimension probe"))
    ensure_collection(
        client, collection, dim, quantization=quantization, hnsw=hnsw,
        multitenant=multitenant, normalized=normalized, recreate=recreate,
    )
    store = QdrantVectorStore(client=client, collection_name=collection)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=store),
        transformations=transformations,
    )
    wait_until_indexed(client, collection)
    return index


def attach_qdrant_index(collection: str, client: QdrantClient | None = None) -> VectorStoreIndex:
    """Open an existing collection as an index (no re-embedding)."""
    store = QdrantVectorStore(client=client or get_qdrant_client(), collection_name=collection)
    return VectorStoreIndex.from_vector_store(store)
