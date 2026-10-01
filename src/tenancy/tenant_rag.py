"""Tenant-isolated ingestion and retrieval over one shared Qdrant collection."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from src.indexing.qdrant_store import TENANT_FIELD, build_qdrant_index
from src.tenancy.auth import TenantContext

COLLECTION = "rag_multitenant"


def load_tenant_documents(root: Path):
    """Each sub-directory of `root` is a tenant; its docs are tagged tenant_id."""
    docs = []
    for tenant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for doc in SimpleDirectoryReader(str(tenant_dir)).load_data():
            doc.metadata[TENANT_FIELD] = tenant_dir.name
            docs.append(doc)
    return docs


def build_tenant_index(root: Path, normalized: bool = False, quantization: str = "scalar"):
    return build_qdrant_index(
        load_tenant_documents(root),
        COLLECTION,
        multitenant=True,
        quantization=quantization,
        normalized=normalized,
    )


def tenant_filters(ctx: TenantContext) -> MetadataFilters:
    """Mandatory filter: only the verified tenants' points are reachable."""
    return MetadataFilters(
        filters=[MetadataFilter(key=TENANT_FIELD, operator=FilterOperator.IN, value=list(ctx.tenants))]
    )


def tenant_retriever(index: VectorStoreIndex, ctx: TenantContext, top_k: int = 3):
    return index.as_retriever(similarity_top_k=top_k, filters=tenant_filters(ctx))


def tenant_query_engine(index: VectorStoreIndex, ctx: TenantContext, top_k: int = 3):
    return index.as_query_engine(similarity_top_k=top_k, filters=tenant_filters(ctx))
