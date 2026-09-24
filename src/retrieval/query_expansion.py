"""Query expansion via LlamaIndex's native `QueryFusionRetriever`.

Given one user query, an LLM generates several reformulations, each is run
against the base retriever, and the combined candidate pool is re-ranked
(reciprocal rank fusion) into a single result list. This widens recall
versus a single literal-query search, entirely through LlamaIndex's own
retriever composition - no hand-written query-rewriting logic.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever


def build_query_expansion_retriever(
    index_or_retriever: VectorStoreIndex | BaseRetriever,
    similarity_top_k: int = 3,
    num_queries: int = 4,
) -> QueryFusionRetriever:
    """Wrap a retriever (or index) with LLM-driven query expansion + fusion."""
    base_retriever = (
        index_or_retriever
        if isinstance(index_or_retriever, BaseRetriever)
        else index_or_retriever.as_retriever(similarity_top_k=similarity_top_k)
    )

    return QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=num_queries,  # 1 original + (num_queries - 1) LLM-generated
        mode="reciprocal_rerank",
        use_async=False,
    )
