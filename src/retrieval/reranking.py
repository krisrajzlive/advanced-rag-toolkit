"""Reranking via LlamaIndex's native `LLMRerank` postprocessor.

`LLMRerank` asks the configured LLM to score each retrieved node's relevance
to the query and keeps only the top-scoring ones, applied as a
`node_postprocessor` on a standard `RetrieverQueryEngine`. Using the LLM
(rather than a downloaded cross-encoder) keeps this demo dependency-light
and free-tier friendly.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever


def build_reranked_query_engine(
    index_or_retriever: VectorStoreIndex | BaseRetriever,
    similarity_top_k: int = 10,
    rerank_top_n: int = 3,
) -> RetrieverQueryEngine:
    """Retrieve a wide candidate pool, then rerank down to `rerank_top_n`."""
    retriever = (
        index_or_retriever
        if isinstance(index_or_retriever, BaseRetriever)
        else index_or_retriever.as_retriever(similarity_top_k=similarity_top_k)
    )

    reranker = LLMRerank(choice_batch_size=5, top_n=rerank_top_n)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
    )
