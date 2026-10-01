"""Graph RAG: answer questions by retrieving entities + relationship paths.

Two native LlamaIndex graph retrievers are combined:
- `VectorContextRetriever`: embeds the question, finds similar entity nodes in
  Neo4j's vector index, then walks `path_depth` hops of relationships.
- `LLMSynonymRetriever`: asks the LLM for keywords/synonyms, matches entity
  names, then walks the same hops.
The retrieved triples (plus the source chunk text) are handed to the LLM.
"""

from __future__ import annotations

from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine

from src.graph.knowledge_graph import graph_llm


def build_graph_rag_query_engine(index: PropertyGraphIndex, path_depth: int = 2) -> RetrieverQueryEngine:
    store = index.property_graph_store
    retriever = index.as_retriever(
        sub_retrievers=[
            VectorContextRetriever(
                store, embed_model=index._embed_model, include_text=True,
                similarity_top_k=4, path_depth=path_depth,
            ),
            LLMSynonymRetriever(
                store, llm=graph_llm(), include_text=True, path_depth=path_depth,
            ),
        ]
    )
    return RetrieverQueryEngine.from_args(retriever, llm=graph_llm())


def build_plain_rag_query_engine(documents, top_k: int = 3) -> RetrieverQueryEngine:
    """Baseline: ordinary chunk-vector RAG over the same corpus."""
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(similarity_top_k=top_k, llm=graph_llm())
