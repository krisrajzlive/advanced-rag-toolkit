"""Vector indexing with pluggable similarity metrics.

LlamaIndex's default in-memory vector store (`SimpleVectorStore`) always
ranks results by cosine similarity. To compare cosine / dot-product /
Euclidean retrieval without writing our own similarity math, this module
leans entirely on two native LlamaIndex building blocks:

- `llama_index.core.base.embeddings.base.similarity(..., mode=SimilarityMode)`
  - the same function LlamaIndex itself uses internally for cosine scoring -
  which already implements all three metrics.
- `llama_index.core.indices.query.embedding_utils.get_top_k_embeddings`,
  which accepts that function as its `similarity_fn` and does the ranking.

`MetricRetriever` is a thin `BaseRetriever` subclass (the standard LlamaIndex
extension point) that plugs a chosen `SimilarityMode` into these two native
utilities.
"""

from __future__ import annotations

from functools import partial

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import SimilarityMode, similarity
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import SimpleVectorStore

METRIC_LABELS = {
    SimilarityMode.DEFAULT: "cosine",
    SimilarityMode.DOT_PRODUCT: "dot_product",
    SimilarityMode.EUCLIDEAN: "euclidean",
}


def build_vector_index(documents: list[Document]) -> VectorStoreIndex:
    """Embed and store `documents` once; metric choice happens at query time.

    All three similarity metrics are computed over the *same* embeddings, so
    a single `SimpleVectorStore` (LlamaIndex's native in-memory vector
    store) is built and reused by every `MetricRetriever` below.
    """
    storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


class MetricRetriever(BaseRetriever):
    """Retrieve nodes from a `VectorStoreIndex` under a chosen `SimilarityMode`."""

    def __init__(
        self,
        index: VectorStoreIndex,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
        similarity_top_k: int = 3,
    ) -> None:
        self._index = index
        self._mode = mode
        self._similarity_top_k = similarity_top_k
        self._embed_model = Settings.embed_model
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)

        vector_store: SimpleVectorStore = self._index.vector_store
        node_ids = list(vector_store.data.embedding_dict.keys())
        embeddings = list(vector_store.data.embedding_dict.values())

        similarity_fn = partial(similarity, mode=self._mode)
        top_scores, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_fn=similarity_fn,
            similarity_top_k=self._similarity_top_k,
            embedding_ids=node_ids,
        )

        docstore = self._index.docstore
        return [
            NodeWithScore(node=docstore.get_node(node_id), score=score)
            for node_id, score in zip(top_ids, top_scores)
        ]


def get_metric_retriever(
    index: VectorStoreIndex,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
    similarity_top_k: int = 3,
) -> MetricRetriever:
    return MetricRetriever(index, mode=mode, similarity_top_k=similarity_top_k)


def compare_metrics(
    index: VectorStoreIndex, query: str, similarity_top_k: int = 3
) -> dict[str, list[NodeWithScore]]:
    """Run the same query under all three native similarity metrics."""
    results: dict[str, list[NodeWithScore]] = {}
    for mode, label in METRIC_LABELS.items():
        retriever = get_metric_retriever(index, mode=mode, similarity_top_k=similarity_top_k)
        results[label] = retriever.retrieve(query)
    return results
