"""Real L2 normalization of embeddings.

`L2NormalizedEmbedding` wraps any LlamaIndex embedding model and rescales every
vector it returns to unit length (||v||_2 = 1). Once vectors are unit length:

- dot product == cosine similarity,
- squared Euclidean distance == 2 - 2 * cosine,

so all three metrics produce the *same ranking*, and vector stores can use the
cheaper dot product. `compare_normalization()` demonstrates this on real data.
"""

from __future__ import annotations

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding


def l2_normalize(vector: Embedding) -> Embedding:
    """Return `vector / ||vector||_2` (a zero vector is returned unchanged)."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else list(vector)


def l2_norms(vectors: list[Embedding]) -> np.ndarray:
    return np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)


class L2NormalizedEmbedding(BaseEmbedding):
    """Embedding-model decorator that L2-normalizes the wrapped model's output."""

    inner: BaseEmbedding

    def __init__(self, inner: BaseEmbedding, **kwargs) -> None:
        super().__init__(
            inner=inner,
            model_name=f"l2({inner.model_name})",
            embed_batch_size=inner.embed_batch_size,
            **kwargs,
        )

    def _get_query_embedding(self, query: str) -> Embedding:
        return l2_normalize(self.inner.get_query_embedding(query))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return l2_normalize(await self.inner.aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> Embedding:
        return l2_normalize(self.inner.get_text_embedding(text))

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return l2_normalize(await self.inner.aget_text_embedding(text))

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return [l2_normalize(e) for e in self.inner.get_text_embedding_batch(texts)]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return [
            l2_normalize(e) for e in await self.inner.aget_text_embedding_batch(texts)
        ]


def compare_normalization(documents, queries: list[str], top_k: int = 3) -> dict:
    """Index `documents` with raw vs L2-normalized embeddings and compare.

    Returns vector norms before/after and, per metric, how often the top-k
    ordering equals the cosine ordering. After normalization every metric must
    agree with cosine; before it, dot-product and Euclidean may diverge.
    """
    from llama_index.core import Settings

    from src.config import get_llamaindex_embed_model
    from src.indexing.similarity_indexes import build_vector_index, compare_metrics

    report: dict = {}
    original = Settings.embed_model
    try:
        for label, normalize in (("raw", False), ("l2_normalized", True)):
            Settings.embed_model = get_llamaindex_embed_model(normalize=normalize)
            index = build_vector_index(documents)
            store = index.vector_store.data.embedding_dict
            norms = l2_norms(list(store.values()))
            agree = {"dot_product": 0, "euclidean": 0}
            for q in queries:
                res = compare_metrics(index, q, similarity_top_k=top_k)
                cos = [n.node.node_id for n in res["cosine"]]
                for m in agree:
                    agree[m] += [n.node.node_id for n in res[m]] == cos
            report[label] = {
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
                "rank_agreement_with_cosine": {m: f"{a}/{len(queries)}" for m, a in agree.items()},
            }
    finally:
        Settings.embed_model = original
    return report
