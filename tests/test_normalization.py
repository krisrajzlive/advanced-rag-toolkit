import numpy as np
from llama_index.core.base.embeddings.base import SimilarityMode, similarity
from llama_index.core.embeddings import MockEmbedding

from src.indexing.normalization import L2NormalizedEmbedding, l2_normalize, l2_norms


def test_l2_normalize_gives_unit_length():
    v = l2_normalize([3.0, 4.0])
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, [0.6, 0.8])


def test_zero_vector_unchanged():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_wrapper_normalizes_text_and_query_embeddings():
    emb = L2NormalizedEmbedding(inner=MockEmbedding(embed_dim=8))
    texts = emb.get_text_embedding_batch(["a", "bb", "ccc"])
    assert np.allclose(l2_norms(texts), 1.0)
    assert np.isclose(np.linalg.norm(emb.get_query_embedding("q")), 1.0)


def test_normalized_vectors_make_all_metrics_agree():
    rng = np.random.default_rng(0)
    docs = [(rng.normal(size=16) * rng.uniform(0.2, 5)).tolist() for _ in range(30)]
    query = rng.normal(size=16).tolist()

    def order(vs, q, mode):
        return sorted(range(len(vs)), key=lambda i: -similarity(q, vs[i], mode=mode))

    # raw (differing magnitudes): dot product disagrees with cosine
    assert order(docs, query, SimilarityMode.DOT_PRODUCT) != order(docs, query, SimilarityMode.DEFAULT)
    ndocs = [l2_normalize(d) for d in docs]
    nq = l2_normalize(query)
    cos = order(ndocs, nq, SimilarityMode.DEFAULT)
    assert order(ndocs, nq, SimilarityMode.DOT_PRODUCT) == cos
    assert order(ndocs, nq, SimilarityMode.EUCLIDEAN) == cos
