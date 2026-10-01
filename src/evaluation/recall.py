"""Measure retrieval recall and tune the pipeline to improve it.

Two levels, both using only native Qdrant / LlamaIndex components:

1. `tune_ann_recall` - *index* recall. Does the approximate index (HNSW +
   quantization) return the same top-k as exact brute-force search? Sweeps
   quantization, rescoring/oversampling and `hnsw_ef`, then recommends the
   fastest configuration that reaches the target recall.

2. `improve_retrieval_recall` - *end-to-end* recall on a labelled question set
   (does the retrieved context contain the expected fact?). Compares plain
   top-k, wider top-k, LLM query expansion and LLM reranking over a wide pool.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.indexing.hnsw import HnswParams, hnsw_config
from src.indexing.quantization import (
    QUANT_KINDS,
    make_collection,
    recall_at_k,
    search_params,
    synthetic_corpus,
    time_queries,
)

# ---------------------------------------------------------------------------
# 1. ANN (index) recall
# ---------------------------------------------------------------------------


def tune_ann_recall(
    client: QdrantClient,
    n: int = 20000,
    dim: int = 1536,
    n_queries: int = 100,
    k: int = 10,
    target: float = 0.95,
) -> tuple[list[dict], dict | None]:
    """Return (all measured configs, fastest config with recall >= target)."""
    vectors, queries = synthetic_corpus(n, dim, n_queries)
    rows: list[dict] = []
    truth = None
    for kind in QUANT_KINDS:
        name = f"tune_{kind}"
        make_collection(client, name, dim, kind=kind, hnsw=hnsw_config(HnswParams()), vectors=vectors)
        if truth is None:
            truth, _ = time_queries(client, name, queries, k, models.SearchParams(exact=True))
        rescoring = ((False, 1.0),) if kind == "none" else tuple(
            (r, o) for r in (False, True) for o in ((1.0,) if not r else (1.0, 2.0, 4.0))
        )
        for rescore, oversampling in rescoring:
            for ef in (16, 64, 256):
                params = search_params(kind, oversampling=oversampling, rescore=rescore, hnsw_ef=ef)
                found, ms = time_queries(client, name, queries, k, params)
                rows.append({
                    "quantization": kind,
                    "rescore": rescore,
                    "oversampling": oversampling if rescore else None,
                    "hnsw_ef": ef,
                    "recall@k": round(recall_at_k(found, truth), 3),
                    "latency_ms": round(ms, 2),
                })
        client.delete_collection(name)
    eligible = [r for r in rows if r["recall@k"] >= target]
    best = min(eligible, key=lambda r: r["latency_ms"]) if eligible else None
    return rows, best


# ---------------------------------------------------------------------------
# 2. End-to-end retrieval recall on labelled questions
# ---------------------------------------------------------------------------

# (question, substring that must appear in the retrieved context). The later
# questions are deliberately paraphrased so plain vector search struggles.
LABELED_QUESTIONS = [
    ("What are the three stages of a RAG pipeline?", "Indexing - documents"),
    ("Which technique widens retrieval recall with reformulated questions?", "Query expansion generates"),
    ("Why is dot product natural for L2-normalized embeddings?", "identical rankings"),
    ("Which similarity metric is used in k-means clustering?", "k-means"),
    ("Which LlamaIndex class reranks with an LLM judge?", "LLMRerank"),
    ("What is the numeric range of cosine similarity?", "[-1, 1]"),
    ("How do I stop the model from making things up?", "reduces hallucination"),
    ("Which metric suits pictures and sound embeddings?", "image/audio"),
    ("How can retrieval first find the right document and then the right section?", "summary and child nodes"),
    ("How do tools let an LLM decide by itself when to search?", "autonomously"),
]


def measure_recall(retrieve, questions=LABELED_QUESTIONS) -> float:
    """Fraction of questions whose retrieved text contains the expected fact."""
    hits = 0
    for question, expected in questions:
        text = " ".join(n.node.get_content() for n in retrieve(question)).lower()
        hits += expected.lower() in text
    return hits / len(questions)


def improve_retrieval_recall(index, llm, documents=None) -> list[dict]:
    """Compare retrieval strategies; each row is one labelled-set recall score."""
    from llama_index.core.postprocessor import LLMRerank
    from llama_index.core.schema import QueryBundle

    from src.retrieval.query_expansion import build_query_expansion_retriever

    base3 = index.as_retriever(similarity_top_k=2)
    base6 = index.as_retriever(similarity_top_k=5)
    base10 = index.as_retriever(similarity_top_k=10)
    expansion = build_query_expansion_retriever(index, similarity_top_k=2, num_queries=4)
    expansion._llm = llm  # keep expansion on the same fast LLM as the rest
    expansion_wide = build_query_expansion_retriever(index, similarity_top_k=5, num_queries=4)
    expansion_wide._llm = llm
    reranker = LLMRerank(choice_batch_size=5, top_n=2, llm=llm)

    def rerank(question):
        return reranker.postprocess_nodes(base10.retrieve(question), QueryBundle(question))

    strategies = []
    if documents is not None:
        # Smaller chunks give each embedding a single focused topic.
        from llama_index.core.node_parser import SentenceSplitter

        from src.indexing.qdrant_store import build_qdrant_index

        small = build_qdrant_index(
            documents, "recall_eval_small", normalized=True, quantization="scalar",
            transformations=[SentenceSplitter(chunk_size=128, chunk_overlap=16)],
        )
        strategies.append(
            ("smaller chunks (128 tok): top-2", small.as_retriever(similarity_top_k=2).retrieve, 0)
        )
    strategies += [
        ("baseline: top-2", base3.retrieve, 0),
        ("wider: top-5", base6.retrieve, 0),
        ("query expansion (4 queries) -> top-2", expansion.retrieve, 1),
        ("query expansion (4 queries) -> top-5", expansion_wide.retrieve, 1),
        ("rerank: top-10 pool -> LLM rerank -> top-2", rerank, 2),
    ]
    return [
        {"strategy": name, "recall": round(measure_recall(fn), 2), "extra_llm_calls_per_query": cost}
        for name, fn, cost in strategies
    ]
