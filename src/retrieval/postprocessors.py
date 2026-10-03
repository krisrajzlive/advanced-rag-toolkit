"""Node postprocessors (LlamaIndex-native) applied after retrieval.

A postprocessor takes the retrieved nodes and filters, re-scores, reorders or
rewrites them before they reach the LLM. Four that earn their place:

- `SimilarityPostprocessor`   drop weak matches; lets the pipeline abstain on
                              out-of-scope questions instead of answering from noise.
- `SentenceTransformerRerank` cross-encoder reranker: scores (query, passage)
                              pairs jointly, far more precise than embedding
                              similarity; runs locally on CPU, no LLM call.
- `LongContextReorder`        puts the best chunks at the start and end of the
                              prompt (LLMs under-use the middle: "lost in the middle").
- `MetadataReplacementPostProcessor`  sentence-window retrieval: embed single
                              sentences (precise), then swap each hit for its
                              surrounding window (enough context).

Chain order matters: apply the similarity cutoff *before* the cross-encoder,
because reranking replaces cosine scores with the cross-encoder's own scale.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    LongContextReorder,
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index.core.schema import NodeWithScore, QueryBundle

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WINDOW_KEY = "window"

# Questions the corpus cannot answer; a good pipeline retrieves nothing for them.
OUT_OF_SCOPE_QUESTIONS = [
    "What is the capital city of Mongolia?",
    "Who won the 2018 football world cup?",
    "How do I bake sourdough bread?",
    "What is the boiling point of liquid nitrogen?",
]


def similarity_cutoff(cutoff: float) -> SimilarityPostprocessor:
    return SimilarityPostprocessor(similarity_cutoff=cutoff)


def long_context_reorder() -> LongContextReorder:
    return LongContextReorder()


def cross_encoder_reranker(top_n: int = 2, model: str = CROSS_ENCODER_MODEL):
    """Cross-encoder reranker (needs `pip install sentence-transformers`)."""
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(model=model, top_n=top_n)
    except ImportError as exc:  # optional heavy dependency (torch)
        raise ImportError(
            "Cross-encoder reranking needs sentence-transformers: "
            "pip install -r requirements-rerank.txt"
        ) from exc


def sentence_window_parser(window_size: int = 3) -> SentenceWindowNodeParser:
    """Index-time parser: one node per sentence, its neighbours stored as metadata."""
    return SentenceWindowNodeParser.from_defaults(
        window_size=window_size, window_metadata_key=WINDOW_KEY, original_text_metadata_key="original_text"
    )


def window_replacement() -> MetadataReplacementPostProcessor:
    """Query-time: replace each sentence hit with its surrounding window."""
    return MetadataReplacementPostProcessor(target_metadata_key=WINDOW_KEY)


def apply_chain(postprocessors, nodes: list[NodeWithScore], question: str) -> list[NodeWithScore]:
    """Run the chain on private copies: rerankers overwrite `score` in place."""
    nodes = [NodeWithScore(node=n.node, score=n.score) for n in nodes]
    bundle = QueryBundle(question)
    for processor in postprocessors:
        nodes = processor.postprocess_nodes(nodes, bundle)
    return nodes


def build_postprocessed_query_engine(index: VectorStoreIndex, postprocessors, top_k: int = 10, llm=None):
    """Retrieve `top_k`, run the chain, then answer (node_postprocessors on the engine)."""
    return index.as_query_engine(similarity_top_k=top_k, node_postprocessors=list(postprocessors), llm=llm)


def out_of_scope_rate(retrieve, questions=OUT_OF_SCOPE_QUESTIONS) -> float:
    """Share of unanswerable questions that still retrieve >= 1 chunk (lower is better)."""
    return sum(bool(retrieve(q)) for q in questions) / len(questions)


def top_score(retrieve, question: str) -> float:
    nodes = retrieve(question)
    return max((n.score or 0.0) for n in nodes) if nodes else 0.0


def compare_chains(index: VectorStoreIndex, window_index: VectorStoreIndex, question: str, cutoff: float = 0.3) -> dict:
    """Show what each postprocessor chain keeps for `question` (no LLM calls)."""
    pool = index.as_retriever(similarity_top_k=10).retrieve(question)
    chains = {
        "raw top-10": pool,
        f"similarity cutoff {cutoff}": apply_chain([similarity_cutoff(cutoff)], pool, question),
        "cutoff -> cross-encoder top-3": apply_chain(
            [similarity_cutoff(cutoff), cross_encoder_reranker(top_n=3)], pool, question
        ),
        "cutoff -> cross-encoder -> long-context reorder": apply_chain(
            [similarity_cutoff(cutoff), cross_encoder_reranker(top_n=3), long_context_reorder()], pool, question
        ),
    }
    window_hits = window_index.as_retriever(similarity_top_k=2).retrieve(question)
    chains["sentence-window: top-2 sentences -> windows"] = apply_chain([window_replacement()], window_hits, question)
    return chains
