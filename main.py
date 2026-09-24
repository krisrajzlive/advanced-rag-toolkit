"""CLI entry point for the Advanced RAG Toolkit demos.

    python main.py load-all                 # load every modality, print doc counts
    python main.py compare-metrics "<query>" # cosine vs dot_product vs euclidean
    python main.py query-expansion "<query>"
    python main.py rerank "<query>"
    python main.py recursive "<query>"
    python main.py agent-llamaindex "<query>"
    python main.py agent-langchain "<query>"
    python main.py evaluate

See README.md for setup and a description of each demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    # Windows consoles default to a legacy codepage (e.g. cp1252) that
    # can't render every character an LLM returns (smart quotes, en/em
    # dashes, etc.), crashing print() with a UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from src.config import configure_llamaindex_settings

DATA_DIR = Path(__file__).parent / "data"


def load_all_documents():
    from src.loaders import (
        load_audio_video,
        load_csv,
        load_images,
        load_json,
        load_pdfs,
        load_text,
    )

    documents = []
    documents += load_text(DATA_DIR / "text")
    documents += load_json(DATA_DIR / "json")
    documents += load_csv(DATA_DIR / "csv")
    documents += load_pdfs(DATA_DIR / "pdf")
    documents += load_images(DATA_DIR / "images")
    documents += load_audio_video(DATA_DIR / "audio_video")
    return documents


def cmd_load_all(_args: argparse.Namespace) -> None:
    documents = load_all_documents()
    by_modality: dict[str, int] = {}
    for doc in documents:
        modality = doc.metadata.get("modality", "unknown")
        by_modality[modality] = by_modality.get(modality, 0) + 1

    print(f"Loaded {len(documents)} documents:")
    for modality, count in sorted(by_modality.items()):
        print(f"  {modality:10s} {count}")


def cmd_compare_metrics(args: argparse.Namespace) -> None:
    from src.indexing import build_vector_index
    from src.indexing.similarity_indexes import compare_metrics

    documents = load_all_documents()
    index = build_vector_index(documents)
    results = compare_metrics(index, args.query, similarity_top_k=3)

    for metric, nodes in results.items():
        print(f"\n--- {metric} ---")
        for node in nodes:
            snippet = node.node.get_content()[:120].replace("\n", " ")
            print(f"  [{node.score:.4f}] {snippet}...")


def cmd_query_expansion(args: argparse.Namespace) -> None:
    from src.indexing import build_vector_index
    from src.retrieval import build_query_expansion_retriever

    documents = load_all_documents()
    index = build_vector_index(documents)
    retriever = build_query_expansion_retriever(index)
    nodes = retriever.retrieve(args.query)

    print(f"Query-fusion retrieval returned {len(nodes)} nodes:")
    for node in nodes:
        snippet = node.node.get_content()[:120].replace("\n", " ")
        print(f"  [{node.score:.4f}] {snippet}...")


def cmd_rerank(args: argparse.Namespace) -> None:
    from src.indexing import build_vector_index
    from src.retrieval import build_reranked_query_engine

    documents = load_all_documents()
    index = build_vector_index(documents)
    query_engine = build_reranked_query_engine(index)
    response = query_engine.query(args.query)

    print(str(response))


def cmd_recursive(args: argparse.Namespace) -> None:
    from src.retrieval import build_recursive_retriever

    documents = load_all_documents()
    retriever = build_recursive_retriever(documents)
    nodes = retriever.retrieve(args.query)

    print(f"Recursive retrieval returned {len(nodes)} nodes:")
    for node in nodes:
        snippet = node.node.get_content()[:160].replace("\n", " ")
        print(f"  [{node.score:.4f}] {snippet}...")


def cmd_agent_llamaindex(args: argparse.Namespace) -> None:
    from src.agents.llamaindex_rag_agent import build_llamaindex_rag_agent, run_agent
    from src.indexing import build_vector_index

    documents = load_all_documents()
    index = build_vector_index(documents)
    agent = build_llamaindex_rag_agent(index)
    print(run_agent(agent, args.query))


def cmd_agent_langchain(args: argparse.Namespace) -> None:
    from src.agents.langchain_rag_agent import build_langchain_rag_agent, run_agent
    from src.indexing import build_vector_index

    documents = load_all_documents()
    index = build_vector_index(documents)
    agent = build_langchain_rag_agent(index)
    print(run_agent(agent, args.query))


def cmd_evaluate(_args: argparse.Namespace) -> None:
    from src.evaluation import evaluate_rag_pipeline
    from src.indexing import build_vector_index

    documents = load_all_documents()
    index = build_vector_index(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    def query_fn(question: str) -> dict:
        response = query_engine.query(question)
        contexts = [n.node.get_content() for n in response.source_nodes]
        return {"answer": str(response), "contexts": contexts}

    results = evaluate_rag_pipeline(query_fn)
    print(results)


COMMANDS = {
    "load-all": (cmd_load_all, False),
    "compare-metrics": (cmd_compare_metrics, True),
    "query-expansion": (cmd_query_expansion, True),
    "rerank": (cmd_rerank, True),
    "recursive": (cmd_recursive, True),
    "agent-llamaindex": (cmd_agent_llamaindex, True),
    "agent-langchain": (cmd_agent_langchain, True),
    "evaluate": (cmd_evaluate, False),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced RAG Toolkit demos")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("query", nargs="?", default="What is RAG?")
    args = parser.parse_args()

    configure_llamaindex_settings()

    handler, _needs_query = COMMANDS[args.command]
    handler(args)


if __name__ == "__main__":
    sys.exit(main())
