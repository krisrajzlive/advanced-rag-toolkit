"""Benchmark parallel querying (async fan-out, sub-questions, router, agent tools, evaluation).

    python scripts/parallel_query.py queries       # sequential vs gather vs semaphore-bounded gather
    python scripts/parallel_query.py subquestion   # SubQuestionQueryEngine use_async False vs True
    python scripts/parallel_query.py router        # RouterQueryEngine (multi-select): sync vs async fan-out
    python scripts/parallel_query.py agent         # FunctionAgent parallel vs sequential tool calls
    python scripts/parallel_query.py eval          # LangSmith evaluate(): max_concurrency 1 vs 4
    python scripts/parallel_query.py all
Options: --concurrency 4 (semaphore bound)

Three indexes are built once in Qdrant (RAG notes, vector-infra notes, the 154-page
threat report) and reused. LLM calls go to the cloud chat provider (Ollama Cloud
when available). Free-tier rate limits cap how much concurrency helps.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.WARNING)

DATA = Path(__file__).resolve().parents[1] / "data"

QUESTIONS = [
    "What are the three stages of a RAG pipeline?",
    "Which technique widens retrieval recall with reformulated questions?",
    "What numeric range does cosine similarity have?",
    "What does reranking do to retrieved passages?",
    "Which algorithm does Qdrant implement for nearest-neighbour search?",
    "Who introduced HNSW?",
    "How does Neo4j store data?",
    "What does L2 normalization do to an embedding?",
    "What does the report say about detecting misuse of AI models?",
    "Give examples of misuse the threat report describes.",
    "What countermeasures does the report describe?",
    "What is a threat intelligence report for?",
]
MULTI_HOP = (
    "Compare how the RAG notes describe reranking with how the vector-infrastructure notes "
    "describe quantization, and summarize what the threat report says about detecting misuse."
)


def build_tools():
    from llama_index.core import Settings

    from src.config import configure_llamaindex_settings, get_llamaindex_embed_model
    from src.graph.knowledge_graph import graph_llm
    from src.loaders import load_csv, load_json, load_pdfs, load_text
    from src.retrieval.parallel_query import get_or_build_index, make_tools

    configure_llamaindex_settings()
    Settings.embed_model = get_llamaindex_embed_model(embed_batch_size=100)  # fewer embedding round trips
    llm = graph_llm()
    from llama_index.core import SimpleDirectoryReader

    indexes = {
        "rag_notes": (
            get_or_build_index("pq_rag_notes", load_text(DATA / "text") + load_json(DATA / "json") + load_csv(DATA / "csv")),
            "RAG pipeline stages, retrieval techniques (query expansion, reranking, recursive retrieval) and similarity metrics.",
        ),
        "vector_infra_notes": (
            get_or_build_index("pq_vector_infra", SimpleDirectoryReader(str(DATA / "graph")).load_data()),
            "Vector databases and infrastructure: Qdrant, HNSW, quantization, Neo4j, LlamaIndex, Keycloak, Graph RAG.",
        ),
        "threat_report": (
            get_or_build_index("pq_threat_report", load_pdfs(DATA / "pdf")),
            "Anthropic's threat intelligence report on detecting and countering misuse of AI models.",
        ),
    }
    return make_tools(indexes, llm), llm


def run_queries(args) -> None:
    import pandas as pd

    from src.retrieval.parallel_query import benchmark_queries

    tools, _ = build_tools()
    engine = tools[0].query_engine  # one index; every question still exercises embed + search + LLM
    print(f"{len(QUESTIONS)} questions\n")
    print(pd.DataFrame(benchmark_queries(engine, QUESTIONS, args.concurrency)).to_string(index=False))


def run_subquestion(_args) -> None:
    from src.retrieval.parallel_query import build_sub_question_engine, median_seconds, time_aquery

    tools, llm = build_tools()
    print(f"Question: {MULTI_HOP}\n(median of 3 runs after 1 warm-up)\n")
    for use_async in (False, True):
        engine = build_sub_question_engine(tools, llm, use_async)
        seconds = median_seconds(lambda: time_aquery(engine, MULTI_HOP))
        print(f"use_async={use_async}: {seconds:.1f}s | {time_aquery(engine, MULTI_HOP)[1][:140].strip()!r}")


def run_router(_args) -> None:
    from src.retrieval.parallel_query import build_router_engine, median_seconds, time_aquery

    tools, llm = build_tools()
    engine = build_router_engine(tools, llm)
    print(f"Question: {MULTI_HOP}\n(median of 3 runs after 1 warm-up; the LLM selector may pick a different number of indexes per run)\n")
    print(f"router multi-select, aquery fan-out: {median_seconds(lambda: time_aquery(engine, MULTI_HOP)):.1f}s")
    print(f"router multi-select, sync query:     {median_seconds(lambda: engine.query(MULTI_HOP)):.1f}s")


def run_agent(_args) -> None:
    from src.agents.parallel_tools_agent import build_parallel_tool_agent, run_agent_timed

    tools, llm = build_tools()
    print(f"Question: {MULTI_HOP}\n(LLM behaviour varies run to run; treat timings as indicative)\n")
    for allow in (False, True):
        result = run_agent_timed(build_parallel_tool_agent(tools, llm, allow_parallel=allow), MULTI_HOP)
        print(f"allow_parallel_tool_calls={allow}: {result['seconds']}s, {result['tool_calls']} tool calls | "
              f"{result['answer'][:120].strip()!r}")


def run_eval(_args) -> None:
    from src.evaluation.langsmith_eval import evaluate_rag_pipeline
    from src.evaluation.recall import LABELED_QUESTIONS

    tools, _ = build_tools()
    engine = tools[0].query_engine
    examples = [
        {"inputs": {"question": q}, "outputs": {"answer": kw, "expected_keyword": kw}}
        for q, kw in LABELED_QUESTIONS[:8]
    ]

    def query_fn(question: str) -> dict:
        response = engine.query(question)
        return {"answer": str(response), "contexts": [n.node.get_content() for n in response.source_nodes]}

    for concurrency in (1, 4):
        start = time.perf_counter()
        evaluate_rag_pipeline(
            query_fn, dataset_name="advanced-rag-toolkit-qa-parallel", examples=examples, provider="ollama",
            experiment_prefix=f"parallel-eval-c{concurrency}", max_concurrency=concurrency,
        )
        print(f"evaluate(max_concurrency={concurrency}): {time.perf_counter() - start:.1f}s for {len(examples)} examples")


MODES = {"queries": run_queries, "subquestion": run_subquestion, "router": run_router,
         "agent": run_agent, "eval": run_eval}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=[*MODES, "all"])
    p.add_argument("--concurrency", type=int, default=4, help="semaphore bound for bounded gather")
    a = p.parse_args()
    for name in (MODES if a.mode == "all" else [a.mode]):
        print(f"\n===== {name} =====")
        MODES[name](a)
