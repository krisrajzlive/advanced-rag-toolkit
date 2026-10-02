"""CLI entry point for the Advanced RAG Toolkit demos.

    python main.py load-all                 # load every modality, print doc counts
    python main.py compare-metrics "<query>" # cosine vs dot_product vs euclidean
    python main.py query-expansion "<query>"
    python main.py rerank "<query>"
    python main.py recursive "<query>"
    python main.py agent-llamaindex "<query>"
    python main.py agent-langchain "<query>"
    python main.py evaluate
    python main.py normalize-demo           # raw vs L2-normalized embeddings
    python main.py quant-bench              # Qdrant scalar/binary/product quantization
    python main.py hnsw-bench               # Qdrant HNSW m / ef sweep
    python main.py tenant-demo "<query>"    # Keycloak JWT -> tenant-isolated retrieval
    python main.py kg-build                 # extract knowledge graph into Neo4j
    python main.py kg-show                  # print graph stats + triples
    python main.py graph-rag "<query>"      # Graph RAG answer
    python main.py graph-vs-vector "<query>"  # Graph RAG vs plain vector RAG
    python main.py recall [ann|retrieval|all] # recall measurement + tuning

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


def cmd_normalize_demo(_args: argparse.Namespace) -> None:
    from src.indexing.normalization import compare_normalization

    queries = [
        "What is cosine similarity?",
        "Which metric is magnitude-sensitive?",
        "How does reranking work?",
        "What is recursive retrieval?",
    ]
    docs = load_all_documents_light()
    for label, stats in compare_normalization(docs, queries).items():
        print(f"{label}: {stats}")


def load_all_documents_light():
    """text + json + csv only (skips the 154-page PDF, images, audio)."""
    from src.loaders import load_csv, load_json, load_text

    return load_text(DATA_DIR / "text") + load_json(DATA_DIR / "json") + load_csv(DATA_DIR / "csv")


def cmd_quant_bench(_args: argparse.Namespace) -> None:
    import pandas as pd

    from src.indexing.qdrant_store import get_qdrant_client
    from src.indexing.quantization import benchmark_quantization

    print(pd.DataFrame(benchmark_quantization(get_qdrant_client())).to_string(index=False))


def cmd_hnsw_bench(_args: argparse.Namespace) -> None:
    import pandas as pd

    from src.indexing.hnsw import benchmark_hnsw
    from src.indexing.qdrant_store import get_qdrant_client

    print(pd.DataFrame(benchmark_hnsw(get_qdrant_client())).to_string(index=False))


def cmd_tenant_demo(args: argparse.Namespace) -> None:
    from src.config import configure_llamaindex_settings
    from src.tenancy import (
        build_tenant_index,
        get_access_token,
        tenant_retriever,
        verify_token,
    )

    configure_llamaindex_settings(normalize=True)
    index = build_tenant_index(DATA_DIR / "tenants", normalized=True)
    for username in ("alice", "bob", "carol"):
        ctx = verify_token(get_access_token(username))
        print(f"\n{ctx.username} (tenants: {', '.join(ctx.tenants)})")
        for n in tenant_retriever(index, ctx).retrieve(args.query):
            snippet = n.node.get_content()[:90].replace("\n", " ")
            print(f"  [{n.node.metadata['tenant_id']:6s} {n.score:.3f}] {snippet}")


def cmd_kg_build(_args: argparse.Namespace) -> None:
    from src.graph import build_knowledge_graph, get_graph_store, graph_stats

    build_knowledge_graph(DATA_DIR / "graph")
    print(graph_stats(get_graph_store()))


def cmd_kg_show(_args: argparse.Namespace) -> None:
    from src.graph import get_graph_store, graph_stats, list_triples

    store = get_graph_store()
    print(graph_stats(store))
    for s, p, o in list_triples(store, limit=80):
        print(f"  ({s}) -[{p}]-> ({o})")
    print("\nBrowse it: http://localhost:7475")


def cmd_graph_rag(args: argparse.Namespace) -> None:
    from src.graph import build_graph_rag_query_engine, load_knowledge_graph

    print(build_graph_rag_query_engine(load_knowledge_graph()).query(args.query))


def cmd_graph_vs_vector(args: argparse.Namespace) -> None:
    from llama_index.core import SimpleDirectoryReader

    from src.graph import (
        build_graph_rag_query_engine,
        build_plain_rag_query_engine,
        load_knowledge_graph,
    )

    docs = SimpleDirectoryReader(str(DATA_DIR / "graph")).load_data()
    print("=== Plain vector RAG ===")
    print(build_plain_rag_query_engine(docs).query(args.query))
    print("\n=== Graph RAG ===")
    print(build_graph_rag_query_engine(load_knowledge_graph()).query(args.query))


def cmd_recall(args: argparse.Namespace) -> None:
    import subprocess

    mode = args.query if args.query in ("ann", "retrieval", "all") else "all"
    script = Path(__file__).parent / "scripts" / "tune_recall.py"
    subprocess.run([sys.executable, str(script), mode], check=True)


def cmd_sync(_args: argparse.Namespace) -> None:
    from src.cdc import sync_directory
    from src.config import configure_llamaindex_settings

    configure_llamaindex_settings(normalize=True)
    result = sync_directory(DATA_DIR / "tenants")
    print(f"unchanged: {result['unchanged']}")
    for key in ("new", "changed", "deleted"):
        print(f"{key}: {result[key] or '-'}")


def cmd_cdc_setup(_args: argparse.Namespace) -> None:
    from src.cdc.debezium import register_connector
    from src.cdc.source_db import setup_source

    setup_source()
    print("Postgres table ready; Debezium connector state:", register_connector())


def cmd_cdc_consume(args: argparse.Namespace) -> None:
    from src.cdc.consumer import consume, open_index
    from src.config import configure_llamaindex_settings

    configure_llamaindex_settings(normalize=True)
    idle = float(args.query) if args.query.replace(".", "").isdigit() else 30.0
    print(f"Consuming change events (exits after {idle:.0f}s idle)...")
    for line in consume(open_index(), idle_timeout=idle):
        print("  ", line)


def cmd_cdc_demo(_args: argparse.Namespace) -> None:
    import time

    from src.cdc import source_db
    from src.cdc.consumer import COLLECTION, consume, open_index
    from src.cdc.debezium import register_connector
    from src.config import configure_llamaindex_settings
    from src.indexing.qdrant_store import get_qdrant_client
    from src.tenancy import TenantContext
    from src.tenancy.tenant_rag import tenant_retriever

    configure_llamaindex_settings(normalize=True)
    source_db.setup_source(seed=False)
    print("Connector:", register_connector())
    index = open_index()
    # Self-contained: this run creates (and finally removes) its own rows.
    esc_id = source_db.insert_doc("acme", "Roadrunner escalation", "The Roadrunner support escalation code is ACME-RR-7741.")
    refund_id = source_db.insert_doc("globex", "Doomsday refunds", "Globex subscriptions can be cancelled within 14 days for a prorated refund.")

    def drain(title: str, idle: float) -> None:
        print(f"\n== {title}")
        for line in consume(index, idle_timeout=idle):
            print("  ", line)

    def show(tenant: str, question: str) -> None:
        nodes = tenant_retriever(index, TenantContext("demo", (tenant,)), 1).retrieve(question)
        text = nodes[0].node.get_content().replace("\n", " ")[:100] if nodes else "(nothing)"
        print(f"  [{tenant}] {question!r} -> {text}")

    client = get_qdrant_client()
    drain("existing rows / new rows arrive as events", idle=20)
    show("acme", "What is the Roadrunner escalation code?")
    show("globex", "What is the refund policy?")
    print("  points in Qdrant:", client.count(COLLECTION, exact=True).count)

    print("\n== applying changes in Postgres (INSERT / UPDATE / DELETE)")
    plan_id = source_db.insert_doc("acme", "Acme Q4 plan", "Acme will migrate vector search to a sharded cluster in Q4.")
    source_db.update_doc(esc_id, "The Roadrunner support escalation code is now ACME-RR-9999.")
    source_db.delete_doc(refund_id)
    time.sleep(2)
    drain("change events streamed by Debezium", idle=20)

    print("\n== retrieval after CDC")
    show("acme", "What is the Roadrunner escalation code?")
    show("acme", "What is planned for Q4?")
    show("globex", "What is the refund policy?")
    print("  points in Qdrant:", client.count(COLLECTION, exact=True).count)
    for row_id in (esc_id, plan_id):  # tidy up; Qdrant catches up on the next run
        source_db.delete_doc(row_id)


COMMANDS = {
    "sync": (cmd_sync, False),
    "cdc-setup": (cmd_cdc_setup, False),
    "cdc-consume": (cmd_cdc_consume, False),
    "cdc-demo": (cmd_cdc_demo, False),
    "normalize-demo": (cmd_normalize_demo, False),
    "quant-bench": (cmd_quant_bench, False),
    "hnsw-bench": (cmd_hnsw_bench, False),
    "tenant-demo": (cmd_tenant_demo, True),
    "kg-build": (cmd_kg_build, False),
    "kg-show": (cmd_kg_show, False),
    "graph-rag": (cmd_graph_rag, True),
    "graph-vs-vector": (cmd_graph_vs_vector, True),
    "recall": (cmd_recall, False),
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
