"""Knowledge-graph construction in Neo4j (LlamaIndex `PropertyGraphIndex`).

An LLM extracts (subject, relation, object) triples from each chunk
(`SimpleLLMPathExtractor`); entities become `__Entity__` nodes, chunks become
`Chunk` nodes linked to their entities by `MENTIONS`, and entity embeddings are
stored in Neo4j's native vector index. Everything is LlamaIndex/Neo4j native.
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from src.config import get_llamaindex_embed_model, get_llamaindex_llm


def graph_llm():
    """Extraction/answer LLM: Ollama Cloud when available (fast, tool-capable)."""
    if os.environ.get("RAG_LLM_PROVIDER") or not os.environ.get("OLLAMA_API_KEY"):
        return get_llamaindex_llm()
    return get_llamaindex_llm("ollama")


def get_graph_store() -> Neo4jPropertyGraphStore:
    return Neo4jPropertyGraphStore(
        username=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ["NEO4J_PASSWORD"],
        url=os.environ.get("NEO4J_URI", "bolt://localhost:7688"),
    )


def build_knowledge_graph(corpus_dir: Path, reset: bool = True) -> PropertyGraphIndex:
    """Extract triples from every file in `corpus_dir` and write them to Neo4j."""
    store = get_graph_store()
    if reset:
        store.structured_query("MATCH (n) DETACH DELETE n")
    documents = SimpleDirectoryReader(str(corpus_dir)).load_data()
    return PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=store,
        llm=graph_llm(),
        embed_model=get_llamaindex_embed_model(),
        kg_extractors=[SimpleLLMPathExtractor(llm=graph_llm(), max_paths_per_chunk=12, num_workers=4)],
        transformations=[SentenceSplitter(chunk_size=300, chunk_overlap=30)],
        embed_kg_nodes=True,
        show_progress=True,
    )


def load_knowledge_graph() -> PropertyGraphIndex:
    """Re-attach to the graph already stored in Neo4j (no LLM calls)."""
    return PropertyGraphIndex.from_existing(
        property_graph_store=get_graph_store(),
        llm=graph_llm(),
        embed_model=get_llamaindex_embed_model(),
    )


def graph_stats(store: Neo4jPropertyGraphStore) -> dict:
    rows = store.structured_query(
        "MATCH (n) RETURN count(n) AS nodes UNION ALL MATCH ()-[r]->() RETURN count(r) AS nodes"
    )
    entities = store.structured_query("MATCH (n:`__Entity__`) RETURN count(n) AS c")[0]["c"]
    return {"nodes": rows[0]["nodes"], "relationships": rows[1]["nodes"], "entities": entities}


def list_triples(store: Neo4jPropertyGraphStore, limit: int = 60) -> list[tuple[str, str, str]]:
    """Entity-to-entity facts (chunk MENTIONS edges excluded)."""
    rows = store.structured_query(
        "MATCH (a:`__Entity__`)-[r]->(b:`__Entity__`) "
        "RETURN a.name AS s, type(r) AS p, b.name AS o ORDER BY s, p LIMIT $limit",
        param_map={"limit": limit},
    )
    return [(r["s"], r["p"], r["o"]) for r in rows]
