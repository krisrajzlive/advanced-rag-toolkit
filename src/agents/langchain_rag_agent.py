"""RAG agent built with LangChain's native `create_react_agent` (LangGraph).

Reuses the LlamaIndex query engine (built by `src.indexing` /
`src.retrieval`) as the retrieval tool, so both frameworks share one
underlying index instead of maintaining two separate vector stores -
this module is about LangChain's agent orchestration, not re-indexing.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from src.config import get_langchain_model

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the "
    "search_knowledge_base tool. Always search before answering questions "
    "about the corpus, and cite the retrieved content in your answer."
)


def build_langchain_rag_agent(
    index_or_query_engine: VectorStoreIndex | BaseQueryEngine,
    provider: str | None = None,
):
    query_engine = (
        index_or_query_engine
        if isinstance(index_or_query_engine, BaseQueryEngine)
        else index_or_query_engine.as_query_engine(similarity_top_k=3)
    )

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the indexed knowledge base (text, JSON, CSV, PDF, image
        captions, audio/video transcripts) for information relevant to the
        query."""
        return str(query_engine.query(query))

    model = get_langchain_model(provider)
    return create_react_agent(model, tools=[search_knowledge_base], prompt=SYSTEM_PROMPT)


def run_agent(agent, query: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
