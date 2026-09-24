"""RAG agent built with LlamaIndex's native `ReActAgent` workflow.

The retriever's query engine is exposed as a single `QueryEngineTool`; the
`ReActAgent` decides when to call it and reasons over the result, rather
than the query engine being called directly.
"""

from __future__ import annotations

import asyncio

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

DEFAULT_TOOL_DESCRIPTION = (
    "Search the indexed knowledge base (text, JSON, CSV, PDF, image captions, "
    "and audio/video transcripts) for information relevant to the query. "
    "Always use this tool before answering questions about the corpus."
)


def build_llamaindex_rag_agent(
    index_or_query_engine: VectorStoreIndex | BaseQueryEngine,
    tool_name: str = "knowledge_base_search",
    tool_description: str = DEFAULT_TOOL_DESCRIPTION,
    system_prompt: str | None = None,
) -> ReActAgent:
    query_engine = (
        index_or_query_engine
        if isinstance(index_or_query_engine, BaseQueryEngine)
        else index_or_query_engine.as_query_engine(similarity_top_k=3)
    )

    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name=tool_name, description=tool_description),
    )

    return ReActAgent(
        tools=[tool],
        llm=Settings.llm,
        system_prompt=system_prompt
        or "You are a helpful assistant that answers questions using the "
        "knowledge_base_search tool. Cite the retrieved content in your answer.",
    )


def run_agent(agent: ReActAgent, query: str) -> str:
    """Synchronous convenience wrapper around the agent's async workflow run."""

    async def _run() -> str:
        handler = agent.run(user_msg=query)
        result = await handler
        return str(result)

    return asyncio.run(_run())
