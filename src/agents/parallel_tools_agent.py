"""Agent that fans tool calls out in parallel (LlamaIndex `FunctionAgent`).

When the LLM asks for several tools in one step (e.g. "search the RAG notes,
the infra notes and the report"), `allow_parallel_tool_calls=True` makes the
agent workflow execute them concurrently (`asyncio.gather` internally) instead
of one after another. Requires a function-calling LLM.
"""

from __future__ import annotations

import asyncio
import time

from llama_index.core.agent.workflow import FunctionAgent, ToolCall

SYSTEM_PROMPT = (
    "You answer questions using the provided search tools. When a question spans "
    "several topics, call ALL relevant tools in the same step, then combine their results."
)


def build_parallel_tool_agent(tools, llm, allow_parallel: bool = True) -> FunctionAgent:
    return FunctionAgent(
        tools=list(tools),
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        allow_parallel_tool_calls=allow_parallel,
    )


async def _run(agent: FunctionAgent, question: str) -> tuple[str, int]:
    handler = agent.run(question)
    tool_calls = 0
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            tool_calls += 1
    return str(await handler), tool_calls


def run_agent_timed(agent: FunctionAgent, question: str) -> dict:
    start = time.perf_counter()
    answer, tool_calls = asyncio.run(_run(agent, question))
    return {"seconds": round(time.perf_counter() - start, 1), "tool_calls": tool_calls, "answer": answer}
