"""Parallel query execution with LlamaIndex's native async engines.

Querying is I/O-bound (embedding call + vector search + LLM call), so the
universal speed-up is concurrency, not processes:

- `engine.aquery()` + `asyncio.gather()`   many independent questions in flight at once
- bounded gather (`asyncio.Semaphore`)     the same, capped to stay under provider rate limits
- `SubQuestionQueryEngine(use_async=True)` one complex question split into sub-questions
                                           that hit different indexes concurrently
- `RouterQueryEngine` + `LLMMultiSelector` fan one question out to several indexes at once

This works in every runtime, including serverless (no multiprocessing needed).
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool

from src.indexing.qdrant_store import attach_qdrant_index, build_qdrant_index, get_qdrant_client


def get_or_build_index(name: str, documents, quantization: str = "scalar") -> VectorStoreIndex:
    """Reuse a Qdrant collection built with the current embedding model, else build it."""
    from llama_index.core import Settings

    client = get_qdrant_client()
    if client.collection_exists(name):
        existing = client.get_collection(name).config.params.vectors.size
        dim = len(Settings.embed_model.get_text_embedding("dimension probe"))
        if existing == dim and client.count(name, exact=True).count > 0:
            return attach_qdrant_index(name, client)
    return build_qdrant_index(documents, name, quantization=quantization)


def make_tools(indexes: dict[str, tuple[VectorStoreIndex, str]], llm, top_k: int = 3) -> list[QueryEngineTool]:
    """`indexes`: name -> (index, description). One QueryEngineTool per index."""
    return [
        QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(similarity_top_k=top_k, llm=llm),
            name=name,
            description=description,
        )
        for name, (index, description) in indexes.items()
    ]


# ---------------------------------------------------------------------------
# Many independent questions: sequential vs gather vs bounded gather
# ---------------------------------------------------------------------------


async def _timed_aquery(engine, question: str, semaphore: asyncio.Semaphore | None):
    """Latency is measured from submission, so queueing behind the semaphore counts."""
    start = time.perf_counter()
    try:
        if semaphore is None:
            await engine.aquery(question)
        else:
            async with semaphore:
                await engine.aquery(question)
        return time.perf_counter() - start, True
    except Exception:  # noqa: BLE001 - e.g. provider 429; counted, not fatal
        return time.perf_counter() - start, False


async def _gather(engine, questions: list[str], concurrency: int | None):
    semaphore = asyncio.Semaphore(concurrency) if concurrency else None
    return await asyncio.gather(*[_timed_aquery(engine, q, semaphore) for q in questions])


def _row(label: str, wall: float, results: list[tuple[float, bool]]) -> dict:
    ok = [lat for lat, good in results if good]
    return {
        "mode": label,
        "wall_s": round(wall, 1),
        "queries_per_s": round(len(results) / wall, 2),
        "p50_s": round(float(np.percentile(ok, 50)), 1) if ok else None,
        "p95_s": round(float(np.percentile(ok, 95)), 1) if ok else None,
        "errors": sum(not good for _, good in results),
    }


def benchmark_queries(engine, questions: list[str], concurrency: int = 4) -> list[dict]:
    """Time the same questions run sequentially, fully concurrent, and bounded."""
    results = []
    start = time.perf_counter()
    for q in questions:
        t0 = time.perf_counter()
        try:
            engine.query(q)
            results.append((time.perf_counter() - t0, True))
        except Exception:  # noqa: BLE001
            results.append((time.perf_counter() - t0, False))
    rows = [_row("sequential (engine.query)", time.perf_counter() - start, results)]

    for label, limit in ((f"gather ({len(questions)} in flight)", None), (f"bounded gather (semaphore {concurrency})", concurrency)):
        start = time.perf_counter()
        results = asyncio.run(_gather(engine, questions, limit))
        rows.append(_row(label, time.perf_counter() - start, results))
    base = rows[0]["wall_s"]
    for row in rows:
        row["speedup"] = round(base / row["wall_s"], 2)
    return rows


# ---------------------------------------------------------------------------
# One complex question fanned out over several indexes
# ---------------------------------------------------------------------------


def build_sub_question_engine(tools, llm, use_async: bool) -> SubQuestionQueryEngine:
    # Explicit text-based generator from llama-index-core: the default one for OpenAI
    # models needs an extra package that pins an older llama-index-core.
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        llm=llm,
        question_gen=LLMQuestionGenerator.from_defaults(llm=llm),
        use_async=use_async,
        verbose=False,
    )


def build_router_engine(tools, llm) -> RouterQueryEngine:
    """Multi-select router: runs every chosen index concurrently under `aquery`."""
    return RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,
        summarizer=TreeSummarize(llm=llm),
    )


def time_aquery(engine, question: str) -> tuple[float, str]:
    start = time.perf_counter()
    response = asyncio.run(engine.aquery(question))
    return time.perf_counter() - start, str(response)


def median_seconds(fn, repeats: int = 3, warmup: int = 1) -> float:
    """Median wall time of `fn()` after warm-up runs (excludes client cold starts)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))
