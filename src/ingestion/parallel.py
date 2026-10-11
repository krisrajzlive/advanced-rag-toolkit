"""Parallel ingestion strategies for `IngestionPipeline` (LlamaIndex-native).

Embedding is the slow step of ingestion (one API round trip per batch), so
there are three independent levers, all built into LlamaIndex:

- `embed_batch_size`   texts per embedding request (fewer, larger round trips)
- async embedding      `pipeline.arun(...)` plus the embed model's `num_workers`
                       sends several batches concurrently from one process
- process workers      `pipeline.run(num_workers=N)` splits the nodes across N
                       spawned processes, each chunking + embedding its share

Each strategy is timed on the same documents into a fresh Qdrant collection.
No custom concurrency code: only these LlamaIndex options are used.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from src.config import get_llamaindex_embed_model
from src.indexing.qdrant_store import get_qdrant_client

COLLECTION = "bench_ingest"


@dataclass(frozen=True)
class Strategy:
    name: str
    embed_batch_size: int = 10
    embed_workers: int | None = None  # concurrent async batches (async mode)
    process_workers: int | None = None  # spawned processes (pipeline.run)
    use_async: bool = False
    gather_batch_size: int | None = None  # split documents into batches run via asyncio.gather
    gather_concurrency: int = 4  # semaphore bound for those batches


def default_strategies(
    process_workers: int = 2, async_workers: int = 8, sweep: tuple[int, ...] = ()
) -> list[Strategy]:
    process_strategies = [Strategy(f"{n} worker processes", process_workers=n) for n in (sweep or (process_workers,))]
    return [
        Strategy("baseline (sequential, batch 10)"),
        Strategy("larger batches (batch 100)", embed_batch_size=100),
        Strategy(f"async ({async_workers} concurrent batches)", embed_workers=async_workers, use_async=True),
        Strategy(f"async + larger batches (50 x {async_workers})", embed_batch_size=50,
                 embed_workers=async_workers, use_async=True),
        Strategy("document batches gathered (50 docs, semaphore 4, batch 100)", embed_batch_size=100,
                 use_async=True, gather_batch_size=50, gather_concurrency=4),
        Strategy(f"async + {process_workers} worker processes", embed_batch_size=50,
                 embed_workers=async_workers, process_workers=process_workers, use_async=True),
        *process_strategies,
    ]


def chunk_list(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


class PeakRss:
    """Context manager sampling peak resident memory (MB) of this process + children."""

    def __enter__(self):
        import threading

        import psutil

        self._psutil, self._proc = psutil, psutil.Process()
        self.peak_mb, self._stop = 0.0, threading.Event()

        def sample():
            while not self._stop.is_set():
                try:
                    rss = self._proc.memory_info().rss + sum(
                        c.memory_info().rss for c in self._proc.children(recursive=True)
                    )
                    self.peak_mb = max(self.peak_mb, rss / 1e6)
                except self._psutil.Error:
                    pass
                self._stop.wait(0.05)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


async def _gather_batches(pipeline, documents, batch_size, concurrency, process_workers):
    """Run `pipeline.arun` over document batches concurrently, bounded by a semaphore."""
    semaphore = asyncio.Semaphore(concurrency)

    async def run(batch):
        async with semaphore:
            return await pipeline.arun(documents=batch, num_workers=process_workers)

    results = await asyncio.gather(*[run(b) for b in chunk_list(documents, batch_size)])
    return [node for batch_nodes in results for node in batch_nodes]


def replicate(documents: list[Document], times: int) -> list[Document]:
    """Copy the corpus `times` times with unique ids (a bigger, realistic load)."""
    out = []
    for i in range(times):
        for doc in documents:
            out.append(Document(text=doc.text, metadata=dict(doc.metadata), id_=f"{doc.doc_id}#{i}"))
    return out


def _fresh_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )


def run_strategy(strategy: Strategy, documents: list[Document], client: QdrantClient | None = None) -> dict:
    """Ingest `documents` with `strategy` into a fresh collection; return timing."""
    client = client or get_qdrant_client()
    embed_kwargs = {"embed_batch_size": strategy.embed_batch_size}
    if strategy.embed_workers:
        embed_kwargs["num_workers"] = strategy.embed_workers
    embed_model = get_llamaindex_embed_model(**embed_kwargs)
    _fresh_collection(client, len(embed_model.get_text_embedding("dimension probe")))

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50), embed_model],
        vector_store=QdrantVectorStore(
            client=client,
            aclient=AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333")),
            collection_name=COLLECTION,
        ),
    )
    with PeakRss() as memory:
        start = time.perf_counter()
        if strategy.gather_batch_size:
            nodes = asyncio.run(_gather_batches(
                pipeline, documents, strategy.gather_batch_size, strategy.gather_concurrency,
                strategy.process_workers,
            ))
        elif strategy.use_async:
            nodes = asyncio.run(pipeline.arun(documents=documents, num_workers=strategy.process_workers))
        else:
            nodes = pipeline.run(documents=documents, num_workers=strategy.process_workers)
        seconds = time.perf_counter() - start

    stored = client.count(COLLECTION, exact=True).count
    client.delete_collection(COLLECTION)
    if stored != len(nodes):
        raise RuntimeError(f"{strategy.name}: {len(nodes)} nodes produced but {stored} stored")
    return {
        "strategy": strategy.name,
        "nodes": len(nodes),
        "seconds": round(seconds, 1),
        "nodes_per_sec": round(len(nodes) / seconds, 1),
        "peak_rss_mb": round(memory.peak_mb),
    }


def benchmark(documents: list[Document], strategies: list[Strategy] | None = None) -> list[dict]:
    strategies = strategies or default_strategies(process_workers=min(2, os.cpu_count() or 1))
    rows = [run_strategy(s, documents) for s in strategies]
    base = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = round(base / row["seconds"], 2)
    return rows
