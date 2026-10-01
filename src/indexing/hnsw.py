"""HNSW graph configuration + recall/latency sweep (Qdrant-native).

Qdrant indexes vectors with HNSW. The knobs:

- m               edges per node (higher = better recall, more RAM)
- ef_construct    candidate-list size while building (higher = better graph, slower build)
- hnsw_ef         candidate-list size at *query* time (the recall/latency dial)

`benchmark_hnsw()` sweeps `hnsw_ef` and (m, ef_construct) against exact search.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from src.indexing.quantization import (
    make_collection,
    recall_at_k,
    synthetic_corpus,
    time_queries,
)


@dataclass(frozen=True)
class HnswParams:
    m: int = 16
    ef_construct: int = 100
    full_scan_threshold: int = 10_000  # KB; below this a plain scan is used
    payload_m: int | None = None  # per-payload-value graphs (multi-tenancy)


def hnsw_config(p: HnswParams) -> models.HnswConfigDiff:
    return models.HnswConfigDiff(
        m=p.m,
        ef_construct=p.ef_construct,
        full_scan_threshold=p.full_scan_threshold,
        payload_m=p.payload_m,
    )


def benchmark_hnsw(
    client: QdrantClient, n: int = 20000, dim: int = 1536, n_queries: int = 100, k: int = 10
) -> list[dict]:
    vectors, queries = synthetic_corpus(n, dim, n_queries)
    rows: list[dict] = []
    truth = None
    for m, efc in ((8, 50), (16, 100), (32, 200)):
        name = f"bench_hnsw_m{m}"
        make_collection(client, name, dim, hnsw=hnsw_config(HnswParams(m=m, ef_construct=efc)),
                        vectors=vectors)
        if truth is None:
            truth, ms_exact = time_queries(client, name, queries, k, models.SearchParams(exact=True))
            rows.append({"config": "exact (brute force)", "recall@k": 1.0,
                         "latency_ms": round(ms_exact, 2)})
        for ef in (8, 32, 128, 512):
            found, ms = time_queries(client, name, queries, k, models.SearchParams(hnsw_ef=ef))
            rows.append({"config": f"HNSW m={m} ef_construct={efc} hnsw_ef={ef}",
                         "recall@k": round(recall_at_k(found, truth), 3),
                         "latency_ms": round(ms, 2)})
        client.delete_collection(name)
    return rows
