"""Vector quantization (Qdrant-native) + a recall/latency/memory benchmark.

Three schemes, all configured on the Qdrant collection (no custom math):

- scalar  - float32 -> int8 (4x smaller), near-lossless.
- binary  - float32 -> 1 bit/dim (32x smaller), fastest, needs rescoring and
            high-dimensional embeddings (>= ~1024 dims) to keep recall.
- product - PQ, 16x smaller (4 float32 dims -> 1 byte); lossy, slowest to build.

All quantization and HNSW work is done by Qdrant itself; this module only
passes its config objects and measures results.

Quantized vectors drive the fast first-pass search kept in RAM; searches can
"oversample" candidates and rescore them against the original float vectors.
`benchmark_quantization()` measures the trade-off on a synthetic corpus.
"""

from __future__ import annotations

import time

import numpy as np
from qdrant_client import QdrantClient, models

QUANT_KINDS = ("none", "scalar", "binary", "product")


def quantization_config(kind: str, always_ram: bool = True):
    """Return the Qdrant `quantization_config` for `kind` (None for 'none')."""
    if kind == "none":
        return None
    if kind == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=always_ram
            )
        )
    if kind == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=always_ram)
        )
    if kind == "product":
        return models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X16, always_ram=always_ram
            )
        )
    raise ValueError(f"Unknown quantization kind {kind!r}; choose from {QUANT_KINDS}")


def search_params(kind: str, oversampling: float = 3.0, rescore: bool = True, hnsw_ef=None):
    quant = None
    if kind != "none":
        quant = models.QuantizationSearchParams(
            ignore=False, rescore=rescore, oversampling=oversampling if rescore else None
        )
    return models.SearchParams(hnsw_ef=hnsw_ef, quantization=quant)


def wait_until_indexed(client: QdrantClient, name: str, timeout: float = 600) -> None:
    """Block until the optimizer has built HNSW + quantized data (status green)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get_collection(name).status == models.CollectionStatus.GREEN:
            return
        time.sleep(1)
    raise TimeoutError(f"{name} not indexed within {timeout}s")


def synthetic_corpus(n: int, dim: int, n_queries: int, seed: int = 7):
    """Unit-norm clustered Gaussian vectors (stand-in for real embeddings)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(50, n // 100), dim))
    assign = rng.integers(0, len(centers), size=n + n_queries)
    x = centers[assign] + 0.6 * rng.normal(size=(n + n_queries, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x[:n].astype(np.float32), x[n:].astype(np.float32)


def time_queries(client, name, queries, k, params):
    """Return (list of top-k id lists, mean latency in ms)."""
    results, t0 = [], time.perf_counter()
    for q in queries:
        hits = client.query_points(
            name, query=q.tolist(), limit=k, search_params=params, with_payload=False
        ).points
        results.append([h.id for h in hits])
    return results, (time.perf_counter() - t0) / len(queries) * 1000


def recall_at_k(found: list[list[int]], truth: list[list[int]]) -> float:
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)]))


def make_collection(client, name, dim, *, kind="none", hnsw=None, vectors=None):
    """(Re)create a benchmark collection, upload `vectors`, wait for indexing."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
        hnsw_config=hnsw,
        quantization_config=quantization_config(kind),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
    )
    if vectors is not None:
        client.upload_collection(name, vectors=vectors, ids=range(len(vectors)), batch_size=256)
        wait_until_indexed(client, name)


def benchmark_quantization(
    client: QdrantClient, n: int = 20000, dim: int = 1536, n_queries: int = 100, k: int = 10
) -> list[dict]:
    """Compare none / scalar / binary / product on recall@k and latency.

    Ground truth is an exact (brute-force) search on the unquantized
    collection. Synthetic data - absolute recall will differ on real corpora.
    """
    vectors, queries = synthetic_corpus(n, dim, n_queries)
    rows: list[dict] = []
    truth = None
    for kind in QUANT_KINDS:
        name = f"bench_quant_{kind}"
        make_collection(client, name, dim, kind=kind, vectors=vectors)
        if truth is None:  # 'none' runs first -> exact ground truth
            truth, _ = time_queries(client, name, queries, k, models.SearchParams(exact=True))
        for rescore in (False, True) if kind != "none" else (False,):
            found, ms = time_queries(client, name, queries, k, search_params(kind, rescore=rescore))
            rows.append({
                "quantization": kind + ("+rescore" if rescore else ""),
                "recall@k": round(recall_at_k(found, truth), 3),
                "latency_ms": round(ms, 2),
            })
        client.delete_collection(name)
    return rows
