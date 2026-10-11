import asyncio
import time

from src.ingestion.parallel import chunk_list, default_strategies
from src.retrieval.parallel_query import _gather, benchmark_queries, median_seconds


class FakeEngine:
    """Stands in for a query engine: 50 ms of 'I/O' per query; tracks peak concurrency."""

    def __init__(self, fail_on=None):
        self.in_flight = self.peak = 0
        self.fail_on = fail_on

    def query(self, question):
        time.sleep(0.05)

    async def aquery(self, question):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.05)
            if question == self.fail_on:
                raise RuntimeError("429 rate limited")
        finally:
            self.in_flight -= 1


def test_gather_is_much_faster_than_sequential():
    rows = benchmark_queries(FakeEngine(), [f"q{i}" for i in range(10)], concurrency=5)
    by_mode = {}
    for r in rows:
        key = "bounded" if "bounded" in r["mode"] else "gather" if r["mode"].startswith("gather") else "sequential"
        by_mode[key] = r
    assert by_mode["sequential"]["wall_s"] >= 0.4
    assert by_mode["gather"]["speedup"] > 3
    assert by_mode["bounded"]["wall_s"] < by_mode["sequential"]["wall_s"]


def test_semaphore_bounds_in_flight_queries():
    engine = FakeEngine()
    asyncio.run(_gather(engine, [f"q{i}" for i in range(12)], concurrency=3))
    assert engine.peak == 3
    unbounded = FakeEngine()
    asyncio.run(_gather(unbounded, [f"q{i}" for i in range(12)], concurrency=None))
    assert unbounded.peak == 12


def test_one_failing_query_is_counted_not_fatal():
    results = asyncio.run(_gather(FakeEngine(fail_on="bad"), ["ok", "bad", "ok"], concurrency=2))
    assert [good for _, good in results] == [True, False, True]


def test_chunk_list_and_median_seconds():
    assert chunk_list(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    calls = []
    assert median_seconds(lambda: calls.append(1), repeats=3, warmup=2) >= 0
    assert len(calls) == 5


def test_default_strategies_cover_gather_async_processes_and_sweep():
    names = [s.name for s in default_strategies(process_workers=2, async_workers=8)]
    assert any("document batches gathered" in n for n in names)
    assert any(n.startswith("async + 2 worker processes") for n in names)
    sweep = [s.process_workers for s in default_strategies(sweep=(1, 4)) if s.name.endswith("worker processes") and s.process_workers in (1, 4)]
    assert sweep == [1, 4]
