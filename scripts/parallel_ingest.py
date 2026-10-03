"""Benchmark parallel ingestion strategies (batching, async, worker processes).

    python scripts/parallel_ingest.py                 # all strategies
    python scripts/parallel_ingest.py --repeat 4      # bigger corpus (default 3)
    python scripts/parallel_ingest.py --workers 3     # worker processes to try
    python scripts/parallel_ingest.py --only async    # strategies whose name contains "async"

Corpus: text + JSON + CSV + the 154-page PDF, replicated --repeat times.
Needs Qdrant (docker compose up -d) and an embedding key in .env.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.WARNING)  # silence pypdf font-map warnings

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    import pandas as pd

    from src.ingestion.parallel import benchmark, default_strategies, replicate
    from src.loaders import load_csv, load_json, load_pdfs, load_text

    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=3, help="corpus replication factor")
    p.add_argument("--workers", type=int, default=2, help="worker processes for the multiprocess strategy")
    p.add_argument("--async-workers", type=int, default=8, help="concurrent embedding batches")
    p.add_argument("--only", default="", help="run only strategies whose name contains this text")
    args = p.parse_args()

    docs = load_text(DATA / "text") + load_json(DATA / "json") + load_csv(DATA / "csv") + load_pdfs(DATA / "pdf")
    docs = replicate(docs, args.repeat)
    strategies = [s for s in default_strategies(args.workers, args.async_workers) if args.only in s.name]
    if args.only and strategies and not strategies[0].name.startswith("baseline"):
        strategies = [default_strategies()[0]] + strategies  # speedup needs the baseline
    print(f"{len(docs)} documents, {len(strategies)} strategies\n")
    print(pd.DataFrame(benchmark(docs, strategies)).to_string(index=False))


if __name__ == "__main__":  # required: worker processes are spawned on Windows
    main()
