"""Measure recall and recommend settings that improve it.

    python scripts/tune_recall.py ann        # index recall: HNSW + quantization sweep
    python scripts/tune_recall.py retrieval  # end-to-end recall: top-k / expansion / rerank
    python scripts/tune_recall.py all
Options: --target 0.95 (ANN recall goal)  --n 20000  --dim 1536
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging  # noqa: E402

logging.disable(logging.WARNING)  # silence pypdf font-map warnings

import pandas as pd  # noqa: E402

from src.config import configure_llamaindex_settings  # noqa: E402


def run_ann(args) -> None:
    from src.evaluation.recall import tune_ann_recall
    from src.indexing.qdrant_store import get_qdrant_client

    rows, best = tune_ann_recall(get_qdrant_client(), n=args.n, dim=args.dim, target=args.target)
    df = pd.DataFrame(rows).sort_values(["recall@k", "latency_ms"], ascending=[False, True])
    print(df.to_string(index=False))
    print(f"\nTarget recall@10 >= {args.target}: ", end="")
    print(f"fastest config -> {best}" if best else "no configuration reached it; raise hnsw_ef/oversampling or use scalar/none")


def run_retrieval(_args) -> None:
    from src.evaluation.recall import improve_retrieval_recall
    from src.graph.knowledge_graph import graph_llm
    from src.indexing.qdrant_store import build_qdrant_index
    from src.loaders import load_csv, load_json, load_pdfs, load_text

    configure_llamaindex_settings(normalize=True)
    data = Path(__file__).resolve().parents[1] / "data"
    docs = load_text(data / "text") + load_json(data / "json") + load_csv(data / "csv") + load_pdfs(data / "pdf")  # PDF = realistic distractors
    index = build_qdrant_index(docs, "recall_eval", normalized=True, quantization="scalar")
    rows = improve_retrieval_recall(index, graph_llm(), documents=docs)
    print(pd.DataFrame(rows).to_string(index=False))
    best = max(rows, key=lambda r: (r["recall"], -r["extra_llm_calls_per_query"]))
    print(f"\nBest: {best['strategy']} (recall {best['recall']}, "
          f"{best['extra_llm_calls_per_query']} extra LLM call(s)/query)")


def run_cutoff(_args) -> None:
    from src.evaluation.recall import sweep_similarity_cutoff
    from src.indexing.qdrant_store import attach_qdrant_index, build_qdrant_index, get_qdrant_client
    from src.loaders import load_csv, load_json, load_pdfs, load_text

    configure_llamaindex_settings(normalize=True)
    client = get_qdrant_client()
    from llama_index.core import Settings

    dim = len(Settings.embed_model.get_text_embedding("dimension probe"))
    existing = client.get_collection("recall_eval").config.params.vectors.size if client.collection_exists("recall_eval") else None
    if existing == dim:  # reuse only when built with the same embedding model
        index = attach_qdrant_index("recall_eval", client)
    else:
        data = Path(__file__).resolve().parents[1] / "data"
        docs = load_text(data / "text") + load_json(data / "json") + load_csv(data / "csv") + load_pdfs(data / "pdf")
        index = build_qdrant_index(docs, "recall_eval", normalized=True, quantization="scalar")
    rows, best = sweep_similarity_cutoff(index)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nRecommended similarity_cutoff = {best} (lowest cutoff that removes out-of-scope noise without losing recall)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ann", "retrieval", "cutoff", "all"])
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--dim", type=int, default=1536)
    a = p.parse_args()
    if a.mode in ("ann", "all"):
        run_ann(a)
    if a.mode in ("retrieval", "all"):
        run_retrieval(a)
    if a.mode in ("cutoff", "all"):
        run_cutoff(a)
