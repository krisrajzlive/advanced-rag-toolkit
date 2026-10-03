# Advanced RAG Toolkit

Production-style Retrieval-Augmented Generation built with **LlamaIndex**
and **LangChain**: multimodal document ingestion, three native vector
similarity metrics, advanced retrieval techniques (query expansion,
reranking, recursive retrieval), RAG agents in both frameworks, and RAG
evaluation with LangSmith.

Everything runs on free-tier backends — Hugging Face Inference Providers,
Ollama Cloud, and (optionally) OpenAI — configured once in `src/config.py`
and shared by every demo.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`.env` was copied from `D:\workspace\prompt-engineering\.env` and already has
`OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`, `OLLAMA_API_KEY`, and the
`LANGSMITH_*` keys filled in. See `.env.example` for the expected shape.
`src/config.py` auto-selects the first available **chat** provider in this
order: Hugging Face → Ollama → OpenAI. **Embeddings** prefer OpenAI when
`OPENAI_API_KEY` is set, falling back to the Hugging Face Inference API
otherwise — `llama-index-embeddings-huggingface-api` opens a new asyncio
event loop per batch while reusing one async HTTP client, which reliably
raises `RuntimeError: Event loop is closed` on Windows' default
ProactorEventLoop once more than a few nodes are embedded (see the
docstring in `src/config.py`). If you don't have an OpenAI key, pass
`embed_batch_size=1` (already the default there) and expect it to be
slower/flakier on Windows specifically.

Optional, heavier local multimodal deps (CPU PyTorch + Transformers for
local BLIP image captioning and local Whisper transcription):

```bash
.venv\Scripts\pip install -r requirements-multimodal.txt
```

Without these, image and audio/video loaders fall back to the free Hugging
Face Inference API automatically — no code changes needed.

## Data

| Modality | Location | Source |
|---|---|---|
| Text | [data/text](data/text) | Hand-written notes on RAG + similarity metrics |
| JSON | [data/json](data/json) | `rag_techniques.json` — structured technique catalog |
| CSV | [data/csv](data/csv) | `similarity_metrics.csv` — metric formulas/use cases |
| PDF | [data/pdf](data/pdf) | Anthropic's ["Detecting and Countering Misuse"](https://www-cdn.anthropic.com/e50be2e51e7695dc4b1366a37a245a597377d3b5/Anthropic-Detecting-and-countering-091026.pdf) threat intelligence report (154 pages) |
| Image | [data/images](data/images) | Generated diagram, captioned into text at load time |
| Audio/video | [data/audio_video](data/audio_video) | Empty — drop your own file in; see that folder's README |

## Architecture

```
src/
  config.py          # provider-agnostic LLM/embedding setup (LlamaIndex + LangChain)
  loaders/            # one loader per modality -> list[Document]
  indexing/            # cosine/dot/euclidean retrieval, L2 normalization, Qdrant store, quantization, HNSW
  retrieval/           # query expansion, reranking, recursive retrieval
  agents/              # LlamaIndex ReActAgent + LangChain/LangGraph agent
  evaluation/           # LangSmith evaluate() + recall measurement/tuning
  ingestion/            # parallel ingestion strategies (batching, async, worker processes)
  cdc/                  # IngestionPipeline sync + Postgres/Debezium/Kafka CDC consumer
  tenancy/              # Keycloak JWT verification + tenant-filtered retrieval
  graph/                # Neo4j knowledge graph + Graph RAG
scripts/                # download_anthropic_pdf.py, tune_recall.py
docker/ + docker-compose.yml  # Qdrant, Neo4j, Keycloak (realm import)
```

Every loader returns plain `llama_index.core.schema.Document` objects, so
all six modalities flow into the same indexing/retrieval/agent code paths.

### Document loading (all native LlamaIndex readers)

- **Text**: `SimpleDirectoryReader`
- **JSON**: `llama_index.core.readers.json.JSONReader`
- **CSV**: `llama_index.readers.file.CSVReader`
- **PDF**: `llama_index.readers.file.PDFReader`
- **Image**: `llama_index.readers.file.ImageCaptionReader` (local BLIP) with
  a Hugging Face Inference API fallback when the multimodal extras aren't
  installed
- **Audio/video**: `llama_index.readers.file.VideoAudioReader` (local
  Whisper) with a Hugging Face Inference API fallback

### Similarity metrics — cosine / dot product / Euclidean

`src/indexing/similarity_indexes.py` builds one `SimpleVectorStore` and
exposes three retrievers over it — one per `SimilarityMode` — using
LlamaIndex's **own** `similarity()` function and `get_top_k_embeddings()`
utility (the same ones LlamaIndex uses internally). No similarity math is
hand-written.

### Advanced retrieval

- **Query expansion** — `QueryFusionRetriever` (LLM-generated query
  reformulations + reciprocal rank fusion)
- **Reranking** — `LLMRerank` node postprocessor (LLM-as-judge relevance
  scoring over a wide candidate pool)
- **Recursive retrieval** — `RecursiveRetriever` over per-document summary
  `IndexNode`s that link to each document's own chunk-level retriever

### RAG agents

- **LlamaIndex**: `llama_index.core.agent.workflow.ReActAgent` wrapping the
  query engine as a `QueryEngineTool`
- **LangChain**: `langgraph.prebuilt.create_react_agent` wrapping the same
  query engine as a `@tool`-decorated function (one shared index, two agent
  frameworks)

### Evaluation

`src/evaluation/langsmith_eval.py` uploads a small QA dataset to LangSmith
(`LANGSMITH_PROJECT` from `.env`) and runs `langsmith.evaluate()` with two
evaluators: a deterministic context-recall check and an LLM-as-judge
answer-correctness grader. Results are inspectable in the LangSmith UI.

## Run

```bash
python main.py load-all
python main.py compare-metrics "What is cosine similarity?"
python main.py query-expansion "How does RAG reduce hallucination?"
python main.py rerank "What is recursive retrieval?"
python main.py recursive "What does Euclidean distance measure?"
python main.py agent-llamaindex "Summarize the RAG pipeline stages."
python main.py agent-langchain "What similarity metric is magnitude-sensitive?"
python main.py evaluate
```

The Qdrant / Neo4j / Keycloak commands are listed in the reference table below.

## Scripts and commands reference

Every demo is a `python main.py <command> "<query>"` command; two helper scripts live in `scripts/`.

| Command / script | What it does | Needs |
|---|---|---|
| `main.py load-all` | Loads every modality (text, JSON, CSV, PDF, image, audio/video) and prints per-modality document counts | - |
| `main.py compare-metrics "<q>"` | Same query under cosine / dot product / Euclidean retrieval | embeddings |
| `main.py query-expansion "<q>"` | LLM-generated query variants fused with reciprocal rank fusion | LLM |
| `main.py rerank "<q>"` | Wide retrieval, then LLM reranking, then answer | LLM |
| `main.py recursive "<q>"` | Summary nodes that link to per-document chunk retrievers | LLM |
| `main.py agent-llamaindex "<q>"` / `agent-langchain "<q>"` | RAG agent in LlamaIndex / LangChain-LangGraph | LLM |
| `main.py evaluate` | LangSmith evaluation (context recall + LLM judge) | LangSmith key |
| `main.py normalize-demo` | Raw vs L2-normalized embeddings: vector norms and metric ranking agreement | embeddings |
| `main.py quant-bench` | Qdrant scalar / binary / product quantization: recall@10 and latency vs exact search | Docker (Qdrant) |
| `main.py hnsw-bench` | Qdrant HNSW `m` / `ef_construct` / `hnsw_ef` sweep vs exact search | Docker (Qdrant) |
| `main.py tenant-demo "<q>"` | Logs in alice, bob and carol via Keycloak, verifies each JWT, and runs tenant-isolated retrieval | Docker (Qdrant, Keycloak) |
| `main.py kg-build` | Extracts a knowledge graph from `data/graph/` into Neo4j (resets the graph first) | Docker (Neo4j), LLM |
| `main.py kg-show` | Prints graph stats and the entity-to-entity triples | Docker (Neo4j) |
| `main.py graph-rag "<q>"` | Graph RAG answer (graph retrieval over Neo4j) | Docker (Neo4j), LLM |
| `main.py graph-vs-vector "<q>"` | Plain vector RAG vs Graph RAG on the same question | Docker (Neo4j), LLM |
| `main.py recall [ann\|retrieval\|all]` | Runs `scripts/tune_recall.py` (below) | Docker (Qdrant), LLM for `retrieval` |
| `main.py sync` | Hash-based incremental sync of `data/tenants/` into Qdrant (`IngestionPipeline` + docstore): new / changed / unchanged / deleted documents. State in `.cdc_state/` | Docker (Qdrant), embeddings |
| `main.py cdc-setup` | Creates the Postgres `documents` table and registers the Debezium connector | Docker `--profile cdc` |
| `main.py cdc-consume [idle_seconds]` | Kafka consumer applying Debezium change events to Qdrant until idle | Docker `--profile cdc` |
| `main.py cdc-demo` | End to end: INSERT / UPDATE / DELETE in Postgres, then shows the change reflected in tenant-filtered retrieval | Docker `--profile cdc` |
| `scripts/tune_recall.py ann\|retrieval\|all` | Measures recall and recommends settings. `ann`: sweeps quantization / rescoring / `hnsw_ef` against exact search and picks the fastest config reaching `--target` (default 0.95; also `--n`, `--dim`). `retrieval`: compares top-k, smaller chunks, query expansion and LLM rerank on 10 labelled questions | Docker (Qdrant) |
| `scripts/parallel_ingest.py [--repeat N] [--workers W] [--async-workers A] [--only text]` | Times five ingestion strategies on the same documents (text, JSON, CSV, 154-page PDF, replicated `--repeat` times) into a fresh Qdrant collection: sequential baseline, larger embedding batches, async concurrent batches, async + larger batches, and worker processes (`IngestionPipeline.run(num_workers=...)`). Prints nodes/sec and speedup vs baseline | Docker (Qdrant), OpenAI key |
| `scripts/download_anthropic_pdf.py` | Downloads the Anthropic threat-report PDF into `data/pdf/` | internet |

## Tests

```bash
python -m pytest tests/ -q
```

## Notes

- Free-tier API calls (especially the Anthropic PDF at 154 pages, and any
  LLM reranking/agent step) can take a while — this is expected.
- The Hugging Face Inference API's model roster changes over time; if a
  fallback model 404s, swap the model id in `src/loaders/image_loader.py`
  or `src/loaders/audio_video_loader.py`, or install
  `requirements-multimodal.txt` to run fully local instead.

## Advanced vector infrastructure (Docker)

```bash
docker compose up -d     # Qdrant :6333, Neo4j :7475/:7688, Keycloak :8081
```

Set `NEO4J_PASSWORD`, `KEYCLOAK_ADMIN_PASSWORD` and `RAG_DEMO_PASSWORD` in
`.env` first (see `.env.example`). Host ports are non-default so they don't
collide with other Neo4j/Keycloak containers. **No local Ollama is used**:
LLMs come from the cloud providers above and embeddings from OpenAI.

| Feature | Where | How it works |
|---|---|---|
| **L2 normalization** | `src/indexing/normalization.py`, `get_llamaindex_embed_model(normalize=True)` | `L2NormalizedEmbedding` wraps any embed model and rescales vectors to unit length; dot == cosine, Euclidean ranks identically. `python main.py normalize-demo`. Note OpenAI vectors are already ~unit length, so the demo is only a visible change for providers that return raw vectors; `tests/test_normalization.py` proves the effect on un-normalized vectors. |
| **Quantization** | `src/indexing/quantization.py` | Qdrant-native scalar (int8), binary and product quantization, with oversampling + rescoring at query time. No custom algorithm - only Qdrant config objects. `python main.py quant-bench` |
| **HNSW** | `src/indexing/hnsw.py` | Qdrant's HNSW index configured via `m`, `ef_construct`, `hnsw_ef`. `python main.py hnsw-bench` |
| **Multi-tenancy** | `src/tenancy/`, `docker/keycloak/rag-realm.json` | One Qdrant collection, `tenant_id` payload with an `is_tenant` index and per-tenant HNSW graphs. Keycloak (realm `rag`, users `alice`/acme, `bob`/globex, `carol`/both) issues JWTs; the tenant comes only from the *verified* token's `groups` claim and becomes a mandatory retrieval filter. `python main.py tenant-demo "What is the refund policy?"` |
| **Knowledge graph** | `src/graph/knowledge_graph.py` | LLM triple extraction (`SimpleLLMPathExtractor`) into Neo4j via `PropertyGraphIndex`; browse at http://localhost:7475. `python main.py kg-build`, `kg-show` |
| **Graph RAG** | `src/graph/graph_rag.py` | `VectorContextRetriever` + `LLMSynonymRetriever` walk 2 relationship hops. `python main.py graph-rag "<q>"`, `graph-vs-vector "<q>"` |
| **Recall tuning** | `scripts/tune_recall.py`, `src/evaluation/recall.py` | `ann`: recall@10 vs exact search over quantization/rescore/`hnsw_ef`, recommends the fastest config reaching `--target`. `retrieval`: end-to-end recall on 10 labelled questions comparing top-k, smaller chunks, query expansion, LLM rerank. `python main.py recall all` |

Measured on this machine (20k synthetic 1536-d vectors, recall@10 vs exact):
scalar+rescore 1.00, binary+rescore 0.998, product+rescore 0.988; without
rescoring scalar 0.93, binary 0.26, product 0.23. HNSW reached ~0.99+ recall
even at `hnsw_ef=8` on this easy synthetic data. On the labelled retrieval set
(real text + the 154-page PDF as distractors) baseline top-2 recall was 0.9
and none of the tried strategies improved on it - the single miss is a heavily
paraphrased question; smaller chunks and LLM reranking were slightly worse.
Treat these as indicative, not general benchmarks.

Graph RAG limitation: the demo corpus (`data/graph/`) is tiny, so plain vector
RAG answers its questions equally well; Graph RAG's multi-hop advantage needs a
larger corpus. The graph is global (not tenant-scoped).

## Change Data Capture (two levels)

Keeps the Qdrant index current as the source data changes.

| | Hash-based sync | Database CDC |
|---|---|---|
| Code | `src/cdc/pipeline_sync.py` | `src/cdc/consumer.py`, `debezium.py`, `source_db.py` |
| Source | files in `data/tenants/<tenant>/` | Postgres table `documents` |
| Detects changes by | content hash in a LlamaIndex docstore (polling: run `main.py sync`) | Postgres write-ahead log read by Debezium, streamed through Kafka (event-driven) |
| Delivery | re-embeds only new/changed documents; deletes removed ones (`UPSERTS_AND_DELETE`) | each INSERT/UPDATE/DELETE becomes a Kafka event; offsets are committed only after Qdrant is updated (at-least-once, idempotent) |
| Collection | `rag_synced` | `rag_cdc` |

Both write into a multi-tenant collection (`tenant_id` payload). Start the CDC
stack with `docker compose --profile cdc up -d` (Postgres with
`wal_level=logical` on :5433, Kafka KRaft on :9094, Debezium Connect on :8083;
needs `POSTGRES_PASSWORD` in `.env`), then `python main.py cdc-demo`.

Limitations: only Qdrant is kept in sync. The Neo4j knowledge graph is not
updated by either path (changed documents would need re-extraction). The CDC
stack is memory-hungry (two JVMs capped at 512 MB each) and is behind a
compose profile so the base stack stays light. The Kafka consumer is a single
process (no scaling or dead-letter handling).

### Parallel ingestion

`scripts/parallel_ingest.py` (code in `src/ingestion/parallel.py`) compares the
three LlamaIndex-native ways to speed up ingestion; embedding API round trips
are the bottleneck, so I/O-level parallelism wins:

- **Larger embedding batches** (`embed_batch_size`): fewer requests.
- **Async embedding** (`pipeline.arun` + the embed model's `num_workers`): several batches in flight from one process. Needs an `AsyncQdrantClient`.
- **Worker processes** (`pipeline.run(num_workers=N)`): splits nodes across spawned processes.

Measured here (398 chunks, OpenAI `text-embedding-3-small`, local Qdrant,
Windows): baseline 16.1 s; batch 100 4.0 s (4.0x); async x8 3.9 s (4.1x);
async + batch 50 x 8 3.1 s (5.2x); 2 worker processes 18.7 s (0.86x, i.e.
slower: on Windows each process is spawned and re-imports LlamaIndex, and the
work is network-bound, not CPU-bound). Worker processes should only pay off for
CPU-heavy transformations (e.g. local embedding models) on larger corpora.
Free-tier rate limits cap how far concurrency can go.
