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
| `main.py postprocess "<q>"` | Shows what each postprocessor chain keeps for a question: raw top-10, similarity cutoff, cutoff + cross-encoder, + long-context reorder, and sentence-window replacement (no LLM calls) | Qdrant, embeddings, `requirements-rerank.txt` |
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
| `scripts/tune_recall.py ann\|retrieval\|cutoff\|all` | Measures recall and recommends settings. `ann`: sweeps quantization / rescoring / `hnsw_ef` against exact search and picks the fastest config reaching `--target` (default 0.95; also `--n`, `--dim`). `retrieval`: compares top-k, smaller chunks, query expansion, LLM rerank, cross-encoder rerank and sentence-window on 10 labelled questions. `cutoff`: sweeps `SimilarityPostprocessor` and recommends a cutoff that removes out-of-scope noise without losing recall | Docker (Qdrant) |
| `scripts/parallel_ingest.py [--repeat N] [--workers W] [--sweep 1,2,4,8] [--async-workers A] [--only text]` | Times five ingestion strategies on the same documents (text, JSON, CSV, 154-page PDF, replicated `--repeat` times) into a fresh Qdrant collection: sequential baseline, larger embedding batches, async concurrent batches, async + larger batches, and worker processes (`IngestionPipeline.run(num_workers=...)`). Prints nodes/sec and speedup vs baseline | Docker (Qdrant), OpenAI key |
| `scripts/parallel_query.py queries / subquestion / router / agent / eval / all [--concurrency N]` | Benchmarks parallel querying on three Qdrant indexes (RAG notes, vector-infra notes, 154-page report): sequential vs `aquery` + `asyncio.gather` vs semaphore-bounded gather; `SubQuestionQueryEngine` `use_async` off/on; `RouterQueryEngine` multi-select fan-out; `FunctionAgent` parallel tool calls; LangSmith `evaluate(max_concurrency)` 1 vs 4 | Docker (Qdrant), chat LLM (use `RAG_LLM_PROVIDER=openai`) |
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

### Postprocessors

`src/retrieval/postprocessors.py` (all LlamaIndex-native node postprocessors):

| Postprocessor | Purpose |
|---|---|
| `SimilarityPostprocessor` | drops weak matches so the system can abstain on out-of-scope questions |
| `SentenceTransformerRerank` | cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, local CPU, no LLM call); needs `pip install -r requirements-rerank.txt` |
| `LongContextReorder` | best chunks first and last in the prompt ("lost in the middle") |
| `MetadataReplacementPostProcessor` | sentence-window retrieval: embed sentences, return their surrounding window |

Order matters: cutoff first (cosine scale), then the cross-encoder (which
replaces scores with its own scale). Measured here: a cutoff of 0.20-0.35
cut out-of-scope questions that still retrieved something from 100% to 0% with
no loss of in-scope recall (0.40 starts losing recall). On the labelled set the
cross-encoder matched the 0.9 baseline with **no LLM calls**, versus 0.8 for the
LLM reranker with 2 calls; sentence-window scored 0.7 here (single-sentence
embeddings lose to the PDF distractors), so it is not a default. Small
labelled set (10 questions) - treat as indicative. Note: rerankers overwrite
`NodeWithScore.score` in place, so `apply_chain` works on copies.

## Choosing the embedding provider

Set `EMBED_PROVIDER` in `.env` (chat LLMs are unaffected and stay on the cloud providers):

| `EMBED_PROVIDER` | Model | Dim | Notes |
|---|---|---|---|
| unset / `openai` | `text-embedding-3-small` | 1536 | default when `OPENAI_API_KEY` is set |
| `huggingface` | `all-MiniLM-L6-v2` via Inference API | 384 | metered credits (can return 402) |
| `local` | `all-MiniLM-L6-v2` in-process (sentence-transformers) | 384 | no server, no network, ~300 MB RAM; `pip install -r requirements-local-embeddings.txt`. Measured: 100 texts in 0.26 s |
| `ollama` | `nomic-embed-text` on a local Ollama | 768 | embeddings only; `keep_alive` (default 1m) unloads the model after use (~376 MB while loaded). `ollama pull nomic-embed-text`; Ollama Cloud does **not** serve embeddings |

Rules when switching: vectors from different models are not comparable and
dimensions differ, so an existing Qdrant collection is rejected with a clear
error (`ensure_collection` checks the size) - delete it (and `.cdc_state/` for
`sync`) or switch back. `scripts/tune_recall.py cutoff` rebuilds its eval
collection automatically. Score scales differ per model (MiniLM scores run lower
than OpenAI), so re-run `scripts/tune_recall.py cutoff` to recalibrate the
similarity cutoff for the model you use.

## Loader cache

Transcripts (audio/video) and image captions come from metered APIs, so
`src/loaders/cache.py` stores them in `.cache/loaders/` (git-ignored), keyed by
file name + size + modified time. A failing provider call (e.g. exhausted
credits) logs a warning and skips that file instead of aborting the load.

### Parallel ingestion: updated results

Same 398-chunk corpus, 8 physical / 16 logical cores, peak memory = this process plus workers:

| Strategy | Time | Speedup | Peak RSS |
|---|---|---|---|
| baseline (sequential, batch 10) | 22.1 s | 1.0x | 416 MB |
| larger batches (100) | 4.5 s | 4.9x | 429 MB |
| async (8 concurrent batches) | 4.0 s | 5.5x | 433 MB |
| async + batch 50 x 8 | 2.8 s | 7.9x | 438 MB |
| **document batches gathered (50 docs, semaphore 4, batch 100)** | **2.5 s** | **8.8x** | 429 MB |
| async + 2 worker processes | 11.3 s | 2.0x | 1121 MB |
| 2 worker processes | 15.2 s | 1.5x | 1096 MB |
| 4 worker processes | 15.4 s | 1.4x | 1665 MB |

Worker processes help a little but cost 2.5-4x the memory, and going from 2 to 4 gave nothing:
embedding is network-bound, so I/O concurrency (batching, async, gathered batches) wins.
(The baseline varied between runs, 16-22 s, so compare strategies within one run.)

### Parallel querying

`src/retrieval/parallel_query.py`, `src/agents/parallel_tools_agent.py`, `scripts/parallel_query.py`.
OpenAI as the chat LLM, 12 questions, medians of 3 for the engine comparisons:

| Technique | Without | With | Speedup |
|---|---|---|---|
| `aquery` + `asyncio.gather` (12 in flight) | 19.2 s | 6.6 s | 2.9x |
| bounded gather (semaphore 4) | 19.2 s | 10.6 s | 1.8x (rate-limit friendly) |
| `SubQuestionQueryEngine(use_async=True)` | 11.1 s | 10.1 s | 1.1x |
| `RouterQueryEngine` multi-select, `aquery` fan-out | 16.8 s | 12.1 s | 1.4x |
| `FunctionAgent(allow_parallel_tool_calls=True)`, 3 tool calls | 16.1 s | 8.8 s | 1.8x |
| LangSmith `evaluate(max_concurrency=4)`, 8 examples | 56.1 s | 33.9 s | 1.7x |

Why the fan-out engines gain little: only the middle step (the sub-queries) runs in
parallel; question generation and answer synthesis are still sequential LLM calls.
Agent timings vary run to run with LLM behaviour.

**Provider concurrency limits matter more than your code.** Ollama Cloud's free tier
served 6 concurrent requests in 11.3 s (about one at a time); OpenAI served the same 6 in
2.6 s. With Ollama Cloud, `gather` gave only 1.3x and per-query latency ballooned (14 s
vs 2.6 s), so use a bounded semaphore sized to your provider's limit.

### Choosing parallelism by runtime

| Runtime | Use | Avoid |
|---|---|---|
| Laptop / VM / EC2 / ECS / GKE (fixed cores) | async + larger batches first; `num_workers` only for CPU-heavy transforms, sized to **physical** cores (`psutil.cpu_count(logical=False)`) | more workers than physical cores; burstable instances |
| AWS Lambda / Cloud Functions | `asyncio.gather` / `use_async=True` only, `num_workers` <= 1 | `ProcessPoolExecutor` (no `/dev/shm`), long single-invocation ingestion (15-minute cap), local-mode vector stores |
| Vercel / edge runtimes | async only | any Python worker processes |

Rules of thumb: parallel workers multiply API calls, so cap concurrency with a
semaphore or your provider's retry/backoff; process pools multiply memory
(about 0.35-0.6 GB per worker here); below about 100 chunks setup overhead dominates.
Async clients are required for async paths: `QdrantVectorStore` needs an
`AsyncQdrantClient` (`src.indexing.qdrant_store.async_client_for` adds it for every
server-backed index). `evaluate_rag_pipeline` now defaults to `max_concurrency=4`.
Note: `llama-index-question-gen-openai` pins an older `llama-index-core`, so the
sub-question engine uses core's `LLMQuestionGenerator` instead.
