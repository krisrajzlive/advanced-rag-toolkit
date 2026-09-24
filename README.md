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
  indexing/            # native cosine / dot_product / euclidean retrieval
  retrieval/           # query expansion, reranking, recursive retrieval
  agents/              # LlamaIndex ReActAgent + LangChain/LangGraph agent
  evaluation/           # LangSmith evaluate() over a small QA dataset
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
