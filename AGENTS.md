# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project shape

This is a demo/reference toolkit, not a service. Every module under `src/`
is a small, composable function (`load_*`, `build_*`, `get_*`) rather than a
class hierarchy — keep additions in that style. `main.py` is a thin CLI that
wires those functions together; it should stay thin.

## Conventions

- **LLM/embedding access always goes through `src/config.py`.** Never
  instantiate an LLM or embedding client directly in a loader/retriever/agent
  module — call `get_llamaindex_llm()`, `get_llamaindex_embed_model()`,
  `get_langchain_model()`, or `configure_llamaindex_settings()`. This keeps
  the free-tier-provider fallback (Hugging Face → Ollama → OpenAI) working
  everywhere.
- **Similarity metrics use LlamaIndex's own `similarity()` /
  `get_top_k_embeddings()`, never hand-written cosine/dot-product/Euclidean
  math.** See `src/indexing/similarity_indexes.py` for the pattern if adding
  a new metric or retriever.
- **Prefer native LlamaIndex readers/retrievers/postprocessors over custom
  implementations.** Only fall back to a hand-written loader step (e.g. the
  Hugging Face Inference API calls in `src/loaders/image_loader.py` and
  `audio_video_loader.py`) when there's no reasonably-light native option,
  and keep it wrapped so it still returns plain `Document` objects.
- Every loader returns `list[llama_index.core.schema.Document]` and tags
  `metadata["modality"]`. Keep that contract when adding a new loader so it
  composes with the rest of the pipeline unmodified.
- Heavy/optional dependencies (torch, transformers, whisper, moviepy) belong
  in `requirements-multimodal.txt`, not `requirements.txt`. Code that uses
  them must degrade gracefully (see the `_caption_locally` /
  `_transcribe_locally` try/except patterns) rather than hard-failing when
  they're absent.

## Running things

```bash
.venv\Scripts\python main.py <command> "<query>"
```

`main.py load-all` is the fastest sanity check after touching a loader —
it prints a per-modality document count with no LLM calls beyond
embedding/captioning/transcription.

## Secrets

`.env` holds real API keys and is git-ignored. Never commit it, never print
its contents, and update `.env.example` (placeholders only) instead when a
new variable is introduced.
