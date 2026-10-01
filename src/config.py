"""Provider-agnostic LLM / embedding setup, shared by every demo.

Both LlamaIndex and LangChain point at the same three free-tier backends:

- Hugging Face Inference Providers (`HUGGINGFACE_API_KEY`), via its
  OpenAI-compatible chat router and its dedicated embeddings endpoint.
- Ollama Cloud (`OLLAMA_API_KEY`), via its OpenAI-compatible chat endpoint.
- OpenAI (`OPENAI_API_KEY`), used only if explicitly requested or if it is
  the only key configured.

`get_llm_provider()` auto-selects the first backend with a configured API
key, in that order, so the whole toolkit runs on free tiers out of the box.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

HF_CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_CHAT_MODEL = "gpt-oss:20b"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

HUGGINGFACE_ROUTER_BASE = "https://router.huggingface.co/v1"
OLLAMA_CLOUD_BASE = "https://ollama.com/v1"


def get_llm_provider() -> str:
    """Pick the first available free-tier provider: huggingface > ollama > openai.

    `RAG_LLM_PROVIDER` forces a specific one.
    """
    forced = os.environ.get("RAG_LLM_PROVIDER")
    if forced:
        return forced
    if os.environ.get("HUGGINGFACE_API_KEY"):
        return "huggingface"
    if os.environ.get("OLLAMA_API_KEY"):
        return "ollama"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM provider configured. Set one of HUGGINGFACE_API_KEY, "
        "OLLAMA_API_KEY, or OPENAI_API_KEY in .env."
    )


# --------------------------------------------------------------------------
# LlamaIndex
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def get_llamaindex_llm(provider: str | None = None):
    """Return a llama_index LLM for the given provider (default: auto-pick)."""
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai_like import OpenAILike

    provider = provider or get_llm_provider()

    if provider == "huggingface":
        return OpenAILike(
            model=HF_CHAT_MODEL,
            api_base=HUGGINGFACE_ROUTER_BASE,
            api_key=os.environ["HUGGINGFACE_API_KEY"],
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=8192,
        )
    if provider == "ollama":
        return OpenAILike(
            model=OLLAMA_CHAT_MODEL,
            api_base=OLLAMA_CLOUD_BASE,
            api_key=os.environ["OLLAMA_API_KEY"],
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=8192,
        )
    if provider == "openai":
        return OpenAI(model=OPENAI_CHAT_MODEL, api_key=os.environ["OPENAI_API_KEY"])

    raise ValueError(f"Unknown provider: {provider}")


@lru_cache(maxsize=None)
def get_llamaindex_embed_model(provider: str | None = None, normalize: bool = False):
    """Return a llama_index embedding model for the given provider.

    With `normalize=True` the model is wrapped so every vector it returns is
    L2-normalized (unit length) - see `src/indexing/normalization.py`.

    Prefers OpenAI when `OPENAI_API_KEY` is set: `HuggingFaceInferenceAPIEmbedding`
    opens a brand-new asyncio event loop per embedding batch
    (llama-index-embeddings-huggingface-api's `_get_text_embeddings`) while
    reusing one persistent async HTTP client across those loops - on
    Windows' default ProactorEventLoop this reliably raises "Event loop is
    closed" once more than a handful of nodes are embedded. OpenAI's
    embedding client is a plain synchronous HTTP call with no such issue.
    """
    if normalize:
        from src.indexing.normalization import L2NormalizedEmbedding

        return L2NormalizedEmbedding(
            inner=get_llamaindex_embed_model(provider, normalize=False)
        )

    provider = provider or get_llm_provider()

    if os.environ.get("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=OPENAI_EMBED_MODEL, api_key=os.environ["OPENAI_API_KEY"]
        )

    # Hugging Face Inference API embeddings: free-tier fallback when no
    # OpenAI key is configured. See the docstring above for a known Windows
    # caveat (https://github.com/run-llama/llama_index Windows/asyncio
    # event-loop reuse issue in this integration).
    from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

    return HuggingFaceInferenceAPIEmbedding(
        model_name=HF_EMBED_MODEL,
        token=os.environ["HUGGINGFACE_API_KEY"],
        embed_batch_size=1,
    )


def configure_llamaindex_settings(
    provider: str | None = None, normalize: bool = False
) -> None:
    """Wire up llama_index's global `Settings` with our LLM + embed model."""
    from llama_index.core import Settings

    Settings.llm = get_llamaindex_llm(provider)
    Settings.embed_model = get_llamaindex_embed_model(provider, normalize=normalize)


# --------------------------------------------------------------------------
# LangChain
# --------------------------------------------------------------------------

HUGGINGFACE_KWARGS = {
    "model_provider": "openai",
    "base_url": HUGGINGFACE_ROUTER_BASE,
    "api_key": os.environ.get("HUGGINGFACE_API_KEY"),
}
OLLAMA_KWARGS = {
    "model_provider": "openai",
    "base_url": OLLAMA_CLOUD_BASE,
    "api_key": os.environ.get("OLLAMA_API_KEY"),
}


def get_langchain_model(provider: str | None = None, **kwargs):
    """Return a LangChain `BaseChatModel` for the given provider."""
    from langchain.chat_models import init_chat_model

    provider = provider or get_llm_provider()

    if provider == "huggingface":
        return init_chat_model(HF_CHAT_MODEL, **HUGGINGFACE_KWARGS, **kwargs)
    if provider == "ollama":
        return init_chat_model(OLLAMA_CHAT_MODEL, **OLLAMA_KWARGS, **kwargs)
    if provider == "openai":
        return init_chat_model(f"openai:{OPENAI_CHAT_MODEL}", **kwargs)

    raise ValueError(f"Unknown provider: {provider}")


def get_langchain_embeddings():
    """Return LangChain embeddings (Hugging Face Inference API, free tier)."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=HF_EMBED_MODEL,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_KEY"),
    )
