"""Image -> text loading for RAG over images.

Two paths, both native (no hand-rolled captioning/embedding math):

1. Preferred, fully local: LlamaIndex's built-in `ImageCaptionReader`
   (llama_index.readers.file), which runs a local BLIP model via
   `transformers`. Enabled by installing requirements-multimodal.txt.
2. Default fallback, free-tier remote: the Hugging Face Inference API's
   image-to-text pipeline (`HUGGINGFACE_API_KEY`), so the demo works with
   only the core requirements.txt installed.

Either way the output is a list of `Document`s (caption text + image path in
metadata) that can be embedded and indexed exactly like any other modality.
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_index.core.schema import Document

from src.loaders.cache import cached_text

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULT_HF_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"


def _caption_locally(path: Path) -> str | None:
    try:
        from llama_index.readers.file import ImageCaptionReader
    except ImportError:
        return None

    try:
        reader = ImageCaptionReader()
    except ImportError:
        # torch/transformers/sentencepiece not installed.
        return None

    docs = reader.load_data(file=path)
    return docs[0].text if docs else None


def _caption_via_hf_inference(path: Path) -> str | None:
    token = os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        return None

    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=token)
    result = client.image_to_text(str(path), model=DEFAULT_HF_CAPTION_MODEL)
    return getattr(result, "generated_text", None) or str(result)


def load_images(directory: str | Path) -> list[Document]:
    """Caption every image in `directory` and wrap each caption in a Document."""
    directory = Path(directory)
    documents: list[Document] = []

    for path in sorted(directory.iterdir()) if directory.exists() else []:
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        caption = cached_text(
            path, "caption", lambda p=path: _caption_locally(p) or _caption_via_hf_inference(p)
        )
        if not caption:
            continue

        documents.append(
            Document(
                text=caption,
                metadata={
                    "modality": "image",
                    "file_name": path.name,
                    "file_path": str(path),
                },
            )
        )

    return documents
