"""Document loaders for every modality, built on LlamaIndex's native readers.

Each loader returns a list of `llama_index.core.schema.Document` objects so
they can all be fed into the same indexing pipeline
(`src/indexing/similarity_indexes.py`) regardless of source modality.
"""

from src.loaders.audio_video_loader import load_audio_video
from src.loaders.image_loader import load_images
from src.loaders.pdf_loader import load_pdfs
from src.loaders.structured_loader import load_csv, load_json
from src.loaders.text_loader import load_text

__all__ = [
    "load_text",
    "load_json",
    "load_csv",
    "load_pdfs",
    "load_images",
    "load_audio_video",
]
