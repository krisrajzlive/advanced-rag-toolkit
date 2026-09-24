"""Plain-text document loading via LlamaIndex's native `SimpleDirectoryReader`."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_text(directory: str | Path, required_exts: list[str] | None = None) -> list[Document]:
    """Load every text file in `directory` as a llama_index `Document`.

    Uses `SimpleDirectoryReader`, LlamaIndex's built-in loader, which infers
    the file type and tags each Document with `file_path` / `file_name`
    metadata automatically.
    """
    directory = Path(directory)
    if not directory.exists() or not any(directory.iterdir()):
        return []

    try:
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            required_exts=required_exts or [".txt", ".md"],
            filename_as_id=True,
        )
        documents = reader.load_data()
    except ValueError:
        # No files matched required_exts in this directory.
        return []
    for doc in documents:
        doc.metadata["modality"] = "text"
    return documents
