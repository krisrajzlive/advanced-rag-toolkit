"""PDF loading via LlamaIndex's native `PDFReader` (llama-index-readers-file).

Used to ingest data/pdf/anthropic-detecting-countering-misuse-2025-09.pdf —
Anthropic's "Detecting and Countering Misuse" threat intelligence report —
alongside any other PDFs dropped into that directory.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader


def load_pdfs(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    reader = PDFReader()
    documents: list[Document] = []
    for path in sorted(directory.glob("*.pdf")):
        for doc in reader.load_data(file=path):
            doc.metadata.update({"modality": "pdf", "file_name": path.name})
            documents.append(doc)
    return documents
