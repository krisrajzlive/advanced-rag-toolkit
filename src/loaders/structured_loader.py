"""JSON and CSV loading via LlamaIndex's native readers.

- `JSONReader` (llama_index.core.readers.json) flattens/streams arbitrary
  JSON into text nodes.
- `CSVReader` (llama_index.readers.file) turns each row into a Document.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document
from llama_index.readers.file import CSVReader


def load_json(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    reader = JSONReader()
    documents: list[Document] = []
    for path in sorted(directory.glob("*.json")):
        for doc in reader.load_data(input_file=str(path)):
            doc.metadata.update({"modality": "json", "file_name": path.name})
            documents.append(doc)
    return documents


def load_csv(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    reader = CSVReader()
    documents: list[Document] = []
    for path in sorted(directory.glob("*.csv")):
        for doc in reader.load_data(file=path):
            doc.metadata.update({"modality": "csv", "file_name": path.name})
            documents.append(doc)
    return documents
