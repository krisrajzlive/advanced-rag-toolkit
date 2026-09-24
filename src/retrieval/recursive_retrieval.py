"""Recursive retrieval via LlamaIndex's native `RecursiveRetriever`.

Each source document gets:
  - a "summary" `IndexNode` in a top-level vector index, and
  - its own fine-grained chunk index (built with LlamaIndex's
    `SentenceSplitter`) reachable through that node's `index_id`.

`RecursiveRetriever` first searches the top-level summary index to find the
right document, then automatically recurses into that document's own
chunk-level retriever to pull the specific passage - all through LlamaIndex's
built-in node/retriever linking, not manual routing logic.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import Document, IndexNode

ROOT_ID = "summary_index"
SUMMARY_CHARS = 300


def build_recursive_retriever(
    documents: list[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 20,
    top_level_top_k: int = 2,
    child_top_k: int = 3,
) -> RecursiveRetriever:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    retriever_dict = {}
    summary_nodes: list[IndexNode] = []

    for i, doc in enumerate(documents):
        doc_id = doc.doc_id or f"doc_{i}"
        chunks = splitter.get_nodes_from_documents([doc])
        child_index = VectorStoreIndex(chunks)
        retriever_dict[doc_id] = child_index.as_retriever(similarity_top_k=child_top_k)

        summary_text = doc.text[:SUMMARY_CHARS].replace("\n", " ")
        summary_nodes.append(
            IndexNode(text=summary_text, index_id=doc_id, metadata=doc.metadata)
        )

    top_index = VectorStoreIndex(summary_nodes)
    retriever_dict[ROOT_ID] = top_index.as_retriever(similarity_top_k=top_level_top_k)

    return RecursiveRetriever(
        root_id=ROOT_ID,
        retriever_dict=retriever_dict,
        verbose=False,
    )
