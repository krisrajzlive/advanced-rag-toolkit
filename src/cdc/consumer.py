"""Event-driven indexer: apply Debezium change events to Qdrant.

Debezium streams every INSERT / UPDATE / DELETE on the Postgres `documents`
table into Kafka (read from the write-ahead log, no polling of the table). This
consumer turns each event into an index change:

- op `c` (insert), `r` (initial snapshot), `u` (update) -> replace the doc's vectors
- op `d` (delete)                                       -> remove the doc's vectors

Offsets are committed only after a change is applied (at-least-once delivery);
replace/delete are idempotent, so a replay is harmless.
"""

from __future__ import annotations

import json
import os
import time

from confluent_kafka import Consumer
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.cdc.debezium import TOPIC
from src.indexing.qdrant_store import TENANT_FIELD, ensure_collection, get_qdrant_client

COLLECTION = "rag_cdc"


def doc_id_for(row_id: int) -> str:
    return f"pg-{row_id}"


def open_index(collection: str = COLLECTION, client=None) -> VectorStoreIndex:
    client = client or get_qdrant_client()
    dim = len(Settings.embed_model.get_text_embedding("dimension probe"))
    ensure_collection(client, collection, dim, multitenant=True, normalized=True, recreate=False)
    return VectorStoreIndex.from_vector_store(QdrantVectorStore(client=client, collection_name=collection))


def apply_event(index: VectorStoreIndex, event: dict) -> str:
    """Apply one Debezium change event; returns a short description."""
    op = event["op"]
    if op == "d":
        row = event["before"]
        index.delete_ref_doc(doc_id_for(row["id"]))
        return f"DELETE  doc {row['id']} ({row.get('tenant_id')})"
    row = event["after"]
    doc_id = doc_id_for(row["id"])
    index.delete_ref_doc(doc_id)  # drop stale vectors of the previous version
    index.insert(
        Document(
            text=f"{row['title']}\n\n{row['body']}",
            id_=doc_id,
            metadata={TENANT_FIELD: row["tenant_id"], "title": row["title"], "source": "postgres"},
        )
    )
    label = {"c": "INSERT", "r": "SNAPSHOT", "u": "UPDATE"}.get(op, op)
    return f"{label:8s} doc {row['id']} ({row['tenant_id']}): {row['title']}"


def consume(index: VectorStoreIndex, idle_timeout: float = 20.0, group: str = "rag-indexer") -> list[str]:
    """Process events until no new message arrives for `idle_timeout` seconds."""
    consumer = Consumer({
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094"),
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([TOPIC])
    applied, last = [], time.time()
    try:
        while time.time() - last < idle_timeout:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise RuntimeError(msg.error())
            if msg.value() is None:  # tombstone
                consumer.commit(msg)
                continue
            applied.append(apply_event(index, json.loads(msg.value())))
            consumer.commit(msg)  # commit only after the change is in Qdrant
            last = time.time()
    finally:
        consumer.close()
    return applied
