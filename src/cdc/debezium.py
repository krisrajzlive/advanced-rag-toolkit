"""Register the Debezium Postgres connector (Kafka Connect REST API)."""

from __future__ import annotations

import os

import requests

CONNECTOR = "rag-documents"
TOPIC = "cdc.public.documents"  # <topic.prefix>.<schema>.<table>


def connector_config() -> dict:
    return {
        "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
        "plugin.name": "pgoutput",
        "database.hostname": "postgres",  # container-to-container address
        "database.port": "5432",
        "database.user": "rag",
        "database.password": os.environ["POSTGRES_PASSWORD"],
        "database.dbname": "rag_source",
        "topic.prefix": "cdc",
        "table.include.list": "public.documents",
        "slot.name": "rag_cdc_slot",
        "publication.autocreate.mode": "filtered",
        "snapshot.mode": "initial",  # existing rows arrive as op='r' events
        "tombstones.on.delete": "false",
        "key.converter": "org.apache.kafka.connect.json.JsonConverter",
        "key.converter.schemas.enable": "false",
        "value.converter": "org.apache.kafka.connect.json.JsonConverter",
        "value.converter.schemas.enable": "false",
    }


def register_connector(timeout: float = 90.0) -> str:
    """Create or update the connector (idempotent); wait until it and its task RUN."""
    import time

    base = os.environ.get("DEBEZIUM_URL", "http://localhost:8083")
    resp = requests.put(f"{base}/connectors/{CONNECTOR}/config", json=connector_config(), timeout=30)
    resp.raise_for_status()
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{base}/connectors/{CONNECTOR}/status", timeout=30)
        if r.ok:
            status = r.json()
            tasks = status.get("tasks", [])
            if any(t["state"] == "FAILED" for t in tasks) or status["connector"]["state"] == "FAILED":
                raise RuntimeError(f"Debezium connector failed: {status}")
            if status["connector"]["state"] == "RUNNING" and tasks and all(t["state"] == "RUNNING" for t in tasks):
                return "RUNNING"
        time.sleep(2)
    raise TimeoutError("Debezium connector did not reach RUNNING")
