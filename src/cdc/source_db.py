"""The CDC *source*: a Postgres `documents` table that Debezium watches."""

from __future__ import annotations

import os

import psycopg

SEED = [
    ("acme", "Roadrunner returns", "Acme accepts returns of unused anvils within 45 days."),
    ("acme", "Roadrunner escalation", "The Roadrunner support escalation code is ACME-RR-7741."),
    ("globex", "Doomsday refunds", "Globex subscriptions can be cancelled within 14 days for a prorated refund."),
]


def connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5433")),
        user="rag",
        password=os.environ["POSTGRES_PASSWORD"],
        dbname="rag_source",
        autocommit=True,
    )


def setup_source(seed: bool = True) -> None:
    """Create the table (idempotent). REPLICA IDENTITY FULL puts the whole old row in delete events."""
    with connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                   id SERIAL PRIMARY KEY,
                   tenant_id TEXT NOT NULL,
                   title TEXT NOT NULL,
                   body TEXT NOT NULL,
                   updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
        )
        conn.execute("ALTER TABLE documents REPLICA IDENTITY FULL")
        if seed and conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0:
            for row in SEED:
                conn.execute("INSERT INTO documents (tenant_id, title, body) VALUES (%s, %s, %s)", row)


def insert_doc(tenant_id: str, title: str, body: str) -> int:
    with connect() as conn:
        return conn.execute(
            "INSERT INTO documents (tenant_id, title, body) VALUES (%s, %s, %s) RETURNING id",
            (tenant_id, title, body),
        ).fetchone()[0]


def update_doc(doc_id: int, body: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE documents SET body=%s, updated_at=now() WHERE id=%s", (body, doc_id))


def delete_doc(doc_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
