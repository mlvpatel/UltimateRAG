"""
Database utilities for rag-multimodal-2026, conversation memory and document
records on Postgres via psycopg 3.

This module keeps the public interface of the SQLite memory layer used by
the earlier rungs of the line (rag-naive-2022, rag-advanced-2023) while
swapping the storage engine to Postgres, the same instance used for pgvector.

Design notes:
- The connection pool is created lazily, on first use, not at import
  time. Importing this module must never require a live database,
  which matters for unit tests and for any process that imports the
  module without intending to touch the database yet.
- Tables are created by init_db(), not at import time. The SQLite
  baseline created tables as a side effect of import; that behavior
  moves into an explicit init_db() call that the application invokes
  during startup.
- All queries are parameterized with psycopg %s placeholders. No SQL
  string interpolation of caller supplied values, ever.
"""

import logging

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.core.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the process wide connection pool, creating it on first use.

    Lazy by design: constructing this module (import src.api.db_utils)
    must not open a database connection. The pool is only created the
    first time a function actually needs to run a query.
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            open=True,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def create_application_logs() -> None:
    """Create the application_logs table and its session_id index if missing."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS application_logs (
                    id           BIGSERIAL PRIMARY KEY,
                    session_id   TEXT NOT NULL,
                    user_query   TEXT NOT NULL,
                    gpt_response TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_application_logs_session_id "
                "ON application_logs (session_id)"
            )


def create_document_store() -> None:
    """Create the document_store table if missing."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_store (
                    id               BIGSERIAL PRIMARY KEY,
                    filename         TEXT NOT NULL,
                    upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)


def init_db() -> None:
    """Create all tables used by this module if they do not already exist.

    Call this once during application startup. Importing this module
    does not do this automatically, so a live database is only ever
    required when init_db() (or another function here) actually runs.
    """
    create_application_logs()
    create_document_store()


def insert_application_logs(
    session_id: str, user_query: str, gpt_response: str, model: str
) -> None:
    """Record one turn of a conversation (one user query plus one AI response)."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO application_logs "
                "(session_id, user_query, gpt_response, model) "
                "VALUES (%s, %s, %s, %s)",
                (session_id, user_query, gpt_response, model),
            )


def get_chat_history(session_id: str, limit: int = 20) -> list:
    """Return the LAST `limit` turns of a session as alternating dicts.

    The window exists because unbounded history grows the prompt with every
    turn of a long session. Rows come back oldest first; each row expands to
    two entries in the returned list, human then ai, in that order.
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_query, gpt_response FROM ("
                "  SELECT id, user_query, gpt_response FROM application_logs"
                "  WHERE session_id = %s ORDER BY id DESC LIMIT %s"
                ") recent ORDER BY id ASC",
                (session_id, limit),
            )
            rows = cur.fetchall()

    messages = []
    for row in rows:
        messages.append({"role": "human", "content": row["user_query"]})
        messages.append({"role": "ai", "content": row["gpt_response"]})
    return messages


def insert_document_record(filename: str) -> int:
    """Insert a document record and return the new integer id."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO document_store (filename) VALUES (%s) RETURNING id",
                (filename,),
            )
            row = cur.fetchone()

    if isinstance(row, dict):
        return row["id"]
    return row[0]


def delete_document_record(file_id: int) -> bool:
    """Delete a document record by id. Returns False and logs on error."""
    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_store WHERE id = %s", (file_id,))
        return True
    except Exception as e:
        logger.error(f"Failed to delete document record {file_id}: {e}")
        return False


def get_all_documents() -> list:
    """Return all document records as dicts, newest upload first."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, upload_timestamp FROM document_store "
                "ORDER BY upload_timestamp DESC"
            )
            rows = cur.fetchall()

    return [dict(row) for row in rows]
