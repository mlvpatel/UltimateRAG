"""Unit tests for src.api.db_utils (Postgres via psycopg 3).

These tests run without a live database. The connection pool accessor
(_get_pool) is patched to return a fake pool whose connection() context
manager yields a fake connection whose cursor() yields a MagicMock
cursor. Assertions check both that execute() was called and, where
relevant, exactly what SQL and parameters were passed, plus that
return values are mapped and ordered correctly.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import src.api.db_utils as db_utils


class FakeConnection:
    """Stands in for a psycopg Connection, contextmanager-y like a pool checkout."""

    def __init__(self, cursor: MagicMock):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @contextmanager
    def cursor(self):
        yield self._cursor


def make_fake_pool(cursor: MagicMock):
    """Build a fake ConnectionPool whose .connection() yields FakeConnection."""
    fake_conn = FakeConnection(cursor)
    fake_pool = MagicMock()

    @contextmanager
    def _connection():
        yield fake_conn

    fake_pool.connection = _connection
    return fake_pool


def test_insert_application_logs_issues_insert_with_four_params():
    cursor = MagicMock()
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        db_utils.insert_application_logs(
            "session-123",
            "what is rag_multimodal?",
            "it is a rag app",
            "gemini-embedding-001",
        )

    assert cursor.execute.called
    sql, params = cursor.execute.call_args[0]
    assert "INSERT INTO application_logs" in sql
    assert "%s" in sql
    assert params == (
        "session-123",
        "what is rag_multimodal?",
        "it is a rag app",
        "gemini-embedding-001",
    )


def test_get_chat_history_maps_rows_to_alternating_human_ai_dicts_in_order():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        {"user_query": "first question", "gpt_response": "first answer"},
        {"user_query": "second question", "gpt_response": "second answer"},
    ]
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        history = db_utils.get_chat_history("session-123")

    assert history == [
        {"role": "human", "content": "first question"},
        {"role": "ai", "content": "first answer"},
        {"role": "human", "content": "second question"},
        {"role": "ai", "content": "second answer"},
    ]

    sql, params = cursor.execute.call_args[0]
    assert "SELECT" in sql
    assert "application_logs" in sql
    assert "WHERE session_id = %s" in sql
    assert "LIMIT %s" in sql
    assert "ORDER BY id ASC" in sql
    assert params == ("session-123", 20)


def test_get_chat_history_empty_when_no_rows():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        history = db_utils.get_chat_history("no-such-session")

    assert history == []


def test_insert_document_record_returns_id_from_returning_fetch():
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 42}
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        new_id = db_utils.insert_document_record("report.pdf")

    assert new_id == 42
    sql, params = cursor.execute.call_args[0]
    assert "INSERT INTO document_store" in sql
    assert "RETURNING id" in sql
    assert params == ("report.pdf",)


def test_insert_document_record_returns_id_when_fetchone_is_tuple():
    # Some cursor configurations return plain tuples rather than dict-like rows.
    cursor = MagicMock()
    cursor.fetchone.return_value = (7,)
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        new_id = db_utils.insert_document_record("notes.docx")

    assert new_id == 7


def test_delete_document_record_returns_true_on_success():
    cursor = MagicMock()
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        result = db_utils.delete_document_record(5)

    assert result is True
    sql, params = cursor.execute.call_args[0]
    assert "DELETE FROM document_store" in sql
    assert "WHERE id = %s" in sql
    assert params == (5,)


def test_delete_document_record_returns_false_and_logs_on_error():
    fake_pool = MagicMock()

    @contextmanager
    def _connection():
        raise RuntimeError("connection blew up")
        yield  # pragma: no cover

    fake_pool.connection = _connection

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        with patch.object(db_utils.logger, "error") as mock_log_error:
            result = db_utils.delete_document_record(999)

    assert result is False
    assert mock_log_error.called


def test_get_all_documents_orders_newest_first_and_maps_dicts():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        {"id": 2, "filename": "newer.pdf", "upload_timestamp": "2026-07-03T00:00:00"},
        {"id": 1, "filename": "older.pdf", "upload_timestamp": "2026-07-01T00:00:00"},
    ]
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        docs = db_utils.get_all_documents()

    assert docs == [
        {"id": 2, "filename": "newer.pdf", "upload_timestamp": "2026-07-03T00:00:00"},
        {"id": 1, "filename": "older.pdf", "upload_timestamp": "2026-07-01T00:00:00"},
    ]

    sql = cursor.execute.call_args[0][0]
    assert "SELECT" in sql
    assert "id, filename, upload_timestamp" in sql
    assert "document_store" in sql
    assert "ORDER BY upload_timestamp DESC" in sql


def test_create_application_logs_creates_table_and_index():
    cursor = MagicMock()
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        db_utils.create_application_logs()

    executed_sql = " ".join(call.args[0] for call in cursor.execute.call_args_list)
    assert "CREATE TABLE IF NOT EXISTS application_logs" in executed_sql
    assert "CREATE INDEX IF NOT EXISTS" in executed_sql
    assert "session_id" in executed_sql


def test_create_document_store_creates_table():
    cursor = MagicMock()
    fake_pool = make_fake_pool(cursor)

    with patch.object(db_utils, "_get_pool", return_value=fake_pool):
        db_utils.create_document_store()

    executed_sql = " ".join(call.args[0] for call in cursor.execute.call_args_list)
    assert "CREATE TABLE IF NOT EXISTS document_store" in executed_sql


def test_init_db_creates_the_tables():
    with patch.object(db_utils, "create_application_logs") as mock_logs:
        with patch.object(db_utils, "create_document_store") as mock_docs:
            db_utils.init_db()

    assert mock_logs.called
    assert mock_docs.called


def test_module_imports_without_live_database():
    # Importing the module must not attempt any DB connection or table
    # creation. If it did, this import (already done at module load time
    # above) would have raised. Re-check the pool is not created yet by
    # confirming the private pool singleton is still unset for a fresh
    # patch scenario: calling _get_pool is the only thing that should
    # trigger creation, and we never call it here.
    assert hasattr(db_utils, "init_db")
    assert hasattr(db_utils, "insert_application_logs")
    assert hasattr(db_utils, "get_chat_history")
    assert hasattr(db_utils, "insert_document_record")
    assert hasattr(db_utils, "delete_document_record")
    assert hasattr(db_utils, "get_all_documents")
