"""Functional tests for the API endpoints (TestClient, services mocked)."""

import pytest
from fastapi.testclient import TestClient

import src.api.main as main


class _FakeProcess:
    def delay(self, *args, **kwargs):
        class _Result:
            id = "tid123"

        return _Result()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "init_db", lambda: None)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_needs_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_needs_no_auth_header(client, monkeypatch):
    """There is no API key: a bare request must reach the engine, not bounce
    with a 401. Network exposure is prevented by loopback-only port binding."""
    monkeypatch.setattr(main, "get_chat_history", lambda session_id: [])
    monkeypatch.setattr(main, "insert_application_logs", lambda *a: None)
    monkeypatch.setattr(
        main,
        "run_multimodal",
        lambda model, q, history: {
            "answer": "ok",
            "sources": [],
            "images": [],
            "steps": [],
        },
    )
    response = client.post("/v1/chat", json={"question": "hi"})
    assert response.status_code == 200


def test_chat_runs_multimodal_and_persists_the_turn(client, monkeypatch):
    monkeypatch.setattr(main, "get_chat_history", lambda session_id: [])
    monkeypatch.setattr(
        main,
        "run_multimodal",
        lambda model, q, history: {
            "answer": "Hello world",
            "steps": [{"step": "retrieve", "text": 2, "images": 1}],
            "images": [
                {"filename": "chart.png", "caption": "a chart", "data_uri": "data:x"}
            ],
            "sources": ["chart.png", "doc.txt"],
        },
    )
    persisted = {}
    monkeypatch.setattr(
        main,
        "insert_application_logs",
        lambda sid, q, a, m: persisted.update({"answer": a, "model": m}),
    )

    response = client.post(
        "/v1/chat",
        json={"question": "hi", "model": "llama3.2:3b"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello world"
    assert body["steps"][0]["step"] == "retrieve"
    assert body["images"][0]["filename"] == "chart.png"
    assert body["sources"] == ["chart.png", "doc.txt"]
    assert persisted["answer"] == "Hello world"


def test_upload_rejects_unsupported_extension(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "process_document", _FakeProcess())
    response = client.post(
        "/v1/upload-doc",
        files={"file": ("malware.exe", b"data")},
    )
    assert response.status_code == 400


def test_upload_accepts_supported_file_and_queues_task(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "process_document", _FakeProcess())
    response = client.post(
        "/v1/upload-doc",
        files={"file": ("notes.txt", b"hello world")},
    )
    assert response.status_code == 200
    assert response.json()["task_id"] == "tid123"
