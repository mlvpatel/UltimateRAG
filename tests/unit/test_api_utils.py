"""Unit tests for the frontend API client (no server, no streamlit)."""

from frontend import api_utils


class _FakeResp:
    def __init__(self, lines=None, payload=None):
        self._lines = lines or []
        self._payload = payload

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line

    def json(self):
        return self._payload


def test_headers_carry_no_credential():
    headers = api_utils._headers()
    assert "X-API-Key" not in headers
    assert headers["Content-Type"] == "application/json"


def test_chat_posts_question_and_returns_json(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp(payload={"answer": "ok", "session_id": "abc", "steps": []})

    monkeypatch.setattr(api_utils.requests, "post", fake_post)

    out = api_utils.chat("hi", None, "gpt-4o-mini")

    assert out["answer"] == "ok"
    assert out["session_id"] == "abc"
    assert captured["url"].endswith("/v1/chat")
    assert captured["json"]["question"] == "hi"


def test_upload_document_posts_file_to_v1(monkeypatch):
    captured = {}

    def fake_post(url, files=None, headers=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        return _FakeResp(payload={"task_id": "t1", "status": "processing"})

    monkeypatch.setattr(api_utils.requests, "post", fake_post)

    out = api_utils.upload_document(b"data", "notes.txt")

    assert out["task_id"] == "t1"
    assert captured["url"].endswith("/v1/upload-doc")
    assert captured["files"]["file"][0] == "notes.txt"
