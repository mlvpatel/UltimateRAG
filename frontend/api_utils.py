"""Client for the rag-multimodal-2026 API, used by the Streamlit frontend.

All calls go through the v1 router. The chat call
posts a question and returns the answer together with the images that
contributed, so the UI can show the pictures the answer drew on.
"""

import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _headers(json_content: bool = True) -> dict:
    headers = {}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def chat(question: str, session_id, model: str) -> dict:
    """Post a question and return the answer plus the agent trace."""
    response = requests.post(
        f"{API_URL}/v1/chat",
        json={"question": question, "session_id": session_id, "model": model},
        headers=_headers(),
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def upload_document(file_bytes: bytes, filename: str) -> dict:
    response = requests.post(
        f"{API_URL}/v1/upload-doc",
        files={"file": (filename, file_bytes)},
        headers=_headers(json_content=False),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def list_documents() -> list:
    response = requests.get(
        f"{API_URL}/v1/list-docs", headers=_headers(json_content=False), timeout=30
    )
    response.raise_for_status()
    return response.json()


def delete_document(file_id: int) -> dict:
    response = requests.post(
        f"{API_URL}/v1/delete-doc",
        json={"file_id": file_id},
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_task_status(task_id: str) -> dict:
    response = requests.get(
        f"{API_URL}/v1/task/{task_id}", headers=_headers(json_content=False), timeout=30
    )
    response.raise_for_status()
    return response.json()
