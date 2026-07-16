"""Unit tests for the multimodal engine helpers (no model, no database)."""

from src.multimodal.engine import is_image


def test_is_image_recognizes_image_extensions():
    assert is_image("chart.png")
    assert is_image("photo.JPG")
    assert is_image("diagram.jpeg")


def test_is_image_rejects_text_documents():
    assert not is_image("notes.txt")
    assert not is_image("report.pdf")
    assert not is_image("page.html")


def test_run_multimodal_retrieves_with_the_standalone_query(monkeypatch):
    """A follow-up must be rewritten before the vector search sees it."""
    from langchain_core.documents import Document
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import src.multimodal.engine as engine

    fake_llm = FakeListChatModel(
        responses=["what does the Nimbus active users chart show?", "the answer"]
    )
    searched = {}

    def fake_search(query, k):
        searched["query"] = query
        return [
            Document(
                page_content="a chart of active users",
                metadata={"modality": "text", "filename": "doc.txt"},
            )
        ]

    monkeypatch.setattr(engine, "_make_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(engine.store, "search", fake_search)

    history = [{"role": "human", "content": "tell me about the Nimbus chart"}]
    result = engine.run_multimodal("m", "what does it show?", history)

    assert searched["query"] == "what does the Nimbus active users chart show?"
    assert result["answer"] == "the answer"
    assert any(s["step"] == "reformulate" for s in result["steps"])


def test_search_embeds_queries_with_the_query_task_embedder(monkeypatch):
    """The store's own embedder is the document one; search must go through
    the query embedder and the by-vector API."""
    from unittest.mock import MagicMock

    import src.multimodal.store as store

    q_embedder = MagicMock()
    q_embedder.embed_query.return_value = [0.1, 0.2]
    content = MagicMock()
    monkeypatch.setattr(
        "src.embeddings.vectorstore_utils.get_query_embeddings", lambda: q_embedder
    )
    monkeypatch.setattr(store, "get_content_store", lambda: content)

    store.search("question", k=5)

    q_embedder.embed_query.assert_called_once_with("question")
    content.similarity_search_by_vector.assert_called_once_with([0.1, 0.2], k=5)
