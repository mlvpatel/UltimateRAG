"""Embedding and document loading utilities for rag-multimodal-2026.

Two things live here: an embedder, used to embed text chunks and image captions
into the shared vector space, and a plain text loader, used to read an uploaded
text document.

Everything heavy is lazy. Importing this module opens no database connection
and loads no embedding client, which keeps unit tests fast.
"""

import logging

from src.core.config import settings

logger = logging.getLogger(__name__)

_query_embeddings = None
_document_embeddings = None


def _sqlalchemy_url() -> str:
    """PGVector runs on a SQLAlchemy engine, which needs the psycopg driver name."""
    url = settings.database_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _build_embeddings(task_type: str = "retrieval_query"):
    """Construct an embedder for the given task type.

    Provider is chosen by settings.embedding_provider. Ollama (a local, no cost
    embedder such as nomic-embed-text) keeps the whole system keyless.
    """
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        task_type=task_type,
    )


def get_document_embeddings():
    """Index-time embedder (retrieval_document task type). Documents and their
    queries are not paraphrases, so index and query sides use different maps."""
    global _document_embeddings
    if _document_embeddings is None:
        _document_embeddings = _build_embeddings("retrieval_document")
    return _document_embeddings


def get_query_embeddings():
    """The embedder used for text chunks, image captions, and queries."""
    global _query_embeddings
    if _query_embeddings is None:
        _query_embeddings = _build_embeddings()
    return _query_embeddings


def load_document_text(file_path: str) -> str:
    """Load a document by file type and return its full text, unsplit.

    Whole documents are read as plain text before chunking, so this
    returns the joined text rather than retrieval sized chunks.
    """
    if file_path.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader

        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.endswith((".txt", ".md")):
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)
