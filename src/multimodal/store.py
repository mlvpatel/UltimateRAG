"""Unified multimodal content store for rag-multimodal-2026.

Text chunks and image captions live in one pgvector collection, so a single
similarity search returns the most relevant content regardless of modality.
Each item carries its modality and, for images, the stored path and caption.

Everything heavy is lazy, so importing this module opens no database connection.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

CONTENT_COLLECTION = "rag_multimodal_content"

_store = None


def get_content_store():
    """The pgvector collection holding text chunks and image captions."""
    global _store
    if _store is None:
        from langchain_postgres import PGVector

        from src.embeddings.vectorstore_utils import (
            _sqlalchemy_url,
            get_document_embeddings,
        )

        # The store's own embedder runs at ADD time, so it is the document-task
        # embedder; queries are embedded separately in search() with the
        # query-task embedder. Asymmetric on purpose: a question and the passage
        # answering it are not paraphrases.
        _store = PGVector(
            embeddings=get_document_embeddings(),
            collection_name=CONTENT_COLLECTION,
            connection=_sqlalchemy_url(),
            use_jsonb=True,
        )
    return _store


def add_text(chunks: List[str], file_id: int, filename: str) -> int:
    """Store text chunks as searchable content. Returns the number stored."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=chunk,
            metadata={"file_id": file_id, "filename": filename, "modality": "text"},
        )
        for chunk in chunks
        if chunk and chunk.strip()
    ]
    if docs:
        get_content_store().add_documents(docs)
    return len(docs)


def add_image(caption: str, file_id: int, filename: str, image_path: str) -> None:
    """Store an image's caption as searchable content, keyed to the image path."""
    from langchain_core.documents import Document

    get_content_store().add_documents(
        [
            Document(
                page_content=caption,
                metadata={
                    "file_id": file_id,
                    "filename": filename,
                    "modality": "image",
                    "image_path": image_path,
                    "caption": caption,
                },
            )
        ]
    )


def search(query: str, k: int):
    """Return the top k content items, text or image, for a query."""
    from src.embeddings.vectorstore_utils import get_query_embeddings

    vector = get_query_embeddings().embed_query(query)
    return get_content_store().similarity_search_by_vector(vector, k=k)


def delete(file_id: int) -> bool:
    """Delete all content (text and images) for a document id."""
    import psycopg

    from src.core.config import settings

    try:
        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding e
                    USING langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND e.cmetadata->>'file_id' = %s
                    """,
                    (CONTENT_COLLECTION, str(file_id)),
                )
        return True
    except Exception as exc:
        logger.error("Failed to delete content for %s: %s", file_id, exc)
        return False
