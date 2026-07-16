"""Multimodal RAG engine for rag-multimodal-2026.

One vector search returns the most relevant text chunks and image captions
together. The answer is grounded in both, and the images that contributed are
returned with the answer so the UI can show them. Keyless on Ollama: a vision
model captions images at index time, and the answer model reasons over text and
those captions at query time.
"""

import base64
import logging
import os
import shutil
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.langchain_utils import _make_llm, _reformulate_query, _to_lc_messages
from src.embeddings.vectorstore_utils import load_document_text
from src.multimodal import store, vision

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

_QA_SYSTEM = (
    "You answer questions for rag-multimodal-2026 using the retrieved context, which "
    "combines text passages and descriptions of images. Use only what is given. "
    "If it does not contain the answer, say you do not have that information "
    "rather than inventing one.\n\nContext:\n{context}"
)


def is_image(path: str) -> bool:
    return path.lower().endswith(_IMAGE_EXTENSIONS)


def _data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode()
    return f"data:image/{ext};base64,{encoded}"


def run_multimodal(model: str, question: str, chat_history=None) -> dict:
    """Answer a question from retrieved text and image content.

    With history present the question is first rewritten to be standalone, so a
    follow-up pronoun resolves before it reaches the vector search.
    """
    steps: List[Dict[str, Any]] = []
    llm = _make_llm(model, temperature=0)

    history = _to_lc_messages(chat_history)
    query = _reformulate_query(llm, question, history)
    if query != question:
        steps.append({"step": "reformulate", "standalone_question": query})

    items = store.search(query, k=settings.top_k)
    text_items = [d for d in items if d.metadata.get("modality") == "text"]
    image_items = [d for d in items if d.metadata.get("modality") == "image"]
    steps.append(
        {"step": "retrieve", "text": len(text_items), "images": len(image_items)}
    )

    parts = [doc.page_content for doc in text_items]
    for doc in image_items:
        parts.append(f"[Image: {doc.metadata.get('filename')}] {doc.page_content}")
    context = "\n\n".join(parts) or "None."

    answer = llm.invoke(
        [
            SystemMessage(content=_QA_SYSTEM.format(context=context)),
            HumanMessage(content=query),
        ]
    ).content
    steps.append({"step": "grounded_answer"})

    images = []
    for doc in image_items:
        path = doc.metadata.get("image_path")
        if path and os.path.exists(path):
            images.append(
                {
                    "filename": doc.metadata.get("filename"),
                    "caption": doc.metadata.get("caption", ""),
                    "data_uri": _data_uri(path),
                }
            )
    sources = sorted(
        {d.metadata.get("filename") for d in items if d.metadata.get("filename")}
    )
    return {"answer": answer, "sources": sources, "images": images, "steps": steps}


def index_document(file_path: str, file_id: int, filename: str) -> bool:
    """Index a text document as chunks, or an image as a captioned, stored item."""
    if is_image(file_path):
        return _index_image(file_path, file_id, filename)
    return _index_text(file_path, file_id, filename)


def _index_text(file_path: str, file_id: int, filename: str) -> bool:
    try:
        text = load_document_text(file_path)
    except Exception as exc:
        logger.error("Could not read text %s: %s", file_path, exc)
        return False
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_text(text)
    stored = store.add_text(chunks, file_id, filename)
    logger.info("Indexed %d text chunks for file_id=%s", stored, file_id)
    return True


def _index_image(file_path: str, file_id: int, filename: str) -> bool:
    try:
        caption = vision.caption_image(
            file_path, settings.vlm_model, settings.ollama_base_url
        )
    except Exception as exc:
        logger.error("Vision captioning failed for %s: %s", file_path, exc)
        return False
    os.makedirs(settings.image_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower() or ".png"
    stored_path = os.path.join(settings.image_dir, f"{file_id}{ext}")
    try:
        shutil.copyfile(file_path, stored_path)
    except Exception as exc:
        logger.error("Could not store image %s: %s", file_path, exc)
        return False
    store.add_image(caption, file_id, filename, stored_path)
    logger.info(
        "Indexed image file_id=%s with a %d char caption", file_id, len(caption)
    )
    return True
