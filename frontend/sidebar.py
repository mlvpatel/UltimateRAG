"""Sidebar component: model choice and document management."""

import streamlit as st

from frontend import api_utils

MODELS = ["qwen2.5:7b-instruct", "llama3.2:3b", "gpt-4o-mini", "gpt-4o"]


def display_sidebar() -> None:
    with st.sidebar:
        st.header("rag-multimodal-2026")
        st.session_state["model"] = st.selectbox("Model", MODELS, index=0)

        st.subheader("Documents")
        uploaded = st.file_uploader(
            "Upload a document or image",
            type=[
                "pdf",
                "docx",
                "html",
                "txt",
                "md",
                "png",
                "jpg",
                "jpeg",
                "gif",
                "webp",
            ],
        )
        if uploaded is not None and st.button("Index document"):
            result = api_utils.upload_document(uploaded.getvalue(), uploaded.name)
            st.success(f"Queued for indexing (task {result.get('task_id')})")

        try:
            documents = api_utils.list_documents()
        except Exception:
            st.caption("API not reachable")
            return

        for document in documents:
            columns = st.columns([3, 1])
            columns[0].write(document["filename"])
            if columns[1].button("Delete", key=f"delete_{document['id']}"):
                api_utils.delete_document(document["id"])
                st.rerun()
