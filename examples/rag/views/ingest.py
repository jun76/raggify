from __future__ import annotations

from typing import Any, Optional

import streamlit as st
from raggify_client import RestAPIClient

from ..logger import logger
from ..state import (
    FeedBack,
    View,
    clear_feedback,
    display_feedback,
    set_feedback,
    set_view,
)
from .common import save_uploaded_files

__all__ = [
    "register_uploaded_files_callback",
    "register_url_callback",
    "register_url_list_callback",
    "render_ingest_view",
]


def register_uploaded_files_callback(
    client: RestAPIClient,
    files: Optional[list[Any]],
    feedback_key: FeedBack,
) -> None:
    """Register knowledge by uploading files.

    Args:
        client (RestAPIClient): REST API client.
        files (Optional[list[Any]]): Uploaded files.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    if not files:
        set_feedback(feedback_key, "warning", "No files uploaded.")
        return

    try:
        with st.spinner("Registering files..."):
            upload_ids = save_uploaded_files(client, files)
            for upload_id in upload_ids:
                client.ingest_path(upload_id=upload_id)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Failed to register files: {e}")
    else:
        set_feedback(feedback_key, "success", "File registration completed.")


def register_url_callback(
    client: RestAPIClient, url_value: str, feedback_key: FeedBack
) -> None:
    """Register knowledge by specifying a URL.

    Args:
        client (RestAPIClient): REST API client.
        url_value (str): URL to ingest.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    url = (url_value or "").strip()
    if not url:
        set_feedback(feedback_key, "warning", "Enter a URL.")
        return

    try:
        with st.spinner("Registering URL..."):
            client.ingest_url(url)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Failed to register URL: {e}")
    else:
        set_feedback(feedback_key, "success", "URL registration completed.")


def register_url_list_callback(
    client: RestAPIClient,
    file_obj: Any,
    feedback_key: FeedBack,
) -> None:
    """Register knowledge by uploading a URL list file.

    Args:
        client (RestAPIClient): REST API client.
        file_obj (Any): Uploaded URL list file.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    if file_obj is None:
        set_feedback(feedback_key, "warning", "No URL list selected.")
        return

    try:
        with st.spinner("Registering URL list..."):
            upload_id = save_uploaded_files(client, [file_obj])[0]
            client.ingest_url_list(upload_id=upload_id)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Failed to register URL list: {e}")
    else:
        set_feedback(feedback_key, "success", "URL list registration completed.")


def render_ingest_view(client: RestAPIClient) -> None:
    """Render the ingestion view.

    Args:
        client (RestAPIClient): REST API client.
    """
    st.title("📝 Knowledge Ingestion")
    st.button("⬅️ Back to menu", on_click=set_view, args=(View.MAIN,))

    st.divider()
    st.subheader("📁 Upload files")
    st.caption("Send files to raggify and register their content.")
    files = st.file_uploader("Select files", accept_multiple_files=True)
    st.button(
        "📁 Register",
        on_click=register_uploaded_files_callback,
        args=(client, files, FeedBack.FB_INGEST_FILES),
    )
    display_feedback(FeedBack.FB_INGEST_FILES)

    st.divider()
    st.subheader("🌐 Register a URL")
    st.caption("Notify raggify of a URL and register its content.")
    url_value = st.text_input("Target URL", key="ingest_url_input")
    st.button(
        "🌐 Register",
        on_click=register_url_callback,
        args=(client, url_value, FeedBack.FB_INGEST_URL),
    )
    display_feedback(FeedBack.FB_INGEST_URL)

    st.divider()
    st.subheader("📚 Upload a URL list")
    st.caption("Send a text file (*.txt) that lists URLs to raggify and register them.")
    url_list = st.file_uploader("Select a URL list", key="url_list_uploader")
    st.button(
        "📚 Register",
        on_click=register_url_list_callback,
        args=(client, url_list, FeedBack.FB_INGEST_URL_LIST),
    )
    display_feedback(FeedBack.FB_INGEST_URL_LIST)
