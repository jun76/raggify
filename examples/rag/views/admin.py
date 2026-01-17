from __future__ import annotations

from typing import Any

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
    "register_local_path_callback",
    "register_path_list_callback",
    "render_admin_view",
]


def register_local_path_callback(
    client: RestAPIClient, path_value: str, feedback_key: FeedBack
) -> None:
    """Run ingestion for a local path.

    Args:
        client (RestAPIClient): REST API client.
        path_value (str): Target path for ingestion.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    path = (path_value or "").strip()
    if not path:
        set_feedback(feedback_key, "warning", "Enter a path.")
        return

    try:
        with st.spinner("Registering path..."):
            client.ingest_path(path)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Failed to register path: {e}")
    else:
        set_feedback(feedback_key, "success", "Path registration completed.")


def register_path_list_callback(
    client: RestAPIClient,
    file_obj: Any,
    feedback_key: FeedBack,
) -> None:
    """Run ingestion for a local path list file.

    Args:
        client (RestAPIClient): REST API client.
        file_obj (Any): Uploaded path list file.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    if file_obj is None:
        set_feedback(feedback_key, "warning", "No path list selected.")
        return

    try:
        with st.spinner("Registering path list..."):
            upload_id = save_uploaded_files(client, [file_obj])[0]
            client.ingest_path_list(upload_id=upload_id)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Failed to register path list: {e}")
    else:
        set_feedback(feedback_key, "success", "Path list registration completed.")


def render_admin_view(client: RestAPIClient) -> None:
    """Render the administrator menu view.

    Args:
        client (RestAPIClient): REST API client.
    """
    st.title("🛠️ Admin Menu")
    st.button("⬅️ Back to menu", key="admin_back", on_click=set_view, args=(View.MAIN,))

    st.divider()
    st.subheader("🗂️ Register a local raggify path")
    st.caption(
        "Register knowledge from files or folders already placed on the raggify host."
    )
    path_value = st.text_input("Target path", key="admin_path")
    st.button(
        "🗂️ Register",
        on_click=register_local_path_callback,
        args=(client, path_value, FeedBack.FB_ADMIN_PATH),
    )
    display_feedback(FeedBack.FB_ADMIN_PATH)

    st.divider()
    st.subheader("📄 Upload a raggify path list")
    st.caption(
        "Register knowledge from a text file (*.txt) listing local files or folders on the raggify host."
    )
    path_list = st.file_uploader("Select a path list", key="admin_path_list")
    st.button(
        "📄 Register",
        on_click=register_path_list_callback,
        args=(client, path_list, FeedBack.FB_ADMIN_PATH_LIST),
    )
    display_feedback(FeedBack.FB_ADMIN_PATH_LIST)
