from __future__ import annotations

from typing import Any

import streamlit as st
from raggify_client import RestAPIClient

from ..logger import logger
from ..state import View, set_view
from .common import emojify_robot

__all__ = ["render_main_menu"]


def _summarize_status(
    server_stat: dict[str, Any],
) -> dict[str, str]:
    """Summarize server status into display text.

    Args:
        server_stat (dict[str, Any]): server status payload.

    Returns:
        dict[str, str]: Service status description.
    """
    details = "\n".join(
        [
            f"- vector store: {server_stat.get('vector store', 'N/A')}",
            f"- embed: {server_stat.get('embed', 'N/A')}",
            f"- rerank: {server_stat.get('rerank', 'N/A')}",
            f"- document store: {server_stat.get('document store', 'N/A')}",
            f"- ingest cache: {server_stat.get('ingest cache', 'N/A')}",
        ]
    )

    return {
        "raggify": (
            f"✅ Online\n{details}"
            if server_stat.get("status") == "ok"
            else "🛑 Offline"
        )
    }


def _refresh_status(client: RestAPIClient) -> None:
    """Refresh service status and store it in the session state.

    Args:
        client (RestAPIClient): REST API client.
    """
    try:
        server_stat = client.status()
        texts = _summarize_status(server_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception:
        logger.warning("raggify is not ready")

        _DEFAULT_STATUS_TEXT = "Unknown"
        st.session_state["status_texts"] = {"raggify": _DEFAULT_STATUS_TEXT}


def _render_status_section(client: RestAPIClient) -> None:
    """Render the status section for the main menu.

    Args:
        client (RestAPIClient): REST API client.
    """
    if st.session_state.get("status_dirty", False):
        _refresh_status(client)

    st.subheader("🩺 Service status")
    texts = st.session_state["status_texts"]
    st.markdown(f"RAG server:\n{texts['raggify']}")
    st.button(
        "🔄 Refresh status",
        on_click=_refresh_status,
        args=(client,),
    )


def render_main_menu(client: RestAPIClient) -> None:
    """Render the main menu view.

    Args:
        client (RestAPIClient): REST API client.
    """
    st.title("📚 RAG System")
    _render_status_section(client)

    st.subheader("🧭 Menu")
    st.button("📝 Go to ingestion", on_click=set_view, args=(View.INGEST,))
    st.button("🔍 Go to DB search", on_click=set_view, args=(View.SEARCH,))
    st.button(
        emojify_robot("🤖 Go to RAG search"), on_click=set_view, args=(View.RAGSEARCH,)
    )
    st.button("🛠️ Go to admin menu", on_click=set_view, args=(View.ADMIN,))
