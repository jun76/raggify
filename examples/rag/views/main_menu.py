from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from raggify.client import RestAPIClient

from ..logger import logger
from ..state import View, set_view
from .common import emojify_robot

__all__ = ["render_main_menu"]


def _summarize_status(
    raggify_stat: Optional[dict[str, Any]],
) -> dict[str, str]:
    """Summarize the health check result into display text.

    Args:
        raggify_stat (Optional[dict[str, Any]]): raggify status payload.

    Returns:
        dict[str, str]: Service status description.
    """
    return {
        "raggify": (
            "✅ Online ("
            + ", ".join(
                [
                    f"store: {raggify_stat.get('store', 'N/A')}",
                    f"embed: {raggify_stat.get('embed', 'N/A')}",
                    f"rerank: {raggify_stat.get('rerank', 'N/A')}",
                ]
            )
            + ")"
            if raggify_stat and raggify_stat.get("status") == "ok"
            else "🛑 Offline"
        )
    }


def _refresh_status(client: RestAPIClient) -> None:
    """Refresh service status and store it in the session state.

    Args:
        client (RestAPIClient): raggify API client.
    """
    try:
        raggify_stat = client.health()
        texts = _summarize_status(raggify_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception:
        logger.warning("raggify is not ready")

        _DEFAULT_STATUS_TEXT = "Unknown"
        st.session_state["status_texts"] = {"raggify": _DEFAULT_STATUS_TEXT}


def _render_status_section(client: RestAPIClient) -> None:
    """Render the status section for the main menu.

    Args:
        client (RestAPIClient): raggify API client.
    """
    if st.session_state.get("status_dirty", False):
        _refresh_status(client)

    st.subheader("🩺 Service status")
    texts = st.session_state["status_texts"]
    st.write(f"RAG server: {texts['raggify']}")
    st.button(
        "🔄 Refresh status",
        on_click=_refresh_status,
        args=(client,),
    )


def render_main_menu(client: RestAPIClient) -> None:
    """Render the main menu view.

    Args:
        client (RestAPIClient): raggify API client.
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
