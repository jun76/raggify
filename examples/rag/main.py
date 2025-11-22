from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "rag"

import streamlit as st
from raggify_client import RestAPIClient

from .config import Config
from .logger import logger
from .state import View, ensure_session_state
from .views.admin import render_admin_view
from .views.ingest import render_ingest_view
from .views.main_menu import render_main_menu
from .views.ragsearch import render_ragsearch_view
from .views.search import render_search_view


def main() -> None:
    """Entry point for the Streamlit application."""

    st.set_page_config(page_title="RAG System", page_icon="📚", layout="wide")
    ensure_session_state()

    client = RestAPIClient(host=Config.host, port=Config.port)

    view = st.session_state.get("view", View.MAIN)
    match view:
        case View.MAIN:
            render_main_menu(client)
        case View.INGEST:
            render_ingest_view(client)
        case View.SEARCH:
            render_search_view(client)
        case View.RAGSEARCH:
            render_ragsearch_view(client)
        case View.ADMIN:
            render_admin_view(client)
        case _:
            st.error("The requested view is not defined.")


if __name__ == "__main__":
    # Set the logger level
    log_level = getattr(logging, Config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    main()
