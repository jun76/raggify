from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "ragclient"

import streamlit as st

from .api_client import RAGgifyClient
from .config.config import Config
from .logger import logger
from .state import View, ensure_session_state
from .views.admin import render_admin_view
from .views.ingest import render_ingest_view
from .views.main_menu import render_main_menu
from .views.ragsearch import render_ragsearch_view
from .views.search import render_search_view


def _init_services() -> tuple[RAGgifyClient, str]:
    """設定を読み込み、API クライアントとヘルスチェック用 URL を初期化する。

    Returns:
        tuple[RAGgifyClient, str]: API クライアントと
            raggify サービスのヘルスチェック URL
    """
    client = RAGgifyClient(Config.raggify_base_url)
    raggify_health = Config.raggify_base_url.rstrip("/") + "/health"

    return client, raggify_health


def main() -> None:
    """Streamlit アプリのエントリポイント。"""

    st.set_page_config(page_title="RAG Client", page_icon="🧠", layout="wide")
    ensure_session_state()

    client, raggify_health = _init_services()

    view = st.session_state.get("view", View.MAIN)
    match view:
        case View.MAIN:
            render_main_menu(raggify_health)
        case View.INGEST:
            render_ingest_view(client)
        case View.SEARCH:
            render_search_view(client)
        case View.RAGSEARCH:
            render_ragsearch_view(client)
        case View.ADMIN:
            render_admin_view(client)
        case _:
            st.error("未定義の画面です")


if __name__ == "__main__":
    # ログレベルを設定
    log_level = getattr(logging, Config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    main()
