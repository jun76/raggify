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
    """ヘルスチェック結果を表示用テキストへまとめる。

    Args:
        raggify_stat (Optional[dict[str, Any]]): raggify の状態

    Returns:
        dict[str, str]: サービスの状態表示テキスト
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
    """サービス状態を再取得し、セッションステートへ保存する。

    Args:
        client (RestAPIClient): raggify API クライアント
    """
    try:
        raggify_stat = client.health()
        texts = _summarize_status(raggify_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception:
        logger.warning("raggify is not ready")

        _DEFAULT_STATUS_TEXT = "不明"
        st.session_state["status_texts"] = {"raggify": _DEFAULT_STATUS_TEXT}


def _render_status_section(client: RestAPIClient) -> None:
    """メインメニューに表示するステータスセクションを描画する。

    Args:
        client (RestAPIClient): raggify API クライアント
    """
    if st.session_state.get("status_dirty", False):
        _refresh_status(client)

    st.subheader("🩺 サービスステータス")
    texts = st.session_state["status_texts"]
    st.write(f"RAG サーバー: {texts['raggify']}")
    st.button(
        "🔄 最新情報を取得",
        on_click=_refresh_status,
        args=(client,),
    )


def render_main_menu(client: RestAPIClient) -> None:
    """メインメニュー画面を描画する。

    Args:
        client (RestAPIClient): raggify API クライアント
    """
    st.title("📚 RAG システム")
    _render_status_section(client)

    st.subheader("🧭 メニュー")
    st.button("📝 ナレッジ登録へ", on_click=set_view, args=(View.INGEST,))
    st.button("🔍 ＤＢ検索画面へ", on_click=set_view, args=(View.SEARCH,))
    st.button(
        emojify_robot("🤖 RAG 検索画面へ"), on_click=set_view, args=(View.RAGSEARCH,)
    )
    st.button("🛠️ 管理メニューへ", on_click=set_view, args=(View.ADMIN,))
