from __future__ import annotations

from typing import Any, Optional

import requests
import streamlit as st

from ..logger import logger
from ..state import View, set_view
from .common import emojify_robot

__all__ = ["render_main_menu"]


def _check_service_health(url: str) -> Optional[dict[str, Any]]:
    """ヘルスチェックエンドポイントへアクセスし、サービス稼働状況を取得する。

    Args:
        url (str): ヘルスチェック URL

    Returns:
        Optional[dict[str, Any]]: 応答 JSON（失敗時は None）
    """
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception:
        logger.warning("no response from raggify")
        return None

    if not isinstance(data, dict):
        logger.warning("health check response is not a dict for %s", url)
        return None

    return data


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


def _refresh_status(raggify_health: str) -> None:
    """サービス状態を再取得し、セッションステートへ保存する。

    Args:
        raggify_health (str): raggify のヘルスチェック URL
    """
    try:
        raggify_stat = _check_service_health(raggify_health)
        texts = _summarize_status(raggify_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception:
        logger.warning("raggify is not ready")

        _DEFAULT_STATUS_TEXT = "不明"
        st.session_state["status_texts"] = {"raggify": _DEFAULT_STATUS_TEXT}


def _render_status_section(raggify_health: str) -> None:
    """メインメニューに表示するステータスセクションを描画する。

    Args:
        raggify_health (str): raggify のヘルスチェック URL
    """
    if st.session_state.get("status_dirty", False):
        _refresh_status(raggify_health)

    st.subheader("🩺 サービスステータス")
    texts = st.session_state["status_texts"]
    st.write(f"RAG サーバー: {texts['raggify']}")
    st.button(
        "🔄 最新情報を取得",
        on_click=_refresh_status,
        args=(raggify_health,),
    )


def render_main_menu(raggify_health: str) -> None:
    """メインメニュー画面を描画する。

    Args:
        raggify_health (str): raggify のヘルスチェック URL
    """
    st.title("📚 RAG Client")
    _render_status_section(raggify_health)

    st.subheader("🧭 メニュー")
    st.button("📝 ナレッジ登録へ", on_click=set_view, args=(View.INGEST,))
    st.button("🔍 ＤＢ検索画面へ", on_click=set_view, args=(View.SEARCH,))
    st.button(
        emojify_robot("🤖 RAG 検索画面へ"), on_click=set_view, args=(View.RAGSEARCH,)
    )
    st.button("🛠️ 管理メニューへ", on_click=set_view, args=(View.ADMIN,))
