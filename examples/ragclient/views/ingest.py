from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from raggify.client import RestAPIClient

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
    """アップロードファイル経由でナレッジ登録を実行する。

    Args:
        client (RestAPIClient): raggify API クライアント
        files (Optional[list[Any]]): アップロードされたファイル群
        feedback_key (FeedBack): フィードバック表示用キー

    """
    clear_feedback(feedback_key)
    if not files:
        set_feedback(feedback_key, "warning", "アップロードされたファイルがありません")
        return

    try:
        with st.spinner("ファイルを取り込み中です..."):
            saved_paths = save_uploaded_files(client, files)
            for path in saved_paths:
                client.ingest_path(path)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"ファイルの取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "取り込みが完了しました")


def register_url_callback(
    client: RestAPIClient, url_value: str, feedback_key: FeedBack
) -> None:
    """URL 指定でナレッジ登録を実行する。

    Args:
        client (RestAPIClient): raggify API クライアント
        url_value (str): 取り込み対象 URL
        feedback_key (FeedBack): フィードバック表示用キー

    """
    clear_feedback(feedback_key)
    url = (url_value or "").strip()
    if not url:
        set_feedback(feedback_key, "warning", "URL を入力してください")
        return

    try:
        with st.spinner("URL を取り込み中です..."):
            client.ingest_url(url)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"URL の取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "URL の取り込みが完了しました")


def register_url_list_callback(
    client: RestAPIClient,
    file_obj: Any,
    feedback_key: FeedBack,
) -> None:
    """URL リストファイル経由でナレッジ登録を実行する。

    Args:
        client (RestAPIClient): raggify API クライアント
        file_obj (Any): アップロードされた URL リスト
        feedback_key (FeedBack): フィードバック表示用キー

    """
    clear_feedback(feedback_key)
    if file_obj is None:
        set_feedback(feedback_key, "warning", "URL リストが選択されていません")
        return

    try:
        with st.spinner("URL リストを取り込み中です..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            client.ingest_url_list(saved)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"URL リストの取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "URL リストの取り込みが完了しました")


def render_ingest_view(client: RestAPIClient) -> None:
    """ナレッジ登録画面を描画する。

    Args:
        client (RestAPIClient): raggify API クライアント

    """
    st.title("📝 ナレッジ登録")
    st.button("⬅️ メニューに戻る", on_click=set_view, args=(View.MAIN,))

    st.divider()
    st.subheader("📁 ファイルをアップロードして登録")
    st.caption("ファイルを raggify に送信し、その内容を登録します。")
    files = st.file_uploader("ファイルを選択", accept_multiple_files=True)
    st.button(
        "📁 登録",
        on_click=register_uploaded_files_callback,
        args=(client, files, FeedBack.FB_INGEST_FILES),
    )
    display_feedback(FeedBack.FB_INGEST_FILES)

    st.divider()
    st.subheader("🌐 URL を指定して登録")
    st.caption("URL を raggify に通知し、その内容を登録します。")
    url_value = st.text_input("対象 URL", key="ingest_url_input")
    st.button(
        "🌐 登録",
        on_click=register_url_callback,
        args=(client, url_value, FeedBack.FB_INGEST_URL),
    )
    display_feedback(FeedBack.FB_INGEST_URL)

    st.divider()
    st.subheader("📚 URL リストをアップロードして登録")
    st.caption(
        "URL を列挙したテキストファイル（*.txt）を raggify に送信し、その内容を登録します。"
    )
    url_list = st.file_uploader("URL リストを選択", key="url_list_uploader")
    st.button(
        "📚 登録",
        on_click=register_url_list_callback,
        args=(client, url_list, FeedBack.FB_INGEST_URL_LIST),
    )
    display_feedback(FeedBack.FB_INGEST_URL_LIST)
