from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Optional

import streamlit as st

from raggify.client import RestAPIClient
from raggify.core import Exts

from ..agent import AgentExecutionError, RagAgentManager
from ..config.config import Config
from ..state import View, set_view
from .common import emojify_robot, save_uploaded_files

__all__ = ["render_ragsearch_view"]


class RagSearchSessionKey(StrEnum):
    ANSWER = auto()
    IMAGE_PATH = auto()
    AUDIO_PATH = auto()


def _save_reference_file(
    client: RestAPIClient,
    file_obj: Optional[Any],
    session_key: RagSearchSessionKey,
) -> Optional[str]:
    """アップロードファイルを raggify に保存しパスを返す。

    Args:
        client (RestAPIClient): raggify API クライアント
        file_obj (Optional[Any]): アップロードされたファイルオブジェクト
        session_key (RagSearchSessionKey): セッションステートに保存するキー

    Raises:
        AgentExecutionError: ファイルのアップロードに失敗した場合

    Returns:
        Optional[str]: 保存されたファイルパス。アップロードが無ければ None
    """
    if file_obj is None:
        st.session_state[session_key] = None
        return None

    try:
        saved = save_uploaded_files(client, [file_obj])
    except Exception as e:
        st.session_state[session_key] = None
        raise AgentExecutionError(f"failed to upload reference file: {e}") from e

    path = saved[0] if saved else None
    st.session_state[session_key] = path
    return path


def render_ragsearch_view(client: RestAPIClient) -> None:
    """RAG 検索画面を描画する。

    Args:
        client (RestAPIClient): raggify API クライアント
    """
    st.title(emojify_robot("🤖 RAG 検索"))
    st.button(
        "⬅️ メニューに戻る", key="ragsearch_back", on_click=set_view, args=(View.MAIN,)
    )
    st.divider()

    question = st.text_area("質問文", key="ragsearch_question")

    ref_file = st.file_uploader(
        "添付ファイル（任意）",
        type=list(Exts.IMAGE | Exts.AUDIO),
        key="ragsearch_image",
    )

    if RagSearchSessionKey.ANSWER not in st.session_state:
        st.session_state[RagSearchSessionKey.ANSWER] = None

    if st.button(emojify_robot("🤖 送信"), key="ragsearch_submit"):
        if not question.strip():
            st.warning("質問文を入力してください")
        else:
            file_path = None
            try:
                file_path = _save_reference_file(
                    client, ref_file, RagSearchSessionKey.IMAGE_PATH
                )
            except AgentExecutionError as e:
                st.error(str(e))
                st.session_state[RagSearchSessionKey.ANSWER] = None
            else:
                manager = RagAgentManager(client=client, model=Config.openai_llm_model)
                try:
                    with st.spinner("RAG 検索を実行しています..."):
                        answer = manager.run(
                            question=question,
                            file_path=file_path,
                        )
                except AgentExecutionError as e:
                    st.session_state[RagSearchSessionKey.ANSWER] = None
                    st.error(f"RAG 検索に失敗しました: {e}")
                else:
                    st.session_state[RagSearchSessionKey.ANSWER] = emojify_robot(answer)
                    st.success("RAG 検索が完了しました")

    final_answer: Optional[str] = st.session_state.get(RagSearchSessionKey.ANSWER)
    if final_answer:
        st.divider()
        st.header("🧠 最終回答")
        st.write(final_answer, unsafe_allow_html=True)
