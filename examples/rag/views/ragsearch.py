from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Optional

import streamlit as st
from raggify_client import RestAPIClient

from raggify.core import Exts

from ..agent import AgentExecutionError, RagAgentManager
from ..config import Config
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
    """Upload a reference file to raggify and return its upload id.

    Args:
        client (RestAPIClient): REST API client.
        file_obj (Optional[Any]): Uploaded file object.
        session_key (RagSearchSessionKey): Session key used to store the upload id.

    Raises:
        AgentExecutionError: Raised when the upload fails.

    Returns:
        Optional[str]: Upload id, or None when no file is uploaded.
    """
    if file_obj is None:
        st.session_state[session_key] = None
        return None

    try:
        saved = save_uploaded_files(client, [file_obj])
    except Exception as e:
        st.session_state[session_key] = None
        raise AgentExecutionError(f"failed to upload reference file: {e}") from e

    upload_id = saved[0] if saved else None
    st.session_state[session_key] = upload_id
    return upload_id


def render_ragsearch_view(client: RestAPIClient) -> None:
    """Render the RAG search view.

    Args:
        client (RestAPIClient): REST API client.
    """
    st.title(emojify_robot("🤖 RAG Search"))
    st.button(
        "⬅️ Back to menu", key="ragsearch_back", on_click=set_view, args=(View.MAIN,)
    )
    st.divider()

    question = st.text_area("Question", key="ragsearch_question")

    ref_file = st.file_uploader(
        "Attachment (optional)",
        type=list(Exts.IMAGE | Exts.AUDIO | Exts.VIDEO),
        key="ragsearch_image",
    )

    if RagSearchSessionKey.ANSWER not in st.session_state:
        st.session_state[RagSearchSessionKey.ANSWER] = None

    if st.button(emojify_robot("🤖 Submit"), key="ragsearch_submit"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            upload_id = None
            try:
                upload_id = _save_reference_file(
                    client, ref_file, RagSearchSessionKey.IMAGE_PATH
                )
            except AgentExecutionError as e:
                st.error(str(e))
                st.session_state[RagSearchSessionKey.ANSWER] = None
            else:
                manager = RagAgentManager(client=client, model=Config.openai_llm_model)
                try:
                    with st.spinner("Running RAG search..."):
                        answer = manager.run(
                            question=question,
                            upload_id=upload_id,
                        )
                except AgentExecutionError as e:
                    st.session_state[RagSearchSessionKey.ANSWER] = None
                    st.error(f"Failed to run RAG search: {e}")
                else:
                    st.session_state[RagSearchSessionKey.ANSWER] = emojify_robot(answer)
                    st.success("RAG search completed.")

    final_answer: Optional[str] = st.session_state.get(RagSearchSessionKey.ANSWER)
    if final_answer:
        st.divider()
        st.header("🧠 Final answer")
        st.write(final_answer, unsafe_allow_html=True)
