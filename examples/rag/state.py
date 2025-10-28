from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Optional

import streamlit as st

from .logger import logger

__all__ = [
    "View",
    "FeedBack",
    "SearchResult",
    "ensure_session_state",
    "set_view",
    "set_feedback",
    "clear_feedback",
    "display_feedback",
    "set_search_result",
    "clear_search_result",
]


class View(StrEnum):
    MAIN = auto()
    INGEST = auto()
    SEARCH = auto()
    RAGSEARCH = auto()
    ADMIN = auto()


class FeedBack(StrEnum):
    # Ingest
    FB_INGEST_FILES = auto()
    FB_INGEST_URL = auto()
    FB_INGEST_URL_LIST = auto()
    # Search
    FB_SEARCH_TEXT_TEXT = auto()
    FB_SEARCH_TEXT_IMAGE = auto()
    FB_SEARCH_IMAGE_IMAGE = auto()
    FB_SEARCH_TEXT_AUDIO = auto()
    FB_SEARCH_AUDIO_AUDIO = auto()
    # RAG Search
    FB_RAGSEARCH_TEXT_TEXT = auto()
    FB_RAGSEARCH_TEXT_IMAGE = auto()
    FB_RAGSEARCH_IMAGE_IMAGE = auto()
    FB_RAGSEARCH_TEXT_AUDIO = auto()
    FB_RAGSEARCH_AUDIO_AUDIO = auto()
    # Admin
    FB_ADMIN_PATH = auto()
    FB_ADMIN_PATH_LIST = auto()


class SearchResult(StrEnum):
    # Search
    SR_SEARCH_TEXT_TEXT = auto()
    SR_SEARCH_TEXT_IMAGE = auto()
    SR_SEARCH_IMAGE_IMAGE = auto()
    SR_SEARCH_TEXT_AUDIO = auto()
    SR_SEARCH_AUDIO_AUDIO = auto()
    # RAG Search
    SR_RAGSEARCH_TEXT_TEXT = auto()
    SR_RAGSEARCH_TEXT_IMAGE = auto()
    SR_RAGSEARCH_IMAGE_IMAGE = auto()
    SR_RAGSEARCH_TEXT_AUDIO = auto()
    SR_RAGSEARCH_AUDIO_AUDIO = auto()


def ensure_session_state() -> None:
    """Initialize the Streamlit session state."""

    _DEFAULT_STATUS_TEXT = "Unknown"
    current_view = st.session_state.get("view")
    if current_view is None:
        st.session_state["view"] = View.MAIN
    elif not isinstance(current_view, View):
        try:
            st.session_state["view"] = View[str(current_view).upper()]
        except KeyError:
            st.session_state["view"] = View.MAIN

    if "status_texts" not in st.session_state:
        st.session_state["status_texts"] = {
            "raggify": _DEFAULT_STATUS_TEXT,
            "embed": _DEFAULT_STATUS_TEXT,
            "rerank": _DEFAULT_STATUS_TEXT,
        }

    if "status_dirty" not in st.session_state:
        st.session_state["status_dirty"] = True

    for key in FeedBack:
        st.session_state.setdefault(key, None)

    for key in SearchResult:
        st.session_state.setdefault(key, None)


def set_view(view: View) -> None:
    """Update the currently displayed view.

    Args:
        view (View): Destination view identifier.
    """
    st.session_state["view"] = view
    if view == View.MAIN:
        st.session_state["status_dirty"] = True


def set_feedback(key: FeedBack | str, category: str, message: str) -> None:
    """Store a feedback message in the session.

    Args:
        key (FeedBack | str): Session state key.
        category (str): Message category.
        message (str): Message text.
    """
    st.session_state[key] = {"category": category, "message": message}


def clear_feedback(key: FeedBack | str) -> None:
    """Clear the feedback message stored under the given key.

    Args:
        key (FeedBack | str): Session state key.
    """
    st.session_state[key] = None


def display_feedback(key: FeedBack | str) -> None:
    """Render the stored feedback message on Streamlit.

    Args:
        key (FeedBack | str): Session state key.
    """
    payload = st.session_state.get(key)
    if not payload:
        return

    category = payload.get("category", "")
    message = payload.get("message", "")

    if category == "success":
        st.success(message)
    elif category == "error":
        st.error(message)
    elif category == "warning":
        st.warning(message)
    elif category == "info":
        st.info(message)
    else:
        logger.warning(f"undefined category: {category}")


def set_search_result(
    key: SearchResult | str, result: Optional[dict[str, Any]]
) -> None:
    """Store the search result in the session.

    Args:
        key (SearchResult | str): Session state key.
        result (Optional[dict[str, Any]]): Search result to store.
    """
    st.session_state[key] = result


def clear_search_result(key: SearchResult | str) -> None:
    """Clear the stored search result."""

    st.session_state[key] = None
