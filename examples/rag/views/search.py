from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from raggify.client import RestAPIClient
from raggify.core import Exts

from ..logger import logger
from ..state import (
    FeedBack,
    SearchResult,
    View,
    clear_feedback,
    clear_search_result,
    display_feedback,
    set_feedback,
    set_search_result,
    set_view,
)
from .common import save_uploaded_files

__all__ = [
    "run_text_text_search_callback",
    "run_text_image_search_callback",
    "run_image_image_search_callback",
    "run_text_audio_search_callback",
    "run_audio_audio_search_callback",
    "render_search_view",
]


def _render_search_section(
    *,
    title: str,
    caption: str,
    input_func: Callable[[], Any],
    button_label: str,
    button_callback: Callable[..., None],
    button_args: Callable[[Any], tuple],
    feedback_key: FeedBack,
    result_key: SearchResult,
    result_renderer: Callable[[dict[str, Any]], None],
) -> None:
    """Render a shared search form block.

    Args:
        title (str): Section title.
        caption (str): Helper text displayed under the title.
        input_func (Callable[[], Any]): Callable that renders the input widget.
        button_label (str): Label for the search button.
        button_callback (Callable[..., None]): Callback executed on submit.
        button_args (Callable[[Any], tuple]): Callable that builds callback args.
        feedback_key (FeedBack): Feedback state key.
        result_key (SearchResult): Session key for the search result.
        result_renderer (Callable[[dict[str, Any]], None]): Renderer for results.
    """
    st.subheader(title)
    if caption:
        st.caption(caption)

    value = input_func()
    st.button(
        button_label,
        on_click=button_callback,
        args=button_args(value),
    )

    display_feedback(feedback_key)
    result = st.session_state.get(result_key)
    if result is not None:
        result_renderer(result)


def _render_search_view_text_text(client: RestAPIClient) -> None:
    """Render the text-to-text search section.

    Args:
        client (RestAPIClient): raggify API client.
    """
    _render_search_section(
        title="📝→📝 Search text with text",
        caption="Find passages similar to the search query. Example: \"employment rules summary\"",
        input_func=lambda: st.text_input("Search query", key="text_text_query"),
        button_label="🔎 Search",
        button_callback=run_text_text_search_callback,
        button_args=lambda query: (
            client,
            query,
            SearchResult.SR_SEARCH_TEXT_TEXT,
            FeedBack.FB_SEARCH_TEXT_TEXT,
        ),
        feedback_key=FeedBack.FB_SEARCH_TEXT_TEXT,
        result_key=SearchResult.SR_SEARCH_TEXT_TEXT,
        result_renderer=lambda data: _render_query_results_text("📝 Search results", data),
    )


def _render_search_view_text_image(client: RestAPIClient) -> None:
    """Render the text-to-image search section.

    Args:
        client (RestAPIClient): raggify API client.
    """
    _render_search_section(
        title="📝→🖼️ Search images with text",
        caption="Find images similar to the search query. Example: \"friends having a conversation\"",
        input_func=lambda: st.text_input("Search query", key="text_image_query"),
        button_label="🔎 Search",
        button_callback=run_text_image_search_callback,
        button_args=lambda query: (
            client,
            query,
            SearchResult.SR_SEARCH_TEXT_IMAGE,
            FeedBack.FB_SEARCH_TEXT_IMAGE,
        ),
        feedback_key=FeedBack.FB_SEARCH_TEXT_IMAGE,
        result_key=SearchResult.SR_SEARCH_TEXT_IMAGE,
        result_renderer=lambda data: _render_query_results_image("🖼️ Search results", data),
    )


def _render_search_view_image_image(client: RestAPIClient) -> None:
    """Render the image-to-image search section.

    Args:
        client (RestAPIClient): raggify API client.
    """
    _render_search_section(
        title="🖼️→🖼️ Search images with an image",
        caption="Upload an image to find similar images.",
        input_func=lambda: st.file_uploader(
            "Select an image to search", key="image_query_uploader"
        ),
        button_label="🔎 Search",
        button_callback=run_image_image_search_callback,
        button_args=lambda file_obj: (
            client,
            file_obj,
            SearchResult.SR_SEARCH_IMAGE_IMAGE,
            FeedBack.FB_SEARCH_IMAGE_IMAGE,
        ),
        feedback_key=FeedBack.FB_SEARCH_IMAGE_IMAGE,
        result_key=SearchResult.SR_SEARCH_IMAGE_IMAGE,
        result_renderer=lambda data: _render_query_results_image("🖼️ Search results", data),
    )


def _render_search_view_text_audio(client: RestAPIClient) -> None:
    """Render the text-to-audio search section.

    Args:
        client (RestAPIClient): raggify API client.
    """
    _render_search_section(
        title="📝→🎤 Search audio with text",
        caption="Find audio similar to the query. Example: \"car horn\"",
        input_func=lambda: st.text_input("Search query", key="text_audio_query"),
        button_label="🔎 Search",
        button_callback=run_text_audio_search_callback,
        button_args=lambda query: (
            client,
            query,
            SearchResult.SR_SEARCH_TEXT_AUDIO,
            FeedBack.FB_SEARCH_TEXT_AUDIO,
        ),
        feedback_key=FeedBack.FB_SEARCH_TEXT_AUDIO,
        result_key=SearchResult.SR_SEARCH_TEXT_AUDIO,
        result_renderer=lambda data: _render_query_results_audio("🎤 Search results", data),
    )


def _render_search_view_audio_audio(client: RestAPIClient) -> None:
    """Render the audio-to-audio search section.

    Args:
        client (RestAPIClient): raggify API client.
    """
    _render_search_section(
        title="🎤→🎤 Search audio with audio",
        caption="Upload audio to find similar clips.",
        input_func=lambda: st.file_uploader(
            "Select audio to search", key="audio_query_uploader"
        ),
        button_label="🔎 Search",
        button_callback=run_audio_audio_search_callback,
        button_args=lambda file_obj: (
            client,
            file_obj,
            SearchResult.SR_SEARCH_AUDIO_AUDIO,
            FeedBack.FB_SEARCH_AUDIO_AUDIO,
        ),
        feedback_key=FeedBack.FB_SEARCH_AUDIO_AUDIO,
        result_key=SearchResult.SR_SEARCH_AUDIO_AUDIO,
        result_renderer=lambda data: _render_query_results_audio("🎤 Search results", data),
    )


def _run_text_search(
    func: Callable[[str], dict[str, Any]],
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """Execute a text-based search."""

    clear_feedback(feedback_key)
    clear_search_result(result_key)

    text = (query or "").strip()
    if not text:
        set_feedback(feedback_key, "warning", "Enter a query.")
        return

    try:
        with st.spinner("Searching..."):
            result = func(text)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Search failed: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "Search completed.")


def run_text_text_search_callback(
    client: RestAPIClient,
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """Call the text-to-text search API."""
    _run_text_search(
        func=client.query_text_text,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_text_image_search_callback(
    client: RestAPIClient,
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """Call the text-to-image search API."""
    _run_text_search(
        func=client.query_text_image,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_image_image_search_callback(
    client: RestAPIClient,
    file_obj: Any,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """Call the image-to-image search API.

    Args:
        client (RestAPIClient): raggify API client.
        file_obj (Any): Uploaded image file.
        result_key (SearchResult): Session key used for the result.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    clear_search_result(result_key)

    if file_obj is None:
        set_feedback(feedback_key, "warning", "No image selected.")
        return

    try:
        with st.spinner("Searching images..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            result = client.query_image_image(saved)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Image search failed: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "Search completed.")


def run_text_audio_search_callback(
    client: RestAPIClient,
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """Call the text-to-audio search API."""
    _run_text_search(
        func=client.query_text_audio,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_audio_audio_search_callback(
    client: RestAPIClient,
    file_obj: Any,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """Call the audio-to-audio search API.

    Args:
        client (RestAPIClient): raggify API client.
        file_obj (Any): Uploaded audio file.
        result_key (SearchResult): Session key used for the result.
        feedback_key (FeedBack): Feedback state key.
    """
    clear_feedback(feedback_key)
    clear_search_result(result_key)

    if file_obj is None:
        set_feedback(feedback_key, "warning", "No audio file selected.")
        return

    try:
        with st.spinner("Searching audio..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            result = client.query_audio_audio(saved)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"Audio search failed: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "Search completed.")


def _render_query_results_text(title: str, result: dict[str, Any]) -> None:
    """Render text search results."""

    st.subheader(title)
    documents = result.get("documents") if isinstance(result, dict) else None
    if not documents:
        st.info("No matching documents.")
        return

    for doc in documents:
        metadata = doc.get("metadata", {})
        content = doc.get("text", "")
        source = (
            metadata.get("base_source", "")
            or metadata.get("url", "")
            or metadata.get("file_path", "")
        )  # Priority order

        st.divider()
        st.markdown("#### Content")
        st.write(content)
        st.markdown("##### Source")
        st.write(source)


def _render_query_results_image(title: str, result: dict[str, Any]) -> None:
    """Render image search results."""
    st.subheader(title)
    documents = result.get("documents") if isinstance(result, dict) else None
    if not documents:
        st.info("No matching images.")
        return

    for doc in documents:
        metadata = doc.get("metadata", {})
        source = metadata.get("url", "") or metadata.get("file_path", "")  # Priority order

        st.divider()
        try:
            st.image(source, width="content")
        except Exception as e:
            logger.warning(f"failed to render result image: {e}")
            st.warning("Unable to display the embedded file.")

        st.markdown("##### Source")
        st.write(source)

        base_source = metadata.get("base_source", "")
        if base_source and source != base_source:
            st.write(f"Reference: {base_source}")


def _render_query_results_audio(title: str, result: dict[str, Any]) -> None:
    """Render audio search results."""
    st.subheader(title)
    documents = result.get("documents") if isinstance(result, dict) else None
    if not documents:
        st.info("No matching audio.")
        return

    for doc in documents:
        metadata = doc.get("metadata", {})
        source = metadata.get("url", "") or metadata.get("file_path", "")  # Priority order

        st.divider()
        try:
            ext = Exts.get_ext(uri=source, dot=False)
            st.audio(data=source, format=f"audio/{ext}")
        except Exception as e:
            logger.warning(f"failed to render result audio: {e}")
            st.warning("Unable to play the embedded file.")

        st.markdown("##### Source")
        st.write(source)

        base_source = metadata.get("base_source", "")
        if base_source and source != base_source:
            st.write(f"Reference: {base_source}")


def render_search_view(client: RestAPIClient) -> None:
    """Render the search view."""

    st.title("🔎 Search")
    st.button(
        "⬅️ Back to menu", key="search_back", on_click=set_view, args=(View.MAIN,)
    )
    st.divider()

    choice_map: dict[str, Callable[[RestAPIClient], None]] = {
        "Text 📝 → Text 📝": _render_search_view_text_text,
        "Text 📝 → Image 🖼️": _render_search_view_text_image,
        "Image 🖼️ → Image 🖼️": _render_search_view_image_image,
        "Text 📝 → Audio 🎤": _render_search_view_text_audio,
        "Audio 🎤 → Audio 🎤": _render_search_view_audio_audio,
    }
    choice = st.sidebar.selectbox(
        "Choose a search option.", list(choice_map.keys())
    )

    renderer = choice_map.get(choice)
    if renderer is not None:
        renderer(client)
