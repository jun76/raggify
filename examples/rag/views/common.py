from __future__ import annotations

from typing import Any, Optional

from raggify.client import RestAPIClient

__all__ = ["emojify_robot", "save_uploaded_files"]


def emojify_robot(s: str) -> str:
    """Ensure the robot emoji renders properly instead of plain text.
    Reference: https://github.com/streamlit/streamlit/issues/11390

    Args:
        s (str): String that may contain the robot emoji.

    Returns:
        str: Updated string with proper emoji presentation.
    """
    return s.replace("\U0001f916", "\U0001f916" + "\ufe0f")  # 🤖


def save_uploaded_files(client: RestAPIClient, files: list[Any]) -> list[str]:
    """Persist uploaded files and return their paths on raggify.

    Args:
        client (RestAPIClient): raggify API client.
        files (list[Any]): Uploaded file objects from Streamlit.

    Returns:
        list[str]: List of saved file paths.

    Raises:
        RuntimeError: Raised when the response payload is invalid.
    """
    payload: list[tuple[str, bytes, Optional[str]]] = []
    for uploaded in files:
        data = uploaded.getvalue()
        payload.append((uploaded.name, data, getattr(uploaded, "type", None)))

    if not payload:
        return []

    response = client.upload(payload)
    entries = response.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("raggify upload response is invalid")

    saved: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("raggify upload response item is invalid")
        save_path = item.get("save_path")
        if not isinstance(save_path, str) or save_path == "":
            raise RuntimeError("raggify upload save_path is invalid")
        saved.append(save_path)

    if len(saved) != len(payload):
        raise RuntimeError("raggify upload file count mismatch")

    return saved
