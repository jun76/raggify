from __future__ import annotations

from typing import Any, Optional

from raggify_client import RestAPIClient

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
    """Persist uploaded files and return their upload ids on raggify.

    Args:
        client (RestAPIClient): REST API client.
        files (list[Any]): Uploaded file objects from Streamlit.

    Returns:
        list[str]: List of upload identifiers.

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

    upload_ids: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("raggify upload response item is invalid")
        upload_id = item.get("upload_id")
        if not isinstance(upload_id, str) or upload_id == "":
            raise RuntimeError("raggify upload_id is invalid")
        upload_ids.append(upload_id)

    if len(upload_ids) != len(payload):
        raise RuntimeError("raggify upload file count mismatch")

    return upload_ids
