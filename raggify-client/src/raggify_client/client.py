from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import requests

__all__ = ["RestAPIClient"]


class RestAPIClient:
    """Client for calling the server REST API."""

    def __init__(
        self, host: Optional[str] = "localhost", port: Optional[int] = 8000
    ) -> None:
        """Constructor.

        Args:
            host (Optional[str]): Hostname of the server. Defaults to "localhost".
            port (Optional[int]): Port number of the server. Defaults to 8000.
        """
        self._base_url = f"http://{host}:{port}/v1"

    def _make_request(
        self, endpoint: str, func: Callable, timeout: int = 120, **kwargs
    ) -> dict[str, Any]:
        """Common request handling and error processing.

        Args:
            endpoint (str): Endpoint path.
            func (Callable): `requests.post` or `requests.get`.
            timeout (int, optional): Timeout in seconds. Defaults to 120.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        url = f"{self._base_url}{endpoint}"
        try:
            res = func(url, timeout=timeout, **kwargs)
            res.raise_for_status()
        except requests.RequestException as e:
            if e.response is not None:
                detail = e.response.text
            else:
                detail = str(e)
            raise RuntimeError(f"failed to call server endpoint: {detail}") from e

        try:
            return res.json()
        except ValueError as e:
            raise RuntimeError(f"server response is not json: {e}") from e

    def _get_json(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Send a GET request and return the JSON response.

        Args:
            endpoint (str): Relative path from base URL.
            **kwargs (Any): Extra query parameters.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        params = kwargs if kwargs else None
        return self._make_request(endpoint=endpoint, func=requests.get, params=params)

    def _post_json(
        self, endpoint: str, payload: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Send a POST request and return the JSON response.

        Args:
            endpoint (str): Relative path from base URL.
            payload (dict[str, Any]): POST body.
            **kwargs (Any): Extra JSON payload fields.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        merged_payload = {**kwargs, **payload} if kwargs else payload
        return self._make_request(
            endpoint=endpoint, func=requests.post, json=merged_payload
        )

    def _post_form_data_json(
        self,
        endpoint: str,
        files: list[tuple[str, tuple[str, bytes, str]]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a multipart/form-data POST and return the JSON response.

        Args:
            endpoint (str): Relative path from base URL.
            files (list[tuple[str, tuple[str, bytes, str]]]): File tuples for multipart upload.
            **kwargs (Any): Extra query parameters.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        params = kwargs if kwargs else None
        return self._make_request(
            endpoint=endpoint, func=requests.post, files=files, params=params
        )

    def status(self, **kwargs: Any) -> dict[str, str]:
        """Get server status.

        Args:
            **kwargs (Any): Extra query parameters.

        Returns:
            dict[str, str]: Response data.
        """
        return self._get_json("/status", **kwargs)

    def reload(self, **kwargs: Any) -> dict[str, str]:
        """Reload the server configuration file.

        Args:
            **kwargs (Any): Extra query parameters.

        Returns:
            dict[str, str]: Response data.
        """
        return self._get_json("/reload", **kwargs)

    def upload(
        self, files: list[tuple[str, bytes, Optional[str]]], **kwargs: Any
    ) -> dict[str, Any]:
        """Call the file upload API.

        Args:
            files (list[tuple[str, bytes, Optional[str]]]): Files to upload.
            **kwargs (Any): Extra query parameters.

        Returns:
            dict[str, Any]: Response data.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If the request fails or JSON parsing fails.
        """
        if not files:
            raise ValueError("files must not be empty")

        files_payload: list[tuple[str, tuple[str, bytes, str]]] = []
        for name, data, content_type in files:
            if not isinstance(name, str) or name == "":
                raise ValueError("file name must be non-empty string")

            if not isinstance(data, bytes):
                raise ValueError("file data must be bytes")

            mime = content_type or "application/octet-stream"
            files_payload.append(("files", (name, data, mime)))

        return self._post_form_data_json("/upload", files_payload, **kwargs)

    def job(self, job_id: str = "", rm: bool = False, **kwargs: Any) -> dict[str, str]:
        """Call the background job API.

        Args:
            job_id (str, optional): Job ID. Defaults to "" for all jobs.
            rm (bool, optional): Remove flag. Defaults to False.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, str]: Response data.

        Note:
            If job_id is empty, returns status for all jobs.
            If job_id is empty and rm is True, removes all completed jobs.
        """
        return self._post_json("/job", {"job_id": job_id, "rm": rm}, **kwargs)

    def ingest_path(
        self, path: Optional[str] = None, force: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the ingest-from-path API.

        Args:
            path (Optional[str]): Target path. Defaults to None.
            force (bool, optional): Force flag. Defaults to False.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        payload["force"] = force

        return self._post_json("/ingest/path", payload=payload, **kwargs)

    def ingest_path_list(
        self, path: Optional[str] = None, force: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the ingest-from-path-list API.

        Args:
            path (Optional[str]): Path to the path list file. Defaults to None.
            force (bool, optional): Force flag. Defaults to False.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        payload["force"] = force

        return self._post_json("/ingest/path_list", payload, **kwargs)

    def ingest_url(
        self, url: str, force: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the ingest-from-URL API.

        Args:
            url (str): Target URL.
            force (bool, optional): Force flag. Defaults to False.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        return self._post_json("/ingest/url", {"url": url, "force": force}, **kwargs)

    def ingest_url_list(
        self, path: Optional[str] = None, force: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the ingest-from-URL-list API.

        Args:
            path (Optional[str]): Path to the URL list file. Defaults to None.
            force (bool, optional): Force flag. Defaults to False.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        payload["force"] = force

        return self._post_json("/ingest/url_list", payload, **kwargs)

    def query_text_text(
        self,
        query: str,
        topk: Optional[int] = None,
        mode: Optional[Literal["vector_only", "bm25_only", "fusion"]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the text->text search API.

        Args:
            query (str): Query string.
            topk (Optional[int], optional): Max count. Defaults to None.
            mode (Optional[Literal["vector_only", "bm25_only", "fusion"]], optional):
                Retrieval mode. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        if mode is not None:
            payload["mode"] = mode

        return self._post_json("/query/text_text", payload, **kwargs)

    def query_text_image(
        self, query: str, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the text->image search API.

        Args:
            query (str): Query string.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_image", payload, **kwargs)

    def query_image_image(
        self, path: Optional[str] = None, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the image->image search API.

        Args:
            path (Optional[str]): Path to the query image file. Defaults to None.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/image_image", payload, **kwargs)

    def query_audio_audio(
        self, path: Optional[str] = None, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the audio->audio search API.

        Args:
            path (Optional[str]): Path to the query audio file. Defaults to None.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/audio_audio", payload, **kwargs)

    def query_text_audio(
        self, query: str, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the text->audio search API.

        Args:
            query (str): Query string.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_audio", payload, **kwargs)

    def query_text_video(
        self, query: str, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the text->video search API.

        Args:
            query (str): Query string.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_video", payload, **kwargs)

    def query_image_video(
        self, path: Optional[str] = None, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the image->video search API.

        Args:
            path (Optional[str]): Path to the query image file. Defaults to None.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/image_video", payload, **kwargs)

    def query_audio_video(
        self, path: Optional[str] = None, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the audio->video search API.

        Args:
            path (Optional[str]): Path to the query audio file. Defaults to None.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/audio_video", payload, **kwargs)

    def query_video_video(
        self, path: Optional[str] = None, topk: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the video->video search API.

        Args:
            path (Optional[str]): Path to the query video file. Defaults to None.
            topk (Optional[int], optional): Max count. Defaults to None.
            **kwargs (Any): Extra JSON payload fields.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {}
        if path is not None:
            payload["path"] = path

        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/video_video", payload, **kwargs)
