from __future__ import annotations

import json
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

    def _get_json(
        self, endpoint: str, request_kwargs: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Send a GET request and return the JSON response.

        Args:
            endpoint (str): Relative path from base URL.
            request_kwargs (Optional[dict[str, Any]]): Extra query parameters.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        params = request_kwargs if request_kwargs else None
        return self._make_request(endpoint=endpoint, func=requests.get, params=params)

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        request_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send a POST request and return the JSON response.

        Args:
            endpoint (str): Relative path from base URL.
            payload (dict[str, Any]): POST body.
            request_kwargs (Optional[dict[str, Any]]): Extra JSON payload fields.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        if request_kwargs:
            merged_payload = {**request_kwargs, **payload}
        else:
            merged_payload = payload
        return self._make_request(
            endpoint=endpoint, func=requests.post, json=merged_payload
        )

    def _post_form_data_json(
        self,
        endpoint: str,
        files: list[tuple[str, tuple[str, bytes, str]]],
        request_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send a multipart/form-data POST and return the JSON response.

        Args:
            endpoint (str): Relative path from base URL.
            files (list[tuple[str, tuple[str, bytes, str]]]): File tuples for multipart upload.
            request_kwargs (Optional[dict[str, Any]]): Extra query parameters.

        Raises:
            RuntimeError: If the request fails or JSON parsing fails.

        Returns:
            dict[str, Any]: JSON response.
        """
        params = request_kwargs if request_kwargs else None
        return self._make_request(
            endpoint=endpoint, func=requests.post, files=files, params=params
        )

    def _parse_request_kwargs(self, payload: Optional[str]) -> dict[str, Any]:
        """Parse JSON payload into request kwargs.

        Args:
            payload (Optional[str]): JSON payload string.

        Raises:
            ValueError: If parsing fails or the payload is not a JSON object.

        Returns:
            dict[str, Any]: Parsed request kwargs.
        """
        if not payload:
            return {}

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("request_kwargs must be a JSON object") from exc

        if not isinstance(data, dict):
            raise ValueError("request_kwargs must be a JSON object")

        return data

    def status(self, request_kwargs: Optional[str] = None) -> dict[str, str]:
        """Get server status.

        Args:
            request_kwargs (Optional[str]): Extra query parameters as JSON string.

        Returns:
            dict[str, str]: Response data.
        """
        parsed_kwargs = self._parse_request_kwargs(request_kwargs)
        return self._get_json("/status", request_kwargs=parsed_kwargs)

    def reload(self, request_kwargs: Optional[str] = None) -> dict[str, str]:
        """Reload the server configuration file.

        Args:
            request_kwargs (Optional[str]): Extra query parameters as JSON string.

        Returns:
            dict[str, str]: Response data.
        """
        parsed_kwargs = self._parse_request_kwargs(request_kwargs)
        return self._get_json("/reload", request_kwargs=parsed_kwargs)

    def upload(
        self,
        files: list[tuple[str, bytes, Optional[str]]],
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the file upload API.

        Args:
            files (list[tuple[str, bytes, Optional[str]]]): Files to upload.
            request_kwargs (Optional[str]): Extra query parameters as JSON string.

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

        return self._post_form_data_json(
            "/upload",
            files_payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def job(
        self,
        job_id: str = "",
        rm: bool = False,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, str]:
        """Call the background job API.

        Args:
            job_id (str, optional): Job ID. Defaults to "" for all jobs.
            rm (bool, optional): Remove flag. Defaults to False.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, str]: Response data.

        Note:
            If job_id is empty, returns status for all jobs.
            If job_id is empty and rm is True, removes all completed jobs.
        """
        return self._post_json(
            "/job",
            {"job_id": job_id, "rm": rm},
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def ingest_path(
        self,
        path: str,
        force: bool = False,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the ingest-from-path API.

        Args:
            path (str): Target path.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        return self._post_json(
            "/ingest/path",
            {"path": path, "force": force},
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def ingest_path_list(
        self,
        path: str,
        force: bool = False,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the ingest-from-path-list API.

        Args:
            path (str): Path to the path list file.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        return self._post_json(
            "/ingest/path_list",
            {"path": path, "force": force},
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def ingest_url(
        self,
        url: str,
        force: bool = False,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the ingest-from-URL API.

        Args:
            url (str): Target URL.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        return self._post_json(
            "/ingest/url",
            {"url": url, "force": force},
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def ingest_url_list(
        self,
        path: str,
        force: bool = False,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the ingest-from-URL-list API.

        Args:
            path (str): Path to the URL list file.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        return self._post_json(
            "/ingest/url_list",
            {"path": path, "force": force},
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_text_text(
        self,
        query: str,
        topk: Optional[int] = None,
        mode: Optional[Literal["vector_only", "bm25_only", "fusion"]] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the text->text search API.

        Args:
            query (str): Query string.
            topk (Optional[int]): Max count.
            mode (Optional[Literal["vector_only", "bm25_only", "fusion"]], optional):
                Retrieval mode. Defaults to None.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        if mode is not None:
            payload["mode"] = mode

        return self._post_json(
            "/query/text_text",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_text_image(
        self,
        query: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the text->image search API.

        Args:
            query (str): Query string.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/text_image",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_image_image(
        self,
        path: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the image->image search API.

        Args:
            path (str): Path to the query image file.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/image_image",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_audio_audio(
        self,
        path: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the audio->audio search API.

        Args:
            path (str): Path to the query audio file.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/audio_audio",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_text_audio(
        self,
        query: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the text->audio search API.

        Args:
            query (str): Query string.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/text_audio",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_text_video(
        self,
        query: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the text->video search API.

        Args:
            query (str): Query string.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/text_video",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_image_video(
        self,
        path: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the image->video search API.

        Args:
            path (str): Path to the query image file.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/image_video",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_audio_video(
        self,
        path: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the audio->video search API.

        Args:
            path (str): Path to the query audio file.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/audio_video",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )

    def query_video_video(
        self,
        path: str,
        topk: Optional[int] = None,
        request_kwargs: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call the video->video search API.

        Args:
            path (str): Path to the query video file.
            topk (Optional[int]): Max count.
            request_kwargs (Optional[str]): Extra JSON payload fields as JSON string.

        Returns:
            dict[str, Any]: Response data.
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json(
            "/query/video_video",
            payload,
            request_kwargs=self._parse_request_kwargs(request_kwargs),
        )
