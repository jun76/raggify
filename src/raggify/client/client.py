from __future__ import annotations

from typing import Any, Callable, Optional

import requests

from ..config.retrieve_config import RetrieveMode

__all__ = ["RestAPIClient"]


class RestAPIClient:
    """raggify サーバの REST API を呼び出すクライアント"""

    def __init__(self, base_url: str) -> None:
        """コンストラクタ

        Args:
            base_url (str): raggify サーバへのベース URL
        """
        self._base_url = base_url.rstrip("/")

    def _make_request(
        self, endpoint: str, func: Callable, timeout: int = 120, **kwargs
    ) -> dict[str, Any]:
        """共通のリクエスト処理とエラーハンドリング。

        Args:
            endpoint (str): エンドポイント
            func (Callable): requests.post/get
            timeout (int, optional): タイムアウト（秒）Defaults to 120.

        Raises:
            RuntimeError: リクエスト失敗または JSON 解析失敗時

        Returns:
            dict[str, Any]: JSON 応答
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
            raise RuntimeError(
                f"failed to call raggify server endpoint: {detail}"
            ) from e

        try:
            return res.json()
        except ValueError as e:
            raise RuntimeError(f"raggify server response is not json: {e}") from e

    def _get_json(self, endpoint: str) -> dict[str, Any]:
        """GET リクエストを送信し、JSON 応答を辞書で返す。

        Args:
            endpoint (str): ベース URL からの相対パス

        Raises:
            RuntimeError: リクエスト失敗または JSON 解析失敗時

        Returns:
            dict[str, Any]: JSON 応答
        """
        return self._make_request(endpoint=endpoint, func=requests.get)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST リクエストを送信し、JSON 応答を辞書で返す。

        Args:
            endpoint (str): ベース URL からの相対パス
            payload (dict[str, Any]): POST ボディ

        Raises:
            RuntimeError: リクエスト失敗または JSON 解析失敗時

        Returns:
            dict[str, Any]: JSON 応答
        """
        return self._make_request(endpoint=endpoint, func=requests.post, json=payload)

    def _post_form_data_json(
        self, endpoint: str, files: list[tuple[str, tuple[str, bytes, str]]]
    ) -> dict[str, Any]:
        """multipart/form-data POST を送信し、JSON 応答を辞書で返す。

        Args:
            endpoint (str): ベース URL からの相対パス
            files (list[tuple[str, tuple[str, bytes, str]]]): multipart/form-data 用ファイル情報

        Raises:
            RuntimeError: リクエスト失敗または JSON 解析失敗時

        Returns:
            dict[str, Any]: JSON 応答
        """
        return self._make_request(endpoint=endpoint, func=requests.post, files=files)

    def health(self) -> dict[str, str]:
        """サーバの稼働状態を取得する。

        Returns:
            dict[str, str]: 応答データ
        """
        return self._get_json("/health")

    def reload(self) -> dict[str, str]:
        """サーバの設定ファイルをリロードする。

        Returns:
            dict[str, str]: 応答データ
        """
        return self._get_json("/reload")

    def upload(self, files: list[tuple[str, bytes, Optional[str]]]) -> dict[str, Any]:
        """ファイルアップロード API を呼び出す。

        Args:
            files (list[tuple[str, bytes, Optional[str]]]): アップロードするファイル情報

        Returns:
            dict[str, Any]: 応答データ

        Raises:
            ValueError: 入力値が不正な場合
            RuntimeError: リクエスト失敗または JSON 解析失敗時
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

        return self._post_form_data_json("/upload", files_payload)

    def job(self, job_id: str = "", rm: bool = False) -> dict[str, str]:
        """バックグラウンドジョブ操作 API を呼び出す。

        Args:
            job_id (str, optional): ジョブ ID。Defaults to "".
            rm (bool, optional): 削除フラグ。Defaults to False.

        Returns:
            dict[str, str]: 応答データ

        Note:
            job_id が空の時、全ジョブの実行状態を取得する。
            job_id が空かつ rm オプション指定の時、完了済みジョブを全削除する。
        """
        return self._post_json("/job", {"job_id": job_id, "rm": rm})

    def ingest_path(self, path: str) -> dict[str, Any]:
        """パス指定の取り込み API を呼び出す。

        Args:
            path (str): 取り込み対象パス

        Returns:
            dict[str, Any]: 応答データ
        """
        return self._post_json("/ingest/path", {"path": path})

    def ingest_path_list(self, path: str) -> dict[str, Any]:
        """パスリスト指定の取り込み API を呼び出す。

        Args:
            path (str): パスリストのファイルパス

        Returns:
            dict[str, Any]: 応答データ
        """
        return self._post_json("/ingest/path_list", {"path": path})

    def ingest_url(self, url: str) -> dict[str, Any]:
        """URL 指定の取り込み API を呼び出す。

        Args:
            url (str): 取り込み対象 URL

        Returns:
            dict[str, Any]: 応答データ
        """
        return self._post_json("/ingest/url", {"url": url})

    def ingest_url_list(self, path: str) -> dict[str, Any]:
        """URL リスト指定の取り込み API を呼び出す。

        Args:
            path (str): URL リストファイルのパス

        Returns:
            dict[str, Any]: 応答データ
        """
        return self._post_json("/ingest/url_list", {"path": path})

    def query_text_text(
        self,
        query: str,
        topk: Optional[int] = None,
        mode: Optional[RetrieveMode] = None,
    ) -> dict[str, Any]:
        """クエリ文字列によるテキストドキュメント検索 API を呼び出す。

        Args:
            query (str): クエリ文字列
            topk (Optional[int]): 上限件数
            mode (Optional[RetrieveMode], optional): 検索モード。Defaults to None.

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        if mode is not None:
            payload["mode"] = mode

        return self._post_json("/query/text_text", payload)

    def query_text_image(
        self, query: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ文字列による画像ドキュメント検索 API を呼び出す。

        Args:
            query (str): クエリ文字列
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_image", payload)

    def query_image_image(
        self, path: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ画像による画像ドキュメント検索 API を呼び出す。

        Args:
            path (str): クエリ画像パス
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/image_image", payload)

    def query_text_audio(
        self, query: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ文字列による音声ドキュメント検索 API を呼び出す。

        Args:
            query (str): クエリ文字列
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_audio", payload)

    def query_audio_audio(
        self, path: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ音声による音声ドキュメント検索 API を呼び出す。

        Args:
            path (str): クエリ音声パス
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/audio_audio", payload)

    def query_text_video(
        self, query: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ文字列による動画ドキュメント検索 API を呼び出す。

        Args:
            query (str): クエリ文字列
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_video", payload)

    def query_image_video(
        self, path: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ画像による動画ドキュメント検索 API を呼び出す。

        Args:
            path (str): クエリ画像パス
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/image_video", payload)

    def query_audio_video(
        self, path: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ音声による動画ドキュメント検索 API を呼び出す。

        Args:
            path (str): クエリ音声パス
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/audio_video", payload)

    def query_video_video(
        self, path: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """クエリ動画による動画ドキュメント検索 API を呼び出す。

        Args:
            path (str): クエリ動画パス
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/video_video", payload)
