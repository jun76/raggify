from __future__ import annotations

import asyncio
import base64
import json
import logging
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from llama_index.embeddings.bedrock import BedrockEmbedding

from raggify.core.exts import Exts

from .multi_modal_base import AudioType, VideoEmbedding, VideoType

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType


logger = logging.getLogger(__name__)


class Models(StrEnum):
    # 本家の Models
    TITAN_EMBEDDING = "amazon.titan-embed-text-v1"
    TITAN_EMBEDDING_V2_0 = "amazon.titan-embed-text-v2:0"
    TITAN_EMBEDDING_G1_TEXT_02 = "amazon.titan-embed-g1-text-02"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    COHERE_EMBED_V4 = "cohere.embed-v4:0"

    # 追加サポート
    NOVA_2_MULTIMODAL_V1 = "amazon.nova-2-multimodal-embeddings-v1:0"


class MultiModalBedrockEmbedding(VideoEmbedding, BedrockEmbedding):
    """BedrockEmbedding のマルチモーダル対応版"""

    def _is_nova_model(self) -> bool:
        """現在のモデルが Nova 系か判定する。

        Returns:
            bool: Nova 系モデルなら True
        """
        return "amazon.nova" in self.model_name.lower()

    @classmethod
    def class_name(cls) -> str:
        """クラス名

        Returns:
            str: クラス名
        """
        return "MultiModalBedrockEmbedding"

    def __init__(
        self,
        model_name: str = Models.NOVA_2_MULTIMODAL_V1,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """コンストラクタ

        Args:
            kwargs (Any): BedrockEmbedding 初期化用
        """
        super().__init__(
            model_name=model_name,
            profile_name=profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            **kwargs,
        )

    def _get_text_embedding(self, text: str) -> Embedding:
        """テキスト埋め込みの同期インタフェース。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        if not self._is_nova_model():
            return super()._get_text_embedding(text)

        trunc_mode = self.additional_kwargs.get("text_truncation_mode", "END")
        payload = {
            "truncationMode": trunc_mode,
            "value": text,
        }
        payload.update(self.additional_kwargs.get("text_payload_overrides", {}))
        request_body = self._build_single_embedding_body(
            media_field="text",
            media_payload=payload,
            params_override_key="text_params_overrides",
        )
        return self._invoke_single_embedding(request_body)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """テキスト埋め込みの非同期インタフェース。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        if not self._is_nova_model():
            return await super()._aget_text_embedding(text)

        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """テキスト埋め込みの同期バッチインタフェース。

        Args:
            texts (list[str]): テキストリスト

        Returns:
            list[Embedding]: 埋め込みベクトルリスト
        """
        if not self._is_nova_model():
            return super()._get_text_embeddings(texts)

        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """テキスト埋め込みの非同期バッチインタフェース。

        Args:
            texts (list[str]): テキストリスト

        Returns:
            list[Embedding]: 埋め込みベクトルリスト
        """
        if not self._is_nova_model():
            return await super()._aget_text_embeddings(texts)

        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """画像埋め込みの同期インタフェース。

        Args:
            img_file_path (ImageType): 画像ファイルパス

        Returns:
            Embedding: 埋め込みベクトル
        """
        encoded, fmt = self._read_media_payload(
            img_file_path,
            expected_exts=Exts.IMAGE,
            fallback_format_key="image_format",
        )
        payload = {
            "format": fmt,
            "source": {"bytes": encoded},
        }
        payload.update(self.additional_kwargs.get("image_payload_overrides", {}))
        request_body = self._build_single_embedding_body(
            media_field="image",
            media_payload=payload,
            params_override_key="image_params_overrides",
        )

        return self._invoke_single_embedding(request_body)

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """画像埋め込みの非同期インタフェース。

        Args:
            img_file_path (ImageType): 画像ファイルパス

        Returns:
            Embedding: 埋め込みベクトル
        """
        return await asyncio.to_thread(self._get_image_embedding, img_file_path)

    def _get_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """音声埋め込みの同期インタフェース。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        vecs: list[Embedding] = []
        for audio in audio_file_paths:
            encoded, fmt = self._read_media_payload(
                audio,
                expected_exts=Exts.AUDIO,
                fallback_format_key="audio_format",
            )
            payload = {
                "format": fmt,
                "source": {"bytes": encoded},
            }
            payload.update(self.additional_kwargs.get("audio_payload_overrides", {}))
            request_body = self._build_single_embedding_body(
                media_field="audio",
                media_payload=payload,
                params_override_key="audio_params_overrides",
            )
            vecs.append(self._invoke_single_embedding(request_body))

        return vecs

    async def _aget_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """音声埋め込みの非同期インタフェース。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        return await asyncio.to_thread(self._get_audio_embeddings, audio_file_paths)

    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """音声埋め込みの非同期バッチインタフェース。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス
            show_progress (bool, optional): 進捗の表示。Defaults to False.

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        return await self._aget_media_embedding_batch(
            audio_file_paths,
            self._aget_audio_embeddings,
            show_progress=show_progress,
        )

    def _get_video_embeddings(
        self, video_file_paths: list[VideoType]
    ) -> list[Embedding]:
        """動画埋め込みの同期インタフェース。

        Args:
            video_file_paths (list[VideoType]): 動画ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        vecs: list[Embedding] = []
        video_overrides = self.additional_kwargs.get("video_payload_overrides", {})
        for video in video_file_paths:
            encoded, fmt = self._read_media_payload(
                video,
                expected_exts=Exts.VIDEO,
                fallback_format_key="video_format",
            )
            payload = {
                "format": fmt,
                "embeddingMode": self.additional_kwargs.get(
                    "video_embedding_mode", "AUDIO_VIDEO_COMBINED"
                ),
                "source": {"bytes": encoded},
            }
            segmentation = self.additional_kwargs.get("video_segmentation_config")
            if segmentation:
                payload["segmentationConfig"] = segmentation

            payload.update(video_overrides)

            request_body = self._build_single_embedding_body(
                media_field="video",
                media_payload=payload,
                params_override_key="video_params_overrides",
            )
            vecs.append(self._invoke_single_embedding(request_body))

        return vecs

    async def _aget_video_embeddings(
        self, video_file_paths: list[VideoType]
    ) -> list[Embedding]:
        """動画埋め込みの非同期インタフェース。

        Args:
            video_file_paths (list[VideoType]): 動画ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        return await asyncio.to_thread(self._get_video_embeddings, video_file_paths)

    async def aget_video_embedding_batch(
        self, video_file_paths: list[VideoType], show_progress: bool = False
    ) -> list[Embedding]:
        """動画埋め込みの非同期バッチインタフェース。

        Args:
            video_file_paths (list[VideoType]): 動画ファイルパス
            show_progress (bool, optional): 進捗の表示。Defaults to False.

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        return await self._aget_media_embedding_batch(
            video_file_paths,
            self._aget_video_embeddings,
            show_progress=show_progress,
        )

    async def _aget_media_embedding_batch(
        self,
        media_file_paths: list[Any],
        worker: Callable[[list[Any]], Awaitable[list[Embedding]]],
        show_progress: bool,
    ) -> list[Embedding]:
        """メディア埋め込みの汎用非同期バッチ処理。

        Args:
            media_file_paths (list[Any]): メディアファイルパス
            worker (Callable[[list[Any]], Awaitable[list[Embedding]]]): 埋め込み実行関数
            show_progress (bool): 進捗表示フラグ

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        from llama_index.core.callbacks.schema import CBEventType, EventPayload

        cur_batch: list[Any] = []
        callback_payloads: list[tuple[str, list[Any]]] = []
        coroutines: list[Awaitable[list[Embedding]]] = []
        for idx, media in enumerate(media_file_paths):
            cur_batch.append(media)
            if (
                idx == len(media_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                coroutines.append(worker(cur_batch))
                cur_batch = []

        if not coroutines:
            return []

        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio

                nested_embeddings = await tqdm_asyncio.gather(
                    *coroutines,
                    total=len(coroutines),
                    desc="Generating embeddings",
                )
            except ImportError:
                nested_embeddings = await asyncio.gather(*coroutines)
        else:
            nested_embeddings = await asyncio.gather(*coroutines)

        flat_embeddings = [emb for chunk in nested_embeddings for emb in chunk]

        for (event_id, payload_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: payload_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return flat_embeddings

    def _read_media_payload(
        self,
        media: AudioType | VideoType | ImageType,
        *,
        expected_exts: set[str],
        fallback_format_key: str,
    ) -> tuple[str, str]:
        """メディアファイルから base64 文字列とフォーマットを取得する。

        Args:
            media (AudioType | VideoType): メディアファイル
            expected_exts (set[str]): 許可された拡張子セット
            fallback_format_key (str): 追加設定で参照するフォーマットのキー

        Returns:
            tuple[str, str]: (base64 文字列, フォーマット文字列)
        """
        file_name: Optional[str] = None
        if isinstance(media, BytesIO):
            media.seek(0)
            data = media.read()
            file_name = getattr(media, "name", None)
        else:
            path = Path(media).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"media file not found: {path}")
            data = path.read_bytes()
            file_name = path.name

        media_format = self._resolve_media_format(
            file_name=file_name,
            expected_exts=expected_exts,
            fallback_format_key=fallback_format_key,
        )
        encoded = base64.b64encode(data).decode("utf-8")

        return encoded, media_format

    def _resolve_media_format(
        self,
        *,
        file_name: Optional[str],
        expected_exts: set[str],
        fallback_format_key: str,
    ) -> str:
        """メディアフォーマットを決定する。

        Args:
            file_name (Optional[str]): ファイル名
            expected_exts (set[str]): 許可された拡張子セット
            fallback_format_key (str): 追加設定で参照するフォーマットのキー

        Returns:
            str: フォーマット名
        """
        if file_name:
            ext = Path(file_name).suffix.lower()
            if ext in expected_exts:
                return ext.lstrip(".")

        override = self.additional_kwargs.get(fallback_format_key)
        if override:
            return str(override).lower()

        raise ValueError(f"unsupported media format: {file_name or 'unknown'}")

    def _build_single_embedding_body(
        self,
        *,
        media_field: str,
        media_payload: dict[str, Any],
        params_override_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Nova への単一埋め込みリクエストボディを構築する。

        Args:
            media_field (str): メディアフィールド名
            media_payload (dict[str, Any]): メディアペイロード
            params_override_key (Optional[str]): 追加設定キー

        Returns:
            dict[str, Any]: リクエストボディ
        """
        default_task_type = "SINGLE_EMBEDDING"
        task_type = self.additional_kwargs.get(
            f"{media_field}_task_type", default_task_type
        )
        params_key = self.additional_kwargs.get(
            f"{media_field}_params_container",
            (
                "singleEmbeddingParams"
                if task_type == default_task_type
                else "segmentedEmbeddingParams"
            ),
        )

        params: dict[str, Any] = {
            "embeddingPurpose": self.additional_kwargs.get(
                "embedding_purpose", "GENERIC_INDEX"
            ),
            media_field: media_payload,
        }
        dimension = self.additional_kwargs.get("embedding_dimension", 3072)
        if dimension is not None:
            params["embeddingDimension"] = dimension

        if params_override_key:
            overrides = self.additional_kwargs.get(params_override_key)
            if overrides:
                params.update(overrides)

        return {
            "taskType": task_type,
            params_key: params,
        }

    def _invoke_single_embedding(self, body: dict[str, Any]) -> Embedding:
        """Bedrock にリクエストを送り埋め込みを取得する。

        Args:
            body (dict[str, Any]): リクエストボディ

        Returns:
            Embedding: 埋め込みベクトル
        """

        if self._client is None:
            self.set_credentials()
            if self._client is None:
                raise RuntimeError("Bedrock client is not initialized")

        response = self._client.invoke_model(
            body=json.dumps(body),
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )

        raw_body = response.get("body")
        if hasattr(raw_body, "read"):
            content = raw_body.read()
        else:
            content = raw_body or b"{}"

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        parsed = json.loads(content)
        embeddings = parsed.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Bedrock response does not include embeddings")

        first = embeddings[0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
        elif isinstance(first, list):
            return first
        else:
            raise RuntimeError(f"Unexpected embedding format: {type(first)}")
