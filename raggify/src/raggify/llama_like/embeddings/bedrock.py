from __future__ import annotations

import asyncio
import base64
import json
import logging
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

try:
    from llama_index.embeddings.bedrock import BedrockEmbedding  # type: ignore
except ImportError:
    from ...core.const import EXTRA_PKG_NOT_FOUND_MSG

    raise ImportError(
        EXTRA_PKG_NOT_FOUND_MSG.format(
            pkg="bedrock",
            extra="video",
            feature="BedrockEmbedding",
        )
    )

from raggify.core.exts import Exts

from .multi_modal_base import AudioType, VideoEmbedding, VideoType

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType


logger = logging.getLogger(__name__)

__all__ = ["BedrockEmbedding", "MultiModalBedrockEmbedding", "BedrockModels"]


class BedrockModels(StrEnum):
    # Official models
    TITAN_EMBEDDING = "amazon.titan-embed-text-v1"
    TITAN_EMBEDDING_V2_0 = "amazon.titan-embed-text-v2:0"
    TITAN_EMBEDDING_G1_TEXT_02 = "amazon.titan-embed-g1-text-02"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    COHERE_EMBED_V4 = "cohere.embed-v4:0"

    # Additional support
    NOVA_2_MULTIMODAL_V1 = "amazon.nova-2-multimodal-embeddings-v1:0"


class MultiModalBedrockEmbedding(VideoEmbedding, BedrockEmbedding):
    """Multimodal-capable variant of BedrockEmbedding."""

    def _is_nova_model(self) -> bool:
        """Return True if the current model is from the Nova series.

        Returns:
            bool: True for Nova models.
        """
        return "amazon.nova" in self.model_name.lower()

    @classmethod
    def class_name(cls) -> str:
        """Class name string.

        Returns:
            str: Class name.
        """
        return "MultiModalBedrockEmbedding"

    def __init__(
        self,
        model_name: str = BedrockModels.NOVA_2_MULTIMODAL_V1,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            model_name (str, optional): Bedrock embedding model name. Defaults to Models.NOVA_2_MULTIMODAL_V1.
            profile_name (Optional[str], optional): AWS profile name. Defaults to None.
            aws_access_key_id (Optional[str], optional): AWS access key. Defaults to None.
            aws_secret_access_key (Optional[str], optional): AWS secret key. Defaults to None.
            aws_session_token (Optional[str], optional): AWS session token. Defaults to None.
            region_name (Optional[str], optional): AWS region name. Defaults to None.
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
        """Synchronous interface for text embeddings.

        Args:
            text (str): Text to embed.

        Returns:
            Embedding: Embedding vector.
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
        """Asynchronous interface for text embeddings.

        Args:
            text (str): Text to embed.

        Returns:
            Embedding: Embedding vector.
        """
        if not self._is_nova_model():
            return await super()._aget_text_embedding(text)

        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_query_embedding(self, query: str) -> Embedding:
        """Synchronous interface for query embeddings.

        Args:
            query (str): Query string.

        Returns:
            Embedding: Embedding vector.
        """
        if not self._is_nova_model():
            return super()._get_query_embedding(query)

        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Asynchronous interface for query embeddings.

        Args:
            query (str): Query string.

        Returns:
            Embedding: Embedding vector.
        """
        if not self._is_nova_model():
            return await super()._aget_query_embedding(query)

        return await self._aget_text_embedding(query)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Synchronous batch interface for text embeddings.

        Args:
            texts (list[str]): Text list.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        if not self._is_nova_model():
            return super()._get_text_embeddings(texts)

        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Asynchronous batch interface for text embeddings.

        Args:
            texts (list[str]): Text list.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        if not self._is_nova_model():
            return await super()._aget_text_embeddings(texts)

        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Synchronous interface for image embeddings.

        Args:
            img_file_path (ImageType): Image file path.

        Returns:
            Embedding: Embedding vector.
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
        """Asynchronous interface for image embeddings.

        Args:
            img_file_path (ImageType): Image file path.

        Returns:
            Embedding: Embedding vector.
        """
        return await asyncio.to_thread(self._get_image_embedding, img_file_path)

    def _embed_media_files(
        self,
        file_paths: list[Any],
        *,
        expected_exts: set[str],
        fallback_format_key: str,
        media_field: str,
        payload_overrides_key: str,
        params_override_key: str,
        payload_builder: Callable[[str, str], dict[str, Any]],
    ) -> list[Embedding]:
        """Embed multiple media files that share a similar payload structure.

        Args:
            file_paths (list[Any]): Media file paths or file-like objects.
            expected_exts (set[str]): Allowed extensions.
            fallback_format_key (str): Additional kwargs fallback key for format.
            media_field (str): Media field identifier (audio/video).
            payload_overrides_key (str): Additional kwargs key for payload overrides.
            params_override_key (str): Additional kwargs key for body overrides.
            payload_builder (Callable[[str, str], dict[str, Any]]):
                Function to build media payload.

        Returns:
            list[Embedding]: Embedded vectors.
        """
        vecs: list[Embedding] = []
        overrides = self.additional_kwargs.get(payload_overrides_key, {})
        for media in file_paths:
            encoded, fmt = self._read_media_payload(
                media,
                expected_exts=expected_exts,
                fallback_format_key=fallback_format_key,
            )
            payload = payload_builder(fmt, encoded)
            payload.update(overrides)
            request_body = self._build_single_embedding_body(
                media_field=media_field,
                media_payload=payload,
                params_override_key=params_override_key,
            )
            vecs.append(self._invoke_single_embedding(request_body))

        return vecs

    def _get_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """Synchronous interface for audio embeddings.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return self._embed_media_files(
            audio_file_paths,
            expected_exts=Exts.AUDIO,
            fallback_format_key="audio_format",
            media_field="audio",
            payload_overrides_key="audio_payload_overrides",
            params_override_key="audio_params_overrides",
            payload_builder=lambda fmt, encoded: {
                "format": fmt,
                "source": {"bytes": encoded},
            },
        )

    async def _aget_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """Asynchronous interface for audio embeddings.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return await asyncio.to_thread(self._get_audio_embeddings, audio_file_paths)

    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """Async batch interface for audio embeddings.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.
            show_progress (bool, optional): Show progress. Defaults to False.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return await self._aget_media_embedding_batch(
            audio_file_paths,
            self._aget_audio_embeddings,
            show_progress=show_progress,
        )

    def _get_video_embeddings(
        self, video_file_paths: list[VideoType]
    ) -> list[Embedding]:
        """Synchronous interface for video embeddings.

        Args:
            video_file_paths (list[VideoType]): Video file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return self._embed_media_files(
            video_file_paths,
            expected_exts=Exts.VIDEO,
            fallback_format_key="video_format",
            media_field="video",
            payload_overrides_key="video_payload_overrides",
            params_override_key="video_params_overrides",
            payload_builder=lambda fmt, encoded: {
                "format": fmt,
                "source": {"bytes": encoded},
                "embeddingMode": self.additional_kwargs.get(
                    "video_embedding_mode", "AUDIO_VIDEO_COMBINED"
                ),
            },
        )

    async def _aget_video_embeddings(
        self, video_file_paths: list[VideoType]
    ) -> list[Embedding]:
        """Asynchronous interface for video embeddings.

        Args:
            video_file_paths (list[VideoType]): Video file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return await asyncio.to_thread(self._get_video_embeddings, video_file_paths)

    async def aget_video_embedding_batch(
        self, video_file_paths: list[VideoType], show_progress: bool = False
    ) -> list[Embedding]:
        """Async batch interface for video embeddings.

        Args:
            video_file_paths (list[VideoType]): Video file paths.
            show_progress (bool, optional): Show progress. Defaults to False.

        Returns:
            list[Embedding]: Embedding vectors.
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
        """Generic async batch processing for media embeddings.

        Args:
            media_file_paths (list[Any]): Media file paths.
            worker (Callable[[list[Any]], Awaitable[list[Embedding]]]):
                Embedding executor.
            show_progress (bool): Whether to show progress.

        Returns:
            list[Embedding]: Embedding vectors.
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
        """Obtain a base64 string and media format from a media file.

        Args:
            media (AudioType | VideoType | ImageType): Media file.
            expected_exts (set[str]): Allowed extension set.
            fallback_format_key (str): Key to look up a format override.

        Returns:
            tuple[str, str]: Base64 string and format string.
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
        """Determine the media format.

        Args:
            file_name (Optional[str]): File name.
            expected_exts (set[str]): Allowed extension set.
            fallback_format_key (str): Key to look up a format override.

        Returns:
            str: Format name.
        """
        if file_name:
            ext = Path(file_name).suffix.lower()
            if ext in expected_exts:
                return self._normalize_media_format(ext.lstrip("."))

        override = self.additional_kwargs.get(fallback_format_key)
        if override:
            return str(override).lower()

        raise ValueError(f"unsupported media format: {file_name or 'unknown'}")

    def _normalize_media_format(self, fmt: str) -> str:
        """Normalize media format strings to the ones accepted by Bedrock.

        Args:
            fmt (str): Format string from file extension.

        Returns:
            str: Normalized format string.
        """
        normalization_map = {
            "jpg": "jpeg",
        }

        return normalization_map.get(fmt, fmt)

    def _build_single_embedding_body(
        self,
        *,
        media_field: str,
        media_payload: dict[str, Any],
        params_override_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a single-embedding request body for Nova.

        Args:
            media_field (str): Media field name.
            media_payload (dict[str, Any]): Media payload.
            params_override_key (Optional[str]): Additional override key.

        Returns:
            dict[str, Any]: Request body.
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
        dim = self.additional_kwargs.get("embedding_dimension")
        params["embeddingDimension"] = dim

        if params_override_key:
            overrides = self.additional_kwargs.get(params_override_key)
            if overrides:
                params.update(overrides)

        return {
            "taskType": task_type,
            params_key: params,
        }

    def _invoke_embeddings(self, body: dict[str, Any]) -> list[Embedding]:
        """Send a request to Bedrock and retrieve embedding list.

        Args:
            body (dict[str, Any]): Request body.

        Returns:
            list[Embedding]: Embedding vectors.
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

        results: list[Embedding] = []
        for emb in embeddings:
            if isinstance(emb, dict) and "embedding" in emb:
                results.append(emb["embedding"])
            elif isinstance(emb, list):
                results.append(emb)
            else:
                raise RuntimeError(f"Unexpected embedding format: {type(emb)}")

        return results

    def _invoke_single_embedding(self, body: dict[str, Any]) -> Embedding:
        """Send a single-embedding request to Bedrock and retrieve the first embedding.

        Args:
            body (dict[str, Any]): Request body.

        Returns:
            Embedding: Embedding vector.
        """
        return self._invoke_embeddings(body)[0]
