from __future__ import annotations

import asyncio
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Coroutine

from .multi_modal_base import AudioEmbedding, AudioType

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding

__all__ = ["ClapEmbedding", "ClapModels"]


class ClapModels(StrEnum):
    EFFECT_SHORT = auto()
    EFFECT_VARLEN = auto()
    MUSIC = auto()
    SPEECH = auto()
    GENERAL = auto()


class _AudioEncoderModel(StrEnum):
    HTSAT_TINY = "HTSAT-tiny"
    HTSAT_BASE = "HTSAT-base"


class _TextEncoderModel(StrEnum):
    ROBERTA = auto()


class ClapEmbedding(AudioEmbedding):
    """Embedding class dedicated to LAION-AI CLAP.

    Implemented with reference to MultiModalEmbedding;
    uses BaseEmbedding -> AudioEmbedding
    because MultiModalEmbedding lacks audio support.
    """

    @classmethod
    def class_name(cls) -> str:
        """Class name.

        Returns:
            str: Class name.
        """
        return "ClapEmbedding"

    def __init__(
        self,
        model_name: str = ClapModels.EFFECT_VARLEN,
        device: str = "cuda",
        embed_batch_size: int = 8,
    ) -> None:
        """Constructor.

        Args:
            model_name (str, optional): Model name (custom enum here). Defaults to "general".
            device (str, optional): Embedding device. Defaults to "cuda".
            embed_batch_size (int, optional): Embed batch size. Defaults to 8.

        Raises:
            ImportError: If laion-clap is not installed.
        """
        try:
            import laion_clap  # type: ignore
        except ImportError:
            from ...core.const import PKG_NOT_FOUND_MSG

            raise ImportError(
                PKG_NOT_FOUND_MSG.format(
                    pkg="laion-clap",
                    extra="localmodel",
                    feature="ClapEmbedding",
                )
            )

        super().__init__(
            model_name=f"clap/{model_name}",
            embed_batch_size=embed_batch_size,
        )

        enable_fusion = False
        tmodel = _TextEncoderModel.ROBERTA
        match model_name:
            case ClapModels.EFFECT_SHORT:
                amodel = _AudioEncoderModel.HTSAT_TINY
                model_id = 1
            case ClapModels.EFFECT_VARLEN:
                enable_fusion = True
                amodel = _AudioEncoderModel.HTSAT_TINY
                model_id = 3
            case ClapModels.MUSIC | ClapModels.SPEECH | ClapModels.GENERAL:
                amodel = _AudioEncoderModel.HTSAT_BASE
                raise NotImplementedError("loading local .pt is not implemented")
            case _:
                raise RuntimeError(f"unexpected model name: {model_name}")

        self._model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion, device=device, amodel=amodel, tmodel=tmodel
        )
        self._model.load_ckpt(model_id=model_id)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Embed a query string asynchronously.

        Args:
            query (str): Query string.

        Returns:
            Embedding: Embedding vector.
        """
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Embed a single text synchronously.

        Args:
            text (str): Text content.

        Returns:
            Embedding: Embedding vector.
        """
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts synchronously.

        Args:
            texts (list[str]): Texts.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        vecs = self._model.get_text_embedding(x=texts)

        return [vec.tolist() for vec in vecs]

    def _get_query_embedding(self, query: str) -> Embedding:
        """Embed a query string synchronously.

        Args:
            query (str): Query string.

        Returns:
            Embedding: Embedding vector.
        """
        return self._get_text_embedding(query)

    def _get_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """Synchronous wrapper for the CLAP audio embedding API.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        vecs = self._model.get_audio_embedding_from_filelist(x=audio_file_paths)

        return [vec.tolist() for vec in vecs]

    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """Async batch interface for audio embeddings
        (modeled after `aget_image_embedding_batch`).

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.
            show_progress (bool, optional): Show progress. Defaults to False.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        from llama_index.core.callbacks.schema import CBEventType, EventPayload

        cur_batch: list[AudioType] = []
        callback_payloads: list[tuple[str, list[AudioType]]] = []
        result_embeddings: list[Embedding] = []
        embeddings_coroutines: list[Coroutine] = []
        for idx, audio_file_path in enumerate(audio_file_paths):
            cur_batch.append(audio_file_path)
            if (
                idx == len(audio_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_audio_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio

                nested_embeddings = await tqdm_asyncio.gather(
                    *embeddings_coroutines,
                    total=len(embeddings_coroutines),
                    desc="Generating embeddings",
                )
            except ImportError:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, audio_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: audio_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings

    async def _aget_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """Async wrapper for the CLAP audio embedding API.

        At implementation time, only synchronous CLAP interfaces exist.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        return await asyncio.to_thread(self._get_audio_embeddings, audio_file_paths)
