from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from llama_index.core.settings import Settings

from ..core.utils import sanitize_str
from ..llama_like.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
    from llama_index.core.schema import ImageType

    from ..llama_like.embeddings.multi_modal_base import AudioType, VideoType

__all__ = ["EmbedManager", "EmbedContainer"]


@dataclass(kw_only=True)
class EmbedContainer:
    """Container for embedding-related parameters per modality."""

    provider_name: str
    embed: BaseEmbedding
    dim: int
    alias: str
    space_key: str = ""


class EmbedManager:
    """Manager class for embeddings."""

    def __init__(
        self,
        conts: dict[Modality, EmbedContainer],
        embed_batch_size: int,
        batch_interval_sec: int,
    ) -> None:
        """Constructor.

        Args:
            conts (dict[Modality, EmbedContainer]):
                Mapping of modality to embedding container.
            embed_batch_size (int): Number of nodes processed per embed batch.
            batch_interval_sec (int): Interval between batches in seconds.
        """
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        self._conts = conts
        self._embed_batch_size = embed_batch_size
        self._batch_interval_sec = batch_interval_sec

        for modality, cont in conts.items():
            cont.space_key = self._generate_space_key(
                provider=cont.provider_name,
                model=cont.alias,
                modality=modality,
            )
            logger.debug(f"space_key: {cont.space_key} generated")

        Settings.embed_model = MockEmbedding(
            embed_dim=1
        )  # disable llama_index default embed model

    @property
    def name(self) -> str:
        """Provider names.

        Returns:
            str: Provider names.
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """Modalities supported by this embedding manager.

        Returns:
            set[Modality]: Modalities.
        """
        return set(self._conts.keys())

    @property
    def space_key_text(self) -> str:
        """Space key for text embeddings.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            str: Space key.
        """
        return self.get_container(Modality.TEXT).space_key

    @property
    def space_key_image(self) -> str:
        """Space key for image embeddings.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            str: Space key.
        """
        return self.get_container(Modality.IMAGE).space_key

    @property
    def space_key_audio(self) -> str:
        """Space key for audio embeddings.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            str: Space key.
        """
        return self.get_container(Modality.AUDIO).space_key

    @property
    def space_key_video(self) -> str:
        """Space key for video embeddings.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            str: Space key.
        """
        return self.get_container(Modality.VIDEO).space_key

    def get_container(self, modality: Modality) -> EmbedContainer:
        """Get the embedding container for a modality.

        Args:
            modality (Modality): Modality.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            EmbedContainer: Embedding container.
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"embed {modality} is not initialized")

        return cont

    async def aembed_batch(
        self,
        modality: Modality,
        inputs: list[Any],
        batcher: Callable[[list[Any]], Awaitable[list[Embedding]]],
    ) -> list[Embedding]:
        """Embed inputs in batches using provided callable.

        Args:
            modality (Modality): Target modality.
            inputs (list[Any]): Items to embed.
            batcher (Callable[[list[Any]], Awaitable[list[Embedding]]]):
                Async callable to embed each chunk.

        Raises:
            RuntimeError: If failed to embed.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        if not inputs:
            return []

        logger.debug(f"now batch embedding {len(inputs)} {modality}s...")

        dims: list[Embedding] = []
        for idx in range(0, len(inputs), self._embed_batch_size):
            batch = inputs[idx : idx + self._embed_batch_size]
            dims.extend(await batcher(batch))

            logger.debug(f"embed total {len(dims)} {modality}s so far...")
            time.sleep(self._batch_interval_sec)

        logger.debug(f"done. dim = {len(dims[0])}, embed {len(dims)} {modality}s")

        return dims

    async def aembed_text(self, texts: list[str]) -> list[Embedding]:
        """Get embedding vectors for text asynchronously.

        Args:
            texts (list[str]): Texts to embed.

        Raises:
            RuntimeError: If failed to embed.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        if Modality.TEXT not in self.modality:
            logger.warning("no text embedding is specified")
            return []

        embed = self.get_container(Modality.TEXT).embed

        async def _batch_call(batch: list[str]) -> list[Embedding]:
            return await embed.aget_text_embedding_batch(
                texts=batch, show_progress=True
            )

        return await self.aembed_batch(
            modality=Modality.TEXT, inputs=texts, batcher=_batch_call
        )

    async def aembed_image(self, paths: list[ImageType]) -> list[Embedding]:
        """Get embedding vectors for images asynchronously.

        Args:
            paths (list[ImageType]): Image paths or base64 payloads.

        Raises:
            RuntimeError: If not an image embedder or failed to embed.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding

        if Modality.IMAGE not in self.modality:
            logger.warning("no image embedding is specified")
            return []

        embed = self.get_container(Modality.IMAGE).embed
        if not isinstance(embed, MultiModalEmbedding):
            raise RuntimeError("multimodal embed model is required")

        async def _batch_call(batch: list[ImageType]) -> list[Embedding]:
            return await embed.aget_image_embedding_batch(
                img_file_paths=batch, show_progress=True
            )

        return await self.aembed_batch(
            modality=Modality.IMAGE, inputs=paths, batcher=_batch_call
        )

    async def aembed_audio(self, paths: list[AudioType]) -> list[Embedding]:
        """Get embedding vectors for audio asynchronously.

        Args:
            paths (list[AudioType]): Audio paths.

        Raises:
            RuntimeError: If not an audio embedder or failed to embed.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        from ..llama_like.embeddings.multi_modal_base import AudioEmbedding

        if Modality.AUDIO not in self.modality:
            logger.warning("no audio embedding is specified")
            return []

        embed = self.get_container(Modality.AUDIO).embed
        if not isinstance(embed, AudioEmbedding):
            raise RuntimeError("audio embed model is required")

        async def _batch_call(batch: list[AudioType]) -> list[Embedding]:
            return await embed.aget_audio_embedding_batch(
                audio_file_paths=batch, show_progress=True
            )

        return await self.aembed_batch(
            modality=Modality.AUDIO, inputs=paths, batcher=_batch_call
        )

    async def aembed_video(self, paths: list[VideoType]) -> list[Embedding]:
        """Get embedding vectors for video asynchronously.

        Args:
            paths (list[VideoType]): Video paths.

        Raises:
            RuntimeError: If not a video embedder or failed to embed.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        from ..llama_like.embeddings.multi_modal_base import VideoEmbedding

        if Modality.VIDEO not in self.modality:
            logger.warning("no video embedding is specified")
            return []

        embed = self.get_container(Modality.VIDEO).embed
        if not isinstance(embed, VideoEmbedding):
            raise RuntimeError("video embed model is required")

        async def _batch_call(batch: list[VideoType]) -> list[Embedding]:
            return await embed.aget_video_embedding_batch(
                video_file_paths=batch, show_progress=True
            )

        return await self.aembed_batch(
            modality=Modality.VIDEO, inputs=paths, batcher=_batch_call
        )

    def _generate_space_key(self, provider: str, model: str, modality: Modality) -> str:
        """Generate a space key string.

        Args:
            provider (str): Provider name.
            model (str): Model name.
            modality (Modality): Modality.

        Raises:
            ValueError: If the space key is too long.

        Returns:
            str: Space key string.
        """
        # Shorten labels
        mod = {
            Modality.TEXT: "te",
            Modality.IMAGE: "im",
            Modality.AUDIO: "au",
            Modality.VIDEO: "vi",
        }
        if mod.get(modality) is None:
            raise ValueError(f"unexpected modality: {modality}")

        return sanitize_str(f"{provider}_{model}_{mod[modality]}")
