from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from llama_index.core.schema import BaseNode, TransformComponent

from ...core.event import async_loop_runner
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType

    from ...embed.embed_manager import EmbedManager
    from ...llama_like.embeddings.multi_modal_base import AudioType, VideoType

__all__ = [
    "make_text_embed_transform",
    "make_image_embed_transform",
    "make_audio_embed_transform",
    "make_video_embed_transform",
]


class _BaseEmbedTransform(TransformComponent):
    """Base transform for embedding."""

    def __init__(
        self,
        batch_embed_fn: Callable[[list], Awaitable[list[list[float]]]],
        extract_fn: Callable[[BaseNode], object],
        name: Optional[str] = None,
    ):
        """Constructor.

        Args:
            batch_embed_fn (Callable[[list], Awaitable[list[list[float]]]]):
                Batch embedding function.
            extract_fn (Callable[[BaseNode], object]): Modality-specific extractor.
            name (str, optional): Name label for cache separation. Defaults to None.
        """
        self._batch_embed_fn = batch_embed_fn
        self._extract_fn = extract_fn
        self._name = name or self.__class__.__name__

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def to_dict(self, **kwargs) -> dict:
        """Return a dict for caching that includes name label.

        Args:
            **kwargs: Unused additional options.

        Returns:
            dict: Serializable payload for hashing.
        """
        return {"class_name": self.class_name(), "name": self._name}

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Synchronous interface.

        Args:
            nodes (list[BaseNode]): Nodes to embed.

        Returns:
            list[BaseNode]: Nodes after embedding.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (list[BaseNode]): Nodes to embed.

        Returns:
            list[BaseNode]: Nodes after embedding.
        """
        from ...core.metadata import MetaKeys as MK

        # Extract inputs (skip missing while keeping back-references to original nodes)
        inputs: list[object] = []
        backrefs: list[int] = []
        for i, node in enumerate(nodes):
            x = self._extract_fn(node)
            if x is None:
                continue

            inputs.append(x)
            backrefs.append(i)

        if not inputs:
            return nodes

        # Batch embedding
        vecs = await self._batch_embed_fn(inputs)
        if not vecs:
            return nodes

        if len(vecs) != len(inputs):
            # Safety: do not write when lengths differ (log at caller)
            return nodes

        # Write back to nodes
        for i, vec in zip(backrefs, vecs):
            nodes[i].embedding = vec

            if nodes[i].metadata.get(MK.TEMP_FILE_PATH):
                # Overwrite file_path with base_source for nodes with temp files
                # (either becomes empty or restores original path kept by
                # custom readers such as PDF)
                nodes[i].metadata[MK.FILE_PATH] = nodes[i].metadata[MK.BASE_SOURCE]

        return nodes


def _get_media_path(node: BaseNode) -> str:
    """Get media path for embedded non-text content.

    Args:
        node (BaseNode): Target node.

    Returns:
        str: Media path.
    """
    from ...core.metadata import MetaKeys as MK

    temp = node.metadata.get(MK.TEMP_FILE_PATH)
    if temp:
        # Temp file fetched
        return temp

    # Local file
    return node.metadata[MK.FILE_PATH]


def make_text_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for text embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from llama_index.core.schema import TextNode

    async def batch_text(texts: list[str]) -> list[Embedding]:
        return await embed.aembed_text(texts)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, TextNode) and node.text and node.text.strip():
            return node.text

        logger.warning("text is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_text, extractor, name="text_embed")


def make_image_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for image embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from llama_index.core.schema import ImageNode

    async def batch_image(paths: list[ImageType]) -> list[Embedding]:
        return await embed.aembed_image(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, ImageNode):
            return _get_media_path(node)

        logger.warning("image is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_image, extractor, name="image_embed")


def make_audio_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for audio embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from ...core.exts import Exts
    from ...core.utils import has_media

    async def batch_audio(paths: list[AudioType]) -> list[Embedding]:
        return await embed.aembed_audio(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        # Can't use isinstance because AudioNode is not known by llama_index
        if has_media(node=node, exts=Exts.AUDIO):
            return _get_media_path(node)

        logger.warning("audio is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_audio, extractor, name="audio_embed")


def make_video_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for video embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from ...core.exts import Exts
    from ...core.utils import has_media

    async def batch_video(paths: list[VideoType]) -> list[Embedding]:
        return await embed.aembed_video(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        # Can't use isinstance because VideoNode is not known by llama_index
        if has_media(node=node, exts=Exts.VIDEO):
            return _get_media_path(node)

        logger.warning("video is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_video, extractor, name="video_embed")
