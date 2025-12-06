from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from llama_index.core.schema import BaseNode, ImageNode, TextNode, TransformComponent

from ...core.event import async_loop_runner
from ...llama_like.core.schema import AudioNode, VideoNode
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType

    from ...embed.embed_manager import EmbedManager
    from ...llama_like.embeddings.multi_modal_base import AudioType, VideoType

__all__ = ["EmbedTransform"]


class EmbedTransform(TransformComponent):
    """Transform to embed various modalities."""

    def __init__(self, embed: EmbedManager) -> None:
        """Constructor.

        Args:
            embed (EmbedManager): Embedding manager.
        """
        self._embed = embed

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Synchronous interface.

        Args:
            nodes (list[BaseNode]): Nodes to summarize.

        Returns:
            list[BaseNode]: Nodes after summarization.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (list[BaseNode]): Nodes to split.

        Returns:
            list[BaseNode]: Nodes after splitting.
        """
        candidate = nodes[0]
        if isinstance(candidate, ImageNode):
            split_nodes = await self._aembed_image(nodes)
        elif isinstance(candidate, AudioNode):
            split_nodes = await self._aembed_audio(nodes)
        elif isinstance(candidate, VideoNode):
            split_nodes = await self._aembed_video(nodes)
        elif isinstance(candidate, TextNode):
            split_nodes = await self._aembed_text(nodes)
        else:
            raise ValueError(f"unsupported node type: {type(candidate)}")

        return split_nodes

    def _get_media_path(self, node: BaseNode) -> str:
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

    async def _aembed_text(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Embed a text node.

        Args:
            nodes (list[BaseNode]): Target text nodes.

        Returns:
            list[BaseNode]: Embedded text nodes.
        """

        async def batch_text(texts: list[str]) -> list[Embedding]:
            return await self._embed.aembed_text(texts)

        def extractor(node: BaseNode) -> Optional[str]:
            if isinstance(node, TextNode) and node.text and node.text.strip():
                return node.text

            logger.warning("text is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_text, extractor)

    async def _aembed_image(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Embed an image node.

        Args:
            nodes (list[BaseNode]): Target image nodes.

        Returns:
            list[BaseNode]: Embedded image nodes.
        """
        from ...core.exts import Exts
        from ...core.utils import has_media

        async def batch_image(paths: list[ImageType]) -> list[Embedding]:
            return await self._embed.aembed_image(paths)

        def extractor(node: BaseNode) -> Optional[str]:
            if has_media(node=node, exts=Exts.IMAGE):
                return self._get_media_path(node)

            logger.warning("image is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_image, extractor)

    async def _aembed_audio(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Embed an audio node.

        Args:
            nodes (list[BaseNode]): Target audio nodes.

        Returns:
            list[BaseNode]: Embedded audio nodes.
        """
        from ...core.exts import Exts
        from ...core.utils import has_media

        async def batch_audio(paths: list[AudioType]) -> list[Embedding]:
            return await self._embed.aembed_audio(paths)

        def extractor(node: BaseNode) -> Optional[str]:
            if has_media(node=node, exts=Exts.AUDIO):
                return self._get_media_path(node)

            logger.warning("audio is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_audio, extractor)

    async def _aembed_video(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Embed a video node.

        Args:
            nodes (list[BaseNode]): Target video nodes.

        Returns:
            list[BaseNode]: Embedded video nodes.
        """
        from ...core.exts import Exts
        from ...core.utils import has_media

        async def batch_video(paths: list[VideoType]) -> list[Embedding]:
            return await self._embed.aembed_video(paths)

        def extractor(node: BaseNode) -> Optional[str]:
            if has_media(node=node, exts=Exts.VIDEO):
                return self._get_media_path(node)

            logger.warning("video is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_video, extractor)

    async def _aexec_transform(
        self,
        nodes: list[BaseNode],
        batch_embed_fn: Callable[[list], Awaitable[list[list[float]]]],
        extract_fn: Callable[[BaseNode], object],
    ) -> list[BaseNode]:
        """Embed nodes using the given batch embedding function and extractor.

        Args:
            nodes (list[BaseNode]): Nodes to embed.
            batch_embed_fn: Function to perform batch embedding.
            extract_fn: Function to extract embedding input from a node.

        Returns:
            list[BaseNode]: Nodes after embedding.
        """
        from ...core.metadata import MetaKeys as MK

        # Extract inputs (skip missing while keeping back-references to original nodes)
        inputs: list[object] = []
        backrefs: list[int] = []
        for i, node in enumerate(nodes):
            x = extract_fn(node)
            if x is None:
                continue

            inputs.append(x)
            backrefs.append(i)

        if not inputs:
            return nodes

        # Batch embedding
        vecs = await batch_embed_fn(inputs)
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
