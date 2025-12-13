from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Sequence

from llama_index.core.schema import BaseNode, ImageNode, TextNode

from ...core.event import async_loop_runner
from ...llama_like.core.schema import AudioNode, VideoNode
from ...logger import logger
from .base_transform import BaseTransform

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import BaseNode, ImageType

    from ...embed.embed_manager import EmbedManager
    from ...llama_like.embeddings.multi_modal_base import AudioType, VideoType

__all__ = ["EmbedTransform"]


class EmbedTransform(BaseTransform):
    """Transform to embed various modalities."""

    def __init__(self, embed: EmbedManager, is_canceled: Callable[[], bool]) -> None:
        """Constructor.

        Args:
            embed (EmbedManager): Embedding manager.
            is_canceled (Callable[[], bool]): Cancellation flag for the job.
        """
        super().__init__(is_canceled)
        self._embed = embed

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Synchronous interface.

        Args:
            nodes (Sequence[BaseNode]): Nodes to summarize.

        Returns:
            Sequence[BaseNode]: Nodes after summarization.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (Sequence[BaseNode]): Nodes to split.

        Returns:
            Sequence[BaseNode]: Nodes after splitting.
        """
        # For subsequent ingestions from known sources, nodes become empty
        if not nodes:
            return nodes

        if self._is_canceled():
            logger.info("Job is canceled, aborting batch processing")
            return []

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

        if self._pipe_callback:
            self._pipe_callback(self, split_nodes)

        return split_nodes

    async def _aembed_text(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Embed a text node.

        Args:
            nodes (Sequence[BaseNode]): Target text nodes.
        Returns:
            Sequence[BaseNode]: Embedded text nodes.
        """

        async def batch_text(texts: list[str]) -> list[Embedding]:
            return await self._embed.aembed_text(texts)

        def extractor(node: BaseNode) -> Optional[str]:
            if isinstance(node, TextNode) and node.text and node.text.strip():
                return node.text

            logger.warning("text is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_text, extractor)

    async def _aembed_image(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Embed an image node.

        Args:
            nodes (Sequence[BaseNode]): Target image nodes.
        Returns:
            Sequence[BaseNode]: Embedded image nodes.
        """
        from ...core.exts import Exts
        from ...core.metadata import MetaKeys as MK
        from ...core.utils import has_media

        async def batch_image(paths: list[ImageType]) -> list[Embedding]:
            return await self._embed.aembed_image(paths)

        def extractor(node: BaseNode) -> Optional[str]:
            if has_media(node=node, exts=Exts.IMAGE):
                return node.metadata[MK.FILE_PATH]

            logger.warning("image is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_image, extractor)

    async def _aembed_audio(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Embed an audio node.

        Args:
            nodes (Sequence[BaseNode]): Target audio nodes.

        Returns:
            Sequence[BaseNode]: Embedded audio nodes.
        """
        from ...core.exts import Exts
        from ...core.metadata import MetaKeys as MK
        from ...core.utils import has_media

        async def batch_audio(paths: list[AudioType]) -> list[Embedding]:
            return await self._embed.aembed_audio(paths)

        def extractor(node: BaseNode) -> Optional[str]:
            if has_media(node=node, exts=Exts.AUDIO):
                return node.metadata[MK.FILE_PATH]

            logger.warning("audio is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_audio, extractor)

    async def _aembed_video(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Embed a video node.

        Args:
            nodes (Sequence[BaseNode]): Target video nodes.

        Returns:
            Sequence[BaseNode]: Embedded video nodes.
        """
        from ...core.exts import Exts
        from ...core.metadata import MetaKeys as MK
        from ...core.utils import has_media

        async def batch_video(paths: list[VideoType]) -> list[Embedding]:
            return await self._embed.aembed_video(paths)

        def extractor(node: BaseNode) -> Optional[str]:
            if has_media(node=node, exts=Exts.VIDEO):
                return node.metadata[MK.FILE_PATH]

            logger.warning("video is not found, skipped")
            return None

        return await self._aexec_transform(nodes, batch_video, extractor)

    async def _aexec_transform(
        self,
        nodes: Sequence[BaseNode],
        batch_embed_fn: Callable[[list], Awaitable[list[list[float]]]],
        extract_fn: Callable[[BaseNode], object],
    ) -> Sequence[BaseNode]:
        """Embed nodes using the given batch embedding function and extractor.

        Args:
            nodes (Sequence[BaseNode]): Nodes to embed.
            batch_embed_fn: Function to perform batch embedding.
            extract_fn: Function to extract embedding input from a node.

        Returns:
            Sequence[BaseNode]: Nodes after embedding.
        """
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

        return nodes
