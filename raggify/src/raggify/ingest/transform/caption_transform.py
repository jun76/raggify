from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from llama_index.core.llms import AudioBlock, ChatMessage, ImageBlock, TextBlock
from llama_index.core.schema import BaseNode

from ...core.event import async_loop_runner
from ...core.metadata import MetaKeys as MK
from ...logger import logger
from .base_transform import BaseTransform

_BlockSequence = Sequence[TextBlock | ImageBlock | AudioBlock]


if TYPE_CHECKING:
    from llama_index.core.llms import LLM
    from llama_index.core.schema import ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode
    from ...llm.llm_manager import LLMManager

__all__ = ["DefaultCaptionTransform", "LLMCaptionTransform"]


class DefaultCaptionTransform(BaseTransform):
    """A placeholder caption transform that returns nodes unchanged."""

    def __init__(self, is_canceled: Callable[[], bool] = lambda: False) -> None:
        """Constructor.

        Args:
            is_canceled (Callable[[], bool], optional):
                Cancellation flag for the job. Defaults to lambda: False.
        """
        super().__init__(is_canceled)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline.

        Args:
            nodes (Sequence[BaseNode]): Nodes to caption.

        Returns:
            Sequence[BaseNode]: Nodes after captioning.
        """
        if self._record_nodes:
            self._record_nodes(self, nodes)

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (Sequence[BaseNode]): Nodes to caption.

        Returns:
            Sequence[BaseNode]: Nodes after captioning.
        """
        return nodes


class LLMCaptionTransform(BaseTransform):
    """Transform to caption multimodal nodes using an LLM."""

    def __init__(
        self,
        llm_manager: LLMManager,
        is_canceled: Callable[[], bool],
        audio_sample_rate: int = 16000,
    ) -> None:
        """Constructor.

        Args:
            llm_manager (LLMManager): LLM manager.
            is_canceled (Callable[[], bool]): Cancellation flag for the job.
            audio_sample_rate (int, optional): Audio sample rate. Defaults to 16000.
        """
        super().__init__(is_canceled)
        self._llm_manager = llm_manager
        self._audio_sample_rate = audio_sample_rate

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline.

        Args:
            nodes (Sequence[BaseNode]): Nodes to caption.

        Returns:
            Sequence[BaseNode]: Nodes after captioning.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (Sequence[BaseNode]): Nodes to caption.

        Returns:
            Sequence[BaseNode]: Nodes after captioning.
        """
        from llama_index.core.schema import ImageNode

        from ...llama_like.core.schema import AudioNode, VideoNode

        if not nodes:
            return nodes

        captioned_nodes: list[BaseNode] = []
        for node in nodes:
            if self._is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return []

            if isinstance(node, ImageNode):
                captioned = await self._acaption_image(node)
            elif isinstance(node, AudioNode):
                captioned = await self._acaption_audio(node)
            elif isinstance(node, VideoNode):
                captioned = await self._acaption_video(node)
            else:
                captioned = node

            captioned_nodes.append(captioned)

        if self._record_nodes:
            self._record_nodes(self, nodes)

        logger.debug(f"captioned {len(captioned_nodes)} nodes")

        return captioned_nodes

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    async def _acaption_image(self, node: ImageNode) -> TextNode:
        """Caption an image node using LLM.

        Args:
            node (ImageNode): Node to caption.

        Returns:
            TextNode: Node after captioning.
        """
        from pathlib import Path

        prompt = """
Please provide a concise description of the image for semantic search purposes.
If the image is not describable,
please return just an empty string (no need for unnecessary comments).
"""
        llm = self._llm_manager.image_captioner

        def _build_blocks(target: TextNode) -> list[TextBlock | ImageBlock]:
            path = target.metadata[MK.FILE_PATH]
            return [
                ImageBlock(path=Path(path)),
                TextBlock(text=prompt),
            ]

        return await self._acaption_with_llm(
            node=node,
            llm=llm,
            block_builder=_build_blocks,
            modality="image",
        )

    async def _acaption_audio(self, node: AudioNode | VideoNode) -> TextNode:
        """Caption an audio node using LLM.

        Args:
            node (AudioNode | VideoNode): Node to caption.

        Returns:
            TextNode: Node after captioning.
        """
        from pathlib import Path

        from ...core.exts import Exts

        prompt = """
Please provide a concise description of the audio for semantic search purposes.
If the audio is not describable,
please return just an empty string (no need for unnecessary comments).
"""
        llm = self._llm_manager.audio_captioner

        def _build_blocks(target: TextNode) -> list[TextBlock | AudioBlock]:
            path = target.metadata[MK.FILE_PATH]
            return [
                AudioBlock(path=Path(path), format=Exts.get_ext(uri=path, dot=False)),
                TextBlock(text=prompt),
            ]

        return await self._acaption_with_llm(
            node=node,
            llm=llm,
            block_builder=_build_blocks,
            modality="audio",
        )

    async def _acaption_video(self, node: VideoNode) -> TextNode:
        """Caption a video node using LLM.

        Args:
            node (VideoNode): Node to caption.

        Returns:
            TextNode: Node after captioning.
        """
        from ...core.metadata import MetaKeys as MK
        from ...llama_like.core.schema import AudioNode
        from ..util import MediaConverter

        path = node.metadata[MK.FILE_PATH]
        try:
            converter = MediaConverter()
        except ImportError as e:
            logger.error(f"ffmpeg not installed, cannot caption video audio: {e}")
            return node

        temp_path = converter.extract_mp3_audio_from_video(
            src=Path(path), sample_rate=self._audio_sample_rate
        )
        if temp_path is not None:
            audio_node = AudioNode(
                text=node.text, metadata={MK.FILE_PATH: str(temp_path)}
            )
            audio_node = await self._acaption_audio(audio_node)
            node.text = audio_node.text

        return node

    async def _acaption_with_llm(
        self,
        node: TextNode,
        llm: LLM,
        block_builder: Callable[[TextNode], _BlockSequence],
        modality: str,
    ) -> TextNode:
        """Run captioning with provided LLM and block builder.

        Args:
            node (TextNode): Target node.
            llm (LLM): LLM instance to use.
            block_builder (Callable[[TextNode], _BlockSequence]):
                Callable that returns chat message blocks for the node.
            modality (str): Modality label for logging.

        Returns:
            TextNode: Node after captioning.
        """
        try:
            blocks = list(block_builder(node))
        except Exception as e:
            logger.error(f"failed to build {modality} summary blocks: {e}")
            return node

        messages = [
            ChatMessage(
                role="user",
                blocks=blocks,
            )
        ]

        summary = ""
        try:
            response = await llm.achat(messages)
            summary = (response.message.content or "").strip()
            if summary:
                node.text = summary
        except Exception as e:
            logger.error(f"failed to caption {modality} node: {e}")

        logger.debug(f"captioned {modality} node: {summary[:50]}...")

        return node
