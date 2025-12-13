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

__all__ = ["DefaultSummarizeTransform", "LLMSummarizeTransform"]


class DefaultSummarizeTransform(BaseTransform):
    """A placeholder summarize transform that returns nodes unchanged."""

    def __init__(self, is_canceled: Callable[[], bool] = lambda: False) -> None:
        """Constructor.

        Args:
            is_canceled (Callable[[], bool], optional):
                Cancellation flag for the job. Defaults to lambda: False.
        """
        super().__init__(is_canceled)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Return nodes unchanged.

        Args:
            nodes (Sequence[BaseNode]): Input nodes.

        Returns:
            Sequence[BaseNode]: Unchanged nodes.
        """
        if self._pipe_callback:
            self._pipe_callback(self, nodes)

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Async wrapper matching the synchronous call.

        Args:
            nodes (Sequence[BaseNode]): Input nodes.

        Returns:
            Sequence[BaseNode]: Unchanged nodes.
        """
        return nodes


class LLMSummarizeTransform(BaseTransform):
    """Transform to summarize multimodal nodes using an LLM."""

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
            nodes (Sequence[BaseNode]): Nodes to summarize.

        Returns:
            Sequence[BaseNode]: Nodes after summarization.
        """
        from llama_index.core.schema import ImageNode, TextNode

        from ...llama_like.core.schema import AudioNode, VideoNode

        if not nodes:
            return nodes

        summarized_nodes: list[BaseNode] = []
        for node in nodes:
            if self._is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return []

            if isinstance(node, ImageNode):
                summarized = await self._asummarize_image(node)
            elif isinstance(node, AudioNode):
                summarized = await self._asummarize_audio(node)
            elif isinstance(node, VideoNode):
                summarized = await self._asummarize_video(node)
            elif isinstance(node, TextNode):
                summarized = await self._asummarize_text(node)
            else:
                raise ValueError(f"unsupported node type: {type(node)}")

            summarized_nodes.append(summarized)

        if self._pipe_callback:
            self._pipe_callback(self, summarized_nodes)

        return summarized_nodes

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    async def _asummarize_text(self, node: TextNode) -> TextNode:
        """Summarize a text node using LLM.

        Args:
            node (TextNode): Node to summarize.

        Returns:
            TextNode: Node after summarization.
        """
        prompt = """
Please extract only the main text useful for semantic search from the following text.
Remove advertisements, copyright notices,
clearly unnecessary text such as headers and footers etc.

Since the extracted text will be shortened later,
DO NOT SUMMARIZE its content SEMANTICALLY here.

If no useful text is available, please return ONLY an empty string (no need for unnecessary comments).

Original text:
{original_text}
"""
        llm = self._llm_manager.text_summarizer

        def _build_blocks(target: TextNode) -> list[TextBlock]:
            return [
                TextBlock(text=prompt.format(original_text=target.text)),
            ]

        return await self._summarize_with_llm(
            node=node,
            llm=llm,
            block_builder=_build_blocks,
            modality="text",
        )

    async def _asummarize_image(self, node: ImageNode) -> TextNode:
        """Summarize an image node using LLM.

        Args:
            node (ImageNode): Node to summarize.

        Returns:
            TextNode: Node after summarization.
        """
        from pathlib import Path

        prompt = """
Please provide a concise description of the image for semantic search purposes.
If the image is not describable,
please return just an empty string (no need for unnecessary comments).
"""
        llm = self._llm_manager.image_summarizer

        def _build_blocks(target: TextNode) -> list[TextBlock | ImageBlock]:
            path = target.metadata[MK.FILE_PATH]
            return [
                ImageBlock(path=Path(path)),
                TextBlock(text=prompt),
            ]

        return await self._summarize_with_llm(
            node=node,
            llm=llm,
            block_builder=_build_blocks,
            modality="image",
        )

    async def _asummarize_audio(self, node: AudioNode | VideoNode) -> TextNode:
        """Summarize an audio node using LLM.

        Args:
            node (AudioNode | VideoNode): Node to summarize.

        Returns:
            TextNode: Node after summarization.
        """
        from pathlib import Path

        from ...core.exts import Exts

        prompt = """
Please provide a concise description of the audio for semantic search purposes.
If the audio is not describable,
please return just an empty string (no need for unnecessary comments).
"""
        llm = self._llm_manager.audio_summarizer

        def _build_blocks(target: TextNode) -> list[TextBlock | AudioBlock]:
            path = target.metadata[MK.FILE_PATH]
            return [
                AudioBlock(path=Path(path), format=Exts.get_ext(uri=path, dot=False)),
                TextBlock(text=prompt),
            ]

        return await self._summarize_with_llm(
            node=node,
            llm=llm,
            block_builder=_build_blocks,
            modality="audio",
        )

    async def _asummarize_video(self, node: VideoNode) -> TextNode:
        """Summarize a video node using LLM.

        Args:
            node (VideoNode): Node to summarize.

        Returns:
            TextNode: Node after summarization.
        """
        from ...core.exts import Exts
        from ...core.metadata import MetaKeys as MK
        from ...core.utils import get_temp_path
        from ...llama_like.core.schema import AudioNode
        from ..util import MediaConverter

        path = node.metadata[MK.FILE_PATH]
        temp_path = Path(get_temp_path(seed=path, suffix=Exts.MP3))
        try:
            converter = MediaConverter()
        except ImportError as e:
            logger.error(f"ffmpeg not installed, cannot summarize video audio: {e}")
            return node

        temp_path = converter.extract_mp3_audio_from_video(
            src=Path(path), dst=temp_path, sample_rate=self._audio_sample_rate
        )
        if temp_path is not None:
            audio_node = AudioNode(
                text=node.text, metadata={MK.FILE_PATH: str(temp_path)}
            )
            audio_node = await self._asummarize_audio(audio_node)
            node.text = audio_node.text

        return node

    async def _summarize_with_llm(
        self,
        node: TextNode,
        llm: LLM,
        block_builder: Callable[[TextNode], _BlockSequence],
        modality: str,
    ) -> TextNode:
        """Run summarization with provided LLM and block builder.

        Args:
            node (TextNode): Target node.
            llm (LLM): LLM instance to use.
            block_builder (Callable[[TextNode], _BlockSequence]):
                Callable that returns chat message blocks for the node.
            modality (str): Modality label for logging.

        Returns:
            TextNode: Node after summarization.
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
            logger.error(f"failed to summarize {modality} node: {e}")

        logger.debug(f"summarized {modality} node: {summary[:50]}...")

        return node
