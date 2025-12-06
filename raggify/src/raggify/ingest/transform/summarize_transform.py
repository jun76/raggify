from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.schema import BaseNode, TransformComponent

from ...core.event import async_loop_runner
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode
    from ...llm.llm import LLMManager

__all__ = ["DefaultSummarizeTransform", "LLMSummarizeTransform"]


class DefaultSummarizeTransform(TransformComponent):
    """A placeholder summarize transform that returns nodes unchanged."""

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Return nodes unchanged.

        Args:
            nodes (list[BaseNode]): Input nodes.

        Returns:
            list[BaseNode]: Unchanged nodes.
        """
        return nodes

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Async wrapper matching the synchronous call.

        Args:
            nodes (list[BaseNode]): Input nodes.

        Returns:
            list[BaseNode]: Unchanged nodes.
        """
        return nodes


class LLMSummarizeTransform(TransformComponent):
    """Transform to summarize multimodal nodes using an LLM."""

    def __init__(self, llm_manager: LLMManager) -> None:
        """Constructor.

        Args:
            llm_manager (LLMManager): LLM manager.
        """
        self._llm_manager = llm_manager

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
            nodes (list[BaseNode]): Nodes to summarize.

        Returns:
            list[BaseNode]: Nodes after summarization.
        """
        from llama_index.core.schema import ImageNode, TextNode

        from ...llama_like.core.schema import AudioNode, VideoNode

        summarized_nodes: list[BaseNode] = []
        for node in nodes:
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

        return summarized_nodes

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    async def _asummarize_text(self, node: TextNode, **kwargs) -> BaseNode:
        """Summarize a text node using LLM.

        Args:
            node (TextNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        llm = self._llm_manager.text_summarize_transform
        prompt = """
Please extract only the main text useful for semantic search from the following text.
Remove advertisements, copyright notices, 
clearly unnecessary text such as headers and footers etc.

Since the extracted text will be shortened later, 
DO NOT SUMMARIZE its content SEMANTICALLY here.

If no useful text is available, please return ONLY an empty string (no need for unnecessary comments).

Original text:
{text}
"""
        try:
            resp = await llm.acomplete(
                prompt=prompt.format(text=node.text), image_documents=[]
            )
            node.text = resp.text.strip()
        except Exception as e:
            logger.error(f"failed to summarize text node: {e}")
            return node

        logger.debug(f"summary: {node.text[:50]}...")

        return node

    async def _asummarize_image(self, node: ImageNode) -> BaseNode:
        """Summarize an image node using LLM.

        Args:
            node (ImageNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        llm = self._llm_manager.image_summarize_transform
        prompt = """
Please provide a concise description of the content of the following image for 
semantic search purposes. If the image is not describable, please return 
just an empty string (no need for unnecessary comments).
"""
        try:
            resp = await llm.acomplete(prompt=prompt, image_documents=[node])
            caption = resp.text.strip()
            if caption:
                node.text = caption
        except Exception as e:
            logger.error(f"failed to summarize image node: {e}")
            return node

        logger.debug(f"caption: {caption}")

        return node

    async def _asummarize_audio(self, node: AudioNode) -> BaseNode:
        """Summarize an audio node using LLM.

        Args:
            node (AudioNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        logger.debug("audio summarization is not implemented yet")
        return node

    async def _asummarize_video(self, node: VideoNode) -> BaseNode:
        """Summarize a video node using LLM.

        Args:
            node (VideoNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        logger.debug("video summarization is not implemented yet")
        return node
