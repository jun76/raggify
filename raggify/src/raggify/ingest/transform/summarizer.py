from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.schema import BaseNode, TransformComponent

from ...core.event import async_loop_runner
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode

__all__ = ["DefaultSummarizer", "LLMSummarizer"]


class DefaultSummarizer(TransformComponent):
    """A placeholder summarizer that returns nodes unchanged."""

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


class LLMSummarizer(TransformComponent):
    """Transform to summarize multimodal nodes using an LLM."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Constructor.

        Args:
            model (str): LLM model name. Defaults to "gpt-4o-mini".
        """
        self._model = model

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Synchronous interface.

        Args:
            nodes (list[BaseNode]): Nodes to summarize.

        Returns:
            list[BaseNode]: Nodes after summarization.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def _summarize_text(self, node: TextNode, **kwargs) -> BaseNode:
        """Summarize a text node using LLM.

        Args:
            node (TextNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        from llama_index.llms.openai import OpenAI

        llm = OpenAI(model=self._model, temperature=0)
        prompt = """
Please extract only the main text useful for semantic search from the following text.
Remove figure captions, short texts under 20 characters, advertisements,
site navigation elements, copyright notices, etc.
If nothing remains after removal, that is acceptable.
In that case, please return an empty string.

Original text:
{text}

Only useful main text:
"""
        resp = llm.complete(prompt.format(text=node.text))
        node.text = resp.text.strip()
        logger.debug(f"summary: {node.text[:50]}...")

        return node

    async def _summarize_image(self, node: ImageNode) -> BaseNode:
        """Summarize an image node using LLM.

        Args:
            node (ImageNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        from llama_index.multi_modal_llms.openai import OpenAIMultiModal

        llm = OpenAIMultiModal(model=self._model, temperature=0)
        prompt = """
Please provide a concise description of the content of the following image for semantic search purposes.
If the image is not describable, please return an empty string.
"""
        resp = llm.complete(prompt=prompt, image_documents=[node])
        caption = resp.text.strip()
        if caption:
            node.text = caption

        logger.debug(f"caption: {caption}")

        return node

    async def _summarize_audio(self, node: AudioNode) -> BaseNode:
        """Summarize an audio node using LLM.

        Args:
            node (AudioNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        logger.debug("audio summarization is not implemented yet")
        return node

    async def _summarize_video(self, node: VideoNode) -> BaseNode:
        """Summarize a video node using LLM.

        Args:
            node (VideoNode): Node to summarize.

        Returns:
            BaseNode: Node after summarization.
        """
        logger.debug("video summarization is not implemented yet")
        return node

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
                summarized = await self._summarize_image(node)
            elif isinstance(node, AudioNode):
                summarized = await self._summarize_audio(node)
            elif isinstance(node, VideoNode):
                summarized = await self._summarize_video(node)
            elif isinstance(node, TextNode):
                summarized = await self._summarize_text(node)
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
