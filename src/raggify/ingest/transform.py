from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from llama_index.core.async_utils import asyncio_run
from llama_index.core.schema import BaseNode, TransformComponent

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType

    from ..embed.embed_manager import EmbedManager
    from ..llama.embeddings.multi_modal_base import AudioType


class AddChunkIndexTransform(TransformComponent):
    """チャンク番号付与用トランスフォーム"""

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """パイプライン側から呼ばれるインタフェース

        Args:
            nodes (list[BaseNode]): 分割済みのノード

        Returns:
            list[BaseNode]: チャンク番号付与後のノード
        """
        from ..core.metadata import MetaKeys as MK

        buckets = defaultdict(list)
        for node in nodes:
            id = node.ref_doc_id
            buckets[id].append(node)

        node: BaseNode
        for id, group in buckets.items():
            for i, node in enumerate(group):
                node.metadata[MK.CHUNK_NO] = i

        return nodes

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        return self.__call__(nodes, **kwargs)


class _BaseEmbedTransform(TransformComponent):
    """埋め込み用トランスフォーム"""

    def __init__(
        self,
        batch_embed_fn: Callable[[list], Awaitable[list[list[float]]]],
        extract_fn: Callable[[BaseNode], object],
    ):
        """コンストラクタ

        Args:
            batch_embed_fn (Callable[[list], Awaitable[list[list[float]]]]): バッチ埋め込み関数
            extract_fn (Callable[[BaseNode], object]): モダリティ別のノード抽出関数
        """
        self._batch_embed_fn = batch_embed_fn
        self._extract_fn = extract_fn

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        return asyncio_run(self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """パイプライン側から呼ばれるインタフェース

        Args:
            nodes (list[BaseNode]): 埋め込み対象ノード

        Returns:
            list[BaseNode]: 埋め込み後のノード
        """
        from ..core.metadata import MetaKeys as MK

        # 入力抽出（欠損はスキップしつつ、元ノードへの逆写像を保持）
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

        # バッチ埋め込み
        vecs = await self._batch_embed_fn(inputs)
        if not vecs:
            return nodes

        if len(vecs) != len(inputs):
            # 安全側: 長さ不一致なら何も書かない（ログは上位で）
            return nodes

        # node へ戻し書き
        for i, vec in zip(backrefs, vecs):
            nodes[i].embedding = vec

            if nodes[i].metadata.get(MK.TEMP_FILE_PATH):
                # 一時ファイルを持つノードの file_path は base_source で上書き
                # （空になるか、PDF 等の独自 reader が退避していた元パスが復元されるか）
                nodes[i].metadata[MK.FILE_PATH] = nodes[i].metadata[MK.BASE_SOURCE]

        return nodes


def _get_media_path(node: BaseNode) -> str:
    """テキスト以外の埋め込み対象メディアのパスを取得する。

    Args:
        node (BaseNode): 対象ノード

    Returns:
        str: メディアのパス
    """
    from ..core.metadata import MetaKeys as MK

    temp = node.metadata.get(MK.TEMP_FILE_PATH)
    if temp:
        # フェッチした一時ファイル
        return temp

    # ローカルファイル
    return node.metadata[MK.FILE_PATH]


def make_text_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """テキストノードの埋め込みトランスフォーム生成用ラッパー

    Args:
        embed (EmbedManager): 埋め込み管理

    Returns:
        _BaseEmbedTransform: トランスフォーム
    """
    from llama_index.core.schema import TextNode

    async def batch_text(texts: list[str]) -> list[Embedding]:
        return await embed.aembed_text(texts)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, TextNode) and node.text and node.text.strip():
            return node.text

        logger.warning("text is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_text, extractor)


def make_image_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """画像ノードの埋め込みトランスフォーム生成用ラッパー

    Args:
        embed (EmbedManager): 埋め込み管理

    Returns:
        _BaseEmbedTransform: トランスフォーム
    """
    from llama_index.core.schema import ImageNode

    async def batch_image(paths: list[ImageType]) -> list[Embedding]:
        return await embed.aembed_image(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, ImageNode):
            return _get_media_path(node)

        logger.warning("image is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_image, extractor)


def make_audio_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """音声ノードの埋め込みトランスフォーム生成用ラッパー

    Args:
        embed (EmbedManager): 埋め込み管理

    Returns:
        _BaseEmbedTransform: トランスフォーム
    """
    from ..llama.core.schema import AudioNode

    async def batch_audio(paths: list[AudioType]) -> list[Embedding]:
        return await embed.aembed_audio(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, AudioNode):
            return _get_media_path(node)

        logger.warning("audio is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_audio, extractor)
