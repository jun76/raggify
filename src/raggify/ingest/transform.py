from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from llama_index.core.schema import BaseNode, ImageNode, TextNode, TransformComponent

from ..llama.core.schema import Modality

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType

    from ..embed.embed_manager import EmbedManager
    from ..llama.embeddings.multi_modal_base import AudioType


# FIXME: 色々未整理
class AddChunkIndexTransform(TransformComponent):
    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        buckets = defaultdict(list)
        for n in nodes:
            doc_id = n.metadata.get("doc_id") or n.metadata.get("file_path")
            buckets[doc_id].append(n)

        for doc_id, group in buckets.items():
            total = len(group)
            for i, n in enumerate(group):
                n.metadata["chunk_index"] = i
                n.metadata["chunk_total"] = total
                # 任意: 安定IDを作るならここで
                # n.id_ = f"{doc_id}:{i}"  # 既存の node_id との整合は方針に合わせて
        return nodes

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        return self.__call__(nodes, **kwargs)


class _BaseEmbedTransform(TransformComponent):
    """nodes -> nodes の非同期Transform。node.embedding を埋める。"""

    def __init__(
        self,
        batch_embed_fn: Callable[[list], Awaitable[list[list[float]]]],
        extract_fn: Callable[[BaseNode], object],
        modality: Modality,
        meta_stamp: str | None = None,
    ):
        self._batch_embed_fn = batch_embed_fn
        self._extract_fn = extract_fn
        self._modality = modality
        self._meta_stamp = meta_stamp

    async def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        # 入力抽出（欠損はスキップしつつ、元ノードへの逆写像を保持）
        inputs: list[object] = []
        backrefs: list[int] = []
        for i, n in enumerate(nodes):
            x = self._extract_fn(n)
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
        for idx, vec in zip(backrefs, vecs):
            nodes[idx].embedding = vec
            if self._meta_stamp:
                nodes[idx].metadata["embedding_model"] = self._meta_stamp
                nodes[idx].metadata["modality"] = self._modality

        return nodes


def make_text_embed_transform(
    embed: EmbedManager, model_stamp: str | None = None
) -> _BaseEmbedTransform:
    async def batch_text(texts: list[str]) -> list[Embedding]:
        return await embed.aembed_text(texts)

    def extractor(n: BaseNode) -> Optional[str]:
        if isinstance(n, TextNode) and n.text and n.text.strip():
            return n.text
        return None

    return _BaseEmbedTransform(batch_text, extractor, Modality.TEXT, model_stamp)


def make_image_embed_transform(
    embed: EmbedManager, model_stamp: str | None = None
) -> _BaseEmbedTransform:
    async def batch_image(paths: list[ImageType]) -> list[Embedding]:
        # paths: file path / PIL.Image / bytes など embed 側の約束に合わせる
        return await embed.aembed_image(paths)

    def extractor(n: BaseNode) -> Optional[str]:
        if isinstance(n, ImageNode):
            # ImageNode に合わせて取得。image_path / image_url / image を見る
            return (
                n.image_path
                or getattr(n, "image_url", None)
                or getattr(n, "image", None)
            )
        # BaseNode+metadata 運用なら:
        # return n.metadata.get("image_path") or n.metadata.get("image_b64")
        return None

    return _BaseEmbedTransform(batch_image, extractor, Modality.IMAGE, model_stamp)


def make_audio_embed_transform(
    embed: EmbedManager, model_stamp: str | None = None
) -> _BaseEmbedTransform:
    async def batch_audio(paths: list[AudioType]) -> list[Embedding]:
        return await embed.aembed_audio(paths)

    def extractor(n: BaseNode) -> Optional[str]:
        # AudioNode が無ければ metadata に audio_path を持たせておく
        return getattr(n, "audio_path", None) or n.metadata.get("audio_path")

    return _BaseEmbedTransform(batch_audio, extractor, Modality.AUDIO, model_stamp)
