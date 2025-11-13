from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from llama_index.core.schema import Document, ImageNode, MediaResource, TextNode

from ...core.exts import Exts
from ...core.metadata import BasicMetaData
from ...core.metadata import MetaKeys as MK
from ...llama.core.schema import AudioNode
from ...logger import logger
from ...runtime import get_runtime as _rt

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


class Loader:
    """ローダー基底クラス"""

    def __init__(self, persist_dir: Optional[Path]) -> None:
        """コンストラクタ

        Args:
            persist_dir (Optional[Path]): 永続化ディレクトリ
        """
        self._persist_dir = persist_dir

    def _finalize_docs(self, docs: list[Document]) -> None:
        """メタデータを調整してドキュメントの内容を確定する。

        Args:
            docs (list[Document]): ドキュメント
        """
        counters: dict[str, int] = defaultdict(int)
        for doc in docs:
            meta = BasicMetaData.from_dict(doc.metadata)

            # IPYNBReader が分割後のドキュメントを全て同一 metadata で返してくるため
            # こちらで chunk_no として連番を付与
            counter_key = meta.temp_file_path or meta.file_path or meta.url
            meta.chunk_no = counters[counter_key]
            counters[counter_key] += 1
            doc.metadata[MK.CHUNK_NO] = meta.chunk_no

            # 一意な ID を付与。二回目以降、同一 ID のドキュメントに対しては
            # IngestionPipeline 内でハッシュ比較が行われ、変更がない場合は戻り値の
            # ノードリストに含まれない。
            doc.id_ = self._generate_doc_id(meta)
            doc.doc_id = doc.id_

            # BM25 が text_resource を参照するため、中身が空なら .text を転記する
            text_resource = getattr(doc, "text_resource", None)
            text_value = getattr(text_resource, "text", None) if text_resource else None
            if not text_value:
                try:
                    doc.text_resource = MediaResource(text=doc.text)
                except Exception as e:
                    logger.debug(
                        f"failed to set text_resource on doc {doc.doc_id}: {e}"
                    )

    def _generate_doc_id(self, meta: BasicMetaData) -> str:
        """doc_id を生成する。

        Args:
            meta (BasicMetaData): メタデータの辞書

        Returns:
            str: doc_id 文字列
        """
        return (
            f"{MK.FILE_PATH}:{meta.file_path}_"
            f"{MK.FILE_SIZE}:{meta.file_size}_"
            f"{MK.FILE_LASTMOD_AT}:{meta.file_lastmod_at}_"
            f"{MK.PAGE_NO}:{meta.page_no}_"
            f"{MK.ASSET_NO}:{meta.asset_no}_"
            f"{MK.CHUNK_NO}:{meta.chunk_no}_"
            f"{MK.URL}:{meta.url}_"
            f"{MK.TEMP_FILE_PATH}:{meta.temp_file_path}"  # PDF 埋め込み画像等の識別用
        )

    async def _aparse_documents(
        self,
        docs: Sequence[Document],
        is_canceled: Callable[[], bool],
    ) -> Sequence[BaseNode]:
        """ドキュメントをノードに分割する。

        Args:
            docs (Sequence[Document]): ドキュメント
            is_canceled (Callable[[], bool]): このジョブがキャンセルされたか。

        Returns:
            Sequence[BaseNode]: 分割後のノード
        """
        if not docs or is_canceled():
            return []

        rt = _rt()
        batch_size = rt.cfg.ingest.batch_size
        total_batches = (len(docs) + batch_size - 1) // batch_size
        nodes = []
        pipe = rt.build_pipeline(persist_dir=self._persist_dir)
        for idx in range(0, len(docs), batch_size):
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return nodes

            batch = docs[idx : idx + batch_size]
            prog = f"{idx // batch_size + 1}/{total_batches}"
            logger.debug(
                f"parse documents pipeline: processing batch {prog} "
                f"({len(batch)} docs)"
            )
            try:
                nodes.extend(await pipe.arun(documents=batch))
            except Exception as e:
                logger.error(f"failed to process batch {prog}, continue: {e}")

        rt.persist_pipeline(pipe=pipe, persist_dir=self._persist_dir)
        logger.debug(f"{len(docs)} docs --pipeline--> {len(nodes)} nodes")

        return nodes

    async def _asplit_docs_modality(
        self,
        docs: list[Document],
        is_canceled: Callable[[], bool],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
        """ドキュメントをモダリティ別に分ける。

        Args:
            docs (list[Document]): 入力ドキュメント
            is_canceled (Callable[[], bool]): このジョブがキャンセルされたか。

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
                テキストノード、画像ノード、音声ノード
        """
        self._finalize_docs(docs)
        nodes = await self._aparse_documents(docs=docs, is_canceled=is_canceled)

        image_nodes = []
        audio_nodes = []
        text_nodes = []
        for node in nodes:
            if isinstance(node, TextNode) and self._is_image_node(node):
                image_nodes.append(
                    ImageNode(
                        text=node.text, ref_doc_id=node.id_, metadata=node.metadata
                    )
                )
            elif isinstance(node, TextNode) and self._is_audio_node(node):
                audio_nodes.append(
                    AudioNode(
                        text=node.text, ref_doc_id=node.id_, metadata=node.metadata
                    )
                )
            elif isinstance(node, TextNode):
                text_nodes.append(node)
            else:
                logger.warning(f"unexpected node type {type(node)}, skipped")

        return text_nodes, image_nodes, audio_nodes

    def _is_image_node(self, node: BaseNode) -> bool:
        """画像ノードか。

        Args:
            node (BaseNode): 対象ノード

        Returns:
            bool: 画像ノードなら True
        """
        # ファイルパスか URL の末尾に画像ファイルの拡張子が含まれるものを画像ノードとする
        path = node.metadata.get(MK.FILE_PATH, "")
        url = node.metadata.get(MK.URL, "")

        # 独自 reader を使用し、temp_file_path に画像ファイルの拡張子が含まれるものも抽出
        temp_file_path = node.metadata.get(MK.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, Exts.IMAGE)
            or Exts.endswith_exts(url, Exts.IMAGE)
            or Exts.endswith_exts(temp_file_path, Exts.IMAGE)
        )

    def _is_audio_node(self, node: BaseNode) -> bool:
        """音声ノードか。

        Args:
            node (BaseNode): 対象ノード

        Returns:
            bool: 音声ノードなら True
        """
        path = node.metadata.get(MK.FILE_PATH, "")
        url = node.metadata.get(MK.URL, "")
        temp_file_path = node.metadata.get(MK.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, Exts.AUDIO)
            or Exts.endswith_exts(url, Exts.AUDIO)
            or Exts.endswith_exts(temp_file_path, Exts.AUDIO)
        )
