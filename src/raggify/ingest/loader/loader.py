from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document, ImageNode, TextNode

from ...core.exts import Exts
from ...core.metadata import BasicMetaData
from ...core.metadata import MetaKeys as MK
from ...llama.core.schema import AudioNode
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode

    from ...document_store.document_store_manager import DocumentStoreManager


class Loader:
    """ローダー基底クラス"""

    def __init__(
        self, document_store: DocumentStoreManager, persist_dir: Optional[Path]
    ) -> None:
        """コンストラクタ

        Args:
            document_store (DocumentStoreManager): ドキュメントストア管理
            persist_dir (Optional[Path]): 永続化ディレクトリ
        """
        self._document_store = document_store
        self._persist_dir = persist_dir

    def _add_doc_id(self, docs: list[Document]) -> None:
        """一意なドキュメント ID を付与する。

        二回目以降、同一 ID のドキュメントに対しては IngestionPipeline 内で
        ハッシュ比較が行われ、変更がない場合は戻り値のノードリストから弾かれる。

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
            doc.id_ = self._generate_doc_id(meta)
            doc.doc_id = doc.id_

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

    def _build_or_load_pipe(self) -> IngestionPipeline:
        """ドキュメント重複管理用パイプラインを新規作成またはロードする。

        Returns:
            IngestionPipeline: パイプライン
        """
        from llama_index.core.ingestion import DocstoreStrategy

        pipe = IngestionPipeline(
            docstore=self._document_store.store,
            docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
        )

        if self._persist_dir and self._persist_dir.exists():
            try:
                pipe.load(str(self._persist_dir))
                self._document_store.store = pipe.docstore
            except Exception as e:
                logger.warning(f"failed to load persist dir: {e}")

        return pipe

    async def _asplit_docs_modality(
        self, docs: list[Document]
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
        """ドキュメントをモダリティ別に分ける。

        Args:
            docs (list[Document]): 入力ドキュメント

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
                テキストノード、画像ノード、音声ノード
        """
        self._add_doc_id(docs)
        pipe = self._build_or_load_pipe()
        nodes = await pipe.arun(documents=docs)

        logger.debug(f"{len(docs)} docs --pipeline--> {len(nodes)} nodes")

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

        if self._persist_dir:
            try:
                pipe.persist(str(self._persist_dir))
            except Exception as e:
                logger.warning(f"failed to persist: {e}")

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
