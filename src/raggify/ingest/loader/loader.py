from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.schema import Document, ImageNode, TextNode

from ...core.exts import Exts
from ...core.metadata import BasicMetaData
from ...core.metadata import MetaKeys as MK
from ...llama.core.schema import AudioNode
from ...logger import logger

if TYPE_CHECKING:
    from ...document_store.document_store_manager import DocumentStoreManager


class Loader:
    """ローダー基底クラス"""

    def __init__(self, document_store: DocumentStoreManager) -> None:
        """コンストラクタ

        Args:
            document_store (DocumentStoreManager): ドキュメントストア管理
        """
        self._document_store = document_store

    def _generate_doc_id(self, meta: BasicMetaData) -> str:
        """doc_id を生成する。

        Args:
            meta (BasicMetaData): メタデータの辞書

        Returns:
            str: doc_id 文字列
        """
        import hashlib
        import json

        # Web ページの場合、現状 URL しかチェックしない
        raw = {
            MK.FILE_PATH: meta.file_path if not meta.temp_file_path else "",
            MK.FILE_SIZE: meta.file_size,
            MK.FILE_LASTMOD_AT: meta.file_lastmod_at,
            MK.PAGE_NO: meta.page_no,
            MK.URL: meta.url,
            MK.BASE_SOURCE: meta.base_source,
        }

        return hashlib.md5(json.dumps(raw, sort_keys=True).encode()).hexdigest()

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
        from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline

        new_docs: list[Document] = []
        for doc in docs:
            meta = BasicMetaData.from_dict(doc.metadata)
            doc.id_ = self._generate_doc_id(meta)
            doc.doc_id = doc.id_
            if self._document_store.store.document_exists(doc.id_):
                logger.info(f"source {meta.file_path or meta.url} exists, skipped")
                continue

            # 新規ソースのドキュメントのみ処理対象とする
            new_docs.append(doc)
            logger.info(f"new source: {meta.file_path or meta.url}")

        if not new_docs:
            return [], [], []

        # 前段パイプ。ドキュメントストアでの重複管理
        doc_pipe = IngestionPipeline(
            docstore=self._document_store.store,
            docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
        )
        await doc_pipe.arun(documents=new_docs)

        image_nodes = []
        audio_nodes = []
        text_nodes = []
        for doc in new_docs:
            meta = BasicMetaData.from_dict(doc.metadata)
            if self._is_image_doc(doc):
                image_nodes.append(
                    ImageNode(text=doc.text, ref_doc_id=doc.id_, metadata=doc.metadata)
                )
                logger.debug(f"add ImageNode: {meta.file_path or meta.url}")
            elif self._is_audio_doc(doc):
                audio_nodes.append(
                    AudioNode(text=doc.text, ref_doc_id=doc.id_, metadata=doc.metadata)
                )
                logger.debug(f"add AudioNode: {meta.file_path or meta.url}")
            else:
                text_nodes.append(
                    TextNode(text=doc.text, ref_doc_id=doc.id_, metadata=doc.metadata)
                )
                logger.debug(f"add TextNode: {meta.file_path or meta.url}")

        return text_nodes, image_nodes, audio_nodes

    def _is_image_doc(self, doc: Document) -> bool:
        """画像ドキュメントか。

        Args:
            doc (Document): 対象ドキュメント

        Returns:
            bool: 画像ドキュメントなら True
        """
        # ファイルパスか URL の末尾に画像ファイルの拡張子が含まれるものを画像ドキュメントとする
        path = doc.metadata.get(MK.FILE_PATH, "")
        url = doc.metadata.get(MK.URL, "")

        # 独自 reader を使用し、temp_file_path に画像ファイルの拡張子が含まれるものも抽出
        temp_file_path = doc.metadata.get(MK.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, Exts.IMAGE)
            or Exts.endswith_exts(url, Exts.IMAGE)
            or Exts.endswith_exts(temp_file_path, Exts.IMAGE)
        )

    def _is_audio_doc(self, doc: Document) -> bool:
        """音声ドキュメントか。

        Args:
            doc (Document): 対象ドキュメント

        Returns:
            bool: 音声ドキュメントなら True
        """
        path = doc.metadata.get(MK.FILE_PATH, "")
        url = doc.metadata.get(MK.URL, "")
        temp_file_path = doc.metadata.get(MK.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, Exts.AUDIO)
            or Exts.endswith_exts(url, Exts.AUDIO)
            or Exts.endswith_exts(temp_file_path, Exts.AUDIO)
        )
