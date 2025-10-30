from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.schema import Document, ImageNode, TextNode

from ...core.exts import Exts
from ...core.metadata import MetaKeys as MK
from ...llama.core.schema import AudioNode

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
        from llama_index.core.ingestion import IngestionPipeline

        # 前段パイプ。ドキュメントストアでの重複管理
        doc_pipe = IngestionPipeline(docstore=self._document_store.store)
        await doc_pipe.arun(documents=docs)

        image_nodes = []
        audio_nodes = []
        text_nodes = []
        for doc in docs:
            if self._is_image_doc(doc):
                image_nodes.append(ImageNode(text=doc.text, metadata=doc.metadata))
            elif self._is_audio_doc(doc):
                audio_nodes.append(AudioNode(text=doc.text, metadata=doc.metadata))
            else:
                text_nodes.append(TextNode(text=doc.text, metadata=doc.metadata))

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
