from __future__ import annotations

from llama_index.core.schema import Document

from ...core.exts import Exts
from ...core.metadata import MetaKeys as MK


class Loader:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """ローダー基底クラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _split_docs_modality(
        self, docs: list[Document]
    ) -> tuple[list[Document], list[Document], list[Document]]:
        """ドキュメントをモダリティ別に分ける。

        Args:
            docs (list[Document]): 入力ドキュメント

        Returns:
            tuple[list[Document], list[Document], list[Document]]:
                テキストドキュメント、画像ドキュメント、音声ドキュメント
        """
        image_docs = []
        audio_docs = []
        text_docs = []
        for doc in docs:
            if self._is_image_doc(doc):
                image_docs.append(doc)
            elif self._is_audio_doc(doc):
                audio_docs.append(doc)
            else:
                text_docs.append(doc)

        return text_docs, image_docs, audio_docs

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
