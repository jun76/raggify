from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.storage.docstore import BaseDocumentStore


class DocumentStoreManager:
    """ドキュメントストアの管理クラス。"""

    def __init__(
        self,
        provider_name: str,
        store: Optional[BaseDocumentStore],
        table_name: Optional[str],
    ) -> None:
        """コンストラクタ

        Args:
            provider_name (str): プロバイダ名
            store (Optional[BaseDocumentStore]): ドキュメントストア
            table_name (Optional[str]): テーブル名
        """
        self._provider_name = provider_name
        self._store = store
        self._table_name = table_name

        logger.debug(f"{provider_name} docstore created")

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return self._provider_name

    @property
    def store(self) -> Optional[BaseDocumentStore]:
        """ドキュメントストア。

        Returns:
            Optional[BaseDocumentStore]: ドキュメントストア
        """
        return self._store

    @store.setter
    def store(self, value: Optional[BaseDocumentStore]) -> None:
        """ドキュメントストアを設定する。

        Args:
            value (Optional[BaseDocumentStore]): 設定するドキュメントストア
        """
        self._store = value

    @property
    def table_name(self) -> Optional[str]:
        """テーブル名。

        Returns:
            Optional[str]: テーブル名
        """
        return self._table_name

    def has_bm25_corpus(self) -> bool:
        """BM25 検索用のテキストコーパスを持っているか。

        pipe.arun(store_doc_text=True) のデフォルト設定になっていれば持っているはず。

        Returns:
            bool: コーパスを持っているか
        """
        if self.store is None:
            return False

        docs_attr = getattr(self.store, "docs", None)

        if docs_attr is None:
            return False

        try:
            return len(docs_attr) > 0
        except Exception:
            # 一部の docstore 実装は __len__ を持たない可能性があるため、
            # 属性が存在するだけで True とみなす。
            return True

    def get_ref_doc_ids(self) -> list[str]:
        """ストアに格納済みの全 ref_doc_info 情報を取得する。

        Returns:
            list[str]: ref_doc_id のリスト
        """
        if self.store is None:
            return []

        infos = self.store.get_all_ref_doc_info()
        if infos is None:
            return []

        return list(infos.keys())

    def delete_all(self, persist_path: Optional[str]) -> None:
        """ストアに格納済みの全 ref_doc と関連ノードを削除する。

        Args:
            persist_path (Optional[str]): 永続化ディレクトリ
        """
        if self.store is None:
            return

        try:
            for doc_id in list(self.store.docs.keys()):
                self.store.delete_document(doc_id, raise_error=False)
        except Exception as e:
            logger.warning(f"failed to delete doc {doc_id}: {e}")
            return

        logger.info("all documents are deleted from document store")

        if persist_path is not None:
            try:
                self.store.persist(persist_path)
            except Exception as e:
                logger.warning(f"failed to persist: {e}")
