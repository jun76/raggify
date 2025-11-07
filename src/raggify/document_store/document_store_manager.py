from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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

        return docs_attr is not None
