from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore


class DocumentStoreManager:
    """ドキュメントストアの管理クラス。"""

    def __init__(
        self, provider_name: str, store: KVDocumentStore, table_name: Optional[str]
    ) -> None:
        """コンストラクタ

        Args:
            provider_name (str): プロバイダ名
            store (KVDocumentStore): ドキュメントストア
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
    def store(self) -> KVDocumentStore:
        """ドキュメントストア。

        Returns:
            KVDocumentStore: ドキュメントストア
        """
        return self._store

    @property
    def table_name(self) -> Optional[str]:
        """テーブル名。

        Returns:
            Optional[str]: テーブル名
        """
        return self._table_name
