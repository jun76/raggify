from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore


class DocumentStoreManager:
    """ドキュメントストアの管理クラス。"""

    def __init__(
        self,
        provider_name: str,
        store: KVDocumentStore,
        table_name: str,
        persist_path: Optional[str] = None,
    ) -> None:
        """コンストラクタ

        Args:
            provider_name (str): プロバイダ名
            store (KVDocumentStore): ドキュメントストア
            table_name (str): テーブル名
            persist_path (Optional[str], optional): 永続化パス。Defaults to None.
        """
        self._provider_name = provider_name
        self._store = store
        self._table_name = table_name
        self._persist_path = persist_path

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
    def table_name(self) -> str:
        """テーブル名。

        Returns:
            str: テーブル名
        """
        return self._table_name

    def persist(self) -> None:
        """ストアを保存する。

        リモートストアの場合等、persist_path 未指定では nop。
        """
        if self._persist_path:
            self.store.persist(self._persist_path)
