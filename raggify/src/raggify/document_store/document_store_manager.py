from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.storage.docstore import BaseDocumentStore

__all__ = ["DocumentStoreManager"]


class DocumentStoreManager:
    """Manager class for the document store."""

    def __init__(
        self,
        provider_name: str,
        store: BaseDocumentStore,
        table_name: Optional[str],
    ) -> None:
        """Constructor.

        Args:
            provider_name (str): Provider name.
            store (BaseDocumentStore): Document store.
            table_name (Optional[str]): Table name.
        """
        self._provider_name = provider_name
        self._store = store
        self._table_name = table_name

        logger.debug(f"{provider_name} docstore created")

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            str: Provider name.
        """
        return self._provider_name

    @property
    def store(self) -> BaseDocumentStore:
        """Document store.

        Returns:
            BaseDocumentStore: Document store.
        """
        return self._store

    @store.setter
    def store(self, value: BaseDocumentStore) -> None:
        """Set the document store.

        Args:
            value (BaseDocumentStore): Document store to set.
        """
        self._store = value

    @property
    def table_name(self) -> Optional[str]:
        """Table name.

        Returns:
            Optional[str]: Table name.
        """
        return self._table_name

    def get_bm25_corpus_size(self) -> int:
        """Return the number of documents stored for BM25 retrieval.

        Returns:
            int: Document count (0 if unavailable).
        """
        docs_attr = getattr(self.store, "docs", None)
        if docs_attr is None:
            return 0

        try:
            return len(docs_attr)
        except Exception:
            return sum(1 for _ in docs_attr)

    def get_ref_doc_ids(self) -> list[str]:
        """Get all ref_doc_info keys stored in the docstore.

        Returns:
            list[str]: List of ref_doc_id values.
        """
        infos = self.store.get_all_ref_doc_info()
        if infos is None:
            return []

        return list(infos.keys())

    def delete_all(self, persist_dir: Optional[Path]) -> None:
        """Delete all ref_docs and related nodes stored.

        Args:
            persist_dir (Optional[Path]): Persist directory.
        """
        from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME

        try:
            for doc_id in list(self.store.docs.keys()):
                self.store.delete_document(doc_id, raise_error=False)
        except Exception as e:
            logger.warning(f"failed to delete doc {doc_id}: {e}")
            return

        logger.info("all documents are deleted from document store")

        if persist_dir is not None:
            try:
                self.store.persist(str(persist_dir / DEFAULT_PERSIST_FNAME))
            except Exception as e:
                logger.warning(f"failed to persist: {e}")
