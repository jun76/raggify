from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.storage.docstore import BaseDocumentStore


class DocumentStoreManager:
    """Manager class for the document store."""

    def __init__(
        self,
        provider_name: str,
        store: Optional[BaseDocumentStore],
        table_name: Optional[str],
    ) -> None:
        """Constructor.

        Args:
            provider_name (str): Provider name.
            store (Optional[BaseDocumentStore]): Document store.
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
    def store(self) -> Optional[BaseDocumentStore]:
        """Document store.

        Returns:
            Optional[BaseDocumentStore]: Document store.
        """
        return self._store

    @store.setter
    def store(self, value: Optional[BaseDocumentStore]) -> None:
        """Set the document store.

        Args:
            value (Optional[BaseDocumentStore]): Document store to set.
        """
        self._store = value

    @property
    def table_name(self) -> Optional[str]:
        """Table name.

        Returns:
            Optional[str]: Table name.
        """
        return self._table_name

    def has_bm25_corpus(self) -> bool:
        """Return whether a BM25 text corpus exists.

        The default `pipe.arun(store_doc_text=True)` should populate it.

        Returns:
            bool: True if the corpus exists.
        """
        if self.store is None:
            return False

        docs_attr = getattr(self.store, "docs", None)

        if docs_attr is None:
            return False

        try:
            return len(docs_attr) > 0
        except Exception:
            # Some docstore implementations may not implement __len__;
            # treat presence as True.
            return True

    def get_ref_doc_ids(self) -> list[str]:
        """Get all ref_doc_info keys stored in the docstore.

        Returns:
            list[str]: List of ref_doc_id values.
        """
        if self.store is None:
            return []

        infos = self.store.get_all_ref_doc_info()
        if infos is None:
            return []

        return list(infos.keys())

    def delete_all(self, persist_path: Optional[str]) -> None:
        """Delete all ref_docs and related nodes stored.

        Args:
            persist_path (Optional[str]): Persist directory.
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
