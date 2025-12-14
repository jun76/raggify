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

    def get_ref_doc_ids(self) -> set[str]:
        """Get all ref_doc_id values stored in the docstore.

        Returns:
            set[str]: Ref doc IDs known to the store.
        """
        docs_attr = getattr(self.store, "docs", None)
        if docs_attr:
            try:
                return set(docs_attr.keys())
            except Exception:
                return set()

        infos = self.store.get_all_ref_doc_info()
        if infos:
            return set(infos.keys())

        return set()

    def is_known_source(self, source: str) -> bool:
        """Check if the source is known based on ref_doc_ids.

        Args:
            source (str): File path or URL.

        Returns:
            bool: True if the source is known, False otherwise.
        """
        from ..core.metadata import MetaKeys as MK

        def doc_id_mask(key: str, value: str) -> str:
            return f"{key}:{value}"

        flattened = "".join(self.get_ref_doc_ids())
        file_path_mask = doc_id_mask(MK.FILE_PATH, source)
        url_mask = doc_id_mask(MK.URL, source)
        base_source_mask = doc_id_mask(MK.BASE_SOURCE, source)
        if (
            file_path_mask in flattened
            or url_mask in flattened
            or base_source_mask in flattened
        ):
            return True

        return False

    def delete_nodes(self, ref_doc_ids: set[str], persist_dir: Optional[Path]) -> None:
        """Delete ref_docs and related nodes stored.

        Args:
            ref_doc_ids (set[str]): Reference document IDs to delete.
            persist_dir (Optional[Path]): Persist directory.
        """
        try:
            sorted_ref_doc_ids = sorted(ref_doc_ids)
            for ref_doc_id in sorted_ref_doc_ids:
                self.store.delete_ref_doc(ref_doc_id)
        except Exception as e:
            logger.error(f"failed to delete ref_doc {ref_doc_id}: {e}")
            return

        logger.info(f"{len(ref_doc_ids)} documents are deleted from document store")

        if persist_dir is not None:
            try:
                from llama_index.core.storage.docstore.types import (
                    DEFAULT_PERSIST_FNAME,
                )

                self.store.persist(str(persist_dir / DEFAULT_PERSIST_FNAME))
            except Exception as e:
                logger.error(f"failed to persist: {e}")

    def delete_all(self, persist_dir: Optional[Path]) -> None:
        """Delete all ref_docs and related nodes stored.

        Args:
            persist_dir (Optional[Path]): Persist directory.
        """
        ref_doc_ids = self.get_ref_doc_ids()
        self.delete_nodes(ref_doc_ids=ref_doc_ids, persist_dir=persist_dir)
