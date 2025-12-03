from __future__ import annotations

from pathlib import Path
from typing import cast

from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME

from raggify.document_store.document_store_manager import DocumentStoreManager
from tests.utils.mock_document_store import (
    FakeDocStore,
    LenErrorDocStore,
    MissingDocsAttrStore,
)

from .config import configure_test_env

configure_test_env()


def _make_manager(store) -> DocumentStoreManager:
    return DocumentStoreManager(
        provider_name="dummy",
        store=cast(BaseDocumentStore, store),
        table_name="tbl",
    )


def test_name_and_store_setter():
    store = FakeDocStore()
    manager = _make_manager(store)
    assert manager.name == "dummy"
    assert manager.table_name == "tbl"
    assert manager.store is store

    new_store = FakeDocStore()
    manager.store = cast(BaseDocumentStore, new_store)
    assert manager.store is new_store


def test_has_bm25_corpus_missing_docs_attr():
    manager = _make_manager(MissingDocsAttrStore())
    assert manager.has_bm25_corpus() is False


def test_has_bm25_corpus_with_docs():
    store = FakeDocStore()
    store.docs = {"doc1": None}
    manager = _make_manager(store)
    assert manager.has_bm25_corpus() is True


def test_has_bm25_corpus_len_error():
    manager = _make_manager(LenErrorDocStore())
    assert manager.has_bm25_corpus() is True


def test_get_ref_doc_ids_none_info():
    store = FakeDocStore()
    store.ref_info = None
    manager = _make_manager(store)
    assert manager.get_ref_doc_ids() == []


def test_get_ref_doc_ids_returns_keys():
    store = FakeDocStore()
    store.ref_info = {"a": {}, "b": {}}
    manager = _make_manager(store)
    assert set(manager.get_ref_doc_ids()) == {"a", "b"}


def test_delete_all_with_persist(tmp_path):
    store = FakeDocStore()
    store.docs = {"a": None, "b": None}
    manager = _make_manager(store)

    persist_dir = Path(tmp_path)
    manager.delete_all(persist_dir=persist_dir)

    assert store.deleted == ["a", "b"]
    assert store.persist_paths == [str(persist_dir / DEFAULT_PERSIST_FNAME)]
