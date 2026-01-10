from __future__ import annotations

from typing import cast

from llama_index.core.storage.docstore import BaseDocumentStore

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
    assert manager.get_bm25_corpus_size() == 0


def test_has_bm25_corpus_with_docs():
    store = FakeDocStore()
    store.docs = {"doc1": None}
    manager = _make_manager(store)
    assert manager.get_bm25_corpus_size() > 0


def test_has_bm25_corpus_len_error():
    manager = _make_manager(LenErrorDocStore())
    assert manager.get_bm25_corpus_size() > 0


def test_bm25_corpus_size_counts_docs():
    store = FakeDocStore()
    store.docs = {"a": None, "b": None}
    manager = _make_manager(store)

    assert manager.get_bm25_corpus_size() == 2


def test_bm25_corpus_size_handles_len_error():
    manager = _make_manager(LenErrorDocStore())
    assert manager.get_bm25_corpus_size() == 1


def test_get_ref_doc_ids_none_info():
    store = FakeDocStore()
    manager = _make_manager(store)
    assert manager.get_ref_doc_ids() == set()


def test_get_ref_doc_ids_returns_keys():
    store = FakeDocStore()
    store.docs = {"a": None, "b": None}
    manager = _make_manager(store)
    assert manager.get_ref_doc_ids() == {"a", "b"}


def test_delete_all_with_persist(tmp_path):
    store = FakeDocStore()
    store.docs = {"a": None, "b": None}
    manager = _make_manager(store)

    manager.delete_all()

    assert store.deleted == ["a", "b"]
