from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from raggify.document_store.document_store_manager import DocumentStoreManager
from raggify.llama_like.core.schema import Modality
from raggify.vector_store.vector_store_manager import (
    VectorStoreContainer,
    VectorStoreManager,
)
from tests.utils.mock_vector_store import (
    DummyEmbedManager,
    DummyVectorStoreIndex,
    FakeVectorStore,
)

from .config import configure_test_env

configure_test_env()


def _make_docstore() -> DocumentStoreManager:
    return cast(DocumentStoreManager, SimpleNamespace(store=SimpleNamespace()))


def _make_manager(modalities: dict[Modality, FakeVectorStore]) -> VectorStoreManager:
    embed = DummyEmbedManager({mod: 16 for mod in modalities})
    conts = {}
    for mod, store in modalities.items():
        container = VectorStoreContainer(
            provider_name="dummy",
            store=cast(BasePydanticVectorStore, store),
            table_name=f"table_{mod.value}",
        )
        conts[mod] = container
    return VectorStoreManager(conts=conts, embed=embed, docstore=_make_docstore())


@pytest.fixture(autouse=True)
def patch_indices(monkeypatch):
    monkeypatch.setattr(
        "llama_index.core.indices.VectorStoreIndex",
        DummyVectorStoreIndex,
    )
    monkeypatch.setattr(
        "llama_index.core.indices.multi_modal.MultiModalVectorStoreIndex",
        DummyVectorStoreIndex,
    )


def test_vector_store_manager_initializes_indices():
    store = FakeVectorStore()
    manager = _make_manager({Modality.TEXT: store})

    assert manager.name == "dummy"
    assert manager.modality == {Modality.TEXT}
    assert manager.table_names == ["table_text"]
    index = manager.get_index(Modality.TEXT)
    assert isinstance(index, DummyVectorStoreIndex)


def test_vector_store_manager_get_container_missing():
    manager = _make_manager({Modality.TEXT: FakeVectorStore()})
    with pytest.raises(RuntimeError):
        manager.get_container(Modality.IMAGE)


def test_vector_store_manager_delete_nodes_and_all():
    store = FakeVectorStore()
    manager = _make_manager({Modality.TEXT: store})

    manager.delete_nodes({"doc1", "doc2"})
    assert store.deleted == {"doc1", "doc2"}

    assert manager.delete_all() is True
    assert store.cleared is True
