from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional


class DummyRedisDocumentStore:
    def __init__(self, host: str, port: int, namespace: str) -> None:
        self.host = host
        self.port = port
        self.namespace = namespace

    @classmethod
    def from_host_and_port(
        cls, host: str, port: int, namespace: str
    ) -> DummyRedisDocumentStore:
        return cls(host=host, port=port, namespace=namespace)


class DummyPostgresDocumentStore:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @classmethod
    def from_params(cls, **kwargs: Any) -> DummyPostgresDocumentStore:
        return cls(kwargs)


class DummySimpleDocumentStore:
    last_from_dir: ClassVar[Optional[str]] = None

    def __init__(self) -> None:
        self.docs: Dict[str, Any] = {}
        self.persist_calls: list[str] = []
        self.loaded_from: Optional[str] = None

    @classmethod
    def from_persist_dir(cls, path: str) -> DummySimpleDocumentStore:
        inst = cls()
        inst.loaded_from = path
        cls.last_from_dir = path
        return inst

    def get_all_ref_doc_info(self) -> dict[str, Any]:
        return {k: None for k in self.docs}

    def delete_document(self, doc_id: str, raise_error: bool = False) -> None:
        self.docs.pop(doc_id, None)

    def persist(self, path: str) -> None:
        self.persist_calls.append(path)


class FakeDocStore:
    def __init__(self) -> None:
        self.docs: Dict[str, Any] = {}
        self.deleted: List[str] = []
        self.persist_paths: List[str] = []
        self.ref_info: Optional[dict[str, Any]] = None

    def get_all_ref_doc_info(self) -> Optional[dict[str, Any]]:
        if self.ref_info is not None:
            return self.ref_info
        return {doc_id: None for doc_id in self.docs}

    def delete_document(self, doc_id: str, raise_error: bool = False) -> None:
        self.deleted.append(doc_id)
        self.docs.pop(doc_id, None)

    def persist(self, path: str) -> None:
        self.persist_paths.append(path)


class MissingDocsAttrStore:
    def get_all_ref_doc_info(self) -> dict[str, Any]:
        return {}


class NoLenDocs(dict):
    def __len__(self) -> int:
        raise TypeError("len not supported")


class LenErrorDocStore:
    def __init__(self) -> None:
        self.docs = NoLenDocs({"a": None})
