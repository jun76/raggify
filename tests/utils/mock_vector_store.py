from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from raggify.embed.embed_manager import EmbedManager, Modality


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: set[str] = set()
        self.cleared = False

    def delete(self, ref_doc_id: str) -> None:
        self.deleted.add(ref_doc_id)

    def clear(self) -> None:
        self.cleared = True


class DummyPGVectorStore(FakeVectorStore):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.params = params

    @classmethod
    def from_params(cls, **kwargs) -> "DummyPGVectorStore":
        return cls(kwargs)


class DummyChromaCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class DummyChromaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.path = path

    def get_or_create_collection(self, name: str) -> DummyChromaCollection:
        return DummyChromaCollection(name)


class DummyChromaVectorStore(FakeVectorStore):
    def __init__(self, chroma_collection: DummyChromaCollection) -> None:
        super().__init__()
        self.collection = chroma_collection


class DummyRedisVectorStore(FakeVectorStore):
    def __init__(self, redis_url: str, schema: Any) -> None:
        super().__init__()
        self.redis_url = redis_url
        self.schema = schema


class DummyIndexSchema:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DummyIndexSchema:
        return cls(value)


class DummyVectorStoreIndex:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @classmethod
    def from_vector_store(cls, **kwargs) -> DummyVectorStoreIndex:
        return cls(**kwargs)


class DummyMultiModalVectorStoreIndex(DummyVectorStoreIndex):
    @classmethod
    def from_vector_store(cls, **kwargs) -> DummyMultiModalVectorStoreIndex:
        return cls(**kwargs)


class DummyEmbedManager(EmbedManager):
    def __init__(self, dims: dict[Modality, int]):
        self._space_keys = {
            Modality.TEXT: "text_space",
            Modality.IMAGE: "image_space",
            Modality.AUDIO: "audio_space",
            Modality.VIDEO: "video_space",
        }
        self._containers = {
            mod: SimpleNamespace(dim=dim, embed=f"{mod.name.lower()}_embed")
            for mod, dim in dims.items()
        }

    @property
    def space_key_text(self) -> str:
        return self._space_keys[Modality.TEXT]

    @property
    def space_key_image(self) -> str:
        return self._space_keys[Modality.IMAGE]

    @property
    def space_key_audio(self) -> str:
        return self._space_keys[Modality.AUDIO]

    @property
    def space_key_video(self) -> str:
        return self._space_keys[Modality.VIDEO]

    def get_container(self, modality: Modality):
        return self._containers[modality]
