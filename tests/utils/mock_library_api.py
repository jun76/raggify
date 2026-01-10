from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Callable, Optional, Sequence
from unittest.mock import AsyncMock, patch

from llama_index.core.schema import NodeWithScore
import asyncio

from raggify.config.retrieve_config import RetrieveMode
from raggify.ingest.ingest import aingest_path as _real_aingest_path
from raggify.ingest.ingest import aingest_path_list as _real_aingest_path_list
from raggify.ingest.ingest import aingest_url as _real_aingest_url
from raggify.ingest.ingest import aingest_url_list as _real_aingest_url_list
from raggify.retrieve.retrieve import aquery_audio_audio as _real_aquery_audio_audio
from raggify.retrieve.retrieve import aquery_audio_video as _real_aquery_audio_video
from raggify.retrieve.retrieve import aquery_image_image as _real_aquery_image_image
from raggify.retrieve.retrieve import aquery_image_video as _real_aquery_image_video
from raggify.retrieve.retrieve import aquery_text_audio as _real_aquery_text_audio
from raggify.retrieve.retrieve import aquery_text_image as _real_aquery_text_image
from raggify.retrieve.retrieve import aquery_text_text as _real_aquery_text_text
from raggify.retrieve.retrieve import aquery_text_video as _real_aquery_text_video
from raggify.retrieve.retrieve import aquery_video_video as _real_aquery_video_video
from tests.utils.mock_document_store import DummyPostgresDocumentStore
from tests.utils.mock_ingest_cache import DummyIngestionCache, DummyPostgresKVStore
from tests.utils.mock_reader import patch_html_fetchers, patch_wikipedia_reader
from tests.utils.mock_vector_store import DummyPGVectorStore
from tests.utils.node_factory import (
    DEFAULT_PATH,
    make_sample_document,
    make_sample_nodes,
)

_EMPTY_NODES: list[NodeWithScore] = []


class DummyPipeline:
    def __init__(
        self,
        transformations,
        vector_store,
        cache,
        docstore,
        docstore_strategy,
        **kwargs,
    ) -> None:
        self.transformations = transformations
        self.vector_store = vector_store
        self.cache = cache
        self.docstore = docstore
        self.docstore_strategy = docstore_strategy
        self.disable_cache = False
        self._nodes = []

    @property
    def nodes(self):
        return self._nodes

    def reset_nodes(self) -> None:
        self._nodes = []

    async def arun(self, nodes):
        return list(nodes)

    def persist(self, path: str) -> None:
        return None


@contextmanager
def _patch_ingest_pipeline() -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "raggify.pipeline.pipeline_manager.TracablePipeline",
                DummyPipeline,
            )
        )
        stack.enter_context(
            patch(
                "llama_index.core.ingestion.pipeline.IngestionPipeline.load",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "llama_index.core.ingestion.pipeline.IngestionPipeline.persist",
                return_value=None,
            )
        )

        stack.enter_context(
            patch(
                "llama_index.core.ingestion.pipeline.IngestionPipeline.arun",
                new=AsyncMock(
                    return_value=[node.node for node in make_sample_nodes(DEFAULT_PATH)]
                ),
            )
        )
        yield


@contextmanager
def _patch_storage_backends() -> Iterator[None]:
    with ExitStack() as stack:
        import importlib
        import sys
        import types

        def _ensure_module(module_name: str, attr_name: str, attr_value: object) -> None:
            module = sys.modules.get(module_name)
            if module is None:
                module = types.ModuleType(module_name)
                sys.modules[module_name] = module
            setattr(module, attr_name, attr_value)

            parent_name, child_name = module_name.rsplit(".", 1)
            try:
                parent = importlib.import_module(parent_name)
            except Exception:
                parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)

        _ensure_module(
            "llama_index.storage.docstore.postgres",
            "PostgresDocumentStore",
            DummyPostgresDocumentStore,
        )
        _ensure_module(
            "llama_index.storage.kvstore.postgres",
            "PostgresKVStore",
            DummyPostgresKVStore,
        )
        _ensure_module(
            "llama_index.vector_stores.postgres",
            "PGVectorStore",
            DummyPGVectorStore,
        )

        stack.enter_context(
            patch("llama_index.core.ingestion.IngestionCache", DummyIngestionCache)
        )
        stack.enter_context(
            patch("llama_index.core.ingestion.cache.IngestionCache", DummyIngestionCache)
        )
        stack.enter_context(
            patch(
                "llama_index.core.ingestion.cache.DEFAULT_CACHE_NAME",
                DummyIngestionCache.default_name,
                create=True,
            )
        )
        yield


@contextmanager
def _patch_file_reader() -> Iterator[None]:
    with patch(
        "llama_index.core.readers.file.base.SimpleDirectoryReader.aload_data",
        new=AsyncMock(return_value=[make_sample_document(DEFAULT_PATH)]),
    ):
        yield


@contextmanager
def _patch_rerank() -> Iterator[None]:
    with patch(
        "llama_index.core.postprocessor.types.BaseNodePostprocessor.apostprocess_nodes",
        new=AsyncMock(return_value=make_sample_nodes()),
    ):
        yield


@contextmanager
def _patch_base_retriever() -> Iterator[None]:
    with patch(
        "llama_index.core.base.base_retriever.BaseRetriever.aretrieve",
        new=AsyncMock(return_value=make_sample_nodes()),
    ):
        yield


@contextmanager
def _patch_multimodal_text_image() -> Iterator[None]:
    with patch(
        "llama_index.core.indices.multi_modal.retriever.MultiModalVectorIndexRetriever.atext_to_image_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_multimodal_image_image() -> Iterator[None]:
    with patch(
        "llama_index.core.indices.multi_modal.retriever.MultiModalVectorIndexRetriever.aimage_to_image_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_audio_text() -> Iterator[None]:
    with patch(
        "raggify.llama_like.core.indices.multi_modal.retriever.AudioRetriever.atext_to_audio_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_audio_audio() -> Iterator[None]:
    with patch(
        "raggify.llama_like.core.indices.multi_modal.retriever.AudioRetriever.aaudio_to_audio_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_video_text() -> Iterator[None]:
    with patch(
        "raggify.llama_like.core.indices.multi_modal.retriever.VideoRetriever.atext_to_video_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_video_image() -> Iterator[None]:
    with patch(
        "raggify.llama_like.core.indices.multi_modal.retriever.VideoRetriever.aimage_to_video_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_video_audio() -> Iterator[None]:
    with patch(
        "raggify.llama_like.core.indices.multi_modal.retriever.VideoRetriever.aaudio_to_video_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


@contextmanager
def _patch_video_video() -> Iterator[None]:
    with patch(
        "raggify.llama_like.core.indices.multi_modal.retriever.VideoRetriever.avideo_to_video_retrieve",
        new=AsyncMock(return_value=_EMPTY_NODES),
    ):
        yield


def ingest_path(
    path: str,
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with _patch_storage_backends(), _patch_ingest_pipeline(), _patch_file_reader():
        asyncio.run(
            _real_aingest_path(
                path=path,
                pipe_batch_size=pipe_batch_size,
                force=force,
                is_canceled=is_canceled,
            )
        )


def ingest_path_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with _patch_storage_backends(), _patch_ingest_pipeline(), _patch_file_reader():
        asyncio.run(
            _real_aingest_path_list(
                lst=lst,
                pipe_batch_size=pipe_batch_size,
                force=force,
                is_canceled=is_canceled,
            )
        )


def ingest_url(
    url: str,
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with (
        _patch_storage_backends(),
        _patch_ingest_pipeline(),
        _patch_file_reader(),
        patch_html_fetchers(),
        patch_wikipedia_reader(),
    ):
        asyncio.run(
            _real_aingest_url(
                url=url,
                pipe_batch_size=pipe_batch_size,
                force=force,
                is_canceled=is_canceled,
            )
        )


def ingest_url_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with (
        _patch_storage_backends(),
        _patch_ingest_pipeline(),
        _patch_file_reader(),
        patch_html_fetchers(),
        patch_wikipedia_reader(),
    ):
        asyncio.run(
            _real_aingest_url_list(
                lst=lst,
                pipe_batch_size=pipe_batch_size,
                force=force,
                is_canceled=is_canceled,
            )
        )


def query_text_text(
    query: str,
    topk: Optional[int] = None,
    mode: Optional[RetrieveMode] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_base_retriever(), _patch_rerank():
        return asyncio.run(
            _real_aquery_text_text(query=query, topk=topk, mode=mode)
        )


def query_text_image(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_multimodal_text_image(), _patch_rerank():
        return asyncio.run(_real_aquery_text_image(query=query, topk=topk))


def query_image_image(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_multimodal_image_image():
        return asyncio.run(_real_aquery_image_image(path=path, topk=topk))


def query_text_audio(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_audio_text(), _patch_rerank():
        return asyncio.run(_real_aquery_text_audio(query=query, topk=topk))


def query_audio_audio(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_audio_audio():
        return asyncio.run(_real_aquery_audio_audio(path=path, topk=topk))


def query_text_video(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_text(), _patch_rerank():
        return asyncio.run(_real_aquery_text_video(query=query, topk=topk))


def query_image_video(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_image():
        return asyncio.run(_real_aquery_image_video(path=path, topk=topk))


def query_audio_video(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_audio():
        return asyncio.run(_real_aquery_audio_video(path=path, topk=topk))


def query_video_video(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_video():
        return asyncio.run(_real_aquery_video_video(path=path, topk=topk))


async def aingest_path(
    path: str,
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with _patch_storage_backends(), _patch_ingest_pipeline(), _patch_file_reader():
        await _real_aingest_path(
            path=path,
            pipe_batch_size=pipe_batch_size,
            force=force,
            is_canceled=is_canceled,
        )


async def aingest_path_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with _patch_storage_backends(), _patch_ingest_pipeline(), _patch_file_reader():
        await _real_aingest_path_list(
            lst=lst,
            pipe_batch_size=pipe_batch_size,
            force=force,
            is_canceled=is_canceled,
        )


async def aingest_url(
    url: str,
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with (
        _patch_storage_backends(),
        _patch_ingest_pipeline(),
        _patch_file_reader(),
        patch_html_fetchers(),
        patch_wikipedia_reader(),
    ):
        await _real_aingest_url(
            url=url,
            pipe_batch_size=pipe_batch_size,
            force=force,
            is_canceled=is_canceled,
        )


async def aingest_url_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    force: bool = False,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    with (
        _patch_storage_backends(),
        _patch_ingest_pipeline(),
        _patch_file_reader(),
        patch_html_fetchers(),
        patch_wikipedia_reader(),
    ):
        await _real_aingest_url_list(
            lst=lst,
            pipe_batch_size=pipe_batch_size,
            force=force,
            is_canceled=is_canceled,
        )


async def aquery_text_text(
    query: str,
    topk: Optional[int] = None,
    mode: Optional[RetrieveMode] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_base_retriever(), _patch_rerank():
        return await _real_aquery_text_text(query=query, topk=topk, mode=mode)


async def aquery_text_image(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_multimodal_text_image(), _patch_rerank():
        return await _real_aquery_text_image(query=query, topk=topk)


async def aquery_image_image(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_multimodal_image_image():
        return await _real_aquery_image_image(path=path, topk=topk)


async def aquery_text_audio(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_audio_text(), _patch_rerank():
        return await _real_aquery_text_audio(query=query, topk=topk)


async def aquery_audio_audio(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_audio_audio():
        return await _real_aquery_audio_audio(path=path, topk=topk)


async def aquery_text_video(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_text(), _patch_rerank():
        return await _real_aquery_text_video(query=query, topk=topk)


async def aquery_image_video(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_image():
        return await _real_aquery_image_video(path=path, topk=topk)


async def aquery_audio_video(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_audio():
        return await _real_aquery_audio_video(path=path, topk=topk)


async def aquery_video_video(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    with _patch_storage_backends(), _patch_video_video():
        return await _real_aquery_video_video(path=path, topk=topk)
