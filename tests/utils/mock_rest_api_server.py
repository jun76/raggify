from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from raggify.llama_like.core.schema import Modality

from .node_factory import make_sample_nodes


@dataclass
class MockServerContext:
    module: ModuleType
    runtime: SimpleNamespace
    worker: Any


def _build_runtime(upload_dir: Path) -> SimpleNamespace:
    general = SimpleNamespace(
        host="localhost",
        port=8000,
        log_level="INFO",
        topk=3,
        mcp=False,
    )
    ingest = SimpleNamespace(upload_dir=str(upload_dir))
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(general=general, ingest=ingest),
        vector_store=SimpleNamespace(name="vector"),
        embed_manager=SimpleNamespace(name="embed", modality=set(Modality)),
        rerank_manager=SimpleNamespace(name="rerank"),
        ingest_cache=SimpleNamespace(name="ingest-cache"),
        document_store=SimpleNamespace(name="document-store"),
    )
    runtime.build = MagicMock()
    return runtime


def _patch_queries(stack: ExitStack) -> None:
    module_path = "raggify.retrieve.retrieve"

    async def _query_stub(**kwargs):
        return make_sample_nodes()

    targets = [
        "aquery_text_text",
        "aquery_text_image",
        "aquery_image_image",
        "aquery_text_audio",
        "aquery_audio_audio",
        "aquery_text_video",
        "aquery_image_video",
        "aquery_audio_video",
        "aquery_video_video",
    ]
    for name in targets:
        stack.enter_context(
            patch(f"{module_path}.{name}", AsyncMock(side_effect=_query_stub))
        )


@contextmanager
def patch_rest_api_server(
    *, module: ModuleType, upload_dir: Path
) -> Iterator[MockServerContext]:
    runtime = _build_runtime(upload_dir)

    from raggify.server import background_worker as bg

    bg._worker = None

    ingest_patch_targets = [
        "aingest_path",
        "aingest_path_list",
        "aingest_url",
        "aingest_url_list",
    ]
    worker_holder: dict[str, Any] = {}

    def worker_factory():
        worker = bg.get_worker()
        worker_holder["worker"] = worker
        return worker

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(module, "_rt", MagicMock(return_value=runtime))
        )
        stack.enter_context(patch.object(module, "console", MagicMock()))
        stack.enter_context(patch.object(module, "logger", MagicMock()))
        stack.enter_context(patch.object(module, "configure_logging", MagicMock()))
        _patch_queries(stack)
        for name in ingest_patch_targets:
            stack.enter_context(
                patch(
                    f"raggify.ingest.ingest.{name}",
                    AsyncMock(return_value=None),
                )
            )
        stack.enter_context(patch.object(module, "_wk", worker_factory))
        yield MockServerContext(
            module=module, runtime=runtime, worker=worker_holder.get("worker")
        )
