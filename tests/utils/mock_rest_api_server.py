from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from raggify.llama.core.schema import Modality

from .node_factory import make_sample_nodes


@dataclass
class MockServerContext:
    """Mock context container for FastAPI tests."""

    module: ModuleType
    runtime: SimpleNamespace
    worker: "DummyWorker"


def _build_runtime(upload_dir: Path) -> SimpleNamespace:
    """Create a runtime stub."""
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


class DummyWorker:
    """Minimal background worker substitute."""

    def __init__(self) -> None:
        self._jobs: dict[str, Any] = {}
        self._counter = 0

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def submit(self, payload: Any):
        self._counter += 1
        job_id = f"job-{self._counter:04d}"
        job = SimpleNamespace(
            job_id=job_id,
            payload=payload,
            status="running",
            created_at="2025-01-01",
            last_update="2025-01-01",
        )
        self._jobs[job_id] = job
        return job

    def get_jobs(self) -> dict[str, Any]:
        return self._jobs.copy()

    def get_job(self, job_id: str) -> Any:
        return self._jobs.get(job_id)

    def remove_completed_jobs(self) -> None:
        self._jobs.clear()

    def remove_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)


def _patch_queries(stack: ExitStack) -> None:
    """Patch query handlers to return sample nodes."""
    module_path = "raggify.retrieve.retrieve"

    async def _query_stub(**kwargs):  # type: ignore[unused-arg]
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
    """Patch FastAPI dependencies for tests."""
    runtime = _build_runtime(upload_dir)
    worker = DummyWorker()

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "_rt", MagicMock(return_value=runtime)))
        stack.enter_context(patch.object(module, "_wk", MagicMock(return_value=worker)))
        stack.enter_context(patch.object(module, "console", MagicMock()))
        stack.enter_context(patch.object(module, "logger", MagicMock()))
        stack.enter_context(
            patch.object(module, "configure_logging", MagicMock())
        )
        _patch_queries(stack)
        yield MockServerContext(module=module, runtime=runtime, worker=worker)
