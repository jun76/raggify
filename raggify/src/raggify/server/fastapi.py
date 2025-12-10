from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel

from ..config.retrieve_config import RetrieveMode
from ..core.const import PROJECT_NAME, VERSION
from ..llama_like.core.schema import Modality
from ..logger import configure_logging, console, logger
from ..runtime import get_runtime as _rt
from .background_worker import JobPayload
from .background_worker import get_worker as _wk

__all__ = ["app"]

logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("unstructured.trace").setLevel(logging.WARNING)


class QueryTextRequest(BaseModel):
    query: str
    topk: Optional[int] = None


class QueryTextTextRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    mode: Optional[RetrieveMode] = None


class QueryMultimodalRequest(BaseModel):
    path: str
    topk: Optional[int] = None


class PathRequest(BaseModel):
    path: str


class URLRequest(BaseModel):
    url: str


class JobRequest(BaseModel):
    job_id: str = ""
    rm: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan hook for pre/post server processing.

    Init processing is deferred for lightweight CLI help,
    but is performed here when starting the server.

    Args:
        app (FastAPI): Server instance.
    """
    configure_logging(_rt().cfg.general.log_level)

    # Initialization
    _setup()
    wk = _wk()
    await wk.start()

    # Begin accepting requests
    try:
        yield
    finally:
        await wk.shutdown()
        console.print(f"🛑 now {PROJECT_NAME} server is stopped.")


# Create FastAPI instance and pass lifespan handler
app = FastAPI(title=PROJECT_NAME, version=VERSION, lifespan=lifespan)

_request_lock = asyncio.Lock()


def _setup() -> None:
    """Create required instances."""
    console.print(f"⏳ {PROJECT_NAME} server is starting up.")
    _rt().build()
    console.print(f"✅ now {PROJECT_NAME} server is online.")


def _nodes_to_response(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """Convert a NodeWithScore list to a JSON-serializable list of dicts.

    Args:
        nodes (list[NodeWithScore]): Nodes to convert.

    Returns:
        list[dict[str, Any]]: Converted node list.
    """
    return [
        {"text": node.text, "metadata": node.metadata, "score": node.score}
        for node in nodes
    ]


@app.get("/v1/status")
async def status() -> dict[str, Any]:
    """Return server status.

    Returns:
        dict[str, Any]: Result.
    """
    logger.debug("exec /v1/status")

    rt = _rt()
    async with _request_lock:
        return {
            "status": "ok",
            "vector store": rt.vector_store.name,
            "embed": rt.embed_manager.name,
            "rerank": rt.rerank_manager.name,
            "ingest cache": rt.ingest_cache.name,
            "document store": rt.document_store.name,
        }


@app.get("/v1/reload")
async def reload() -> dict[str, Any]:
    """Reload the server configuration file.

    Returns:
        dict[str, Any]: Result.
    """
    logger.debug("exec /v1/reload")

    _setup()

    return {"status": "ok"}


@app.post("/v1/upload", operation_id="upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload files from a client.

    Args:
        files (list[UploadFile], optional): Files to upload. Defaults to File(...).

    Raises:
        HTTPException(500): When initialization or file creation fails.
        HTTPException(400): When filename is missing.

    Returns:
        dict[str, Any]: Result.
    """
    logger.debug("exec /v1/upload")

    try:
        upload_dir = Path(_rt().cfg.ingest.upload_dir).absolute()
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        msg = "mkdir failure"
        logger.error(f"{msg}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=msg)

    async with _request_lock:
        results = []
        for f in files:
            if f.filename is None:
                msg = "filename is not specified"
                logger.error(msg)
                raise HTTPException(status_code=400, detail=msg)

            try:
                safe = Path(f.filename).name
                path = upload_dir / safe
                async with aiofiles.open(path, "wb") as buf:
                    while True:
                        chunk = await f.read(1024 * 1024)
                        if not chunk:
                            break
                        await buf.write(chunk)
            except Exception as e:
                msg = "write failure"
                logger.error(f"{msg}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=msg)
            finally:
                await f.close()

            results.append(
                {
                    "filename": safe,
                    "content_type": f.content_type,
                    "save_path": str(path),
                }
            )

        return {"files": results}


@app.post("/v1/job")
async def job(payload: JobRequest) -> dict[str, Any]:
    """Return job status managed by the background worker.

    Args:
        payload (JobRequest):
            job_id: Job ID (all jobs if empty).
            rm: If True, remove completed jobs (when job_id is empty) or the specified job.

    Raises:
        HTTPException(400): Invalid job ID.

    Returns:
        dict[str, Any]: Result.
    """
    logger.debug("exec /v1/job")

    wk = _wk()
    async with _request_lock:
        if not payload.job_id:
            if payload.rm:
                wk.remove_completed_jobs()

            jobs = wk.get_jobs()
            res = {}
            for job_id, job in jobs.items():
                res[job_id] = job.status
        else:
            job = wk.get_job(payload.job_id)
            if job is None:
                msg = "invalid job id"
                logger.error(msg)
                raise HTTPException(status_code=400, detail=msg)

            if payload.rm:
                wk.remove_job(payload.job_id)
                res = {"status": "removed"}
            else:
                res = {
                    "status": job.status,
                    "kind": job.payload.kind,
                    "created_at": job.created_at,
                    "last_update": job.last_update,
                }
                for k, arg in job.payload.kwargs.items():
                    res[k] = arg

        return res


@app.post("/v1/ingest/path", operation_id="ingest_path")
async def ingest_path(payload: PathRequest) -> dict[str, str]:
    """Collect, embed, and store content from a local path (file or directory).

    Directories are traversed recursively to ingest multiple files.

    Args:
        payload (PathRequest): Target path.

    Returns:
        dict[str, str]: Result.
    """
    logger.debug("exec /v1/ingest/path")

    job = _wk().submit(JobPayload(kind="ingest_path", kwargs={"path": payload.path}))

    return {"status": "accepted", "job_id": job.job_id}


@app.post("/v1/ingest/path_list", operation_id="ingest_path_list")
async def ingest_path_list(payload: PathRequest) -> dict[str, str]:
    """Collect, embed, and store content from multiple paths listed in a file.

    Args:
        payload (PathRequest): Path to a list file (text file; comment lines
            starting with # and blank lines are skipped).

    Returns:
        dict[str, str]: Result.
    """
    logger.debug("exec /v1/ingest/path_list")

    job = _wk().submit(
        JobPayload(kind="ingest_path_list", kwargs={"lst": payload.path})
    )

    return {"status": "accepted", "job_id": job.job_id}


@app.post("/v1/ingest/url", operation_id="ingest_url")
async def ingest_url(payload: URLRequest) -> dict[str, str]:
    """Collect, embed, and store content from a URL.

    For sitemaps (.xml), traverse the tree to ingest multiple sites.

    Args:
        payload (URLRequest): Target URL.

    Returns:
        dict[str, str]: Result.
    """
    logger.debug("exec /v1/ingest/url")

    job = _wk().submit(JobPayload(kind="ingest_url", kwargs={"url": payload.url}))

    return {"status": "accepted", "job_id": job.job_id}


@app.post("/v1/ingest/url_list", operation_id="ingest_url_list")
async def ingest_url_list(payload: PathRequest) -> dict[str, str]:
    """Collect, embed, and store content from multiple URLs listed in a file.

    Args:
        payload (PathRequest): Path to a URL list file (text file; comment lines
            starting with # and blank lines are skipped).

    Returns:
        dict[str, str]: Result.
    """
    logger.debug("exec /v1/ingest/url_list")

    job = _wk().submit(JobPayload(kind="ingest_url_list", kwargs={"lst": payload.path}))

    return {"status": "accepted", "job_id": job.job_id}


async def _query_handler(
    modality: Modality, query_func: Callable, operation_name: str, **kwargs
) -> dict[str, Any]:
    """Common handler for query endpoints.

    Args:
        modality (Modality): Modality.
        query_func (Callable): Query function.
        operation_name (str): Operation label for logging.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    if modality not in _rt().embed_manager.modality:
        msg = f"{modality.value} embeddings is not available in current setting"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    async with _request_lock:
        try:
            nodes = await query_func(**kwargs)
        except Exception as e:
            msg = f"{operation_name} failure"
            logger.error(f"{msg}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=msg)

    return {"documents": _nodes_to_response(nodes)}


@app.post("/v1/query/text_text", operation_id="query_text_text")
async def query_text_text(payload: QueryTextTextRequest) -> dict[str, Any]:
    """Search text documents by text query.

    Args:
        payload (QueryTextTextRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_text_text

    logger.debug("exec /v1/query/text_text")

    return await _query_handler(
        modality=Modality.TEXT,
        query_func=aquery_text_text,
        operation_name="query text text",
        query=payload.query,
        topk=payload.topk,
        mode=payload.mode,
    )


@app.post("/v1/query/text_image", operation_id="query_text_image")
async def query_text_image(payload: QueryTextRequest) -> dict[str, Any]:
    """Search image documents by text query.

    Args:
        payload (QueryTextRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_text_image

    logger.debug("exec /v1/query/text_image")

    return await _query_handler(
        modality=Modality.IMAGE,
        query_func=aquery_text_image,
        operation_name="query text image",
        query=payload.query,
        topk=payload.topk,
    )


@app.post("/v1/query/image_image", operation_id="query_image_image")
async def query_image_image(payload: QueryMultimodalRequest) -> dict[str, Any]:
    """Search image documents by image query.

    Args:
        payload (QueryMultimodalRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_image_image

    logger.debug("exec /v1/query/image_image")

    return await _query_handler(
        modality=Modality.IMAGE,
        query_func=aquery_image_image,
        operation_name="query image image",
        path=payload.path,
        topk=payload.topk,
    )


@app.post("/v1/query/text_audio", operation_id="query_text_audio")
async def query_text_audio(payload: QueryTextRequest) -> dict[str, Any]:
    """Search audio documents by text query.

    Args:
        payload (QueryTextRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_text_audio

    logger.debug("exec /v1/query/text_audio")

    return await _query_handler(
        modality=Modality.AUDIO,
        query_func=aquery_text_audio,
        operation_name="query text audio",
        query=payload.query,
        topk=payload.topk,
    )


@app.post("/v1/query/audio_audio", operation_id="query_audio_audio")
async def query_audio_audio(payload: QueryMultimodalRequest) -> dict[str, Any]:
    """Search audio documents by audio query.

    Args:
        payload (QueryMultimodalRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_audio_audio

    logger.debug("exec /v1/query/audio_audio")

    return await _query_handler(
        modality=Modality.AUDIO,
        query_func=aquery_audio_audio,
        operation_name="query audio audio",
        path=payload.path,
        topk=payload.topk,
    )


@app.post("/v1/query/text_video", operation_id="query_text_video")
async def query_text_video(payload: QueryTextRequest) -> dict[str, Any]:
    """Search video documents by text query.

    Args:
        payload (QueryTextRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_text_video

    logger.debug("exec /v1/query/text_video")

    return await _query_handler(
        modality=Modality.VIDEO,
        query_func=aquery_text_video,
        operation_name="query text video",
        query=payload.query,
        topk=payload.topk,
    )


@app.post("/v1/query/image_video", operation_id="query_image_video")
async def query_image_video(payload: QueryMultimodalRequest) -> dict[str, Any]:
    """Search video documents by image query.

    Args:
        payload (QueryMultimodalRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_image_video

    logger.debug("exec /v1/query/image_video")

    return await _query_handler(
        modality=Modality.VIDEO,
        query_func=aquery_image_video,
        operation_name="query image video",
        path=payload.path,
        topk=payload.topk,
    )


@app.post("/v1/query/audio_video", operation_id="query_audio_video")
async def query_audio_video(payload: QueryMultimodalRequest) -> dict[str, Any]:
    """Search video documents by audio query.

    Args:
        payload (QueryMultimodalRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_audio_video

    logger.debug("exec /v1/query/audio_video")

    return await _query_handler(
        modality=Modality.VIDEO,
        query_func=aquery_audio_video,
        operation_name="query audio video",
        path=payload.path,
        topk=payload.topk,
    )


@app.post("/v1/query/video_video", operation_id="query_video_video")
async def query_video_video(payload: QueryMultimodalRequest) -> dict[str, Any]:
    """Search video documents by video query.

    Args:
        payload (QueryMultimodalRequest): Query content.

    Raises:
        HTTPException: When the search processing fails.

    Returns:
        dict[str, Any]: Search results.
    """
    from ..retrieve.retrieve import aquery_video_video

    logger.debug("exec /v1/query/video_video")

    return await _query_handler(
        modality=Modality.VIDEO,
        query_func=aquery_video_video,
        operation_name="query video video",
        path=payload.path,
        topk=payload.topk,
    )
