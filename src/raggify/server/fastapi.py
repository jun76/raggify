from __future__ import annotations

import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel

from ..config.retrieve_config import RetrieveMode
from ..core.const import PROJECT_NAME, VERSION
from ..llama.core.schema import Modality
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

warnings.filterwarnings(
    "ignore",
    message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
    category=UserWarning,
)


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
    """サーバ起動前後の処理用ライフスパン。

    CLI のヘルプコマンド等を軽量に済ませるために初期化処理を遅延しているが、
    サーバとして起動する場合はここで先に済ませておく。

    Args:
        app (FastAPI): サーバインスタンス
    """
    configure_logging()
    logger.setLevel(_rt().cfg.general.log_level)

    # 初期化処理
    _setup()
    wk = _wk()
    await wk.start()

    # リクエストの受付開始
    try:
        yield
    finally:
        await wk.shutdown()
        console.print(f"🛑 now {PROJECT_NAME} server is stopped.")


# FastAPIインスタンスを作成し、lifespanを渡す
app = FastAPI(title=PROJECT_NAME, version=VERSION, lifespan=lifespan)

_request_lock = asyncio.Lock()


def _setup() -> None:
    """各種インスタンスを生成"""
    console.print(f"⏳ {PROJECT_NAME} server is starting up.")
    _rt().build()
    console.print(f"✅ now {PROJECT_NAME} server is online.")


def _nodes_to_response(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """NodeWithScore リストを JSON 返却可能な辞書リストへ変換する。

    Args:
        nodes (list[NodeWithScore]): 変換対象ノード

    Returns:
        list[dict[str, Any]]: JSON 変換済みノードリスト
    """
    return [
        {"text": node.text, "metadata": node.metadata, "score": node.score}
        for node in nodes
    ]


@app.get("/v1/health")
async def health() -> dict[str, Any]:
    """サーバの稼働状態を返却する。

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("exec /v1/health")

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
    """サーバの設定ファイルをリロードする。

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("exec /v1/reload")

    _setup()

    return {"status": "ok"}


@app.post("/v1/upload", operation_id="upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """ファイルを（クライアントから）アップロードする。

    Args:
        files (list[UploadFile], optional): ファイル群。Defaults to File(...).

    Raises:
        HTTPException(500): 初期化やファイル作成に失敗
        HTTPException(400): ファイル名が空

    Returns:
        dict[str, Any]: 結果
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
    """バックグラウンドワーカーが保持するジョブの実行状態を返却する。

    Args:
        payload (JobRequest):
            job_id: ジョブ ID（未指定の場合全件）
            rm: True の場合、完了済みジョブ（job_id 未指定時）または指定ジョブを削除

    Raises:
        HTTPException(400): 不正なジョブ ID

    Returns:
        dict[str, Any]: 結果
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
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        payload (PathRequest): 対象パス

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/path")

    job = _wk().submit(JobPayload(kind="ingest_path", kwargs={"path": payload.path}))

    return {"status": "accepted", "job_id": job.job_id}


@app.post("/v1/ingest/path_list", operation_id="ingest_path_list")
async def ingest_path_list(payload: PathRequest) -> dict[str, str]:
    """パスリストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): パスリストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/path_list")

    job = _wk().submit(
        JobPayload(kind="ingest_path_list", kwargs={"lst": payload.path})
    )

    return {"status": "accepted", "job_id": job.job_id}


@app.post("/v1/ingest/url", operation_id="ingest_url")
async def ingest_url(payload: URLRequest) -> dict[str, str]:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        payload (URLRequest): 対象 URL

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/url")

    job = _wk().submit(JobPayload(kind="ingest_url", kwargs={"url": payload.url}))

    return {"status": "accepted", "job_id": job.job_id}


@app.post("/v1/ingest/url_list", operation_id="ingest_url_list")
async def ingest_url_list(payload: PathRequest) -> dict[str, str]:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/url_list")

    job = _wk().submit(JobPayload(kind="ingest_url_list", kwargs={"lst": payload.path}))

    return {"status": "accepted", "job_id": job.job_id}


async def _query_handler(
    modality: Modality, query_func: Callable, operation_name: str, **kwargs
) -> dict[str, Any]:
    """query 系コマンドの共通ハンドラ。

    Args:
        modality (Modality): モダリティ
        query_func (Callable): query 系コマンド
        operation_name (str): 表示用

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        payload (QueryTextTextRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ文字列による画像ドキュメント検索。

    Args:
        payload (QueryTextRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ画像による画像ドキュメント検索。

    Args:
        payload (QueryMultimodalRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ文字列による音声ドキュメント検索。

    Args:
        payload (QueryTextRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ音声による音声ドキュメント検索。

    Args:
        payload (QueryMultimodalRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ文字列による動画ドキュメント検索。

    Args:
        payload (QueryTextRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ画像による動画ドキュメント検索。

    Args:
        payload (QueryMultimodalRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ音声による動画ドキュメント検索。

    Args:
        payload (QueryMultimodalRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
    """クエリ動画による動画ドキュメント検索。

    Args:
        payload (QueryMultimodalRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
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
