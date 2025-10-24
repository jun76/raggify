from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .. import runtime
from ..config import cfg
from ..ingest import ingest
from ..llama.core.schema import Modality
from ..logger import console, logger

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

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


class QueryMultimodalRequest(BaseModel):
    path: str
    topk: Optional[int] = None


class PathRequest(BaseModel):
    path: str


class URLRequest(BaseModel):
    url: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバ起動前後の処理用ライフスパン。

    CLI のヘルプコマンド等を軽量に済ませるために初期化処理を遅延しているが、
    サーバとして起動する場合はここで先に済ませておく。

    Args:
        app (FastAPI): サーバインスタンス
    """
    logger.setLevel(cfg.general.log_level)

    # 初期化処理
    _setup()

    # リクエストの受付開始
    yield
    console.print(f"🛑 now {cfg.project_name} server is stopped.")


# FastAPIインスタンスを作成し、lifespanを渡す
app = FastAPI(title=cfg.project_name, version=cfg.version, lifespan=lifespan)

_request_lock = asyncio.Lock()


def _setup(reload: bool = False) -> None:
    """各種インスタンスを生成

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.
    """
    console.print(f"⏳ {cfg.project_name} server is starting up.")

    if reload:
        runtime.reload()

    runtime.get_embed_manager()
    runtime.get_meta_store()
    runtime.get_vector_store()
    runtime.get_rerank_manager()
    runtime.get_file_loader()
    runtime.get_html_loader()

    console.print(f"✅ now {cfg.project_name} server is online.")


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

    async with _request_lock:
        return {
            "status": "ok",
            "store": runtime.get_vector_store().name,
            "embed": runtime.get_embed_manager().name,
            "rerank": runtime.get_rerank_manager().name,
        }


@app.get("/v1/reload")
async def reload() -> dict[str, Any]:
    """サーバの設定ファイルをリロードする。

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("exec /v1/reload")

    _setup(True)

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
        upload_dir = Path(cfg.ingest.upload_dir).absolute()
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


@app.post("/v1/ingest/path", operation_id="ingest_path")
async def ingest_path(payload: PathRequest) -> dict[str, str]:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        payload (PathRequest): 対象パス

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/path")

    async with _request_lock:
        try:
            await ingest.aingest_path(path=payload.path)
        except Exception as e:
            msg = "ingest path failure"
            logger.error(f"{msg}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=msg)

    return {"status": "ok"}


@app.post("/v1/ingest/path_list", operation_id="ingest_path_list")
async def ingest_path_list(payload: PathRequest) -> dict[str, str]:
    """パスリストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): パスリストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/path_list")

    async with _request_lock:
        try:
            await ingest.aingest_path_list(payload.path)
        except Exception as e:
            msg = "ingest path list failure"
            logger.error(f"{msg}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=msg)

    return {"status": "ok"}


@app.post("/v1/ingest/url", operation_id="ingest_url")
async def ingest_url(payload: URLRequest) -> dict[str, str]:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        payload (URLRequest): 対象 URL

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/url")

    async with _request_lock:
        try:
            await ingest.aingest_url(payload.url)
        except Exception as e:
            logger.error(f"ingest url failure: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"ingest url failure")

    return {"status": "ok"}


@app.post("/v1/ingest/url_list", operation_id="ingest_url_list")
async def ingest_url_list(payload: PathRequest) -> dict[str, str]:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("exec /v1/ingest/url_list")

    async with _request_lock:
        try:
            await ingest.aingest_url_list(payload.path)
        except Exception as e:
            msg = "ingest url list failure"
            logger.error(f"{msg}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=msg)

    return {"status": "ok"}


@app.post("/v1/query/text_text", operation_id="query_text_text")
async def query_text_text(payload: QueryTextRequest) -> dict[str, Any]:
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        payload (QueryTextRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
    """
    from ..retrieve.retrieve import aquery_text_text

    logger.debug("exec /v1/query/text_text")

    if Modality.TEXT not in runtime.get_embed_manager().modality:
        msg = "text embeddings is not available in current setting"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    async with _request_lock:
        try:
            nodes = await aquery_text_text(
                query=payload.query,
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception:
            msg = "query text text failure"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

    return {"documents": _nodes_to_response(nodes)}


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

    if Modality.IMAGE not in runtime.get_embed_manager().modality:
        msg = "image embeddings is not available in current setting"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    async with _request_lock:
        try:
            nodes = await aquery_text_image(
                query=payload.query,
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception:
            msg = "query text image failure"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

    return {"documents": _nodes_to_response(nodes)}


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

    if Modality.IMAGE not in runtime.get_embed_manager().modality:
        msg = "image embeddings is not available in current setting"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    async with _request_lock:
        try:
            nodes = await aquery_image_image(
                path=payload.path,
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception:
            msg = "query image image failure"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

    return {"documents": _nodes_to_response(nodes)}


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

    if Modality.AUDIO not in runtime.get_embed_manager().modality:
        msg = "audio embeddings is not available in current setting"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    async with _request_lock:
        try:
            nodes = await aquery_text_audio(
                query=payload.query,
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception:
            msg = "query text audio failure"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

    return {"documents": _nodes_to_response(nodes)}


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

    if Modality.AUDIO not in runtime.get_embed_manager().modality:
        msg = "audio embeddings is not available in current setting"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    async with _request_lock:
        try:
            nodes = await aquery_audio_audio(
                path=payload.path,
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception:
            msg = "query audio audio failure"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

    return {"documents": _nodes_to_response(nodes)}
