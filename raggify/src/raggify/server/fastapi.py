from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..config import cfg
from ..ingest import ingest
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

    from raggify.embed.embed_manager import EmbedManager
    from raggify.rerank.rerank_manager import RerankManager
    from raggify.vector_store.vector_store_manager import VectorStoreManager

    from ..ingest.loader.file_loader import FileLoader
    from ..ingest.loader.html_loader import HTMLLoader
    from ..meta_store.structured.structured import Structured

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
    logger.info(f"{cfg.project_name} server is starting...")
    _setup()
    # リクエストの受付開始
    yield

    logger.info(f"{cfg.project_name} server is stopped.")


# FastAPIインスタンスを作成し、lifespanを渡す
app = FastAPI(title=cfg.project_name, version=cfg.version, lifespan=lifespan)

_embed: Optional[EmbedManager] = None
_meta_store: Optional[Structured] = None
_vector_store: Optional[VectorStoreManager] = None
_rerank: Optional[RerankManager] = None
_file_loader: Optional[FileLoader] = None
_html_loader: Optional[HTMLLoader] = None

_init_lock = threading.RLock()
_request_lock = asyncio.Lock()


def _setup(reload: bool = False) -> None:
    """各種インスタンスを生成

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.
    """
    _get_embed_manager(reload)
    _get_meta_store(reload)
    _get_vector_store(reload)
    _get_rerank_manager(reload)
    _get_file_loader(reload)
    _get_html_loader(reload)


def _get_embed_manager(reload: bool = False) -> EmbedManager:
    """埋め込み管理のインスタンスを取得する。

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.

    Raises:
        RuntimeError: インスタンス生成失敗

    Returns:
        EmbedManager: インスタンス
    """
    from ..embed.embed import create_embed_manager

    global _embed

    with _init_lock:
        if _embed is None or reload:
            try:
                _embed = create_embed_manager()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("failed to create embed manager") from e

            logger.info(f"{_embed.name} embed initialized")

    return _embed


def _get_meta_store(reload: bool = False) -> Structured:
    """メタデータ用ストアのインスタンスを取得する。

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.

    Raises:
        RuntimeError: インスタンス生成失敗

    Returns:
        Structured: インスタンス
    """
    from ..meta_store.meta_store import create_meta_store

    global _meta_store

    with _init_lock:
        if _meta_store is None or reload:
            try:
                _meta_store = create_meta_store()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("failed to create meta store") from e

            logger.info("meta store initialized")

    return _meta_store


def _get_vector_store(reload: bool = False) -> VectorStoreManager:
    """ベクトルストア管理のインスタンスを取得する。

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.

    Raises:
        RuntimeError: インスタンス生成失敗

    Returns:
        VectorStoreManager: インスタンス
    """
    from ..vector_store.vector_store import create_vector_store_manager

    global _vector_store

    with _init_lock:
        if _vector_store is None or reload:
            try:
                _vector_store = create_vector_store_manager(
                    embed=_get_embed_manager(), meta_store=_get_meta_store()
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("failed to create vector store manager") from e

            logger.info("vector store initialized")

    return _vector_store


def _get_rerank_manager(reload: bool = False) -> RerankManager:
    """リランク管理のインスタンスを取得する。

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.

    Raises:
        RuntimeError: インスタンス生成失敗

    Returns:
        RerankManager: インスタンス
    """
    from ..rerank.rerank import create_rerank_manager

    global _rerank

    with _init_lock:
        if _rerank is None or reload:
            try:
                _rerank = create_rerank_manager()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("failed to create rerank manager") from e

            logger.info(f"{_rerank.name} rerank initialized")

    return _rerank


def _get_file_loader(reload: bool = False) -> FileLoader:
    """ファイルローダーのインスタンスを取得する。

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.

    Raises:
        RuntimeError: インスタンス生成失敗

    Returns:
        FileLoader: インスタンス
    """
    from ..ingest.loader.file_loader import FileLoader

    global _file_loader

    with _init_lock:
        if _file_loader is None or reload:
            try:
                _file_loader = FileLoader(
                    chunk_size=cfg.ingest.chunk_size,
                    chunk_overlap=cfg.ingest.chunk_overlap,
                    store=_get_vector_store(),
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("failed to create file loader") from e

            logger.info("file loader initialized")

    return _file_loader


def _get_html_loader(reload: bool = False) -> HTMLLoader:
    """HTML ローダーのインスタンスを取得する。

    Args:
        reload (bool, optional): 再生成するか。Defaults to False.

    Raises:
        RuntimeError: インスタンス生成失敗

    Returns:
        HTMLLoader: インスタンス
    """
    from ..ingest.loader.html_loader import HTMLLoader

    global _html_loader

    with _init_lock:
        if _html_loader is None or reload:
            try:
                _html_loader = HTMLLoader(
                    chunk_size=cfg.ingest.chunk_size,
                    chunk_overlap=cfg.ingest.chunk_overlap,
                    file_loader=_get_file_loader(),
                    store=_get_vector_store(),
                    user_agent=cfg.ingest.user_agent,
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("failed to create HTML loader") from e

            logger.info("HTML loader initialized")

        return _html_loader


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
    logger.info("exec /v1/health")

    async with _request_lock:
        return {
            "status": "ok",
            "store": _get_vector_store().name,
            "embed": _get_embed_manager().name,
            "rerank": _get_rerank_manager().name,
        }


@app.get("/v1/reload")
async def reload() -> dict[str, Any]:
    """サーバの設定ファイルをリロードする。

    Returns:
        dict[str, Any]: 結果
    """
    logger.info("exec /v1/reload")

    cfg.reload()
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
    logger.info("exec /v1/upload")

    try:
        upload_dir = Path(cfg.ingest.upload_dir).absolute()
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"upload init failure: {e}") from e

    async with _request_lock:
        results = []
        for f in files:
            if f.filename is None:
                raise HTTPException(status_code=400, detail="filename is not specified")

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
                traceback.print_exc()
                raise HTTPException(
                    status_code=500, detail=f"upload failure: {e}"
                ) from e
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
    logger.info("exec /v1/ingest/path")

    async with _request_lock:
        try:
            await ingest.aingest_path(
                path=payload.path,
                store=_get_vector_store(),
                file_loader=_get_file_loader(),
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e

    return {"status": "ok"}


@app.post("/v1/ingest/path_list", operation_id="ingest_path_list")
async def ingest_path_list(payload: PathRequest) -> dict[str, str]:
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.info("exec /v1/ingest/path_list")

    async with _request_lock:
        try:
            await ingest.aingest_path_list(
                list_path=payload.path,
                store=_get_vector_store(),
                file_loader=_get_file_loader(),
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e

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
    logger.info("exec /v1/ingest/url")

    async with _request_lock:
        try:
            await ingest.aingest_url(
                url=payload.url,
                store=_get_vector_store(),
                html_loader=_get_html_loader(),
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e

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
    logger.info("exec /v1/ingest/url_list")

    async with _request_lock:
        try:
            await ingest.aingest_url_list(
                list_path=payload.path,
                store=_get_vector_store(),
                html_loader=_get_html_loader(),
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e

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

    logger.info("exec /v1/query/text_text")

    if Modality.TEXT not in _get_embed_manager().modality:
        raise HTTPException(
            status_code=400,
            detail="text embeddings is not available in current setting",
        )

    async with _request_lock:
        try:
            nodes = await aquery_text_text(
                query=payload.query,
                store=_get_vector_store(),
                topk=payload.topk or cfg.rerank.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e

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

    logger.info("exec /v1/query/text_image")

    if Modality.IMAGE not in _get_embed_manager().modality:
        raise HTTPException(
            status_code=400,
            detail="image embeddings is not available in current setting",
        )

    async with _request_lock:
        try:
            nodes = await aquery_text_image(
                query=payload.query,
                store=_get_vector_store(),
                topk=payload.topk or cfg.rerank.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e

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

    logger.info("exec /v1/query/image_image")

    if Modality.IMAGE not in _get_embed_manager().modality:
        raise HTTPException(
            status_code=400,
            detail="image embeddings is not available in current setting",
        )

    async with _request_lock:
        try:
            nodes = await aquery_image_image(
                path=payload.path,
                store=_get_vector_store(),
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e

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

    logger.info("exec /v1/query/text_audio")

    if Modality.AUDIO not in _get_embed_manager().modality:
        raise HTTPException(
            status_code=400,
            detail="audio embeddings is not available in current setting",
        )

    async with _request_lock:
        try:
            nodes = await aquery_text_audio(
                query=payload.query,
                store=_get_vector_store(),
                topk=payload.topk or cfg.rerank.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e

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

    logger.info("exec /v1/query/audio_audio")

    if Modality.AUDIO not in _get_embed_manager().modality:
        raise HTTPException(
            status_code=400,
            detail="audio embeddings is not available in current setting",
        )

    async with _request_lock:
        try:
            nodes = await aquery_audio_audio(
                path=payload.path,
                store=_get_vector_store(),
                topk=payload.topk or cfg.rerank.topk,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e

    return {"documents": _nodes_to_response(nodes)}
