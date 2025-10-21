from __future__ import annotations

from asyncio import run as aiorun
from typing import Any

import typer
import uvicorn
from llama_index.core.schema import NodeWithScore

from ..config.general_config import GeneralConfig
from ..config.ingest_config import IngestConfig
from ..embed.embed import create_embed_manager
from ..ingest import ingest
from ..ingest.loader.file_loader import FileLoader
from ..ingest.loader.html_loader import HTMLLoader
from ..meta_store.meta_store import create_meta_store
from ..rerank.rerank import RerankManager, create_rerank_manager
from ..retrieve import retrieve
from ..server.fastapi import app as fastapi
from ..server.mcp import app as mcp
from ..vector_store.vector_store import create_vector_store_manager
from ..vector_store.vector_store_manager import VectorStoreManager

__all__ = ["app"]


app = typer.Typer()


def _setup_ingest() -> tuple[VectorStoreManager, FileLoader, HTMLLoader]:
    """ingest 用インスタンス生成ヘルパー。

    Returns:
        tuple[VectorStoreManager, FileLoader, HTMLLoader]: 各種インスタンス
    """
    embed = create_embed_manager()
    meta_store = create_meta_store()
    vector_store = create_vector_store_manager(embed=embed, meta_store=meta_store)
    file_loader = FileLoader(
        chunk_size=IngestConfig.chunk_size,
        chunk_overlap=IngestConfig.chunk_overlap,
        store=vector_store,
    )
    html_loader = HTMLLoader(
        chunk_size=IngestConfig.chunk_size,
        chunk_overlap=IngestConfig.chunk_overlap,
        file_loader=file_loader,
        store=vector_store,
        user_agent=IngestConfig.user_agent,
    )

    return vector_store, file_loader, html_loader


def _setup_retrieve() -> tuple[VectorStoreManager, RerankManager]:
    """retrieve 用インスタンス生成ヘルパー。

    Returns:
        tuple[VectorStoreManager, RerankManager]: 各種インスタンス
    """
    embed = create_embed_manager()
    meta_store = create_meta_store()
    vector_store = create_vector_store_manager(embed=embed, meta_store=meta_store)
    rerank = create_rerank_manager()

    return vector_store, rerank


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


@app.command(help="Start as a local server.")
def server(host: str, port: int, as_rest_api: bool, as_mcp: bool):
    """ローカルサーバとして常駐させる。

    Args:
        host (str): ホスト
        port (int): ポート番号
        as_rest_api (bool): REST API サーバとして公開するか
        as_mcp (bool): MCP サーバとして公開するか
    """
    if as_mcp:
        mcp.mount_http()

    if as_rest_api:
        uvicorn.run(
            app=fastapi,
            host=host,
            port=port,
            log_level=GeneralConfig.log_level.lower(),
        )


@app.command(help="Ingest from local path.")
def ingest_from_path(path: str):
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        path (str): 対象パス

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, file_loader, _ = _setup_ingest()
        aiorun(
            ingest.aingest_from_path(path=path, store=store, file_loader=file_loader)
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)


@app.command(help="Ingest from local path-list.")
def ingest_from_path_list(list_path: str):
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): path リストのパス

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, file_loader, _ = _setup_ingest()
        aiorun(
            ingest.aingest_from_path_list(
                list_path=list_path, store=store, file_loader=file_loader
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)


@app.command(help="Ingest from URL.")
def ingest_from_url(url: str):
    """URL からコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        url (str): 対象 URL

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, _, html_loader = _setup_ingest()
        aiorun(ingest.aingest_from_url(url=url, store=store, html_loader=html_loader))
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)


@app.command(help="Ingest from URL-list.")
def ingest_from_url_list(list_path: str):
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): URL リストのパス

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, _, html_loader = _setup_ingest()
        aiorun(
            ingest.aingest_from_url_list(
                list_path=list_path, store=store, html_loader=html_loader
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)


@app.command(help="Search text documents by text query.")
def query_text_text(query: str, topk: int):
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int): 取得件数

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, rerank = _setup_retrieve()
        nodes = aiorun(
            retrieve.aquery_text_text(
                query=query,
                store=store,
                topk=topk,
                rerank=rerank,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    print(_nodes_to_response(nodes))


@app.command(help="Search image documents by text query.")
def query_text_image(query: str, topk: int):
    """クエリ文字列による画像ドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int): 取得件数

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, rerank = _setup_retrieve()
        nodes = aiorun(
            retrieve.aquery_text_image(
                query=query,
                store=store,
                topk=topk,
                rerank=rerank,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    print(_nodes_to_response(nodes))


@app.command(help="Search image documents by image query.")
def query_image_image(path: str, topk: int):
    """クエリ画像による画像ドキュメント検索。

    Args:
        path (str): クエリ画像パス
        topk (int): 取得件数

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, _ = _setup_retrieve()
        nodes = aiorun(
            retrieve.aquery_image_image(
                path=path,
                store=store,
                topk=topk,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    print(_nodes_to_response(nodes))


@app.command(help="Search audio documents by text query.")
def query_text_audio(query: str, topk: int):
    """クエリ文字列による音声ドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int): 取得件数

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, rerank = _setup_retrieve()
        nodes = aiorun(
            retrieve.aquery_text_audio(
                query=query,
                store=store,
                topk=topk,
                rerank=rerank,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    print(_nodes_to_response(nodes))


@app.command(help="Search audio documents by audio query.")
def query_audio_audio(path: str, topk: int):
    """クエリ音声による音声ドキュメント検索。

    Args:
        path (str): クエリ音声パス
        topk (int): 取得件数

    Raises:
        typer.Exit: エラー発生
    """
    try:
        store, _ = _setup_retrieve()
        nodes = aiorun(
            retrieve.aquery_audio_audio(
                path=path,
                store=store,
                topk=topk,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    print(_nodes_to_response(nodes))


if __name__ == "__main__":
    app()
