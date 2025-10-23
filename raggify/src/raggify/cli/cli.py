from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional, Protocol

import typer
import uvicorn

from raggify.config import cfg
from raggify.logger import logger

if TYPE_CHECKING:
    from raggify.client.client import RestAPIClient

__all__ = ["app"]


app = typer.Typer()


def _get_server_base_url() -> str:
    """raggify サーバのベース URL 文字列を取得する。

    プロトコル等修正する場合は uvicorn.run 側と形式を合わせること。

    Returns:
        str: ベース URL 文字列
    """
    return f"http://{cfg.general.host}:{cfg.general.port}"


def _create_rest_client():
    """REST API クライアントを生成する。

    Returns:
        RestAPIClient: REST API クライアント
    """
    from ..client.client import RestAPIClient

    return RestAPIClient(_get_server_base_url())


def _echo_json(data: dict[str, Any]) -> None:
    """JSON 文字列として整形出力する。

    Args:
        data (dict[str, Any]): 出力データ
    """
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


@app.command(help="Start as a local server.")
def server(
    host: str = typer.Option(cfg.general.host),
    port: int = typer.Option(cfg.general.port),
    mcp: bool = typer.Option(cfg.general.mcp),
):
    """ローカルサーバとして常駐させる。

    Args:
        host (str, optional): ホスト。Defaults to cfg.general.host.
        port (int, optional): ポート番号。Defaults to cfg.general.port.
        mcp (bool, optional): MCP サーバとして公開するか。Defaults to cfg.general.mcp.
    """
    from ..server.fastapi import app as fastapi

    if mcp:
        from ..server.mcp import app as _mcp

        _mcp.mount_http()

    logger.info(f"up server @ host = {host}, port = {port}")
    uvicorn.run(
        app=fastapi,
        host=host,
        port=port,
        log_level=cfg.general.log_level.lower(),
    )


# 以下、REST API Client のラッパーコマンドを定義


class ClientCommand(Protocol):
    def __call__(
        self, client: "RestAPIClient", *args: Any, **kwargs: Any
    ) -> dict[str, Any]: ...


def _execute_client_command(
    command_func: ClientCommand, *args: Any, **kwargs: Any
) -> None:
    try:
        client = _create_rest_client()
        result = command_func(client, *args, **kwargs)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


@app.command(help="Ingest from local path.")
def ingest_path(path: str):
    logger.info(f"path = {path}")
    _execute_client_command(lambda client: client.ingest_path(path))


@app.command(help="Ingest from local-path list.")
def ingest_path_list(list_path: str):
    logger.info(f"list_path = {list_path}")
    _execute_client_command(lambda client: client.ingest_path_list(list_path))


@app.command(help="Ingest from URL.")
def ingest_url(url: str):
    logger.info(f"url = {url}")
    _execute_client_command(lambda client: client.ingest_url(url))


@app.command(help="Ingest from URL list.")
def ingest_url_list(list_path: str):
    logger.info(f"list_path = {list_path}")
    _execute_client_command(lambda client: client.ingest_url_list(list_path))


@app.command(help="Search text documents by text query.")
def query_text_text(query: str, topk: int):
    logger.info(f"query = {query}, topk = {topk}")
    _execute_client_command(lambda client: client.query_text_text(query, topk))


@app.command(help="Search image documents by text query.")
def query_text_image(query: str, topk: int):
    logger.info(f"query = {query}, topk = {topk}")
    _execute_client_command(lambda client: client.query_text_image(query, topk))


@app.command(help="Search image documents by image query.")
def query_image_image(path: str, topk: int):
    logger.info(f"path = {path}, topk = {topk}")
    _execute_client_command(lambda client: client.query_image_image(path, topk))


@app.command(help="Search audio documents by text query.")
def query_text_audio(query: str, topk: int):
    logger.info(f"query = {query}, topk = {topk}")
    _execute_client_command(lambda client: client.query_text_audio(query, topk))


@app.command(help="Search audio documents by audio query.")
def query_audio_audio(path: str, topk: int):
    logger.info(f"path = {path}, topk = {topk}")
    _execute_client_command(lambda client: client.query_audio_audio(path, topk))


if __name__ == "__main__":
    app()
