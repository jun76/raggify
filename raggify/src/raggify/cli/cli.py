from __future__ import annotations

import json
from typing import Any

import typer
import uvicorn

__all__ = ["app"]


app = typer.Typer()


def _get_server_base_url() -> str:
    """raggify サーバのベース URL 文字列を取得する。

    Returns:
        str: ベース URL 文字列
    """
    from raggify.config.general_config import GeneralConfig

    return f"http://{GeneralConfig.host}:{GeneralConfig.port}/v1"


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
def server(host: str, port: int, as_rest_api: bool, as_mcp: bool):
    """ローカルサーバとして常駐させる。

    Args:
        host (str): ホスト
        port (int): ポート番号
        as_rest_api (bool): REST API サーバとして公開するか
        as_mcp (bool): MCP サーバとして公開するか
    """
    if as_mcp:
        from ..server.mcp import app as mcp

        mcp.mount_http()

    if as_rest_api:
        from ..config.general_config import GeneralConfig
        from ..server.fastapi import app as fastapi

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
        client = _create_rest_client()
        result = client.ingest_path(path)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


@app.command(help="Ingest from local-path list.")
def ingest_from_path_list(list_path: str):
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): path リストのパス

    Raises:
        typer.Exit: エラー発生
    """
    try:
        client = _create_rest_client()
        result = client.ingest_path_list(list_path)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


@app.command(help="Ingest from URL.")
def ingest_from_url(url: str):
    """URL からコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        url (str): 対象 URL

    Raises:
        typer.Exit: エラー発生
    """
    try:
        client = _create_rest_client()
        result = client.ingest_url(url)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


@app.command(help="Ingest from URL list.")
def ingest_from_url_list(list_path: str):
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): URL リストのパス

    Raises:
        typer.Exit: エラー発生
    """
    try:
        client = _create_rest_client()
        result = client.ingest_url_list(list_path)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


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
        client = _create_rest_client()
        result = client.query_text_text(query=query, topk=topk)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


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
        client = _create_rest_client()
        result = client.query_text_image(query=query, topk=topk)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


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
        client = _create_rest_client()
        result = client.query_image_image(path=path, topk=topk)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


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
        client = _create_rest_client()
        result = client.query_text_audio(query=query, topk=topk)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


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
        client = _create_rest_client()
        result = client.query_audio_audio(path=path, topk=topk)
    except Exception as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)

    _echo_json(result)


if __name__ == "__main__":
    app()
