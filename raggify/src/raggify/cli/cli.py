from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

import typer
import uvicorn

from raggify.logger import console, logger
from raggify.runtime import get_runtime

if TYPE_CHECKING:
    from raggify.client import RestAPIClient

__all__ = ["app"]

_cfg = get_runtime().cfg

logger.setLevel(_cfg.general.log_level)
app = typer.Typer(
    help="raggify CLI: Interface to ingest/query knowledge into/from raggify server. "
    f"User config is {_cfg.user_config_path}."
)


def _get_server_base_url() -> str:
    """raggify サーバのベース URL 文字列を取得する。

    プロトコル等修正する場合は uvicorn.run 側と形式を合わせること。

    Returns:
        str: ベース URL 文字列
    """
    return f"http://{_cfg.general.host}:{_cfg.general.port}/v1"


def _create_rest_client() -> RestAPIClient:
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
    console.print(json.dumps(data, ensure_ascii=False, indent=2))


@app.command(help="Show version.")
def version() -> None:
    """バージョンコマンド"""
    console.print(f"{_cfg.project_name} version {_cfg.version}")


@app.command(help="Start as a local server.")
def server(
    host: str = typer.Option(default=_cfg.general.host, help="Server hostname."),
    port: int = typer.Option(default=_cfg.general.port, help="Server port number."),
    mcp: bool = typer.Option(default=_cfg.general.mcp, help="Up server also as MCP."),
) -> None:
    """ローカルサーバとして起動する。

    Args:
        host (str, optional): ホスト。Defaults to cfg.general.host.
        port (int, optional): ポート番号。Defaults to cfg.general.port.
        mcp (bool, optional): MCP サーバとして公開するか。Defaults to cfg.general.mcp.
    """
    from ..server.fastapi import app as fastapi

    if mcp:
        from ..server.mcp import app as _mcp

        _mcp.mount_http()

    logger.debug(f"up server @ host = {host}, port = {port}")
    uvicorn.run(
        app=fastapi,
        host=host,
        port=port,
        log_level=_cfg.general.log_level.lower(),
    )


@app.command(help=f"Show current config file.")
def config() -> None:
    _echo_json(_cfg.get_dict())


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
        console.print(e)
        console.print(
            f"❌ Command failed. If you haven't already started the server, run '{_cfg.project_name} server'."
        )
        raise typer.Exit(code=1)

    _echo_json(result)


@app.command(name="stat", help="Get server status.")
def health():
    logger.debug("")
    _execute_client_command(lambda client: client.health())


@app.command(name="reload", help="Reload config file.")
def reload():
    logger.debug("")
    _execute_client_command(lambda client: client.reload())


@app.command(name="ip", help=f"Ingest from local Path.")
def ingest_path(path: str = typer.Argument(help="Target path.")):
    logger.debug(f"path = {path}")
    _execute_client_command(lambda client: client.ingest_path(path))


@app.command(name="ipl", help="Ingest from local Path List.")
def ingest_path_list(
    list_path: str = typer.Argument(
        help="Target path-list path. The list can include comment(#) or blank line."
    ),
):
    logger.debug(f"list_path = {list_path}")
    _execute_client_command(lambda client: client.ingest_path_list(list_path))


@app.command(name="iu", help="Ingest from Url.")
def ingest_url(url: str = typer.Argument(help="Target url.")):
    logger.debug(f"url = {url}")
    _execute_client_command(lambda client: client.ingest_url(url))


@app.command(name="iul", help="Ingest from Url List.")
def ingest_url_list(
    list_path: str = typer.Argument(
        help="Target url-list path. The list can include comment(#) or blank line."
    ),
):
    logger.debug(f"list_path = {list_path}")
    _execute_client_command(lambda client: client.ingest_url_list(list_path))


@app.command(
    name="qtt",
    help="Query Text -> Text documents.",
)
def query_text_text(
    query: str = typer.Argument(help="Query string."),
    topk: int = typer.Option(default=_cfg.rerank.topk, help="Show top-k results."),
):
    logger.debug(f"query = {query}, topk = {topk}")
    _execute_client_command(lambda client: client.query_text_text(query, topk))


@app.command(
    name="qti",
    help="Query Text -> Image documents.",
)
def query_text_image(
    query: str = typer.Argument(help="Query string."),
    topk: int = typer.Option(default=_cfg.rerank.topk, help="Show top-k results."),
):
    logger.debug(f"query = {query}, topk = {topk}")
    _execute_client_command(lambda client: client.query_text_image(query, topk))


@app.command(
    name="qii",
    help="Query Image -> Image documents.",
)
def query_image_image(
    path: str = typer.Argument(help="Query image path."),
    topk: int = typer.Option(default=_cfg.rerank.topk, help="Show top-k results."),
):
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(lambda client: client.query_image_image(path, topk))


@app.command(
    name="qta",
    help="Query Text -> Audio documents.",
)
def query_text_audio(
    query: str = typer.Argument(help="Query string."),
    topk: int = typer.Option(default=_cfg.rerank.topk, help="Show top-k results."),
):
    logger.debug(f"query = {query}, topk = {topk}")
    _execute_client_command(lambda client: client.query_text_audio(query, topk))


@app.command(
    name="qaa",
    help="Query Audio -> Audio documents.",
)
def query_audio_audio(
    path: str = typer.Argument(help="Query audio path."),
    topk: int = typer.Option(default=_cfg.rerank.topk, help="Show top-k results."),
):
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(lambda client: client.query_audio_audio(path, topk))


if __name__ == "__main__":
    app()
