from __future__ import annotations

import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Optional, Protocol

import typer
import uvicorn

from ..config.retrieve_config import RetrieveMode
from ..core.const import PROJECT_NAME, USER_CONFIG_PATH, VERSION
from ..logger import console, logger

if TYPE_CHECKING:
    from ..client import RestAPIClient
    from ..config.config_manager import ConfigManager

__all__ = ["app"]


def _cfg() -> ConfigManager:
    """Getter for lazy-loading the runtime config.

    Returns:
        ConfigManager: Config manager.
    """
    from ..runtime import get_runtime

    cfg = get_runtime().cfg
    logger.setLevel(cfg.general.log_level)

    return cfg


warnings.filterwarnings(
    "ignore",
    message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
    category=UserWarning,
)
app = typer.Typer(
    help="raggify CLI: Interface to ingest/query knowledge into/from raggify server. "
    f"User config is {USER_CONFIG_PATH}."
)


def _get_server_base_url() -> str:
    """Get the base URL string of the raggify server.

    Keep the format aligned with the settings passed to `uvicorn.run`
    if protocol, etc. are adjusted.

    Returns:
        str: Base URL string.
    """
    cfg = _cfg()

    return f"http://{cfg.general.host}:{cfg.general.port}/v1"


def _create_rest_client() -> RestAPIClient:
    """Create a REST API client.

    Returns:
        RestAPIClient: REST API client instance.
    """
    from ..client.client import RestAPIClient

    return RestAPIClient(_get_server_base_url())


def _echo_json(data: dict[str, Any]) -> None:
    """Pretty-print data as JSON.

    Args:
        data (dict[str, Any]): Data to output.
    """
    console.print(json.dumps(data, ensure_ascii=False, indent=2))


@app.command(help="Show version.")
def version() -> None:
    """Version command."""
    console.print(f"{PROJECT_NAME} version {VERSION}")


@app.command(help="Start as a local server.")
def server(
    host: Optional[str] = typer.Option(
        default=None, help="Server hostname (defaults to config)."
    ),
    port: Optional[int] = typer.Option(
        default=None, help="Server port number (defaults to config)."
    ),
    mcp: Optional[bool] = typer.Option(
        default=None, help="Up server also as MCP (defaults to config)."
    ),
) -> None:
    """Start the application as a local server.

    Args:
        host (str, optional): Hostname. Defaults to cfg.general.host.
        port (int, optional): Port number. Defaults to cfg.general.port.
        mcp (bool, optional): Whether to expose as MCP. Defaults to cfg.general.mcp.
    """
    from ..server.fastapi import app as fastapi

    logger.debug("exec cli server command")
    cfg = _cfg()
    host = host or cfg.general.host
    port = port or cfg.general.port
    mcp = mcp or cfg.general.mcp
    logger.debug(f"up server @ host = {host}, port = {port}")

    if mcp:
        from ..server.mcp import app as _mcp

        _mcp.mount_http()

    uvicorn.run(
        app=fastapi,
        host=host,
        port=port,
        log_level=cfg.general.log_level.lower(),
    )


@app.command(help=f"Show current config file.")
def config() -> None:
    cfg = _cfg()
    _echo_json(cfg.get_dict())

    if not os.path.exists(USER_CONFIG_PATH):
        cfg.write_yaml()


# Define wrapper commands for the REST API client


class ClientCommand(Protocol):
    def __call__(
        self, client: RestAPIClient, *args: Any, **kwargs: Any
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
            f"❌ Command failed. If you haven't already started the server, run '{PROJECT_NAME} server'."
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


@app.command(name="job", help="Access background jobs.")
def job(
    job_id: str = typer.Argument(default="", help="Job id to get status."),
    rm: bool = typer.Option(
        default=False,
        help="With no id, all completed tasks will be removed from the job queue.",
    ),
):
    logger.debug(f"id = {job_id}")
    _execute_client_command(lambda client: client.job(job_id=job_id, rm=rm))


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
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    mode: Optional[RetrieveMode] = typer.Option(
        default=None, help="You can specify text retrieve mode."
    ),
):
    logger.debug(f"query = {query}, topk = {topk}, mode = {mode}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(
        lambda client: client.query_text_text(query=query, topk=topk, mode=mode)
    )


@app.command(
    name="qti",
    help="Query Text -> Image documents.",
)
def query_text_image(
    query: str = typer.Argument(help="Query string."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"query = {query}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_text_image(query, topk))


@app.command(
    name="qii",
    help="Query Image -> Image documents.",
)
def query_image_image(
    path: str = typer.Argument(help="Query image path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"path = {path}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_image_image(path, topk))


@app.command(
    name="qta",
    help="Query Text -> Audio documents.",
)
def query_text_audio(
    query: str = typer.Argument(help="Query string."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"query = {query}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_text_audio(query, topk))


@app.command(
    name="qaa",
    help="Query Audio -> Audio documents.",
)
def query_audio_audio(
    path: str = typer.Argument(help="Query audio path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"path = {path}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_audio_audio(path, topk))


@app.command(
    name="qtv",
    help="Query Text -> Video documents.",
)
def query_text_video(
    query: str = typer.Argument(help="Query string."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"query = {query}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_text_video(query, topk))


@app.command(
    name="qiv",
    help="Query Image -> Video documents.",
)
def query_image_video(
    path: str = typer.Argument(help="Query image path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"path = {path}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_image_video(path, topk))


@app.command(
    name="qav",
    help="Query Audio -> Video documents.",
)
def query_audio_video(
    path: str = typer.Argument(help="Query audio path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"path = {path}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_audio_video(path, topk))


@app.command(
    name="qmv",
    help="Query Video -> Video documents.",
)
def query_video_video(
    path: str = typer.Argument(help="Query video path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
):
    logger.debug(f"path = {path}, topk = {topk}")
    topk = topk or _cfg().rerank.topk
    _execute_client_command(lambda client: client.query_video_video(path, topk))


if __name__ == "__main__":
    app()
