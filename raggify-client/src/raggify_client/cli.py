from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

import typer

from .config_manager import ConfigManager
from .const import PROJECT_BASE_NAME, PROJECT_NAME, VERSION
from .logger import configure_logging, console, logger

if TYPE_CHECKING:
    from .client import RestAPIClient

__all__ = ["app"]

cfg = ConfigManager()
configure_logging(cfg.general.log_level)

app = typer.Typer(
    help="raggify-client CLI: Interface to ingest/query knowledge into/from raggify server. "
    f"User config is {cfg.config_path}."
)

_REQUEST_KWARG_HELP = (
    "Additional request parameters as JSON string. "
    'Example: --kwargs \'{"arg1":"aaa","arg2":123}\''
)


def _create_rest_client() -> RestAPIClient:
    """Create a REST API client.

    Returns:
        RestAPIClient: REST API client instance.
    """
    from .client import RestAPIClient

    return RestAPIClient(host=cfg.general.host, port=cfg.general.port)


def _echo_json(data: dict[str, Any]) -> None:
    """Pretty-print data as JSON.

    Args:
        data (dict[str, Any]): Data to output.
    """
    console.print(json.dumps(data, ensure_ascii=False, indent=2))


@app.command(help="Show version.")
def version() -> None:
    """Version command."""
    logger.debug("")
    console.print(f"{PROJECT_NAME} version {VERSION}")


@app.command(help="(Not supported in client CLI)")
def server() -> None:
    logger.debug("")
    console.print(
        "❌ Client CLI does not support server sub command. "
        f"Please use `{PROJECT_BASE_NAME} server` instead."
    )


@app.command(help=f"Show current config file.")
def config() -> None:
    logger.debug("")
    _echo_json(cfg.get_dict())

    if not os.path.exists(cfg.config_path):
        cfg.write_yaml()


# Define wrapper commands for the REST API client


class ClientCommand(Protocol):
    def __call__(self, client: RestAPIClient, *args, **kwargs) -> dict[str, Any]: ...


def _execute_client_command(command_func: ClientCommand, *args, **kwargs) -> None:
    try:
        client = _create_rest_client()
        result = command_func(client, *args, **kwargs)
    except Exception as e:
        console.print(e)
        console.print(
            "❌ Command failed. If you haven't already started the server, "
            f"run '{PROJECT_BASE_NAME} server'."
        )
        raise typer.Exit(code=1)

    _echo_json(result)


@app.command(name="stat", help="Get server status.")
def status(
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    )
):
    logger.debug("")
    _execute_client_command(lambda client: client.status(request_kwargs=request_kwargs))


@app.command(name="reload", help=f"(Not supported in client CLI)")
def reload():
    logger.debug("")
    console.print(
        "❌ Client CLI does not support reload sub command. "
        f"Please use `{PROJECT_BASE_NAME} reload` instead."
    )


@app.command(name="job", help="Access background jobs.")
def job(
    job_id: str = typer.Argument(default="", help="Job id to get status."),
    rm: bool = typer.Option(
        default=False,
        help="With no id, all completed tasks will be removed from the job queue.",
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    logger.debug(f"id = {job_id}, rm = {rm}")
    _execute_client_command(
        lambda client: client.job(job_id=job_id, rm=rm, request_kwargs=request_kwargs)
    )


@app.command(name="ip", help=f"(Not supported in client CLI)")
def ingest_path():
    logger.debug("")
    console.print(
        "❌ Client CLI does not support ingest-path sub command. "
        f"Please use `{PROJECT_BASE_NAME} ip` instead."
    )


@app.command(name="ipl", help=f"(Not supported in client CLI)")
def ingest_path_list():
    logger.debug("")
    console.print(
        "❌ Client CLI does not support ingest-path-list sub command. "
        f"Please use `{PROJECT_BASE_NAME} ipl` instead."
    )


@app.command(name="iu", help="Ingest from Url.")
def ingest_url(
    url: str = typer.Argument(help="Target url."),
    force: bool = typer.Option(
        default=False, help="Reingest even if the source was already processed."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    logger.debug(f"url = {url}, force = {force}")
    _execute_client_command(
        lambda client: client.ingest_url(
            url, force=force, request_kwargs=request_kwargs
        )
    )


@app.command(name="iul", help="Ingest from Url List.")
def ingest_url_list(
    list_path: str = typer.Argument(
        help="Target url-list path. The list can include comment(#) or blank line."
    ),
    force: bool = typer.Option(
        default=False, help="Reingest even if the source was already processed."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    logger.debug(f"list_path = {list_path}, force = {force}")
    _execute_client_command(
        lambda client: client.ingest_url_list(
            list_path, force=force, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qtt",
    help="Query Text -> Text documents.",
)
def query_text_text(
    query: str = typer.Argument(help="Query string."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    mode: Optional[Literal["vector_only", "bm25_only", "fusion"]] = typer.Option(
        default=None, help="You can specify text retrieve mode."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    logger.debug(f"query = {query}, topk = {topk}, mode = {mode}")
    topk = topk or cfg.general.topk
    _execute_client_command(
        lambda client: client.query_text_text(
            query=query, topk=topk, mode=mode, request_kwargs=request_kwargs
        )
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
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"query = {query}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_text_image(
            query, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qii",
    help="Query Image -> Image documents.",
)
def query_image_image(
    path: str = typer.Argument(help="Query image path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_image_image(
            path, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qta",
    help="Query Text -> Audio documents.",
)
def query_text_audio(
    query: str = typer.Argument(help="Query string."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"query = {query}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_text_audio(
            query, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qaa",
    help="Query Audio -> Audio documents.",
)
def query_audio_audio(
    path: str = typer.Argument(help="Query audio path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_audio_audio(
            path, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qtv",
    help="Query Text -> Video documents.",
)
def query_text_video(
    query: str = typer.Argument(help="Query string."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"query = {query}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_text_video(
            query, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qiv",
    help="Query Image -> Video documents.",
)
def query_image_video(
    path: str = typer.Argument(help="Query image path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_image_video(
            path, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qav",
    help="Query Audio -> Video documents.",
)
def query_audio_video(
    path: str = typer.Argument(help="Query audio path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_audio_video(
            path, topk, request_kwargs=request_kwargs
        )
    )


@app.command(
    name="qmv",
    help="Query Video -> Video documents.",
)
def query_video_video(
    path: str = typer.Argument(help="Query video path."),
    topk: Optional[int] = typer.Option(
        default=None, help="Show top-k results (defaults to config)."
    ),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    topk = topk or cfg.general.topk
    logger.debug(f"path = {path}, topk = {topk}")
    _execute_client_command(
        lambda client: client.query_video_video(
            path, topk, request_kwargs=request_kwargs
        )
    )


if __name__ == "__main__":
    app()
