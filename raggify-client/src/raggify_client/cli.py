from __future__ import annotations

import json
import mimetypes
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


def _parse_request_kwargs(payload: Optional[str]) -> dict[str, Any]:
    """Parse JSON payload into request kwargs.

    Args:
        payload (Optional[str]): JSON payload string.

    Returns:
        dict[str, Any]: Parsed request kwargs.
    """
    if not payload:
        return {}

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("kwargs must be a valid JSON object") from exc

    if not isinstance(data, dict):
        raise typer.BadParameter("kwargs must be a JSON object")

    return data


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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(lambda client: client.status(**parsed_kwargs))


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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.job(job_id=job_id, rm=rm, **parsed_kwargs)
    )


@app.command(name="upload", help="Upload files.")
def upload(
    paths: list[str] = typer.Argument(help="File paths to upload."),
    request_kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        help=_REQUEST_KWARG_HELP,
    ),
):
    logger.debug(f"paths = {paths}")
    if not paths:
        raise typer.BadParameter("paths must not be empty")

    files: list[tuple[str, bytes, Optional[str]]] = []
    for path in paths:
        if not os.path.exists(path):
            raise typer.BadParameter(f"file not found: {path}")

        filename = os.path.basename(path)
        if not filename:
            raise typer.BadParameter(f"invalid file path: {path}")

        with open(path, "rb") as handle:
            data = handle.read()

        content_type, _ = mimetypes.guess_type(path)
        files.append((filename, data, content_type))

    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(lambda client: client.upload(files, **parsed_kwargs))


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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.ingest_url(url, force=force, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.ingest_url_list(list_path, force=force, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_text_text(
            query=query, topk=topk, mode=mode, **parsed_kwargs
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_text_image(query, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_image_image(path, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_text_audio(query, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_audio_audio(path, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_text_video(query, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_image_video(path, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_audio_video(path, topk, **parsed_kwargs)
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
    parsed_kwargs = _parse_request_kwargs(request_kwargs)
    _execute_client_command(
        lambda client: client.query_video_video(path, topk, **parsed_kwargs)
    )


if __name__ == "__main__":
    app()
