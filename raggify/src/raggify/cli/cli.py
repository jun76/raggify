from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional, Protocol

import typer
import uvicorn
from raggify_client import app as client_app

from ..core.const import DEFAULT_CONFIG_PATH, PROJECT_NAME, VERSION
from ..logger import configure_logging, console, logger

if TYPE_CHECKING:
    from raggify_client import RestAPIClient

    from ..config.config_manager import ConfigManager

__all__ = ["app"]

_HELP_MSG = (
    "raggify CLI: Interface to ingest/query knowledge into/from raggify server. "
    "User config is {config_path}."
)

# Re-use the client app
# Override help message and add server commands
app = client_app
app.info.help = _HELP_MSG.format(config_path=DEFAULT_CONFIG_PATH)
configure_logging()


def _cfg() -> ConfigManager:
    """Getter for lazy-loading the runtime config.

    Returns:
        ConfigManager: Config manager.
    """
    from ..runtime import get_runtime

    cfg = get_runtime().cfg
    app.info.help = _HELP_MSG.format(config_path=cfg.config_path)
    configure_logging(cfg.general.log_level)

    return cfg


def _create_rest_client() -> RestAPIClient:
    """Create a REST API client.

    Returns:
        RestAPIClient: REST API client instance.
    """
    from raggify_client import RestAPIClient

    cfg = _cfg()

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

    logger.debug(f"host = {host}, port = {port}, mcp = {mcp}")
    cfg = _cfg()
    host = host or cfg.general.host
    port = port or cfg.general.port
    mcp = mcp or cfg.general.mcp

    if mcp:
        from ..server.mcp import app as _mcp

        logger.debug("mount MCP HTTP server")
        _mcp.mount_http()

    uvicorn.run(
        app=fastapi,
        host=host,
        port=port,
        log_level=cfg.general.log_level.lower(),
    )


@app.command(help=f"Show current config file.")
def config() -> None:
    logger.debug("")
    cfg = _cfg()
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
            f"run '{PROJECT_NAME} server'."
        )
        raise typer.Exit(code=1)

    _echo_json(result)


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


if __name__ == "__main__":
    app()
