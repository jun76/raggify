from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

runner_server = CliRunner()


def _build_cfg_mock() -> MagicMock:
    cfg = MagicMock()
    cfg.general = MagicMock()
    cfg.general.host = "localhost"
    cfg.general.port = 8000
    cfg.general.mcp = False
    cfg.general.log_level = "INFO"
    cfg.general.topk = 3
    cfg.get_dict.return_value = {}
    cfg.write_yaml.return_value = None
    return cfg


@contextmanager
def patch_server_cli() -> Iterator[None]:
    cfg_mock = _build_cfg_mock()
    client_cfg_mock = _build_cfg_mock()
    with ExitStack() as stack:
        stack.enter_context(patch("raggify.cli.cli.console", MagicMock()))
        stack.enter_context(patch("raggify.cli.cli.logger", MagicMock()))
        stack.enter_context(patch("raggify_client.cli.console", MagicMock()))
        stack.enter_context(patch("raggify_client.cli.logger", MagicMock()))
        stack.enter_context(
            patch("raggify.cli.cli._cfg", MagicMock(return_value=cfg_mock))
        )
        stack.enter_context(patch("raggify_client.cli.cfg", client_cfg_mock))
        stack.enter_context(
            patch("raggify_client.cli._create_rest_client", MagicMock())
        )
        stack.enter_context(
            patch("raggify_client.cli._execute_client_command", MagicMock())
        )
        stack.enter_context(
            patch("raggify.cli.cli._execute_client_command", MagicMock())
        )
        stack.enter_context(patch("raggify.cli.cli.uvicorn.run", MagicMock()))
        yield


@contextmanager
def patch_server_config_command() -> Iterator[None]:
    cfg_mock = _build_cfg_mock()
    cfg_mock.config_path = "/tmp/dummy.yaml"
    runtime_mock = MagicMock()
    runtime_mock.cfg = cfg_mock

    with ExitStack() as stack:
        stack.enter_context(
            patch("raggify.runtime.get_runtime", MagicMock(return_value=runtime_mock))
        )
        stack.enter_context(patch("raggify.cli.cli.console", MagicMock()))
        stack.enter_context(patch("raggify.cli.cli.logger", MagicMock()))
        stack.enter_context(patch("raggify.cli.cli._echo_json", MagicMock()))
        stack.enter_context(
            patch(
                "raggify.cli.cli.os.path.exists",
                MagicMock(side_effect=lambda path: path == "/tmp/dummy.yaml"),
            )
        )
        yield
