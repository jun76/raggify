from __future__ import annotations

import importlib
import warnings

import pytest
from pydantic.warnings import PydanticDeprecatedSince211

from tests.utils.mock_client_cli import patch_client_cli, runner_client_cli
from tests.utils.mock_server_cli import patch_server_cli, runner_server

# Suppressing the __get_pydantic_core_schema__ warning in Pydantic v2.11 and later
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
pytestmark = pytest.mark.filterwarnings(
    "ignore::pydantic.warnings.PydanticDeprecatedSince211"
)


def _run_commands(runner, app, commands):
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            command,
            result.stdout,
            result.stderr,
        )


def _load_client_cli():
    module = importlib.import_module("raggify_client.cli")
    module = importlib.reload(module)
    return module.app, module


def _load_server_cli():
    module = importlib.import_module("raggify.cli.cli")
    module = importlib.reload(module)
    return module.app


def test_server_cli():
    server_commands = [
        ["version"],
        ["config"],
        ["server"],
        ["server", "--mcp"],
        ["reload"],
        ["ip", "/tmp/sample.txt"],
        ["ipl", "/tmp/list.txt"],
        ["iu", "https://some.site.com"],
        ["iul", "/tmp/urls.txt"],
    ]
    server_app = _load_server_cli()
    with patch_server_cli():
        _run_commands(runner_server, server_app, server_commands)


def test_client_cli():
    client_commands = [
        ["version"],
        ["server"],
        ["config"],
        ["stat"],
        ["reload"],
        ["job"],
        ["iu", "https://some.site.com"],
        ["iul", "/tmp/urls.txt"],
        ["qtt", "hello"],
        ["qti", "hello"],
        ["qii", "/tmp/image.png"],
        ["qta", "hello"],
        ["qaa", "/tmp/audio.wav"],
        ["qtv", "hello"],
        ["qiv", "/tmp/image.png"],
        ["qav", "/tmp/audio.wav"],
        ["qmv", "/tmp/video.mp4"],
    ]
    client_app, client_module = _load_client_cli()
    with patch_client_cli(module=client_module):
        _run_commands(runner_client_cli, client_app, client_commands)
