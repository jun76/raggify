from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import raggify_client.cli as client_cli_module
from typer.testing import CliRunner

runner_client_cli = CliRunner()


def _build_cfg_mock() -> MagicMock:
    cfg = MagicMock()
    cfg.general.host = "localhost"
    cfg.general.port = 8000
    cfg.general.topk = 3
    cfg.get_dict.return_value = {}
    cfg.write_yaml.return_value = None
    return cfg


@contextmanager
def patch_client_cli(module: ModuleType | None = None) -> Iterator[ModuleType]:
    target_module = module or client_cli_module
    cfg_mock = _build_cfg_mock()
    with ExitStack() as stack:
        stack.enter_context(patch.object(target_module, "console", MagicMock()))
        stack.enter_context(patch.object(target_module, "logger", MagicMock()))
        stack.enter_context(patch.object(target_module, "cfg", cfg_mock))
        stack.enter_context(
            patch.object(target_module, "_execute_client_command", MagicMock())
        )
        stack.enter_context(patch("uvicorn.run", MagicMock()))
        yield target_module
