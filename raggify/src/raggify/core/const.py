from __future__ import annotations

from pathlib import Path

PROJECT_NAME: str = "raggify"
PJNAME_ALIAS: str = "rg"
VERSION: str = "0.2.4"
DEFAULT_CONFIG_PATH: str = f"/etc/{PROJECT_NAME}/config.yaml"
DEFAULT_KNOWLEDGEBASE_NAME: str = "default_kb"
DEFAULT_WORKSPACE_PATH: Path = Path.home() / ".local" / "share" / PROJECT_NAME
TEMP_FILE_PREFIX = f"tmp_{PROJECT_NAME}_"
EXTRA_PKG_NOT_FOUND_MSG = (
    "{pkg} package(s) not found, please install "
    "raggify with '{extra}' extra to use {feature}: "
    "`pip install raggify[{extra}]`"
)
PKG_NOT_FOUND_MSG = "{pkg} package not found, please install as follows: `{cmd}`"
