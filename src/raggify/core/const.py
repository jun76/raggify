from __future__ import annotations

from pathlib import Path

PROJECT_NAME: str = "raggify"
PJNAME_ALIAS: str = "rg"
VERSION: str = "0.1.1"
USER_CONFIG_PATH: str = f"/etc/{PROJECT_NAME}/config.yaml"
DEFAULT_KNOWLEDGEBASE_NAME: str = "default_kb"
DEFAULT_WORKSPACE_PATH: Path = Path.home() / ".local" / "share" / PROJECT_NAME
TEMP_FILE_PREFIX = f"tmp_{PROJECT_NAME}_"
