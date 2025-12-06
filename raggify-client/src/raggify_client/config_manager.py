from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv
from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import SerializationStrategy

from .const import DEFAULT_CLIENT_CONFIG_PATH
from .general_config import GeneralConfig

logger = logging.getLogger(__name__)

__all__ = ["ConfigManager"]


class _PathSerializationStrategy(SerializationStrategy):
    """Strategy class for Path <-> str conversion via mashumaro."""

    def serialize(self, value: Path) -> str:
        return str(value)

    def deserialize(self, value: str) -> Path:
        return Path(value).expanduser()


@dataclass(kw_only=True)
class _AppConfig(DataClassDictMixin):
    """Root config dataclass to keep all sections together."""

    general: GeneralConfig = field(default_factory=GeneralConfig)

    class Config(BaseConfig):
        serialization_strategy = {Path: _PathSerializationStrategy()}


class ConfigManager:
    """Configuration manager."""

    def __init__(self) -> None:
        load_dotenv()
        self._config = _AppConfig()

        self._config_path = (
            os.getenv("RG_CLIENT_CONFIG_PATH") or DEFAULT_CLIENT_CONFIG_PATH
        )
        if not os.path.exists(self._config_path):
            self.write_yaml()
        else:
            self.read_yaml()

    def read_yaml(self) -> None:
        """Read YAML config and map it into _AppConfig.

        Raises:
            RuntimeError: If reading fails.
        """
        try:
            with open(self._config_path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except OSError as e:
            raise RuntimeError("failed to read config file") from e

        try:
            self._config = _AppConfig.from_dict(data)
        except Exception as e:
            logger.warning(f"failed to load config, using defaults: {e}")
            self._config = _AppConfig()

    def write_yaml(self) -> None:
        """Write the current configuration as YAML."""
        config_dir = os.path.dirname(self._config_path)
        try:
            os.makedirs(config_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"failed to prepare config directory: {e}")
            return

        data = self._config.to_dict()
        try:
            with open(self._config_path, "w", encoding="utf-8") as fp:
                yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=True)
        except OSError as e:
            logger.warning(f"failed to write config file: {e}")

    @property
    def general(self) -> GeneralConfig:
        return self._config.general

    @property
    def config_path(self) -> str:
        return self._config_path

    def get_dict(self) -> dict[str, object]:
        """Get the current configuration as a dictionary.

        Returns:
            dict[str, object]: Dictionary form.
        """
        return self._config.to_dict()
