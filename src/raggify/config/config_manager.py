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

from ..core.const import USER_CONFIG_PATH
from .document_store_config import DocumentStoreConfig
from .embed_config import EmbedConfig
from .general_config import GeneralConfig
from .ingest_cache_config import IngestCacheConfig
from .ingest_config import IngestConfig
from .rerank_config import RerankConfig
from .retrieve_config import RetrieveConfig
from .vector_store_config import VectorStoreConfig

logger = logging.getLogger(__name__)


class PathSerializationStrategy(SerializationStrategy):
    """Strategy class for Path <-> str conversion via mashumaro."""

    def serialize(self, value: Path) -> str:
        return str(value)

    def deserialize(self, value: str) -> Path:
        return Path(value).expanduser()


@dataclass(kw_only=True)
class AppConfig(DataClassDictMixin):
    """Root config dataclass to keep all sections together."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    document_store: DocumentStoreConfig = field(default_factory=DocumentStoreConfig)
    ingest_cache: IngestCacheConfig = field(default_factory=IngestCacheConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)

    class Config(BaseConfig):
        serialization_strategy = {Path: PathSerializationStrategy()}


class ConfigManager:
    """Configuration manager."""

    def __init__(self) -> None:
        load_dotenv()
        self._config = AppConfig()

        if not os.path.exists(USER_CONFIG_PATH):
            self.write_yaml()
        else:
            self.read_yaml()

    def read_yaml(self) -> None:
        """Read YAML config and map it into AppConfig.

        Raises:
            RuntimeError: If reading fails.
        """
        try:
            with open(USER_CONFIG_PATH, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except OSError as e:
            raise RuntimeError("failed to read config file") from e

        try:
            self._config = AppConfig.from_dict(data)
        except Exception as e:
            logger.warning(f"failed to load config, using defaults: {e}")
            self._config = AppConfig()

    def write_yaml(self) -> None:
        """Write the current configuration as YAML."""
        config_dir = os.path.dirname(USER_CONFIG_PATH)
        try:
            os.makedirs(config_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"failed to prepare config directory: {e}")
            return

        data = self._config.to_dict()
        try:
            with open(USER_CONFIG_PATH, "w", encoding="utf-8") as fp:
                yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=True)
        except OSError as e:
            logger.warning(f"failed to write config file: {e}")

    @property
    def general(self) -> GeneralConfig:
        return self._config.general

    @property
    def vector_store(self) -> VectorStoreConfig:
        return self._config.vector_store

    @property
    def document_store(self) -> DocumentStoreConfig:
        return self._config.document_store

    @property
    def ingest_cache(self) -> IngestCacheConfig:
        return self._config.ingest_cache

    @property
    def embed(self) -> EmbedConfig:
        return self._config.embed

    @property
    def ingest(self) -> IngestConfig:
        return self._config.ingest

    @property
    def rerank(self) -> RerankConfig:
        return self._config.rerank

    @property
    def retrieve(self) -> RetrieveConfig:
        return self._config.retrieve

    def get_dict(self) -> dict[str, object]:
        """Get the current configuration as a dictionary.

        Returns:
            dict[str, object]: Dictionary form.
        """
        return self._config.to_dict()
