import enum
import os
from dataclasses import fields
from typing import Any, Optional, get_args, get_origin

import yaml
from pydantic import SecretStr

from raggify.config.embed_config import EmbedConfig
from raggify.config.general_config import GeneralConfig
from raggify.config.ingest_config import IngestConfig
from raggify.config.rerank_config import RerankConfig
from raggify.config.settings import Settings
from raggify.config.vector_store_config import VectorStoreConfig

from ..logger import logger


class ConfigManager:
    def __init__(self) -> None:
        self._general: GeneralConfig
        self._vector_store: VectorStoreConfig
        self._embed: EmbedConfig
        self._ingest: IngestConfig
        self._rerank: RerankConfig

        self._user_config_path = f"/etc/{Settings.PROJECT_NAME}/config.yaml"

        if not os.path.exists(self._user_config_path):
            self._load_default()
            self._write_user_config()

        self._read_user_config()

    def _load_default(self) -> None:
        self._general = GeneralConfig()
        self._vector_store = VectorStoreConfig()
        self._embed = EmbedConfig()
        self._ingest = IngestConfig()
        self._rerank = RerankConfig()

    def _read_user_config(self) -> None:
        self._load_default()
        try:
            with open(self._user_config_path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except OSError as e:
            logger.warning(f"failed to read config file: {e}")
            return

        self._general = self._apply_section(data.get("general"), GeneralConfig, self._general)
        self._vector_store = self._apply_section(
            data.get("vector_store"), VectorStoreConfig, self._vector_store
        )
        self._embed = self._apply_section(data.get("embed"), EmbedConfig, self._embed)
        self._ingest = self._apply_section(data.get("ingest"), IngestConfig, self._ingest)
        self._rerank = self._apply_section(data.get("rerank"), RerankConfig, self._rerank)

    def _write_user_config(self) -> None:
        config_dir = os.path.dirname(self._user_config_path)
        try:
            os.makedirs(config_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"failed to prepare config directory: {e}")
            return

        data = {
            "general": self._serialize_dataclass(self._general),
            "vector_store": self._serialize_dataclass(self._vector_store),
            "embed": self._serialize_dataclass(self._embed),
            "ingest": self._serialize_dataclass(self._ingest),
            "rerank": self._serialize_dataclass(self._rerank),
        }

        try:
            with open(self._user_config_path, "w", encoding="utf-8") as fp:
                yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=True)
        except OSError as e:
            logger.warning(f"failed to write config file: {e}")

    @property
    def general(self) -> GeneralConfig:
        return self._general

    @property
    def vector_store(self) -> VectorStoreConfig:
        return self._vector_store

    @property
    def embed(self) -> EmbedConfig:
        return self._embed

    @property
    def ingest(self) -> IngestConfig:
        return self._ingest

    @property
    def rerank(self) -> RerankConfig:
        return self._rerank

    def reload(self) -> None:
        if os.path.exists(self._user_config_path):
            self._read_user_config()
        else:
            logger.warning(f"{self._user_config_path} is not found.")

    def _apply_section(self, section: Any, schema, default_instance):
        if not isinstance(section, dict):
            return default_instance

        values: dict[str, Any] = {}
        for field in fields(schema):
            current = getattr(default_instance, field.name)
            raw = section.get(field.name, current)

            enum_cls = self._extract_enum(field.type)
            if enum_cls is not None:
                values[field.name] = self._load_enum(enum_cls, raw, current)
            else:
                values[field.name] = raw

        try:
            return schema(**values)
        except Exception as e:
            logger.warning(
                f"failed to instantiate {schema.__name__} from config. "
                f"falling back to defaults. error: {e}"
            )
            return default_instance

    def _extract_enum(self, type_hint: Any) -> Optional[type[enum.Enum]]:
        origin = get_origin(type_hint)
        if origin is None:
            return type_hint if isinstance(type_hint, type) and issubclass(type_hint, enum.Enum) else None

        if origin is Union:
            for arg in get_args(type_hint):
                if isinstance(arg, type) and issubclass(arg, enum.Enum):
                    return arg
        return None

    def _load_enum(
        self,
        enum_cls: type[enum.Enum],
        raw: Any,
        default: enum.Enum,
    ) -> enum.Enum:
        if isinstance(raw, enum_cls):
            return raw
        if raw is None:
            return default
        try:
            return enum_cls[raw]
        except (KeyError, TypeError):
            try:
                return enum_cls(raw)
            except Exception:
                logger.warning(f"invalid enum value '{raw}' for {enum_cls.__name__}. using default.")
                return default

    def _serialize_dataclass(self, instance: Any) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for field in fields(type(instance)):
            value = getattr(instance, field.name)
            if isinstance(value, SecretStr):
                continue
            if isinstance(value, enum.Enum):
                serialized[field.name] = value.name
            else:
                serialized[field.name] = value
        return serialized
