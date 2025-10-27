from __future__ import annotations

import enum
import logging
import os
from dataclasses import fields
from typing import Any, Optional

import yaml

from ..config.default_settings import DefaultSettings
from ..config.embed_config import EmbedConfig
from ..config.general_config import GeneralConfig
from ..config.ingest_config import IngestConfig
from ..config.meta_store_config import MetaStoreConfig
from ..config.rerank_config import RerankConfig
from ..config.vector_store_config import VectorStoreConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """設定管理クラス

    このクラスの保持する値がマスターであり、周辺クラスはそれぞれ以下の役割。
        Settings: デフォルトの設定値を定義
        *Config: 各所に設定値を公開するためのインタフェース
    """

    def __init__(self) -> None:
        """コンストラクタ"""
        self._general: GeneralConfig
        self._vector_store: VectorStoreConfig
        self._meta_store: MetaStoreConfig
        self._embed: EmbedConfig
        self._ingest: IngestConfig
        self._rerank: RerankConfig

        self._user_config_path = DefaultSettings.USER_CONFIG_PATH

        if not os.path.exists(self._user_config_path):
            self._load_default()
            self._write_user_config()

        self._read_user_config()

    def _load_default(self) -> None:
        """デフォルトの設定値を読み込む。"""
        self._general = GeneralConfig()
        self._vector_store = VectorStoreConfig()
        self._meta_store = MetaStoreConfig()
        self._embed = EmbedConfig()
        self._ingest = IngestConfig()
        self._rerank = RerankConfig()

    def _read_user_config(self) -> None:
        """設定ファイルから設定値を読み込む。"""
        self._load_default()
        try:
            with open(self._user_config_path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except OSError as e:
            logger.warning(f"failed to read config file: {e}")
            return

        self._general = self._apply_section(
            data.get("general"), GeneralConfig, self._general
        )
        self._vector_store = self._apply_section(
            data.get("vector_store"), VectorStoreConfig, self._vector_store
        )
        self._meta_store = self._apply_section(
            data.get("meta_store"), MetaStoreConfig, self._meta_store
        )
        self._embed = self._apply_section(data.get("embed"), EmbedConfig, self._embed)
        self._ingest = self._apply_section(
            data.get("ingest"), IngestConfig, self._ingest
        )
        self._rerank = self._apply_section(
            data.get("rerank"), RerankConfig, self._rerank
        )

    def _write_user_config(self) -> None:
        """現状の設定値を設定ファイルに書き出す。"""
        config_dir = os.path.dirname(self._user_config_path)
        try:
            os.makedirs(config_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"failed to prepare config directory: {e}")
            return

        data = self.get_dict()
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
    def meta_store(self) -> MetaStoreConfig:
        return self._meta_store

    @property
    def embed(self) -> EmbedConfig:
        return self._embed

    @property
    def ingest(self) -> IngestConfig:
        return self._ingest

    @property
    def rerank(self) -> RerankConfig:
        return self._rerank

    @property
    def project_name(self) -> str:
        return DefaultSettings.PROJECT_NAME

    @property
    def version(self) -> str:
        return DefaultSettings.VERSION

    @property
    def user_config_path(self) -> str:
        return DefaultSettings.USER_CONFIG_PATH

    def reload(self) -> None:
        """設定ファイルを再読み込みする。"""
        if os.path.exists(self._user_config_path):
            self._read_user_config()
        else:
            logger.warning(f"{self._user_config_path} is not found.")

    def save(self) -> None:
        """現状の設定をファイルに保存する。"""
        self._write_user_config()

    def get_dict(self) -> dict[str, Any]:
        """現状の設定を辞書形式で返す。

        Returns:
            dict[str, Any]: 設定の辞書
        """
        return {
            "general": self._serialize_dataclass(self._general),
            "vector_store": self._serialize_dataclass(self._vector_store),
            "meta_store": self._serialize_dataclass(self._meta_store),
            "embed": self._serialize_dataclass(self._embed),
            "ingest": self._serialize_dataclass(self._ingest),
            "rerank": self._serialize_dataclass(self._rerank),
        }

    def _apply_section(
        self,
        section: Any,
        schema: type[Any],
        default_instance: Any,
    ) -> Any:
        """YAML セクションを dataclass インスタンスへ適用する。

        Args:
            section (Any): YAML から読み込んだセクションデータ
            schema (type): 対象の dataclass 型
            default_instance (Any): デフォルト値として利用するインスタンス

        Returns:
            Any: セクションから生成した dataclass インスタンス
        """
        if not isinstance(section, dict):
            return default_instance

        values: dict[str, Any] = {}
        for field in fields(schema):
            current = getattr(default_instance, field.name)
            raw = section.get(field.name, current)

            if isinstance(current, enum.Enum):
                values[field.name] = self._load_enum(type(current), raw, current)
            elif isinstance(current, dict):
                values[field.name] = self._merge_dict(current, raw)
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

    def _load_enum(
        self,
        enum_cls: type[enum.Enum],
        raw: Any,
        default: enum.Enum,
    ) -> Optional[enum.Enum]:
        """列挙値に変換する。

        Args:
            enum_cls (type[enum.Enum]): 対象列挙型
            raw (Any): YAML から取得した raw 値
            default (enum.Enum): 変換失敗時に使用する既定値

        Returns:
            enum.Enum: 変換後の列挙値
        """
        if isinstance(raw, enum_cls):
            return raw

        if raw is None:
            return None

        try:
            return enum_cls[raw]
        except (KeyError, TypeError):
            try:
                return enum_cls(raw)
            except Exception:
                logger.warning(
                    f"invalid enum value '{raw}' for {enum_cls.__name__}. using default."
                )

                return default

    def _merge_dict(self, default: dict[str, Any], raw: Any) -> dict[str, Any]:
        """辞書の設定値をマージする。

        Args:
            default (dict[str, Any]): 既定値
            raw (Any): YAML から取得した raw 値

        Returns:
            dict[str, Any]: マージ後の辞書
        """
        if raw is None:
            return dict(default)

        if not isinstance(raw, dict):
            logger.warning(f"invalid dict value '{raw}'. using default.")
            return dict(default)

        return {
            **default,
            **{k: v for k, v in raw.items() if v is not None},
        }

    def _serialize_dataclass(self, instance: Any) -> dict[str, Any]:
        """dataclass インスタンスを YAML 書き込み用 dict に変換する。

        Args:
            instance (Any): 対象の dataclass インスタンス

        Returns:
            dict[str, Any]: シリアライズ済み辞書
        """
        serialized: dict[str, Any] = {}
        for field in fields(type(instance)):
            value = getattr(instance, field.name)
            if isinstance(value, enum.Enum):
                serialized[field.name] = value.name
            else:
                serialized[field.name] = value

        return serialized
