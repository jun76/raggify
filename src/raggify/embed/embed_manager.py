from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llama_index.core.settings import Settings

from ..core.util import sanitize_str
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
    from llama_index.core.schema import ImageType

    from ..llama.embeddings.multi_modal_base import AudioType, MovieType


@dataclass(kw_only=True)
class EmbedContainer:
    """モダリティ毎の埋め込み関連パラメータを集約"""

    provider_name: str
    embed: BaseEmbedding
    dim: int
    alias: str
    space_key: str = ""


class EmbedManager:
    """埋め込みの管理クラス。"""

    def __init__(self, conts: dict[Modality, EmbedContainer]) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, EmbedContainer]): 埋め込みコンテナの辞書
        """
        self._conts = conts

        for modality, cont in conts.items():
            cont.space_key = self._generate_space_key(
                provider=cont.provider_name,
                model=cont.alias,
                modality=modality,
            )
            logger.debug(f"space_key: {cont.space_key} generated")

        if Modality.TEXT in self._conts:
            Settings.embed_model = self._conts[Modality.TEXT].embed

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """この埋め込み管理がサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return set(self._conts.keys())

    @property
    def space_key_text(self) -> str:
        """テキスト埋め込みの空間キー。

        Raises:
            RuntimeError: 未初期化

        Returns:
            str: 空間キー
        """
        return self.get_container(Modality.TEXT).space_key

    @property
    def space_key_image(self) -> str:
        """画像埋め込みの空間キー。

        Raises:
            RuntimeError: 未初期化

        Returns:
            str: 空間キー
        """
        return self.get_container(Modality.IMAGE).space_key

    @property
    def space_key_audio(self) -> str:
        """音声埋め込みの空間キー。

        Raises:
            RuntimeError: 未初期化

        Returns:
            str: 空間キー
        """
        return self.get_container(Modality.AUDIO).space_key

    @property
    def space_key_movie(self) -> str:
        """動画埋め込みの空間キー。

        Raises:
            RuntimeError: 未初期化

        Returns:
            str: 空間キー
        """
        return self.get_container(Modality.MOVIE).space_key

    def get_container(self, modality: Modality) -> EmbedContainer:
        """モダリティ別の埋め込みコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            EmbedContainer: 埋め込みコンテナ
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"embed {modality} is not initialized")

        return cont

    async def aembed_text(self, texts: list[str]) -> list[Embedding]:
        """テキストの埋め込みベクトルを取得する。

        Args:
            texts (list[str]): テキスト

        Raises:
            RuntimeError: 未初期化

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        if Modality.TEXT not in self.modality:
            logger.warning("no text embedding is specified")
            return []

        embed = self.get_container(Modality.TEXT).embed

        logger.debug(f"now batch embedding {len(texts)} texts...")
        dims = await embed.aget_text_embedding_batch(texts=texts, show_progress=True)

        if dims:
            logger.debug(f"dim = {len(dims[0])}, embed {len(dims)} texts")

        return dims

    async def aembed_image(self, paths: list[ImageType]) -> list[Embedding]:
        """画像の埋め込みベクトルを取得する。

        Args:
            paths (list[ImageType]): 画像のパス（または base64 画像の直渡しでも OK）

        Raises:
            RuntimeError: 未初期化または画像埋め込み器でない

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding

        if Modality.IMAGE not in self.modality:
            logger.warning("no image embedding is specified")
            return []

        embed = self.get_container(Modality.IMAGE).embed
        if not isinstance(embed, MultiModalEmbedding):
            raise RuntimeError("multimodal embed model is required")

        logger.debug(f"now batch embedding {len(paths)} images...")
        dims = await embed.aget_image_embedding_batch(
            img_file_paths=paths, show_progress=True
        )

        if dims:
            logger.debug(f"dim = {len(dims[0])}, embed {len(dims)} images")

        return dims

    async def aembed_audio(self, paths: list[AudioType]) -> list[Embedding]:
        """音声の埋め込みベクトルを取得する。

        Args:
            paths (list[AudioType]): 音声のパス

        Raises:
            RuntimeError: 未初期化または音声埋め込み器でない

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        from ..llama.embeddings.multi_modal_base import AudioEmbedding

        if Modality.AUDIO not in self.modality:
            logger.warning("no audio embedding is specified")
            return []

        embed = self.get_container(Modality.AUDIO).embed
        if not isinstance(embed, AudioEmbedding):
            raise RuntimeError("audio embed model is required")

        logger.debug(f"now batch embedding {len(paths)} audios...")
        dims = await embed.aget_audio_embedding_batch(
            audio_file_paths=paths, show_progress=True
        )

        if dims:
            logger.debug(f"dim = {len(dims[0])}, embed {len(dims)} audios")

        return dims

    async def aembed_movie(self, paths: list[MovieType]) -> list[Embedding]:
        """動画の埋め込みベクトルを取得する。

        Args:
            paths (list[MovieType]): 動画のパス

        Raises:
            RuntimeError: 未初期化または動画埋め込み器でない

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        from ..llama.embeddings.multi_modal_base import MovieEmbedding

        if Modality.MOVIE not in self.modality:
            logger.warning("no movie embedding is specified")
            return []

        embed = self.get_container(Modality.MOVIE).embed
        if not isinstance(embed, MovieEmbedding):
            raise RuntimeError("movie embed model is required")

        logger.debug(f"now batch embedding {len(paths)} movies...")
        dims = await embed.aget_movie_embedding_batch(
            movie_file_paths=paths, show_progress=True
        )

        if dims:
            logger.debug(f"dim = {len(dims[0])}, embed {len(dims)} movies")

        return dims

    def _generate_space_key(self, provider: str, model: str, modality: Modality) -> str:
        """空間キー文字列を生成する。

        Args:
            provider (str): プロバイダ名
            model (str): モデル名
            modality (Modality): モダリティ

        Raises:
            ValueError: 長すぎる空間キー

        Returns:
            str: 空間キー文字列
        """
        # 字数節約
        mod = {
            Modality.TEXT: "te",
            Modality.IMAGE: "im",
            Modality.AUDIO: "au",
            Modality.MOVIE: "mo",
        }
        if mod.get(modality) is None:
            raise ValueError(f"unexpected modality: {modality}")

        return sanitize_str(f"{provider}_{model}_{mod[modality]}")
