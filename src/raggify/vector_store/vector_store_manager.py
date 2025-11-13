from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..document_store.document_store_manager import DocumentStoreManager
from ..embed.embed_manager import EmbedManager, Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.indices import VectorStoreIndex
    from llama_index.core.vector_stores.types import BasePydanticVectorStore


@dataclass(kw_only=True)
class VectorStoreContainer:
    """モダリティ毎のベクトルストア関連パラメータを集約"""

    provider_name: str
    store: BasePydanticVectorStore
    table_name: str
    index: Optional[VectorStoreIndex] = None


class VectorStoreManager:
    """ベクトルストアの管理クラス。

    空間キーごとにテーブルを一つ割り当て、ノードを管理する想定。"""

    def __init__(
        self,
        conts: dict[Modality, VectorStoreContainer],
        embed: EmbedManager,
        docstore: DocumentStoreManager,
    ) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, VectorStoreContainer]): ベクトルストアコンテナの辞書
            embed (EmbedManager): 埋め込み管理
            docstore (DocumentStoreManager): ドキュメントストア管理
        """
        self._conts = conts
        self._embed = embed
        self._docstore = docstore

        for modality, cont in self._conts.items():
            cont.index = self._create_index(modality)
            logger.debug(f"{cont.provider_name} {modality} index created")

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """このベクトルストアがサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return set(self._conts.keys())

    @property
    def table_names(self) -> list[str]:
        """このベクトルストアが保持するテーブル名一覧。

        Returns:
            list[str]: テーブル名一覧
        """
        return [cont.table_name for cont in self._conts.values()]

    def get_index(self, modality: Modality) -> VectorStoreIndex:
        """ストレージから生成したインデックス。

        Raises:
            RuntimeError: 未初期化

        Returns:
            VectorStoreIndex: インデックス
        """
        index = self.get_container(modality).index
        if index is None:
            raise RuntimeError(f"index for {modality} is not initialized")

        return index

    def get_container(self, modality: Modality) -> VectorStoreContainer:
        """モダリティ別のベクトルストアコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            VectorStoreContainer: ベクトルストアコンテナ
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"store {modality} is not initialized")

        return cont

    def _create_index(self, modality: Modality) -> VectorStoreIndex:
        """インデックスを生成する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: コンテナ未初期化

        Returns:
            VectorStoreIndex: 生成したインデックス
        """
        from llama_index.core.indices import VectorStoreIndex
        from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

        match modality:
            case Modality.TEXT:
                return VectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.TEXT).store,
                    embed_model=self._embed.get_container(Modality.TEXT).embed,
                )
            case Modality.IMAGE:
                return MultiModalVectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.TEXT).store,
                    embed_model=self._embed.get_container(Modality.TEXT).embed,
                    image_vector_store=self.get_container(Modality.IMAGE).store,
                    image_embed_model=self._embed.get_container(Modality.IMAGE).embed,
                )
            case Modality.AUDIO:
                return VectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.AUDIO).store,
                    embed_model=self._embed.get_container(Modality.AUDIO).embed,
                )
            case Modality.MOVIE:
                return VectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.MOVIE).store,
                    embed_model=self._embed.get_container(Modality.MOVIE).embed,
                )
            case _:
                raise RuntimeError("unexpected modality")
