from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..document_store.document_store_manager import DocumentStoreManager
from ..embed.embed_manager import EmbedManager, Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.indices import VectorStoreIndex
    from llama_index.core.vector_stores.types import BasePydanticVectorStore


__all__ = ["VectorStoreManager", "VectorStoreContainer"]


@dataclass(kw_only=True)
class VectorStoreContainer:
    """Aggregate vector store parameters per modality."""

    provider_name: str
    store: BasePydanticVectorStore
    table_name: str
    index: Optional[VectorStoreIndex] = None


class VectorStoreManager:
    """Manager class for vector stores.

    One table is allocated per space key to manage nodes."""

    def __init__(
        self,
        conts: dict[Modality, VectorStoreContainer],
        embed: EmbedManager,
        docstore: DocumentStoreManager,
    ) -> None:
        """Constructor.

        Args:
            conts (dict[Modality, VectorStoreContainer]): Map of vector store containers.
            embed (EmbedManager): Embedding manager.
            docstore (DocumentStoreManager): Document store manager.
        """
        self._conts = conts
        self._embed = embed
        self._docstore = docstore

        for modality, cont in self._conts.items():
            cont.index = self._create_index(modality)
            logger.debug(f"{cont.provider_name} {modality} index created")

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            str: Provider name.
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """Set of modalities supported by this vector store.

        Returns:
            set[Modality]: Modalities.
        """
        return set(self._conts.keys())

    @property
    def table_names(self) -> list[str]:
        """List of table names maintained by this vector store.

        Returns:
            list[str]: Table names.
        """
        return [cont.table_name for cont in self._conts.values()]

    def get_index(self, modality: Modality) -> VectorStoreIndex:
        """Index generated from the underlying storage.

        Raises:
            RuntimeError: Index is not initialized.

        Returns:
            VectorStoreIndex: Index instance.
        """
        index = self.get_container(modality).index
        if index is None:
            raise RuntimeError(f"index for {modality} is not initialized")

        return index

    def get_container(self, modality: Modality) -> VectorStoreContainer:
        """Get the vector store container for the given modality.

        Args:
            modality (Modality): Modality.

        Raises:
            RuntimeError: Container is not initialized.

        Returns:
            VectorStoreContainer: Vector store container.
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"store {modality} is not initialized")

        return cont

    def _create_index(self, modality: Modality) -> VectorStoreIndex:
        """Create an index for the given modality.

        Args:
            modality (Modality): Modality.

        Raises:
            RuntimeError: Container is not initialized.

        Returns:
            VectorStoreIndex: Created index.
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
            case Modality.VIDEO:
                return VectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.VIDEO).store,
                    embed_model=self._embed.get_container(Modality.VIDEO).embed,
                )
            case _:
                raise RuntimeError("unexpected modality")

    def delete_nodes(self, ref_doc_ids: set[str]) -> None:
        """Delete nodes from the vector store by ref_doc_id.

        Args:
            ref_doc_ids (set[str]): Reference document IDs to delete.
        """
        for mod in self.modality:
            store = self.get_container(mod).store
            try:
                for ref_doc_id in ref_doc_ids:
                    store.delete(ref_doc_id)
            except Exception as e:
                logger.warning(f"failed to delete {ref_doc_id}: {e}")
                return

        logger.info(f"{len(ref_doc_ids)} nodes are deleted from vector store")

    def delete_all(self) -> bool:
        """Delete all nodes.

        Note that Redis does not implement clear.

        Returns:
            bool: True if the deletion succeeds.
        """
        try:
            for mod in self.modality:
                self.get_container(mod).store.clear()
        except Exception as e:
            logger.warning(f"failed to clear {mod} store: {e}")
            return False

        logger.info("all nodes are deleted from vector store")

        return True
