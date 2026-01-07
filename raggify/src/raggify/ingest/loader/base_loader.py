from __future__ import annotations

from collections import defaultdict
from functools import partial

from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageNode,
    MediaResource,
    NodeRelationship,
    TextNode,
)

from ...config.ingest_config import IngestConfig
from ...core.exts import Exts
from ...core.metadata import BasicMetaData
from ...core.metadata import MetaKeys as MK
from ...core.utils import has_media
from ...llama_like.core.schema import AudioNode, VideoNode
from ...logger import logger

__all__ = ["BaseLoader"]


class BaseLoader:
    """Base loader class."""

    def __init__(self, cfg: IngestConfig) -> None:
        """Constructor.

        Args:
            cfg (IngestConfig): Ingest configuration.
        """
        self._cfg = cfg

    @staticmethod
    def _build_hierarchy_node_id(level: int, index: int, node: BaseNode) -> str:
        """Build a stable node ID for hierarchical parsing.

        Args:
            level (int): Hierarchy level index.
            index (int): Chunk index within the level.
            node (BaseNode): Source node.

        Returns:
            str: Stable node ID.
        """
        base_id = node.id_ or node.hash or "node"
        return f"{base_id}-L{level}-C{index}"

    def _build_text_hierarchy_nodes(
        self, docs: list[Document]
    ) -> tuple[list[BaseNode], list[TextNode]]:
        """Build hierarchical text nodes and store them in the docstore.

        Args:
            docs (list[Document]): Input documents.

        Returns:
            tuple[list[BaseNode], list[TextNode]]:
                Hierarchical text nodes excluding leaves and leaf text nodes for vector ingestion.
        """
        from llama_index.core.node_parser import (
            HierarchicalNodeParser,
            SentenceSplitter,
            get_leaf_nodes,
        )

        if not docs:
            return [], []

        node_parser_ids = []
        node_parser_map = {}
        for level, chunk_size in enumerate(self._cfg.hierarchy_chunk_sizes):
            node_parser_id = f"chunk_size_{chunk_size}"
            node_parser_ids.append(node_parser_id)
            node_parser_map[node_parser_id] = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=self._cfg.text_chunk_overlap,
                include_metadata=True,
                id_func=partial(self._build_hierarchy_node_id, level),
            )

        parser = HierarchicalNodeParser(
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
            include_metadata=True,
        )

        text_tree_nodes = parser.get_nodes_from_documents(docs)
        if not text_tree_nodes:
            return [], []

        text_leaf_nodes = [
            node
            for node in get_leaf_nodes(text_tree_nodes)
            if isinstance(node, TextNode)
        ]

        # Like other modalities, it suffices to register itself instead of registering
        # the parent document in ref_doc_id. As a result, only the text path undergoes
        # duplicate detection at the node level.
        for node in text_leaf_nodes:
            node.relationships[NodeRelationship.SOURCE] = node.as_related_node_info()

        # Registering the entire tree to the docstore causes even new leaves
        # to be treated as already registered during pipeline execution,
        # so we separate them here.
        leaf_ids = {node.node_id for node in text_leaf_nodes}
        text_tree_nodes = [
            node for node in text_tree_nodes if node.node_id not in leaf_ids
        ]

        return text_tree_nodes, text_leaf_nodes

    def _finalize_docs(self, docs: list[Document]) -> None:
        """Adjust metadata and finalize documents.

        Args:
            docs (list[Document]): Documents.
        """
        counters: dict[str, int] = defaultdict(int)
        for doc in docs:
            meta = BasicMetaData.from_dict(doc.metadata)

            # IPYNBReader returns all split documents with identical metadata;
            # assign chunk_no here.
            counter_key = meta.temp_file_path or meta.file_path or meta.url
            meta.chunk_no = counters[counter_key]
            counters[counter_key] += 1
            doc.metadata[MK.CHUNK_NO] = meta.chunk_no

            # Assign a unique ID;
            # subsequent runs compare hashes in IngestionPipeline and skip unchanged docs.
            doc.id_ = self._generate_doc_id(meta)
            doc.doc_id = doc.id_

            # BM25 refers to text_resource; if empty, copy .text into it.
            text_resource = getattr(doc, "text_resource", None)
            text_value = getattr(text_resource, "text", None) if text_resource else None
            if not text_value:
                try:
                    doc.text_resource = MediaResource(text=doc.text)
                except Exception as e:
                    logger.debug(
                        f"failed to set text_resource on doc {doc.doc_id}: {e}"
                    )

    def _generate_doc_id(self, meta: BasicMetaData) -> str:
        """Generate a doc_id string.

        Args:
            meta (BasicMetaData): Metadata container.

        Returns:
            str: Doc ID string.
        """
        return (
            f"{MK.FILE_PATH}:{meta.file_path}_"
            f"{MK.FILE_SIZE}:{meta.file_size}_"
            f"{MK.FILE_LASTMOD_AT}:{meta.file_lastmod_at}_"
            f"{MK.PAGE_NO}:{meta.page_no}_"
            f"{MK.ASSET_NO}:{meta.asset_no}_"
            f"{MK.CHUNK_NO}:{meta.chunk_no}_"
            f"{MK.URL}:{meta.url}_"
            f"{MK.BASE_SOURCE}:{meta.base_source}_"
            f"{MK.TEMP_FILE_PATH}:{meta.temp_file_path}"  # To identify embedded images in PDFs, etc.
        )

    async def _asplit_docs_modality(self, docs: list[Document]) -> tuple[
        list[BaseNode],
        list[TextNode],
        list[ImageNode],
        list[AudioNode],
        list[VideoNode],
    ]:
        """Split documents by modality.

        Args:
            docs (list[Document]): Input documents.

        Returns:
            tuple[
                list[BaseNode],
                list[TextNode],
                list[ImageNode],
                list[AudioNode],
                list[VideoNode],
            ]: Text tree, text leaf, image, audio, and video nodes.
        """
        self._finalize_docs(docs)

        image_nodes = []
        audio_nodes = []
        video_nodes = []
        text_docs: list[Document] = []
        for doc in docs:
            if has_media(node=doc, exts=Exts.IMAGE):
                image_nodes.append(
                    ImageNode(
                        text=doc.text,
                        image_path=doc.metadata.get(
                            MK.FILE_PATH
                        ),  # for caption transform use
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            elif has_media(node=doc, exts=Exts.AUDIO):
                audio_nodes.append(
                    AudioNode(
                        text=doc.text,
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            elif has_media(node=doc, exts=Exts.VIDEO):
                video_nodes.append(
                    VideoNode(
                        text=doc.text,
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            elif isinstance(doc, Document):
                text_docs.append(doc)
            else:
                logger.warning(f"unexpected node type {type(doc)}, skipped")

        text_tree_nodes, text_leaf_nodes = self._build_text_hierarchy_nodes(text_docs)
        logger.debug(
            f"split into {len(text_leaf_nodes)} text, "
            f"{len(image_nodes)} image, "
            f"{len(audio_nodes)} audio, "
            f"{len(video_nodes)} video nodes"
        )

        return text_tree_nodes, text_leaf_nodes, image_nodes, audio_nodes, video_nodes
