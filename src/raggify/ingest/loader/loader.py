from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from llama_index.core.schema import Document, ImageNode, MediaResource, TextNode

from ...core.exts import Exts
from ...core.metadata import BasicMetaData
from ...core.metadata import MetaKeys as MK
from ...llama.core.schema import AudioNode, VideoNode
from ...logger import logger
from ...runtime import get_runtime as _rt

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


class Loader:
    """Base loader class."""

    def __init__(self, persist_dir: Optional[Path]) -> None:
        """Constructor.

        Args:
            persist_dir (Optional[Path]): Persist directory.
        """
        self._persist_dir = persist_dir

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
            f"{MK.TEMP_FILE_PATH}:{meta.temp_file_path}"  # To identify embedded images in PDFs, etc.
        )

    async def _aparse_documents(
        self,
        docs: Sequence[Document],
        is_canceled: Callable[[], bool],
    ) -> Sequence[BaseNode]:
        """Split documents into nodes.

        Args:
            docs (Sequence[Document]): Documents.
            is_canceled (Callable[[], bool]): Cancellation flag for the job.

        Returns:
            Sequence[BaseNode]: Parsed nodes.
        """
        if not docs or is_canceled():
            return []

        rt = _rt()
        batch_size = rt.cfg.ingest.batch_size
        total_batches = (len(docs) + batch_size - 1) // batch_size
        nodes = []
        pipe = rt.build_pipeline(persist_dir=self._persist_dir)
        for idx in range(0, len(docs), batch_size):
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return nodes

            batch = docs[idx : idx + batch_size]
            prog = f"{idx // batch_size + 1}/{total_batches}"
            logger.debug(
                f"parse documents pipeline: processing batch {prog} "
                f"({len(batch)} docs)"
            )
            try:
                nodes.extend(await pipe.arun(documents=batch))
            except Exception as e:
                logger.error(f"failed to process batch {prog}, continue: {e}")

        rt.persist_pipeline(pipe=pipe, persist_dir=self._persist_dir)
        logger.debug(f"{len(docs)} docs --pipeline--> {len(nodes)} nodes")

        return nodes

    async def _asplit_docs_modality(
        self,
        docs: list[Document],
        is_canceled: Callable[[], bool],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Split documents by modality.

        Args:
            docs (list[Document]): Input documents.
            is_canceled (Callable[[], bool]): Cancellation flag for the job.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        self._finalize_docs(docs)
        nodes = await self._aparse_documents(docs=docs, is_canceled=is_canceled)

        image_nodes = []
        audio_nodes = []
        video_nodes = []
        text_nodes = []
        for node in nodes:
            if isinstance(node, TextNode) and self._is_image_node(node):
                image_nodes.append(
                    ImageNode(
                        text=node.text,
                        ref_doc_id=node.ref_doc_id,
                        metadata=node.metadata,
                    )
                )
            elif isinstance(node, TextNode) and self._is_audio_node(node):
                audio_nodes.append(
                    AudioNode(
                        text=node.text,
                        ref_doc_id=node.ref_doc_id,
                        metadata=node.metadata,
                    )
                )
            elif isinstance(node, TextNode) and self._is_video_node(node):
                video_nodes.append(
                    VideoNode(
                        text=node.text,
                        ref_doc_id=node.ref_doc_id,
                        metadata=node.metadata,
                    )
                )
            elif isinstance(node, TextNode):
                text_nodes.append(node)
            else:
                logger.warning(f"unexpected node type {type(node)}, skipped")

        return text_nodes, image_nodes, audio_nodes, video_nodes

    def _is_multimodal_node(self, node: BaseNode, exts: set[str]) -> bool:
        """Return True if the node matches multimodal extensions.

        Args:
            node (BaseNode): Target node.
            exts (set[str]): Extension set.

        Returns:
            bool: True if matched.
        """
        # Treat nodes whose file path or URL ends with specific extensions as multimodal
        path = node.metadata.get(MK.FILE_PATH, "")
        url = node.metadata.get(MK.URL, "")

        # Include those whose temp_file_path
        # (via custom readers) contains relevant extensions
        temp_file_path = node.metadata.get(MK.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, exts)
            or Exts.endswith_exts(url, exts)
            or Exts.endswith_exts(temp_file_path, exts)
        )

    def _is_image_node(self, node: BaseNode) -> bool:
        """Return True if the node is an image node.

        Args:
            node (BaseNode): Target node.

        Returns:
            bool: True when the node represents an image.
        """
        return self._is_multimodal_node(node=node, exts=Exts.IMAGE)

    def _is_audio_node(self, node: BaseNode) -> bool:
        """Return True if the node is an audio node.

        Args:
            node (BaseNode): Target node.

        Returns:
            bool: True when the node represents audio.
        """
        return self._is_multimodal_node(node=node, exts=Exts.AUDIO)

    def _is_video_node(self, node: BaseNode) -> bool:
        """Return True if the node is a video node.

        Args:
            node (BaseNode): Target node.

        Returns:
            bool: True when the node represents video.
        """
        return self._is_multimodal_node(node=node, exts=Exts.VIDEO)
