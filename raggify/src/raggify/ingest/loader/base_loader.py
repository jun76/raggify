from __future__ import annotations

from collections import defaultdict

from llama_index.core.schema import Document, ImageNode, MediaResource, TextNode

from ...core.exts import Exts
from ...core.metadata import BasicMetaData
from ...core.metadata import MetaKeys as MK
from ...core.utils import has_media
from ...llama_like.core.schema import AudioNode, VideoNode
from ...logger import logger

__all__ = ["BaseLoader"]


class BaseLoader:
    """Base loader class."""

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

    async def _asplit_docs_modality(
        self, docs: list[Document]
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Split documents by modality.

        Args:
            docs (list[Document]): Input documents.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        self._finalize_docs(docs)

        image_nodes = []
        audio_nodes = []
        video_nodes = []
        text_nodes = []
        for doc in docs:
            if has_media(node=doc, exts=Exts.IMAGE):
                image_nodes.append(
                    ImageNode(
                        text=doc.text,
                        image_path=doc.metadata.get(
                            MK.FILE_PATH
                        ),  # for summarize transform use
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        ref_doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            elif has_media(node=doc, exts=Exts.AUDIO):
                audio_nodes.append(
                    AudioNode(
                        text=doc.text,
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        ref_doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            elif has_media(node=doc, exts=Exts.VIDEO):
                video_nodes.append(
                    VideoNode(
                        text=doc.text,
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        ref_doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            elif isinstance(doc, Document):
                text_nodes.append(
                    TextNode(
                        text=doc.text,
                        id_=doc.id_,
                        doc_id=doc.doc_id,
                        ref_doc_id=doc.doc_id,
                        metadata=doc.metadata,
                    )
                )
            else:
                logger.warning(f"unexpected node type {type(doc)}, skipped")

        return text_nodes, image_nodes, audio_nodes, video_nodes
